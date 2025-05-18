import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

file_path = 'conjunto_de_dados_limpos/Campeonato_Brasileiro_de_futebol_limpo.csv'

if not os.path.exists(file_path):
    raise FileNotFoundError(f'O arquivo {file_path} não foi encontrado. Verifique o caminho e tente novamente.')

df = pd.read_csv(file_path)

df['data'] = pd.to_datetime(df['data'])

def define_resultado(row):
    if row['mandante_placar'] > row['visitante_placar']:
        return 'vitoria'
    elif row['mandante_placar'] < row['visitante_placar']:
        return 'derrota'
    else:
        return 'empate'
        
df['resultado'] = df.apply(define_resultado, axis=1)

df = df.sort_values(by='data')

def calcular_medias_moveis_gols(dataframe, n_jogos=5):
    """
    Calcula a média móvel de gols marcados e sofridos por cada time nos últimos n_jogos.
    
    Args:
        dataframe: DataFrame com os jogos
        n_jogos: Número de jogos anteriores para calcular a média
        
    Returns:
        DataFrame com as novas features adicionadas
    """
    historico_gols_marcados = {}
    historico_gols_sofridos = {}
    
    dataframe['mandante_media_gols_marcados'] = 0.0
    dataframe['mandante_media_gols_sofridos'] = 0.0
    dataframe['visitante_media_gols_marcados'] = 0.0
    dataframe['visitante_media_gols_sofridos'] = 0.0
    
    for idx, row in dataframe.iterrows():
        time_mandante = row['mandante']
        time_visitante = row['visitante']
        gols_mandante = row['mandante_placar']
        gols_visitante = row['visitante_placar']
        
        if time_mandante not in historico_gols_marcados:
            historico_gols_marcados[time_mandante] = []
            historico_gols_sofridos[time_mandante] = []
        
        if time_visitante not in historico_gols_marcados:
            historico_gols_marcados[time_visitante] = []
            historico_gols_sofridos[time_visitante] = []
        
        if len(historico_gols_marcados[time_mandante]) > 0:
            ultimos_n_gols_marcados = historico_gols_marcados[time_mandante][-n_jogos:]
            ultimos_n_gols_sofridos = historico_gols_sofridos[time_mandante][-n_jogos:]
            
            dataframe.at[idx, 'mandante_media_gols_marcados'] = sum(ultimos_n_gols_marcados) / len(ultimos_n_gols_marcados)
            dataframe.at[idx, 'mandante_media_gols_sofridos'] = sum(ultimos_n_gols_sofridos) / len(ultimos_n_gols_sofridos)
        
        if len(historico_gols_marcados[time_visitante]) > 0:
            ultimos_n_gols_marcados = historico_gols_marcados[time_visitante][-n_jogos:]
            ultimos_n_gols_sofridos = historico_gols_sofridos[time_visitante][-n_jogos:]
            
            dataframe.at[idx, 'visitante_media_gols_marcados'] = sum(ultimos_n_gols_marcados) / len(ultimos_n_gols_marcados)
            dataframe.at[idx, 'visitante_media_gols_sofridos'] = sum(ultimos_n_gols_sofridos) / len(ultimos_n_gols_sofridos)
        
        historico_gols_marcados[time_mandante].append(gols_mandante)
        historico_gols_sofridos[time_mandante].append(gols_visitante)
        
        historico_gols_marcados[time_visitante].append(gols_visitante)
        historico_gols_sofridos[time_visitante].append(gols_mandante)
    
    return dataframe

df = calcular_medias_moveis_gols(df, n_jogos=5)

df['ano'] = df['data'].dt.year
treino = df[df['ano'] < 2023]
teste = df[df['ano'] == 2023]

le_mandante = LabelEncoder()
le_visitante = LabelEncoder()

treino = treino.copy()
teste = teste.copy()

treino.loc[:, 'mandante_le'] = le_mandante.fit_transform(treino['mandante'])
treino.loc[:, 'visitante_le'] = le_visitante.fit_transform(treino['visitante'])

teste.loc[:, 'mandante_le'] = le_mandante.transform(teste['mandante'])
teste.loc[:, 'visitante_le'] = le_visitante.transform(teste['visitante'])

features = [
    'mandante_le', 'visitante_le', 'rodata',
    'mandante_media_gols_marcados', 'mandante_media_gols_sofridos',
    'visitante_media_gols_marcados', 'visitante_media_gols_sofridos'
]

X_train = treino[features]
y_train = treino['resultado']
X_test = teste[features]
y_test = teste['resultado']

model = RandomForestClassifier(n_estimators=100, random_state=42)

cv = KFold(n_splits=5, shuffle=True, random_state=42)

cv_accuracy = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
cv_precision = cross_val_score(model, X_train, y_train, cv=cv, scoring='precision_weighted')
cv_recall = cross_val_score(model, X_train, y_train, cv=cv, scoring='recall_weighted')
cv_f1 = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1_weighted')


print("\nResultados da Validação Cruzada (5-fold):")
print(f"Accuracy: {cv_accuracy.mean():.4f}")
print(f"Precision: {cv_precision.mean():.4f}")
print(f"Recall: {cv_recall.mean():.4f}")
print(f"F1-Score: {cv_f1.mean():.4f}\n")

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel('Previsto')
plt.ylabel('Real')
plt.title('Matriz de Confusão')
plt.show()

print("\nRealizando calibração de hiperparâmetros...")

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,  
    scoring='f1_weighted',
    verbose=1
)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

print("\nResultados do modelo calibrado:\n")
print(classification_report(y_test, y_pred))

'''
print("\nImportância das Features:")
feature_importance = pd.DataFrame({'Feature': features, 'Importance': best_model.feature_importances_})
feature_importance = feature_importance.sort_values('Importance', ascending=False)
print(feature_importance)
'''

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=best_model.classes_, yticklabels=best_model.classes_)
plt.xlabel('Previsto')
plt.ylabel('Real')
plt.title('Matriz de Confusão (Modelo Calibrado)')
plt.show()

acertos_teste = (y_pred == y_test.values).sum()
total_teste = len(y_test)
print(f"\nPrevisões acertadas no conjunto de teste (modelo calibrado): {acertos_teste} de {total_teste} ({acertos_teste/total_teste:.2%})")

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("\nMétricas do Modelo:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}\n")