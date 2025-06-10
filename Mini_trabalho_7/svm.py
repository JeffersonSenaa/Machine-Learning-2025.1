import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Carregando e preparando os dados
file_path = 'conjunto_de_dados_limpos/Campeonato_Brasileiro_de_futebol_limpo.csv'
if not os.path.exists(file_path):
    raise FileNotFoundError(f'O arquivo {file_path} não foi encontrado.')

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
df['ano'] = df['data'].dt.year

# Separando treino e teste
treino = df[df['ano'] < 2023].copy()
teste = df[df['ano'] == 2023].copy()

# Encoding dos times
le_mandante = LabelEncoder()
le_visitante = LabelEncoder()
treino['mandante_le'] = le_mandante.fit_transform(treino['mandante'])
treino['visitante_le'] = le_visitante.fit_transform(treino['visitante'])
teste['mandante_le'] = le_mandante.transform(teste['mandante'])
teste['visitante_le'] = le_visitante.transform(teste['visitante'])

# Features
features = ['mandante_le', 'visitante_le']
X_train = treino[features]
y_train = treino['resultado']
X_test = teste[features]
y_test = teste['resultado']

# Escalonamento
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Otimização dos hiperparâmetros
param_grid = {
    'C': [1, 10],
    'gamma': ['scale'],
    'kernel': ['rbf']
}

svc = SVC(class_weight='balanced', probability=False)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(svc, param_grid, cv=skf, scoring='f1_macro', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

print("\nMelhores hiperparâmetros encontrados:")
print(grid_search.best_params_)

# Treinamento do modelo final com os melhores hiperparâmetros
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_scaled)

# Métricas do modelo
print("\nMétricas do Modelo:")
print(classification_report(y_test, y_pred, zero_division=0))

# Cálculo das métricas com 4 casas decimais
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

print("\nMétricas Detalhadas:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# Matriz de Confusão
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=best_model.classes_,
            yticklabels=best_model.classes_)
plt.title('Matriz de Confusão')
plt.xlabel('Previsto')
plt.ylabel('Real')
plt.tight_layout()
plt.savefig('matriz_confusao.png')
plt.close()

# Resultados da Validação Cruzada
print("\n=== Resultados da Validação Cruzada ===")
cv_results = pd.DataFrame(grid_search.cv_results_)
print("\nMédia e Desvio Padrão do F1-Score nos folds:")
print(f"Média: {cv_results['mean_test_score'].mean():.4f}")
print(f"Desvio Padrão: {cv_results['mean_test_score'].std():.4f}")

# Salvando resultados em um arquivo
with open('resultados_svm.txt', 'w') as f:
    f.write("=== Resultados do Modelo SVM ===\n\n")
    f.write("Melhores hiperparâmetros:\n")
    f.write(str(grid_search.best_params_) + "\n\n")
    f.write("Métricas do Modelo:\n")
    f.write(classification_report(y_test, y_pred, zero_division=0))
    f.write("\nResultados da Validação Cruzada:\n")
    f.write(f"Média F1-Score: {cv_results['mean_test_score'].mean():.4f}\n")
    f.write(f"Desvio Padrão F1-Score: {cv_results['mean_test_score'].std():.4f}\n") 