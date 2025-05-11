import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import os
import matplotlib.pyplot as plt
import seaborn as sns

file_path = 'C://Users//Usuario//Desktop//Mini_trabalho_4//conjunto_de_dados_limpos//Campeonato_Brasileiro_de_futebol_limpo.csv'

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

features = ['mandante_le', 'visitante_le', 'rodata']

X_train = treino[features]
y_train = treino['resultado']
X_test = teste[features]
y_test = teste['resultado']

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Resultados das previsões em 15 jogos
df_sample = teste.head(15).copy()
X_sample = df_sample[features]
y_sample = df_sample['resultado']
y_pred_sample = model.predict(X_sample)

df_resultados = df_sample[['mandante', 'visitante', 'mandante_placar', 'visitante_placar']].copy()
df_resultados['real'] = y_sample.values
df_resultados['previsto'] = y_pred_sample
print('\nResultados das Previsões (15 jogos):')
print(df_resultados.to_string(index=False))

# Acertos na amostra de 15 jogos
acertos_sample = (y_pred_sample == y_sample.values).sum()
total_sample = len(y_sample)
print(f"\nPrevisões acertadas na amostra (15 jogos): {acertos_sample} de {total_sample} ({acertos_sample/total_sample:.2%})\n")

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

acertos_teste = (y_pred == y_test.values).sum()
total_teste = len(y_test)
print(f"\nPrevisões acertadas no conjunto de teste: {acertos_teste} de {total_teste} ({acertos_teste/total_teste:.2%})")

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("\nMétricas Adicionais:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}\n")
