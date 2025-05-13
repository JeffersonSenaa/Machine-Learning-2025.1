import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import os
import matplotlib.pyplot as plt
import seaborn as sns

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

# Normalização dos dados
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Criando e treinando o modelo SVM com ajustes para lidar com classes desbalanceadas
model = SVC(
    kernel='rbf',
    probability=True,
    random_state=42,
    class_weight='balanced',  
    C=1.0,  # Parâmetro de regularização
    gamma='scale'  # Escala automática do parâmetro gamma
)
model.fit(X_train_scaled, y_train)

# Resultados das previsões em 15 jogos
df_sample = teste.head(15).copy()
X_sample = df_sample[features]
X_sample_scaled = scaler.transform(X_sample)
y_sample = df_sample['resultado']
y_pred_sample = model.predict(X_sample_scaled)

df_resultados = df_sample[['mandante', 'visitante', 'mandante_placar', 'visitante_placar']].copy()
df_resultados['real'] = y_sample.values
df_resultados['previsto'] = y_pred_sample
print('\nResultados das Previsões (15 jogos):')
print(df_resultados.to_string(index=False))

# Acertos na amostra de 15 jogos
acertos_sample = (y_pred_sample == y_sample.values).sum()
total_sample = len(y_sample)
print(f"\nPrevisões acertadas na amostra (15 jogos): {acertos_sample} de {total_sample} ({acertos_sample/total_sample:.2%})\n")

# Avaliação completa do modelo
y_pred = model.predict(X_test_scaled)
print("Relatório de Classificação:")
print(classification_report(y_test, y_pred, zero_division=0))

# Matriz de Confusão
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel('Previsto')
plt.ylabel('Real')
plt.title('Matriz de Confusão - SVM')
plt.show()

# Métricas gerais
acertos_teste = (y_pred == y_test.values).sum()
total_teste = len(y_test)
print(f"\nPrevisões acertadas no conjunto de teste: {acertos_teste} de {total_teste} ({acertos_teste/total_teste:.2%})")

# Métricas detalhadas com zero_division=0
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

print("\nMétricas Detalhadas:")
print(f"Accuracy (Precisão Geral): {accuracy:.4f}")
print(f"Precision (Macro): {precision:.4f}")
print(f"Recall (Macro): {recall:.4f}")
print(f"F1-Score (Macro): {f1:.4f}\n") 