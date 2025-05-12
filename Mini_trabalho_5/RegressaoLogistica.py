import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Caminho para o arquivo de dados
file_path = '/home/pedro/Machine-Learning-2025.1/Mini_trabalho_5/conjunto_de_dados_limpos/Campeonato_Brasileiro_de_futebol_limpo.csv'

if not os.path.exists(file_path):
    raise FileNotFoundError(f'O arquivo {file_path} não foi encontrado. Verifique o caminho e tente novamente.')

# Carregamento dos dados
df = pd.read_csv(file_path)

# Conversão da coluna de data para o formato datetime
df['data'] = pd.to_datetime(df['data'])

# Função para definir o resultado da partida na perspectiva do time mandante
def define_resultado(row):
    if row['mandante_placar'] > row['visitante_placar']:
        return 'vitoria'
    elif row['mandante_placar'] < row['visitante_placar']:
        return 'derrota'
    else:
        return 'empate'
        
# Aplicação da função para criar a coluna 'resultado'
df['resultado'] = df.apply(define_resultado, axis=1)

# Divisão dos dados em treino e teste por ano
df['ano'] = df['data'].dt.year
treino = df[df['ano'] < 2023]  # Dados até 2022 para treino
teste = df[df['ano'] == 2023]  # Dados de 2023 para teste

# Codificação dos times usando LabelEncoder
le_mandante = LabelEncoder()
le_visitante = LabelEncoder()

treino = treino.copy()
teste = teste.copy()

treino.loc[:, 'mandante_le'] = le_mandante.fit_transform(treino['mandante'])
treino.loc[:, 'visitante_le'] = le_visitante.fit_transform(treino['visitante'])

teste.loc[:, 'mandante_le'] = le_mandante.transform(teste['mandante'])
teste.loc[:, 'visitante_le'] = le_visitante.transform(teste['visitante'])

# Definição das features e target
# Adicionando mais features para melhorar o modelo
features = ['mandante_le', 'visitante_le', 'rodata']

# Calculando estatísticas por time para usar como features adicionais
# Média de gols marcados e sofridos por time mandante
mandante_stats = treino.groupby('mandante').agg({
    'mandante_placar': 'mean',
    'visitante_placar': 'mean'
}).reset_index()
mandante_stats.columns = ['mandante', 'mandante_media_gols_pro', 'mandante_media_gols_contra']

# Média de gols marcados e sofridos por time visitante
visitante_stats = treino.groupby('visitante').agg({
    'visitante_placar': 'mean',
    'mandante_placar': 'mean'
}).reset_index()
visitante_stats.columns = ['visitante', 'visitante_media_gols_pro', 'visitante_media_gols_contra']

# Adicionando as estatísticas ao conjunto de treino e teste
treino = treino.merge(mandante_stats, on='mandante', how='left')
treino = treino.merge(visitante_stats, on='visitante', how='left')
teste = teste.merge(mandante_stats, on='mandante', how='left')
teste = teste.merge(visitante_stats, on='visitante', how='left')

# Preenchendo valores NaN (times que não aparecem no conjunto de treino)
# Usando apenas as colunas numéricas relevantes
for col in ['mandante_media_gols_pro', 'mandante_media_gols_contra', 'visitante_media_gols_pro', 'visitante_media_gols_contra']:
    if treino[col].isna().any():
        treino[col].fillna(treino[col].mean(), inplace=True)
    if teste[col].isna().any():
        teste[col].fillna(treino[col].mean(), inplace=True)

# Atualizando as features para incluir as novas estatísticas
features = ['mandante_le', 'visitante_le', 'rodata', 
           'mandante_media_gols_pro', 'mandante_media_gols_contra',
           'visitante_media_gols_pro', 'visitante_media_gols_contra']

X_train = treino[features]
y_train = treino['resultado']
X_test = teste[features]
y_test = teste['resultado']

# Criação e treinamento do modelo de Regressão Logística
# Utilizando o solver 'saga' que funciona bem para problemas multiclasse e permite regularização
# Ajustando C para controlar a regularização e class_weight para lidar com o desbalanceamento
model = LogisticRegression(max_iter=2000, random_state=42, solver='saga', 
                          multi_class='multinomial', class_weight='balanced',
                          C=0.5, penalty='l1')
model.fit(X_train, y_train)

# Resultados das previsões em 15 jogos (amostra)
df_sample = teste.head(15).copy()
X_sample = df_sample[features]
y_sample = df_sample['resultado']
y_pred_sample = model.predict(X_sample)

# Criação de um DataFrame para mostrar os resultados da amostra
df_resultados = df_sample[['mandante', 'visitante', 'mandante_placar', 'visitante_placar']].copy()
df_resultados['real'] = y_sample.values
df_resultados['previsto'] = y_pred_sample
print('\nResultados das Previsões (15 jogos):')
print(df_resultados.to_string(index=False))

# Cálculo de acertos na amostra
acertos_sample = (y_pred_sample == y_sample.values).sum()
total_sample = len(y_sample)
print(f"\nPrevisões acertadas na amostra (15 jogos): {acertos_sample} de {total_sample} ({acertos_sample/total_sample:.2%})\n")

# Previsões no conjunto de teste completo
y_pred = model.predict(X_test)

# Relatório de classificação
print(classification_report(y_test, y_pred))

# Matriz de confusão
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel('Previsto')
plt.ylabel('Real')
plt.title('Matriz de Confusão - Regressão Logística')
plt.show()

# Cálculo de acertos no conjunto de teste completo
acertos_teste = (y_pred == y_test.values).sum()
total_teste = len(y_test)
print(f"\nPrevisões acertadas no conjunto de teste: {acertos_teste} de {total_teste} ({acertos_teste/total_teste:.2%})")

# Métricas adicionais
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("\nMétricas Adicionais:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")


