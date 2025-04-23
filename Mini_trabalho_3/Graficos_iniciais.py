import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Carrega o dataset
df = pd.read_csv("C:/Users/Pichau/Documents/comecouOstrabalhos/MiniTrabalho-3-Grupo-5/conjunto_de_dados_limpos/Campeonato_Brasileiro_de_futebol_limpo.csv")

# Cria coluna com resultado da partida
df['resultado'] = df.apply(
    lambda row: 'vitória_mandante' if row['mandante_placar'] > row['visitante_placar']
    else ('vitória_visitante' if row['mandante_placar'] < row['visitante_placar']
          else 'empate'),
    axis=1
)

# Cria coluna com total de gols por partida;
df['gols_total'] = df['mandante_placar'] + df['visitante_placar']

# Configura o estilo dos gráficos
sns.set(style="whitegrid")

# 1. Gráfico de barras: Frequência de resultados
plt.figure(figsize=(10, 6))  # Cria uma nova figura
sns.countplot(x='resultado', hue='resultado', data=df, palette='Set2', legend=False)
plt.title('Frequência de Resultados')
plt.xlabel('Resultado')
plt.ylabel('Número de Partidas')
plt.show()  # Exibe o gráfico

# 2. Gráfico de barras: Distribuição de gols por partida
plt.figure(figsize=(10, 6))  # Cria uma nova figura
# Conta quantas partidas tiveram cada total de gols
gols_count = df['gols_total'].value_counts().sort_index()
sns.barplot(x=gols_count.index, y=gols_count.values, color='skyblue')
plt.title('Distribuição de Gols por Partida')
plt.xlabel('Total de Gols')
plt.ylabel('Frequência')
plt.show()  # Exibe o gráfico

# 3. Gráfico de barras: Desempenho mandante vs visitante (Top 10)
plt.figure(figsize=(10, 6))  # Cria uma nova figura
mandante_vitorias = df[df['resultado'] == 'vitória_mandante']['mandante'].value_counts()
visitante_vitorias = df[df['resultado'] == 'vitória_visitante']['visitante'].value_counts()

desempenho_df = pd.DataFrame({
    'Mandante': mandante_vitorias,
    'Visitante': visitante_vitorias
}).fillna(0).astype(int).sort_values(by='Mandante', ascending=False).head(25)

desempenho_df.plot(kind='bar', ax=plt.gca(), color=['#4CAF50', '#FFC107'])
plt.title('Top 10 Times - Vitórias Mandante vs Visitante')
plt.xlabel('Time')
plt.ylabel('Número de Vitórias')
plt.xticks(rotation=45)
plt.show()  # Exibe o gráfico

