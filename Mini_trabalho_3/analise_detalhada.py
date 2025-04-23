import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Configuração para melhor visualização
plt.style.use('seaborn-v0_8-whitegrid')
sns.set(font_scale=1.2)

# Função para garantir diretório de gráficos
def garantir_diretorio_graficos():
    diretorio_graficos = os.path.join(os.path.dirname(__file__), 'graficos_analise')
    if not os.path.exists(diretorio_graficos):
        os.makedirs(diretorio_graficos)
    return diretorio_graficos

# Carrega o dataset
df = pd.read_csv("conjunto_de_dados_limpos/Campeonato_Brasileiro_de_futebol_limpo.csv")

# Cria coluna com resultado da partida se não existir
if 'resultado' not in df.columns:
    df['resultado'] = df.apply(
        lambda row: 'vitória_mandante' if row['mandante_placar'] > row['visitante_placar']
        else ('vitória_visitante' if row['mandante_placar'] < row['visitante_placar']
            else 'empate'),
        axis=1
    )

# Cria coluna com total de gols por partida se não existir
if 'gols_total' not in df.columns:
    df['gols_total'] = df['mandante_placar'] + df['visitante_placar']

# Extrai o ano da data se não existir
if 'ano' not in df.columns:
    df['ano'] = pd.to_datetime(df['data'], errors='coerce').dt.year

# Diretório para salvar os gráficos
diretorio_graficos = garantir_diretorio_graficos()

print(f"Total de partidas no dataset: {len(df)}")
print(f"Período analisado: {df['ano'].min()} a {df['ano'].max()}")

# 1. ANÁLISE DA VANTAGEM DE JOGAR EM CASA
print("\n=== VANTAGEM DE JOGAR EM CASA ===")
resultados_count = df['resultado'].value_counts()
total_jogos = len(df)
print(f"Vitórias do mandante: {resultados_count.get('vitória_mandante', 0)} ({resultados_count.get('vitória_mandante', 0)/total_jogos*100:.2f}%)")
print(f"Empates: {resultados_count.get('empate', 0)} ({resultados_count.get('empate', 0)/total_jogos*100:.2f}%)")
print(f"Vitórias do visitante: {resultados_count.get('vitória_visitante', 0)} ({resultados_count.get('vitória_visitante', 0)/total_jogos*100:.2f}%)")

# Gráfico de pizza para resultados
plt.figure(figsize=(10, 7))
plt.pie(resultados_count, labels=resultados_count.index, autopct='%1.1f%%', 
        colors=['#4CAF50', '#FFC107', '#F44336'], startangle=90)
plt.title('Distribuição de Resultados no Campeonato Brasileiro (2003-2023)')
plt.axis('equal')
plt.tight_layout()
plt.savefig(os.path.join(diretorio_graficos, 'distribuicao_resultados_pizza.png'), dpi=300, bbox_inches='tight')

# 2. TENDÊNCIA DE GOLS AO LONGO DOS ANOS
print("\n=== TENDÊNCIA DE GOLS AO LONGO DOS ANOS ===")
gols_por_ano = df.groupby('ano').agg({
    'mandante_placar': 'sum',
    'visitante_placar': 'sum',
    'gols_total': 'mean'
}).reset_index()

print(gols_por_ano)

plt.figure(figsize=(14, 7))
plt.plot(gols_por_ano['ano'], gols_por_ano['gols_total'], marker='o', linewidth=2, color='#1976D2')
plt.title('Média de Gols por Partida ao Longo dos Anos')
plt.xlabel('Ano')
plt.ylabel('Média de Gols por Partida')
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(gols_por_ano['ano'], rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(diretorio_graficos, 'media_gols_por_ano.png'), dpi=300, bbox_inches='tight')

# 3. CLUBES COM DESEMPENHO FORA DA MÉDIA (OUTLIERS)
print("\n=== CLUBES COM DESEMPENHO FORA DA MÉDIA ===")

# Aproveitamento geral dos times (considerando apenas times com pelo menos 30 jogos)
times_mandante = df.groupby('mandante').agg({
    'resultado': lambda x: (x == 'vitória_mandante').mean() * 100,
    'mandante': 'count'
}).rename(columns={'resultado': 'aproveitamento_mandante', 'mandante': 'jogos_mandante'})

times_visitante = df.groupby('visitante').agg({
    'resultado': lambda x: (x == 'vitória_visitante').mean() * 100,
    'visitante': 'count'
}).rename(columns={'resultado': 'aproveitamento_visitante', 'visitante': 'jogos_visitante'})

# Mescla os dados de mandante e visitante
aproveitamento_times = pd.merge(times_mandante, times_visitante, 
                               left_index=True, right_index=True)

# Calcula o aproveitamento geral (média ponderada pelo número de jogos)
aproveitamento_times['aproveitamento_geral'] = (
    (aproveitamento_times['aproveitamento_mandante'] * aproveitamento_times['jogos_mandante'] + 
     aproveitamento_times['aproveitamento_visitante'] * aproveitamento_times['jogos_visitante']) / 
    (aproveitamento_times['jogos_mandante'] + aproveitamento_times['jogos_visitante'])
)

aproveitamento_times['total_jogos'] = aproveitamento_times['jogos_mandante'] + aproveitamento_times['jogos_visitante']

# Filtra apenas times com pelo menos 30 jogos
aproveitamento_filtrado = aproveitamento_times[aproveitamento_times['total_jogos'] >= 30].sort_values('aproveitamento_geral', ascending=False)

print("Top 10 times com melhor aproveitamento geral:")
print(aproveitamento_filtrado[['aproveitamento_mandante', 'aproveitamento_visitante', 'aproveitamento_geral', 'total_jogos']].head(10))

print("\nTimes com pior aproveitamento geral:")
print(aproveitamento_filtrado[['aproveitamento_mandante', 'aproveitamento_visitante', 'aproveitamento_geral', 'total_jogos']].tail(10))

# Identificação de outliers usando o método IQR
Q1 = aproveitamento_filtrado['aproveitamento_geral'].quantile(0.25)
Q3 = aproveitamento_filtrado['aproveitamento_geral'].quantile(0.75)
IQR = Q3 - Q1

outliers_superior = aproveitamento_filtrado[aproveitamento_filtrado['aproveitamento_geral'] > Q3 + 1.5 * IQR]
outliers_inferior = aproveitamento_filtrado[aproveitamento_filtrado['aproveitamento_geral'] < Q1 - 1.5 * IQR]

print("\nOutliers superiores (times com desempenho muito acima da média):")
if not outliers_superior.empty:
    print(outliers_superior[['aproveitamento_geral', 'total_jogos']])
else:
    print("Nenhum outlier superior identificado pelo método IQR")

print("\nOutliers inferiores (times com desempenho muito abaixo da média):")
if not outliers_inferior.empty:
    print(outliers_inferior[['aproveitamento_geral', 'total_jogos']])
else:
    print("Nenhum outlier inferior identificado pelo método IQR")

# Gráfico de boxplot para visualizar outliers
plt.figure(figsize=(12, 8))
sns.boxplot(y=aproveitamento_filtrado['aproveitamento_geral'])
plt.title('Distribuição do Aproveitamento Geral dos Times (Boxplot)')
plt.ylabel('Aproveitamento Geral (%)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(diretorio_graficos, 'boxplot_aproveitamento.png'), dpi=300, bbox_inches='tight')

# 4. MUDANÇAS ESTRUTURAIS AO LONGO DO TEMPO
print("\n=== MUDANÇAS ESTRUTURAIS AO LONGO DO TEMPO ===")

# Número de times por temporada
times_por_temporada = df.groupby('ano').agg({
    'mandante': lambda x: len(x.unique()),
    'id': 'count'
}).rename(columns={'mandante': 'num_times', 'id': 'num_partidas'})

print("Número de times e partidas por temporada:")
print(times_por_temporada)

# Vantagem do mandante ao longo do tempo
vantagem_mandante_anual = df.groupby('ano')['resultado'].apply(
    lambda x: (x == 'vitória_mandante').mean() * 100
).reset_index()
vantagem_mandante_anual.columns = ['ano', 'percentual_vitorias_mandante']

print("\nPercentual de vitórias do mandante ao longo dos anos:")
print(vantagem_mandante_anual)

plt.figure(figsize=(14, 7))
plt.plot(vantagem_mandante_anual['ano'], vantagem_mandante_anual['percentual_vitorias_mandante'], 
         marker='o', linewidth=2, color='#4CAF50')
plt.axhline(y=vantagem_mandante_anual['percentual_vitorias_mandante'].mean(), color='r', linestyle='--', 
           label=f'Média: {vantagem_mandante_anual["percentual_vitorias_mandante"].mean():.1f}%')
plt.title('Percentual de Vitórias do Mandante ao Longo dos Anos')
plt.xlabel('Ano')
plt.ylabel('Vitórias do Mandante (%)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(vantagem_mandante_anual['ano'], rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(diretorio_graficos, 'vantagem_mandante_anual.png'), dpi=300, bbox_inches='tight')

# 5. QUALIDADE DOS DADOS
print("\n=== QUALIDADE DOS DADOS ===")

# Verificar valores ausentes
valores_ausentes = df.isnull().sum()
print("Valores ausentes por coluna:")
print(valores_ausentes[valores_ausentes > 0])

# Verificar consistência de nomes de times
times_mandante = set(df['mandante'].unique())
times_visitante = set(df['visitante'].unique())
times_diferentes = times_mandante.symmetric_difference(times_visitante)

print(f"\nTotal de times únicos como mandante: {len(times_mandante)}")
print(f"Total de times únicos como visitante: {len(times_visitante)}")
if times_diferentes:
    print(f"Times que aparecem apenas como mandante ou apenas como visitante: {times_diferentes}")

# Estatísticas gerais do dataset
print("\nEstatísticas gerais do dataset:")
print(df.describe())

print("\nAnálise concluída! Os gráficos foram salvos no diretório:", diretorio_graficos)
