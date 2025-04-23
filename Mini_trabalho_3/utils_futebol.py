import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(filepath="C:/Users/Pichau/Documents/comecouOstrabalhos/MiniTrabalho-3-Grupo-5/conjunto_de_dados_limpos/Campeonato_Brasileiro_de_futebol_limpo.csv"):
    """
    Carrega e prepara os dados do Campeonato Brasileiro
    """
    # Carrega o dataset
    df = pd.read_csv(filepath)
    
    # Cria coluna com resultado da partida
    df['resultado'] = df.apply(
        lambda row: 'vitória_mandante' if row['mandante_placar'] > row['visitante_placar']
        else ('vitória_visitante' if row['mandante_placar'] < row['visitante_placar']
              else 'empate'),
        axis=1
    )
    
    # Cria coluna com total de gols por partida
    df['gols_total'] = df['mandante_placar'] + df['visitante_placar']
    
    # Extrair o ano da coluna de data
    if 'ano' not in df.columns:
        df['ano'] = pd.to_datetime(df['data'], errors='coerce').dt.year
    
    # Configuração padrão para os gráficos
    sns.set(style="whitegrid")
    
    return df

def calcular_aproveitamento(df, time_col, resultado_vitoria, group_cols):
    """
    Calcula o aproveitamento (percentual de vitórias) de times
    
    Parâmetros:
    df - DataFrame com os dados
    time_col - Nome da coluna que contém o nome do time
    resultado_vitoria - Valor na coluna 'resultado' que indica vitória
    group_cols - Lista de colunas para agrupar (geralmente [time_col, 'ano'])
    """
    # Seleciona apenas jogos do time específico
    time_data = df[df[time_col].notna()]
    
    # Agrupa por time e ano, contando vitórias
    vitorias = time_data[time_data['resultado'] == resultado_vitoria].groupby(group_cols).size()
    
    # Conta total de jogos por time e ano
    total_jogos = time_data.groupby(group_cols).size()
    
    # Calcula aproveitamento (percentual de vitórias)
    aproveitamento = (vitorias / total_jogos * 100).reset_index()
    aproveitamento.columns = group_cols + ['aproveitamento']
    
    return aproveitamento

def get_regioes_dict():
    """
    Retorna dicionário de mapeamento de estados para regiões
    """
    return {
        'SP': 'Sudeste', 'RJ': 'Sudeste', 'MG': 'Sudeste', 'ES': 'Sudeste',
        'RS': 'Sul', 'PR': 'Sul', 'SC': 'Sul',
        'DF': 'Centro-Oeste', 'GO': 'Centro-Oeste', 'MT': 'Centro-Oeste', 'MS': 'Centro-Oeste',
        'BA': 'Nordeste', 'PE': 'Nordeste', 'CE': 'Nordeste', 'AL': 'Nordeste', 'PB': 'Nordeste', 
        'RN': 'Nordeste', 'SE': 'Nordeste', 'PI': 'Nordeste', 'MA': 'Nordeste',
        'PA': 'Norte', 'AM': 'Norte', 'RO': 'Norte', 'AC': 'Norte', 'AP': 'Norte', 'RR': 'Norte', 'TO': 'Norte'
    }
