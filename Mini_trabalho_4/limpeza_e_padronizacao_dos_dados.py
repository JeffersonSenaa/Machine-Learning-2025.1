import pandas as pd
from sklearn.preprocessing import StandardScaler

# Arquivos CSV de entrada e saída
arquivos = {
    'conjunto_de_dados/Brazilian_Soccer_Database.csv': 'conjunto_de_dados_limpos/Brazilian_Soccer_Database_limpo.csv',
    'conjunto_de_dados/Brazilian Soccer_Brasileirão_Brasileirao.csv': 'conjunto_de_dados_limpos/Brazilian Soccer_Brasileirâo_Brasileirao_limpo.csv',
    'conjunto_de_dados/Campeonato_Brasileiro_de_futebol.csv': 'conjunto_de_dados_limpos/Campeonato_Brasileiro_de_futebol_limpo.csv'
}

# Dicionário de padronização dos nomes de times
mapa_times = {
    'América-MG': 'America-MG', 'America-MG': 'America-MG',
    'América-RN': 'America-RN', 'America-RN': 'America-RN',
    'Atlético-MG': 'Atletico-MG', 'Atletico-MG': 'Atletico-MG',
    'Atlético-GO': 'Atletico-GO', 'Atletico-GO': 'Atletico-GO',
    'Athletico-PR': 'Athletico-PR', 'Atletico-PR': 'Athletico-PR',
    'Avaí': 'Avai', 'Avai': 'Avai', 'Avai-SC': 'Avai',
    'Bahia': 'Bahia', 'Bahia-BA': 'Bahia',
    'Botafogo': 'Botafogo', 'Botafogo-RJ': 'Botafogo',
    'Bragantino': 'Bragantino', 'Red Bull Bragantino-SP': 'Bragantino',
    'Ceará': 'Ceara', 'Ceara': 'Ceara', 'Ceara-CE': 'Ceara',
    'Chapecoense': 'Chapecoense', 'Chapecoense-SC': 'Chapecoense',
    'Corinthians': 'Corinthians', 'Corinthians-SP': 'Corinthians',
    'Coritiba': 'Coritiba', 'Coritiba-PR': 'Coritiba',
    'Criciúma': 'Criciuma', 'Criciuma': 'Criciuma', 'Criciuma-SC': 'Criciuma',
    'Cruzeiro': 'Cruzeiro', 'Cruzeiro-MG': 'Cruzeiro',
    'CSA': 'CSA', 'Csa-AL': 'CSA',
    'Cuiabá': 'Cuiaba', 'Cuiaba': 'Cuiaba', 'Cuiaba-MT': 'Cuiaba',
    'Figueirense': 'Figueirense', 'Figueirense-SC': 'Figueirense',
    'Flamengo': 'Flamengo', 'Flamengo-RJ': 'Flamengo',
    'Fluminense': 'Fluminense', 'Fluminense-RJ': 'Fluminense',
    'Fortaleza': 'Fortaleza', 'Fortaleza-CE': 'Fortaleza',
    'Goiás': 'Goias', 'Goias': 'Goias', 'Goias-GO': 'Goias',
    'Grêmio': 'Gremio', 'Gremio': 'Gremio', 'Gremio-RS': 'Gremio',
    'Internacional': 'Internacional', 'Internacional-RS': 'Internacional',
    'Joinville': 'Joinville', 'Joinville-SC': 'Joinville',
    'Juventude': 'Juventude', 'Juventude-RS': 'Juventude',
    'Náutico': 'Nautico', 'Nautico': 'Nautico', 'Nautico-PE': 'Nautico',
    'Palmeiras': 'Palmeiras', 'Palmeiras-SP': 'Palmeiras',
    'Paraná': 'Parana', 'Parana': 'Parana', 'Parana-PR': 'Parana',
    'Ponte Preta': 'Ponte Preta', 'Ponte Preta-SP': 'Ponte Preta',
    'Portuguesa': 'Portuguesa', 'Portuguesa-SP': 'Portuguesa',
    'Santa Cruz': 'Santa Cruz', 'Santa Cruz-PE': 'Santa Cruz',
    'Santo André': 'Santo Andre', 'Santo Andre': 'Santo Andre',
    'Santos': 'Santos', 'Santos-SP': 'Santos',
    'São Paulo': 'Sao Paulo', 'Sao Paulo': 'Sao Paulo', 'Sao Paulo-SP': 'Sao Paulo',
    'São Caetano': 'Sao Caetano',
    'Sport': 'Sport', 'Sport-PE': 'Sport',
    'Vasco': 'Vasco', 'Vasco da Gama-RJ': 'Vasco',
    'Vitória': 'Vitoria', 'Vitoria': 'Vitoria', 'Vitoria-BA': 'Vitoria',
    'Ipatinga': 'Ipatinga', 'Guarani': 'Guarani', 'Paysandu': 'Paysandu',
    'Grêmio Prudente': 'Gremio Prudente', 'Gremio Prudente': 'Gremio Prudente',
    'Barueri': 'Barueri', 'Brasiliense': 'Brasiliense'
}

# Função para padronizar nomes de colunas
def padronizar_colunas(df):
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('-', '_')
    return df

# Função para tratar datas e separar data e hora
def tratar_datas(df, colunas_data):
    for col in colunas_data:
        if col in df.columns:
            # Garantir que a coluna está no formato datetime
            df[col] = pd.to_datetime(df[col], errors='coerce', dayfirst=True)
            
            # Se a coluna for "datetime", vamos separá-la em duas
            if col == 'datetime':
                df['data'] = df[col].dt.date  # Criando a coluna de data
                df['hora'] = df[col].dt.strftime('%H:%M:%S')  # Criando a coluna de hora
                df.drop(columns=['datetime'], inplace=True)  # Remover a coluna original datetime

    return df

# Função para tratar valores ausentes e valores "-"
def tratar_ausentes(df):
    # Substitui os valores "-" por "ausente" em todo o DataFrame
    df = df.replace('-', 'ausente')
    # Preenche os valores NaN com "ausente"
    df = df.fillna('ausente')
    return df

# Função para converter para numérico
def converter_numericos(df, colunas):
    for col in colunas:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
    return df

# Função para padronizar os nomes dos times
def padronizar_times(df):
    colunas_times = ['home_team', 'away_team', 'hometeam', 'visitingteam', 'mandante', 'visitante']
    for col in colunas_times:
        if col in df.columns:
            df[col] = df[col].map(mapa_times).fillna(df[col])
    return df

#""" Função para codificar variáveis categóricas com One-Hot Encoding """
#def codificar_categoricas(df):
#    colunas_categoricas = df.select_dtypes(include=['object']).columns.tolist()
#   
#    colunas_para_codificar = [col for col in colunas_categoricas if df[col].nunique() <= 20]
#    
#    df = pd.get_dummies(df, columns=colunas_para_codificar, drop_first=True)
#    return df

""" Função para padronizar colunas numéricas """
def padronizar_numericas(df, colunas):
    scaler = StandardScaler()
    for col in colunas:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    colunas_validas = [col for col in colunas if col in df.columns and df[col].notna().all()]
    
    if colunas_validas:
        df[colunas_validas] = scaler.fit_transform(df[colunas_validas])
    return df

# Colunas específicas
colunas_ausentes = ['formacao_mandante', 'formacao_visitante', 'tecnico_mandante', 'tecnico_visitante']
colunas_numericas = [
    'rodada', 'placar_mandante', 'placar_visitante', 'temporada',
    'goalsht', 'goalsvt', 'home_goal', 'away_goal'
]
colunas_data = ['data', 'data_jogo', 'data_partida', 'datetime']

# Processamento dos arquivos
for arquivo_entrada, arquivo_saida in arquivos.items():
    df = pd.read_csv(arquivo_entrada)
    df = padronizar_colunas(df)

    # Verifica se há colunas separadas de year, month e day
    if all(col in df.columns for col in ['year', 'month', 'day']):
        df['data_partida'] = pd.to_datetime(dict(year=df['year'], month=df['month'], day=df['day']), errors='coerce')
        df.drop(columns=['year', 'month', 'day'], inplace=True)
        colunas_data.append('data_partida')

    # Pipeline de limpeza e transformação
    df = tratar_datas(df, colunas_data)
    df = tratar_ausentes(df)
    df = converter_numericos(df, colunas_numericas)
    df = padronizar_times(df)
    #df = codificar_categoricas(df) 
    df = padronizar_numericas(df, colunas_numericas) 
    
    df.to_csv(arquivo_saida, index=False)
    print(f'{arquivo_entrada} processado e salvo como {arquivo_saida}')

print("Todos os arquivos foram tratados, padronizados e salvos com sucesso.")
