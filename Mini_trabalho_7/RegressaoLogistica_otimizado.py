import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time

file_path = 'conjunto_de_dados_limpos/Campeonato_Brasileiro_de_futebol_limpo.csv'

if not os.path.exists(file_path):
    raise FileNotFoundError(f'O arquivo {file_path} não foi encontrado. Verifique o caminho e tente novamente.')


print("Carregando os dados...")
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

# Ordenar por data para cálculos temporais
df = df.sort_values(by='data')

# Função para calcular percentual de vitórias, empates e derrotas nos últimos N jogos
def calcular_historico_recente(df, n_jogos=5):
    # Cria cópia dos dados
    dados = df.copy().sort_values('id')
    
    # Inicializa as colunas de histórico
    dados['mandante_pct_vitorias'] = 0.0
    dados['mandante_pct_empates'] = 0.0
    dados['mandante_pct_derrotas'] = 0.0
    dados['visitante_pct_vitorias'] = 0.0
    dados['visitante_pct_empates'] = 0.0
    dados['visitante_pct_derrotas'] = 0.0
    
    # Lista de todos os times únicos
    times = list(set(dados['mandante'].unique()) | set(dados['visitante'].unique()))
    
    for time in times:
        # Pega todos os jogos de cada time (como mandante ou visitante)
        jogos_como_mandante = dados[dados['mandante'] == time].copy()
        jogos_como_visitante = dados[dados['visitante'] == time].copy()
        
        # Para cada jogo do time como mandante
        for idx, jogo in jogos_como_mandante.iterrows():
            # Pega os n_jogos anteriores a este jogo para este time
            jogos_anteriores = dados[(dados['id'] < jogo['id']) & 
                                    ((dados['mandante'] == time) | (dados['visitante'] == time))].tail(n_jogos)
            
            if len(jogos_anteriores) > 0:
                # Calcula resultados para este time
                vitorias = empates = derrotas = 0
                
                for _, j_anterior in jogos_anteriores.iterrows():
                    if j_anterior['mandante'] == time:
                        if j_anterior['mandante_placar'] > j_anterior['visitante_placar']:
                            vitorias += 1
                        elif j_anterior['mandante_placar'] == j_anterior['visitante_placar']:
                            empates += 1
                        else:
                            derrotas += 1
                    else:  # time é visitante
                        if j_anterior['visitante_placar'] > j_anterior['mandante_placar']:
                            vitorias += 1
                        elif j_anterior['visitante_placar'] == j_anterior['mandante_placar']:
                            empates += 1
                        else:
                            derrotas += 1
                
                # Atualiza os percentuais
                total_jogos = len(jogos_anteriores)
                dados.at[idx, 'mandante_pct_vitorias'] = vitorias / total_jogos
                dados.at[idx, 'mandante_pct_empates'] = empates / total_jogos
                dados.at[idx, 'mandante_pct_derrotas'] = derrotas / total_jogos
        
        # Para cada jogo do time como visitante
        for idx, jogo in jogos_como_visitante.iterrows():
            # Pega os n_jogos anteriores a este jogo para este time
            jogos_anteriores = dados[(dados['id'] < jogo['id']) & 
                                    ((dados['mandante'] == time) | (dados['visitante'] == time))].tail(n_jogos)
            
            if len(jogos_anteriores) > 0:
                # Calcula resultados para este time
                vitorias = empates = derrotas = 0
                
                for _, j_anterior in jogos_anteriores.iterrows():
                    if j_anterior['mandante'] == time:
                        if j_anterior['mandante_placar'] > j_anterior['visitante_placar']:
                            vitorias += 1
                        elif j_anterior['mandante_placar'] == j_anterior['visitante_placar']:
                            empates += 1
                        else:
                            derrotas += 1
                    else:  # time é visitante
                        if j_anterior['visitante_placar'] > j_anterior['mandante_placar']:
                            vitorias += 1
                        elif j_anterior['visitante_placar'] == j_anterior['mandante_placar']:
                            empates += 1
                        else:
                            derrotas += 1
                
                # Atualiza os percentuais
                total_jogos = len(jogos_anteriores)
                dados.at[idx, 'visitante_pct_vitorias'] = vitorias / total_jogos
                dados.at[idx, 'visitante_pct_empates'] = empates / total_jogos
                dados.at[idx, 'visitante_pct_derrotas'] = derrotas / total_jogos
                
    return dados

# Divisão dos dados em treino e teste por ano
df['ano'] = df['data'].dt.year
treino = df[df['ano'] < 2023]  # Dados até 2022 para treino
teste = df[df['ano'] == 2023]  # Dados de 2023 para teste

print(f"Tamanho do conjunto de treino: {len(treino)}")
print(f"Tamanho do conjunto de teste: {len(teste)}")

# Aplicando a função de histórico recente aos dados de treino e teste
print("Calculando histórico recente dos times...")
treino = calcular_historico_recente(treino)
teste = calcular_historico_recente(teste)

le_mandante = LabelEncoder()
le_visitante = LabelEncoder()

treino = treino.copy()
teste = teste.copy()

treino.loc[:, 'mandante_le'] = le_mandante.fit_transform(treino['mandante'])
treino.loc[:, 'visitante_le'] = le_visitante.fit_transform(treino['visitante'])

teste.loc[:, 'mandante_le'] = le_mandante.transform(teste['mandante'])
teste.loc[:, 'visitante_le'] = le_visitante.transform(teste['visitante'])


print("Calculando estatísticas dos times...")


mandante_stats = treino.groupby('mandante').agg({
    'mandante_placar': 'mean',
    'visitante_placar': 'mean'
}).reset_index()
mandante_stats.columns = ['mandante', 'mandante_media_gols_pro', 'mandante_media_gols_contra']


visitante_stats = treino.groupby('visitante').agg({
    'visitante_placar': 'mean',
    'mandante_placar': 'mean'
}).reset_index()
visitante_stats.columns = ['visitante', 'visitante_media_gols_pro', 'visitante_media_gols_contra']


treino = treino.merge(mandante_stats, on='mandante', how='left')
treino = treino.merge(visitante_stats, on='visitante', how='left')
teste = teste.merge(mandante_stats, on='mandante', how='left')
teste = teste.merge(visitante_stats, on='visitante', how='left')

for col in ['mandante_media_gols_pro', 'mandante_media_gols_contra', 'visitante_media_gols_pro', 'visitante_media_gols_contra']:
    if treino[col].isna().any():
        treino[col].fillna(treino[col].mean(), inplace=True)
    if teste[col].isna().any():
        teste[col].fillna(treino[col].mean(), inplace=True)


features_basicas = ['mandante_le', 'visitante_le', 'rodata']
features_avancadas = ['mandante_le', 'visitante_le', 'rodata', 
                     'mandante_media_gols_pro', 'mandante_media_gols_contra',
                     'visitante_media_gols_pro', 'visitante_media_gols_contra',
                     'mandante_pct_vitorias', 'mandante_pct_empates', 'mandante_pct_derrotas',
                     'visitante_pct_vitorias', 'visitante_pct_empates', 'visitante_pct_derrotas']


X_train_basico = treino[features_basicas]
X_train_avancado = treino[features_avancadas]
y_train = treino['resultado']

X_test_basico = teste[features_basicas]
X_test_avancado = teste[features_avancadas]
y_test = teste['resultado']


def avaliar_modelo(modelo, X_train, y_train, X_test, y_test, nome_modelo):
    inicio = time.time()
    modelo.fit(X_train, y_train)
    fim = time.time()
    tempo_treino = fim - inicio
    
    # Previsões
    y_pred = modelo.predict(X_test)
    
    # Métricas
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Matriz de confusão
    cm = confusion_matrix(y_test, y_pred)
    
    # Exibindo resultados
    print(f"\n--- Resultados do modelo {nome_modelo} ---")
    print(f"Tempo de treinamento: {tempo_treino:.2f} segundos")
    print(f"Acurácia: {accuracy:.4f}")
    print(f"Precisão: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    print("\nMatriz de Confusão:")
    print(cm)
    

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=modelo.classes_, yticklabels=modelo.classes_)
    plt.xlabel('Previsto', fontsize=14)
    plt.ylabel('Real', fontsize=14)
    plt.title(f'Matriz de Confusão - {nome_modelo}', fontsize=16)
    plt.savefig(f'matriz_confusao_{nome_modelo.replace(" ", "_").lower()}.png', 
                dpi=300, bbox_inches='tight')
    
    return {
        'modelo': nome_modelo,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tempo_treino': tempo_treino
    }

print("\n--- Treinando modelo básico sem otimização ---")
modelo_basico = LogisticRegression(max_iter=1000, random_state=42)
resultados_basico = avaliar_modelo(modelo_basico, X_train_basico, y_train, X_test_basico, y_test, "Regressão Logística Básica")


print("\n--- Realizando validação cruzada no modelo básico ---")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores_cv = cross_val_score(modelo_basico, X_train_basico, y_train, cv=cv, scoring='accuracy')
print(f"Acurácia média na validação cruzada: {scores_cv.mean():.4f} (±{scores_cv.std():.4f})")

#Otimização de Hiperparâmetros usando GridSearchCV
print("\n--- Iniciando otimização de hiperparâmetros ---")
param_grid = {
    'C': [0.1, 0.5, 1.0, 2.0],
    'penalty': ['l1', 'l2'],
    'solver': ['saga'],  # saga suporta tanto l1 quanto l2
    'class_weight': ['balanced', None],
    'multi_class': ['multinomial']
}

grid_search = GridSearchCV(
    LogisticRegression(max_iter=2000, random_state=42),
    param_grid,
    cv=cv,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

inicio_grid = time.time()
grid_search.fit(X_train_avancado, y_train)
fim_grid = time.time()

print(f"\nTempo total de otimização: {(fim_grid - inicio_grid)/60:.2f} minutos")
print(f"Melhores parâmetros: {grid_search.best_params_}")
print(f"Melhor acurácia na validação cruzada: {grid_search.best_score_:.4f}")


print("\n--- Treinando modelo otimizado com os melhores parâmetros ---")
modelo_otimizado = grid_search.best_estimator_
resultados_otimizado = avaliar_modelo(modelo_otimizado, X_train_avancado, y_train, X_test_avancado, y_test, "Regressão Logística Otimizada")


print("\n--- Realizando validação cruzada no modelo otimizado ---")
scores_cv_otimizado = cross_val_score(modelo_otimizado, X_train_avancado, y_train, cv=cv, scoring='accuracy')
print(f"Acurácia média na validação cruzada (modelo otimizado): {scores_cv_otimizado.mean():.4f} (±{scores_cv_otimizado.std():.4f})")

print("\n--- Comparação dos modelos ---")
comparacao = pd.DataFrame([resultados_basico, resultados_otimizado])
print(comparacao[['modelo', 'accuracy', 'precision', 'recall', 'f1', 'tempo_treino']])

if hasattr(modelo_otimizado, 'coef_'):
    coefs = modelo_otimizado.coef_
    classes = modelo_otimizado.classes_
    
    plt.figure(figsize=(12, 8))
    for i, classe in enumerate(classes):
        plt.subplot(len(classes), 1, i+1)
        importancias = pd.Series(coefs[i], index=features_avancadas)
        importancias = importancias.sort_values(ascending=False)
        importancias.plot(kind='bar')
        plt.title(f'Importância das Features para a classe: {classe}')
    
    plt.tight_layout()
    plt.savefig('importancia_features_regressao_logistica.png', 
                dpi=300, bbox_inches='tight')

print("\nAnálise completa! Imagens salvas no diretório do projeto.")
