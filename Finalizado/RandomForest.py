import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib
from datetime import datetime

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

# Função para calcular percentual de vitórias, empates e derrotas nos últimos N jogos
def calcular_historico_recente(df, n_jogos=5):
    # Cria cópia dos dados
    dados = df.copy().sort_values('id')
    
    # Inicializa as colunas de histórico para mandante
    dados['mandante_pct_vitorias'] = 0.0
    dados['mandante_pct_empates'] = 0.0
    dados['mandante_pct_derrotas'] = 0.0
    # Inicializa as colunas de histórico para visitante
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

# Aplicando a função e adicionando as novas features
treino = calcular_historico_recente(treino)
teste = calcular_historico_recente(teste)

treino.loc[:, 'mandante_le'] = le_mandante.fit_transform(treino['mandante'])
treino.loc[:, 'visitante_le'] = le_visitante.fit_transform(treino['visitante'])

teste.loc[:, 'mandante_le'] = le_mandante.transform(teste['mandante'])
teste.loc[:, 'visitante_le'] = le_visitante.transform(teste['visitante'])

features = [
    'mandante_le', 'visitante_le', 'rodata',
    'mandante_media_gols_marcados', 'mandante_media_gols_sofridos',
    'visitante_media_gols_marcados', 'visitante_media_gols_sofridos',
    'mandante_pct_vitorias', 'mandante_pct_empates', 'mandante_pct_derrotas',
    'visitante_pct_vitorias', 'visitante_pct_empates', 'visitante_pct_derrotas'
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
'''
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel('Previsto')
plt.ylabel('Real')
plt.title('Matriz de Confusão')
plt.savefig('matriz_confusao.png')  # Salva a imagem
#plt.show()'''

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

print("\nImportância das Features:")
feature_importance = pd.DataFrame({'Feature': features, 'Importance': best_model.feature_importances_})
feature_importance = feature_importance.sort_values('Importance', ascending=False)
print(feature_importance)

cm = confusion_matrix(y_test, y_pred)
'''
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=best_model.classes_, yticklabels=best_model.classes_)
plt.xlabel('Previsto')
plt.ylabel('Real')
plt.title('Matriz de Confusão (Modelo Calibrado)')
plt.savefig('matriz_confusao_calibrado.png')  # Salva a imagem
#plt.show()'''

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

# =====================================================
# FUNCIONALIDADE DE EXPORTAÇÃO DO MODELO TREINADO
# =====================================================

def salvar_modelo_completo(modelo, encoders, features_list, caminho_modelo='modelo_futebol.pkl'):
    """
    Salva o modelo treinado junto com os encoders e lista de features usando joblib.
    
    Args:
        modelo: Modelo treinado (RandomForestClassifier)
        encoders: Dicionário com os encoders (le_mandante, le_visitante)
        features_list: Lista com os nomes das features utilizadas
        caminho_modelo: Caminho onde salvar o modelo
    """
    try:
        # Cria um dicionário com todos os componentes necessários
        modelo_completo = {
            'modelo': modelo,
            'le_mandante': encoders['le_mandante'],
            'le_visitante': encoders['le_visitante'],
            'features': features_list,
            'data_treinamento': datetime.now(),
            'versao': '1.0',
            'parametros': modelo.get_params(),
            'metricas': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
        }
        
        # Salva o modelo completo
        joblib.dump(modelo_completo, caminho_modelo)
        print(f"✅ Modelo salvo com sucesso em: {caminho_modelo}")
        print(f"📊 Métricas incluídas: Accuracy={accuracy:.4f}, F1-Score={f1:.4f}")
        print(f"📅 Data do treinamento: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
        
        # Mostra informações do arquivo salvo
        tamanho_arquivo = os.path.getsize(caminho_modelo) / (1024 * 1024)  # MB
        print(f"💾 Tamanho do arquivo: {tamanho_arquivo:.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro ao salvar o modelo: {e}")
        return False

def carregar_modelo_completo(caminho_modelo='modelo_futebol.pkl'):
    """
    Carrega o modelo treinado e todos os componentes necessários.
    
    Args:
        caminho_modelo: Caminho do arquivo do modelo
        
    Returns:
        Dicionário com o modelo e componentes carregados
    """
    try:
        if not os.path.exists(caminho_modelo):
            print(f"❌ Arquivo do modelo não encontrado: {caminho_modelo}")
            return None
            
        # Carrega o modelo completo
        modelo_completo = joblib.load(caminho_modelo)
        
        print(f"✅ Modelo carregado com sucesso de: {caminho_modelo}")
        print(f"📅 Data do treinamento: {modelo_completo['data_treinamento'].strftime('%d/%m/%Y %H:%M:%S')}")
        print(f"🔢 Versão: {modelo_completo['versao']}")
        print(f"📊 Métricas do modelo:")
        for metrica, valor in modelo_completo['metricas'].items():
            print(f"   - {metrica.capitalize()}: {valor:.4f}")
            
        return modelo_completo
        
    except Exception as e:
        print(f"❌ Erro ao carregar o modelo: {e}")
        return None

def fazer_predicao_com_modelo_salvo(time_mandante, time_visitante, dados_historicos, caminho_modelo='modelo_futebol.pkl'):
    """
    Faz predição usando o modelo salvo, sem necessidade de retreinamento.
    
    Args:
        time_mandante: Nome do time que joga em casa
        time_visitante: Nome do time visitante
        dados_historicos: DataFrame com histórico dos jogos
        caminho_modelo: Caminho do arquivo do modelo
        
    Returns:
        Dicionário com as probabilidades de cada resultado
    """
    # Carrega o modelo
    modelo_carregado = carregar_modelo_completo(caminho_modelo)
    
    if modelo_carregado is None:
        return None
    
    modelo = modelo_carregado['modelo']
    le_mandante = modelo_carregado['le_mandante']
    le_visitante = modelo_carregado['le_visitante']
    features_esperadas = modelo_carregado['features']
    
    try:
        # Verifica se os times estão no dataset
        todos_times = set(dados_historicos['mandante'].unique()) | set(dados_historicos['visitante'].unique())
        
        if time_mandante not in todos_times:
            print(f"❌ Time '{time_mandante}' não encontrado no dataset.")
            return None
            
        if time_visitante not in todos_times:
            print(f"❌ Time '{time_visitante}' não encontrado no dataset.")
            return None
        
        # Calcula estatísticas recentes dos times
        stats_mandante = calcular_stats_time(dados_historicos, time_mandante)
        stats_visitante = calcular_stats_time(dados_historicos, time_visitante)
        
        # Codifica os times
        mandante_encoded = le_mandante.transform([time_mandante])[0]
        visitante_encoded = le_visitante.transform([time_visitante])[0]
        
        # Estima a rodada
        ultima_rodada = dados_historicos['rodata'].max()
        rodata_estimada = ultima_rodada + 1
        
        # Cria o vetor de features
        features_jogo = [
            mandante_encoded,
            visitante_encoded,
            rodata_estimada,
            stats_mandante['media_gols_marcados'],
            stats_mandante['media_gols_sofridos'],
            stats_visitante['media_gols_marcados'],
            stats_visitante['media_gols_sofridos'],
            stats_mandante['pct_vitorias'],
            stats_mandante['pct_empates'],
            stats_mandante['pct_derrotas'],
            stats_visitante['pct_vitorias'],
            stats_visitante['pct_empates'],
            stats_visitante['pct_derrotas']
        ]
        
        # Faz a predição
        features_array = np.array(features_jogo).reshape(1, -1)
        probabilidades = modelo.predict_proba(features_array)[0]
        classes = modelo.classes_
        
        # Organiza os resultados
        resultado_dict = {}
        for i, classe in enumerate(classes):
            resultado_dict[classe] = probabilidades[i]
        
        print(f"🔮 Predição realizada com modelo pré-treinado (sem retreinamento)")
        return resultado_dict
        
    except Exception as e:
        print(f"❌ Erro durante a predição: {e}")
        return None

# Salva o modelo treinado
print("\n" + "="*60)
print("SALVANDO MODELO TREINADO")
print("="*60)

encoders_dict = {
    'le_mandante': le_mandante,
    'le_visitante': le_visitante
}

sucesso = salvar_modelo_completo(best_model, encoders_dict, features)

if sucesso:
    print("\n🚀 MODELO PRONTO PARA USO EM PRODUÇÃO!")
    print("💡 Vantagens da exportação:")
    print("   ✓ Sem necessidade de retreinamento")
    print("   ✓ Carregamento rápido em sistemas externos") 
    print("   ✓ Entrega acelerada do serviço")
    print("   ✓ Consistência entre ambientes")
    
    # Demonstra o carregamento e uso do modelo salvo
    print("\n" + "="*60)
    print("DEMONSTRAÇÃO: CARREGAMENTO E USO DO MODELO SALVO")
    print("="*60)
    
    # Exemplo de uso do modelo salvo
    exemplo_mandante = df['mandante'].iloc[0]
    exemplo_visitante = df['visitante'].iloc[0]
    
    print(f"\n🔄 Testando predição com modelo salvo...")
    print(f"📋 Jogo de exemplo: {exemplo_mandante} vs {exemplo_visitante}")
    
    resultado_teste = fazer_predicao_com_modelo_salvo(
        exemplo_mandante, 
        exemplo_visitante, 
        df
    )
    
    if resultado_teste:
        print(f"\n🎯 Resultado da predição:")
        for resultado_tipo, probabilidade in sorted(resultado_teste.items(), key=lambda x: x[1], reverse=True):
            if resultado_tipo == 'vitoria':
                print(f"   🏆 Vitória {exemplo_mandante}: {probabilidade:.1%}")
            elif resultado_tipo == 'empate':
                print(f"   🤝 Empate: {probabilidade:.1%}")
            elif resultado_tipo == 'derrota':
                print(f"   ⚽ Vitória {exemplo_visitante}: {probabilidade:.1%}")
else:
    print("❌ Falha ao salvar o modelo.")

# Função para calcular estatísticas recentes de um time específico
def calcular_stats_time(df, time_nome, n_jogos=5):
    """Calcula estatísticas recentes de um time específico"""
    # Filtra jogos do time (como mandante ou visitante)
    jogos_time = df[(df['mandante'] == time_nome) | (df['visitante'] == time_nome)].copy()
    jogos_time = jogos_time.sort_values('data').tail(n_jogos)
    
    if len(jogos_time) == 0:
        return {
            'media_gols_marcados': 0.0,
            'media_gols_sofridos': 0.0,
            'pct_vitorias': 0.0,
            'pct_empates': 0.0,
            'pct_derrotas': 0.0
        }
    
    gols_marcados = []
    gols_sofridos = []
    resultados = []
    
    for _, jogo in jogos_time.iterrows():
        if jogo['mandante'] == time_nome:
            # Time jogando em casa
            gols_marcados.append(jogo['mandante_placar'])
            gols_sofridos.append(jogo['visitante_placar'])
            
            if jogo['mandante_placar'] > jogo['visitante_placar']:
                resultados.append('vitoria')
            elif jogo['mandante_placar'] < jogo['visitante_placar']:
                resultados.append('derrota')
            else:
                resultados.append('empate')
        else:
            # Time jogando fora
            gols_marcados.append(jogo['visitante_placar'])
            gols_sofridos.append(jogo['mandante_placar'])
            
            if jogo['visitante_placar'] > jogo['mandante_placar']:
                resultados.append('vitoria')
            elif jogo['visitante_placar'] < jogo['mandante_placar']:
                resultados.append('derrota')
            else:
                resultados.append('empate')
    
    # Calcula estatísticas
    media_gols_marcados = sum(gols_marcados) / len(gols_marcados)
    media_gols_sofridos = sum(gols_sofridos) / len(gols_sofridos)
    
    vitorias = resultados.count('vitoria')
    empates = resultados.count('empate')
    derrotas = resultados.count('derrota')
    total = len(resultados)
    
    return {
        'media_gols_marcados': media_gols_marcados,
        'media_gols_sofridos': media_gols_sofridos,
        'pct_vitorias': vitorias / total,
        'pct_empates': empates / total,
        'pct_derrotas': derrotas / total
    }

# Função para fazer predição interativa
def prever_jogo(time_mandante, time_visitante):
    """Faz predição para um jogo específico"""
    try:
        # Verifica se os times estão no dataset
        todos_times = set(df['mandante'].unique()) | set(df['visitante'].unique())
        
        if time_mandante not in todos_times:
            print(f"Erro: Time '{time_mandante}' não encontrado no dataset.")
            print(f"Times disponíveis: {sorted(list(todos_times))}")
            return None
            
        if time_visitante not in todos_times:
            print(f"Erro: Time '{time_visitante}' não encontrado no dataset.")
            print(f"Times disponíveis: {sorted(list(todos_times))}")
            return None
        
        # Calcula estatísticas recentes dos times
        stats_mandante = calcular_stats_time(df, time_mandante)
        stats_visitante = calcular_stats_time(df, time_visitante)
        
        # Codifica os times
        try:
            mandante_encoded = le_mandante.transform([time_mandante])[0]
            visitante_encoded = le_visitante.transform([time_visitante])[0]
        except ValueError as e:
            print(f"Erro ao codificar times: {e}")
            return None
        
        # Estima a rodada (usando a última rodada + 1 como padrão)
        ultima_rodada = df['rodata'].max()
        rodata_estimada = ultima_rodada + 1
        
        # Cria o vetor de features
        features_jogo = [
            mandante_encoded,
            visitante_encoded,
            rodata_estimada,
            stats_mandante['media_gols_marcados'],
            stats_mandante['media_gols_sofridos'],
            stats_visitante['media_gols_marcados'],
            stats_visitante['media_gols_sofridos'],
            stats_mandante['pct_vitorias'],
            stats_mandante['pct_empates'],
            stats_mandante['pct_derrotas'],
            stats_visitante['pct_vitorias'],
            stats_visitante['pct_empates'],
            stats_visitante['pct_derrotas']
        ]
        
        # Faz a predição
        features_array = np.array(features_jogo).reshape(1, -1)
        probabilidades = best_model.predict_proba(features_array)[0]
        classes = best_model.classes_
        
        # Organiza os resultados
        resultado_dict = {}
        for i, classe in enumerate(classes):
            resultado_dict[classe] = probabilidades[i]
        
        return resultado_dict
        
    except Exception as e:
        print(f"Erro durante a predição: {e}")
        return None

# Sistema interativo
def sistema_predicao():
    """Sistema interativo para predição de jogos"""
    print("\n" + "="*60)
    print("SISTEMA DE PREDIÇÃO DE JOGOS - CAMPEONATO BRASILEIRO")
    print("="*60)
    
    # Lista todos os times disponíveis
    todos_times = sorted(list(set(df['mandante'].unique()) | set(df['visitante'].unique())))
    print(f"\nTimes disponíveis no dataset ({len(todos_times)} times):")
    for i, time in enumerate(todos_times, 1):
        print(f"{i:2d}. {time}")
    
    while True:
        print("\n" + "-"*60)
        print("Digite os times para fazer a predição (ou 'sair' para encerrar):")
        
        # Input do time mandante
        time_mandante = input("\nTime MANDANTE (casa): ").strip()
        if time_mandante.lower() == 'sair':
            break
            
        # Input do time visitante  
        time_visitante = input("Time VISITANTE (fora): ").strip()
        if time_visitante.lower() == 'sair':
            break
        
        # Faz a predição
        resultado = prever_jogo(time_mandante, time_visitante)
        
        if resultado:
            print(f"\n{'='*50}")
            print(f"PREDIÇÃO: {time_mandante} (casa) vs {time_visitante} (fora)")
            print(f"{'='*50}")
            
            # Ordena por probabilidade
            resultado_ordenado = sorted(resultado.items(), key=lambda x: x[1], reverse=True)
            
            for resultado_tipo, probabilidade in resultado_ordenado:
                if resultado_tipo == 'vitoria':
                    print(f"🏆 Vitória do {time_mandante}: {probabilidade:.1%}")
                elif resultado_tipo == 'empate':
                    print(f"🤝 Empate: {probabilidade:.1%}")
                elif resultado_tipo == 'derrota':
                    print(f"⚽ Vitória do {time_visitante}: {probabilidade:.1%}")
        
        print("\n" + "-"*60)
        continuar = input("Deseja fazer outra predição? (s/n): ").strip().lower()
        if continuar not in ['s', 'sim', 'y', 'yes']:
            break
    
    print("\nObrigado por usar o sistema de predição! ⚽")

# Executa o sistema interativo
sistema_predicao()

# Salvar o modelo treinado
data_atual = datetime.now().strftime("%Y%m%d")
hora_atual = datetime.now().strftime("%H%M%S")
nome_arquivo_modelo = f"modelo_rf_{data_atual}_{hora_atual}.joblib"

joblib.dump(best_model, nome_arquivo_modelo)
print(f"\nModelo salvo como '{nome_arquivo_modelo}'")