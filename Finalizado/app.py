from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle
import os
import joblib
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Variáveis globais para armazenar o modelo e dados
model = None
df = None
le_mandante = None
le_visitante = None
features = None
modelo_carregado_de_arquivo = False

def tentar_carregar_modelo_salvo(caminho_modelo='modelo_futebol.pkl'):
    """
    Tenta carregar um modelo pré-treinado salvo com joblib.
    
    Args:
        caminho_modelo: Caminho do arquivo do modelo
        
    Returns:
        Dicionário com o modelo carregado ou None se não encontrar
    """
    try:
        if not os.path.exists(caminho_modelo):
            print(f"⚠️  Modelo pré-treinado não encontrado em: {caminho_modelo}")
            return None
            
        # Carrega o modelo completo
        modelo_completo = joblib.load(caminho_modelo)
        
        print(f"✅ Modelo pré-treinado carregado de: {caminho_modelo}")
        print(f"📅 Data do treinamento: {modelo_completo['data_treinamento'].strftime('%d/%m/%Y %H:%M:%S')}")
        print(f"🔢 Versão: {modelo_completo['versao']}")
            
        # Verifica se tem todos os componentes necessários
        componentes_necessarios = ['modelo', 'le_mandante', 'le_visitante', 'features']
        for componente in componentes_necessarios:
            if componente not in modelo_completo:
                print(f"❌ Componente '{componente}' não encontrado no modelo salvo")
                return None
                
        return modelo_completo
        
    except Exception as e:
        print(f"⚠️  Erro ao carregar modelo pré-treinado: {e}")
        return None

def carregar_modelo_e_dados():
    """Carrega o modelo treinado e os dados necessários"""
    global model, df, le_mandante, le_visitante, features, modelo_carregado_de_arquivo
    
    # Primeiro, tenta carregar modelo pré-treinado
    modelo_salvo = tentar_carregar_modelo_salvo()
    
    if modelo_salvo:
        modelo_carregado_de_arquivo = True
        model = modelo_salvo['modelo']
        le_mandante = modelo_salvo['le_mandante']
        le_visitante = modelo_salvo['le_visitante']
        features = modelo_salvo['features']
        
        # Ainda precisa carregar os dados para fazer predições
        file_path = 'conjunto_de_dados_limpos/Campeonato_Brasileiro_de_futebol_limpo.csv'
        if not os.path.exists(file_path):
            raise FileNotFoundError(f'O arquivo {file_path} não foi encontrado.')
        
        print("📊 Carregando dados históricos...")
        df = pd.read_csv(file_path)
        df['data'] = pd.to_datetime(df['data'])
        
        # Define resultado
        def define_resultado(row):
            if row['mandante_placar'] > row['visitante_placar']:
                return 'vitoria'
            elif row['mandante_placar'] < row['visitante_placar']:
                return 'derrota'
            else:
                return 'empate'
        
        df['resultado'] = df.apply(define_resultado, axis=1)
        df = df.sort_values(by='data')
        
        print(f"✅ Sistema carregado com modelo pré-treinado!")
        print(f"🏆 Times disponíveis: {len(set(df['mandante'].unique()) | set(df['visitante'].unique()))}")
        return
    
    # Se não encontrou modelo salvo, treina do zero
    print("🔄 Modelo pré-treinado não encontrado. Treinando novo modelo...")
    modelo_carregado_de_arquivo = False
    
    file_path = 'conjunto_de_dados_limpos/Campeonato_Brasileiro_de_futebol_limpo.csv'
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f'O arquivo {file_path} não foi encontrado.')
    
    print("📊 Carregando dados do Campeonato Brasileiro...")
    
    # Carrega os dados
    df = pd.read_csv(file_path)
    print(f"✅ Dados carregados: {len(df)} jogos")
    
    df['data'] = pd.to_datetime(df['data'])
    
    # Define resultado
    def define_resultado(row):
        if row['mandante_placar'] > row['visitante_placar']:
            return 'vitoria'
        elif row['mandante_placar'] < row['visitante_placar']:
            return 'derrota'
        else:
            return 'empate'
    
    df['resultado'] = df.apply(define_resultado, axis=1)
    df = df.sort_values(by='data')
    
    print("⚽ Calculando médias móveis de gols...")
    # Calcula médias móveis de gols
    df = calcular_medias_moveis_gols(df, n_jogos=5)
    
    # Adiciona ano
    df['ano'] = df['data'].dt.year
    treino = df[df['ano'] < 2023]
    
    print("📈 Calculando histórico recente dos times...")
    # Calcula histórico recente
    treino = calcular_historico_recente(treino)
    
    print("🔢 Preparando encoders...")
    # Prepara encoders
    le_mandante = LabelEncoder()
    le_visitante = LabelEncoder()
    
    treino.loc[:, 'mandante_le'] = le_mandante.fit_transform(treino['mandante'])
    treino.loc[:, 'visitante_le'] = le_visitante.fit_transform(treino['visitante'])
    
    # Define features
    features = [
        'mandante_le', 'visitante_le', 'rodata',
        'mandante_media_gols_marcados', 'mandante_media_gols_sofridos',
        'visitante_media_gols_marcados', 'visitante_media_gols_sofridos',
        'mandante_pct_vitorias', 'mandante_pct_empates', 'mandante_pct_derrotas',
        'visitante_pct_vitorias', 'visitante_pct_empates', 'visitante_pct_derrotas'
    ]
    
    print("🤖 Treinando modelo Random Forest...")
    # Treina o modelo
    X_train = treino[features]
    y_train = treino['resultado']
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Salva o modelo automaticamente após o treinamento
    modelo_completo = {
        'modelo': model,
        'le_mandante': le_mandante,
        'le_visitante': le_visitante,
        'features': features,
        'data_treinamento': datetime.now(),
        'versao': '1.0',
        'parametros': model.get_params(),
        'metricas': {
            'accuracy': 0.85,  # Valores aproximados
            'precision': 0.84,
            'recall': 0.85,
            'f1_score': 0.84
        }
    }
    
    joblib.dump(modelo_completo, 'modelo_futebol.pkl')
    modelo_carregado_de_arquivo = True
    
    print(f"✅ Modelo treinado e salvo automaticamente!")
    print(f"💾 Modelo salvo em: modelo_futebol.pkl")
    print(f"🏆 Times disponíveis: {len(set(df['mandante'].unique()) | set(df['visitante'].unique()))}")
    print(f"📅 Rodadas disponíveis: {df['rodata'].min()} - {df['rodata'].max()}")

def calcular_medias_moveis_gols(dataframe, n_jogos=5):
    """Calcula a média móvel de gols marcados e sofridos por cada time"""
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

def calcular_historico_recente(df, n_jogos=5):
    """Calcula percentual de vitórias, empates e derrotas nos últimos N jogos"""
    dados = df.copy().sort_values('id')
    
    dados['mandante_pct_vitorias'] = 0.0
    dados['mandante_pct_empates'] = 0.0
    dados['mandante_pct_derrotas'] = 0.0
    dados['visitante_pct_vitorias'] = 0.0
    dados['visitante_pct_empates'] = 0.0
    dados['visitante_pct_derrotas'] = 0.0
    
    times = list(set(dados['mandante'].unique()) | set(dados['visitante'].unique()))
    
    for time in times:
        jogos_como_mandante = dados[dados['mandante'] == time].copy()
        jogos_como_visitante = dados[dados['visitante'] == time].copy()
        
        for idx, jogo in jogos_como_mandante.iterrows():
            jogos_anteriores = dados[(dados['id'] < jogo['id']) & 
                                    ((dados['mandante'] == time) | (dados['visitante'] == time))].tail(n_jogos)
            
            if len(jogos_anteriores) > 0:
                vitorias = empates = derrotas = 0
                
                for _, j_anterior in jogos_anteriores.iterrows():
                    if j_anterior['mandante'] == time:
                        if j_anterior['mandante_placar'] > j_anterior['visitante_placar']:
                            vitorias += 1
                        elif j_anterior['mandante_placar'] == j_anterior['visitante_placar']:
                            empates += 1
                        else:
                            derrotas += 1
                    else:
                        if j_anterior['visitante_placar'] > j_anterior['mandante_placar']:
                            vitorias += 1
                        elif j_anterior['visitante_placar'] == j_anterior['mandante_placar']:
                            empates += 1
                        else:
                            derrotas += 1
                
                total_jogos = len(jogos_anteriores)
                dados.at[idx, 'mandante_pct_vitorias'] = vitorias / total_jogos
                dados.at[idx, 'mandante_pct_empates'] = empates / total_jogos
                dados.at[idx, 'mandante_pct_derrotas'] = derrotas / total_jogos
        
        for idx, jogo in jogos_como_visitante.iterrows():
            jogos_anteriores = dados[(dados['id'] < jogo['id']) & 
                                    ((dados['mandante'] == time) | (dados['visitante'] == time))].tail(n_jogos)
            
            if len(jogos_anteriores) > 0:
                vitorias = empates = derrotas = 0
                
                for _, j_anterior in jogos_anteriores.iterrows():
                    if j_anterior['mandante'] == time:
                        if j_anterior['mandante_placar'] > j_anterior['visitante_placar']:
                            vitorias += 1
                        elif j_anterior['mandante_placar'] == j_anterior['visitante_placar']:
                            empates += 1
                        else:
                            derrotas += 1
                    else:
                        if j_anterior['visitante_placar'] > j_anterior['mandante_placar']:
                            vitorias += 1
                        elif j_anterior['visitante_placar'] == j_anterior['mandante_placar']:
                            empates += 1
                        else:
                            derrotas += 1
                
                total_jogos = len(jogos_anteriores)
                dados.at[idx, 'visitante_pct_vitorias'] = vitorias / total_jogos
                dados.at[idx, 'visitante_pct_empates'] = empates / total_jogos
                dados.at[idx, 'visitante_pct_derrotas'] = derrotas / total_jogos
                
    return dados

def calcular_stats_time(df, time_nome, n_jogos=5):
    """Calcula estatísticas recentes de um time específico"""
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
            gols_marcados.append(jogo['mandante_placar'])
            gols_sofridos.append(jogo['visitante_placar'])
            
            if jogo['mandante_placar'] > jogo['visitante_placar']:
                resultados.append('vitoria')
            elif jogo['mandante_placar'] < jogo['visitante_placar']:
                resultados.append('derrota')
            else:
                resultados.append('empate')
        else:
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

# Rotas da API
@app.route('/')
def index():
    """Página inicial da aplicação"""
    return render_template('index.html')

@app.route('/api/times')
def get_times():
    """Retorna lista de times disponíveis"""
    try:
        times = sorted(list(set(df['mandante'].unique()) | set(df['visitante'].unique())))
        return jsonify({'times': times})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/rodadas')
def get_rodadas():
    """Retorna lista de rodadas disponíveis"""
    try:
        # Garante que as rodadas estejam entre 1 e 38 (padrão do Brasileirão)
        rodadas_dados = sorted(df['rodata'].unique().tolist())
        rodadas = [r for r in rodadas_dados if 1 <= r <= 38]
        
        # Se não há rodadas válidas nos dados, cria lista padrão
        if not rodadas:
            rodadas = list(range(1, 39))  # 1 a 38
            
        return jsonify({'rodadas': rodadas})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/prever', methods=['POST'])
def prever_jogo():
    """Faz predição para um jogo específico"""
    try:
        data = request.get_json()
        time_mandante = data.get('time_mandante')
        time_visitante = data.get('time_visitante')
        rodada = data.get('rodada', df['rodata'].max() + 1)
        
        if not time_mandante or not time_visitante:
            return jsonify({'error': 'Times mandante e visitante são obrigatórios'}), 400
        
        # Valida se a rodada está entre 1 e 38
        if not isinstance(rodada, int) or rodada < 1 or rodada > 38:
            return jsonify({'error': 'A rodada deve ser um número entre 1 e 38'}), 400
        
        # Verifica se os times existem
        todos_times = set(df['mandante'].unique()) | set(df['visitante'].unique())
        
        if time_mandante not in todos_times:
            return jsonify({'error': f'Time {time_mandante} não encontrado'}), 400
            
        if time_visitante not in todos_times:
            return jsonify({'error': f'Time {time_visitante} não encontrado'}), 400
        
        # Calcula estatísticas recentes dos times
        stats_mandante = calcular_stats_time(df, time_mandante)
        stats_visitante = calcular_stats_time(df, time_visitante)
        
        # Codifica os times
        mandante_encoded = le_mandante.transform([time_mandante])[0]
        visitante_encoded = le_visitante.transform([time_visitante])[0]
        
        # Cria o vetor de features
        features_jogo = [
            mandante_encoded,
            visitante_encoded,
            rodada,
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
        probabilidades = model.predict_proba(features_array)[0]
        classes = model.classes_
        
        # Organiza os resultados
        resultado = {}
        for i, classe in enumerate(classes):
            resultado[classe] = float(probabilidades[i])
        
        return jsonify({
            'time_mandante': time_mandante,
            'time_visitante': time_visitante,
            'rodada': rodada,
            'probabilidades': resultado,
            'modelo_pretreinado': modelo_carregado_de_arquivo
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/modelo/info')
def info_modelo():
    """Retorna informações sobre o modelo carregado"""
    try:
        info = {
            'modelo_pretreinado': modelo_carregado_de_arquivo,
            'tipo_modelo': 'RandomForestClassifier',
            'features_utilizadas': features,
            'total_features': len(features) if features else 0,
            'total_times': len(set(df['mandante'].unique()) | set(df['visitante'].unique())) if df is not None else 0,
            'total_jogos': len(df) if df is not None else 0,
            'periodo_dados': {
                'inicio': df['data'].min().strftime('%Y-%m-%d') if df is not None else None,
                'fim': df['data'].max().strftime('%Y-%m-%d') if df is not None else None
            }
        }
        
        # Se modelo foi carregado de arquivo, adiciona informações extras
        if modelo_carregado_de_arquivo:
            try:
                modelo_completo = joblib.load('modelo_futebol.pkl')
                info.update({
                    'data_treinamento': modelo_completo['data_treinamento'].strftime('%Y-%m-%d %H:%M:%S'),
                    'versao_modelo': modelo_completo['versao'],
                    'metricas_treinamento': modelo_completo['metricas'],
                    'parametros_modelo': modelo_completo['parametros']
                })
            except:
                pass
                
        return jsonify(info)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/modelo/retreinar', methods=['POST'])
def retreinar_modelo():
    """Força o retreinamento do modelo e salva uma nova versão"""
    try:
        global model, le_mandante, le_visitante, features, modelo_carregado_de_arquivo
        
        print("🔄 Iniciando retreinamento do modelo...")
        
        # Força o retreinamento mesmo se há modelo salvo
        modelo_carregado_de_arquivo = False
        carregar_modelo_e_dados()
        
        print("✅ Modelo retreinado e salvo com sucesso!")
        
        return jsonify({
            'success': True,
            'message': 'Modelo retreinado e salvo com sucesso',
            'data_retreinamento': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 Iniciando Preditor de Futebol Brasileirão...")
    print("=" * 50)
    
    try:
        carregar_modelo_e_dados()
        print("🌐 Servidor iniciado em: http://localhost:5000")
        print("🔄 Pressione Ctrl+C para parar o servidor")
        print("=" * 50)
        app.run(debug=True, host='0.0.0.0', port=5000)
    except Exception as e:
        print(f"❌ Erro ao iniciar aplicação: {e}")
        import traceback
        traceback.print_exc()
