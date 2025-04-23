import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from utils_futebol import load_data, calcular_aproveitamento


def garantir_diretorio_graficos():
    
    diretorio_graficos = os.path.join(os.path.dirname(__file__), 'graficos')
    
    
    if not os.path.exists(diretorio_graficos):
        os.makedirs(diretorio_graficos)
    
    return diretorio_graficos

def plotar_graficos_aproveitamento():
    
    df = load_data()
    

    # Verificar quais anos estão disponíveis na base
    anos_disponiveis = sorted(df['ano'].unique())
    print(f"Anos disponíveis na base: {anos_disponiveis}")
    print(f"Analisando período: {min(anos_disponiveis)} a {max(anos_disponiveis)}")
    
    #  mandante
    aproveitamento_mandante = calcular_aproveitamento(df, 'mandante', 'vitória_mandante', ['mandante', 'ano'])
    aproveitamento_mandante['tipo'] = 'Mandante'
    
    # visitante
    aproveitamento_visitante = calcular_aproveitamento(df, 'visitante', 'vitória_visitante', ['visitante', 'ano']) 
    aproveitamento_visitante['tipo'] = 'Visitante'
    
    
    aproveitamento_mandante.rename(columns={'mandante': 'time'}, inplace=True)
    aproveitamento_visitante.rename(columns={'visitante': 'time'}, inplace=True)
    
    
    aproveitamento_total = pd.concat([aproveitamento_mandante, aproveitamento_visitante])
    
    # Incluir times que apareceram em pelo menos X partidas por ano 
    min_jogos_por_ano = 5
    times_freq = df.groupby(['mandante', 'ano']).size().reset_index()
    times_freq.columns = ['time', 'ano', 'jogos']
    times_freq = times_freq[times_freq['jogos'] >= min_jogos_por_ano]['time'].unique()
    
    
    aproveitamento_total = aproveitamento_total[aproveitamento_total['time'].isin(times_freq)]
    
    # Encontrar os 12 times com maior média de aproveitamento geral para destacar
    top_times = aproveitamento_total.groupby('time')['aproveitamento'].mean().nlargest(12).index.tolist()
    
    # Extrair anos para o eixo x para garantir que todos anos sejam mostrados
    anos_disponiveis = sorted(df['ano'].unique())
    intervalo_anos = max(1, len(anos_disponiveis) // 10)  # Mostrar no máximo 10 rótulos
    anos_a_mostrar = anos_disponiveis[::intervalo_anos] + [anos_disponiveis[-1]]
    
    
    plotar_aproveitamento_mandante(aproveitamento_total, top_times, anos_a_mostrar, anos_disponiveis)
    plotar_aproveitamento_visitante(aproveitamento_total, top_times, anos_a_mostrar, anos_disponiveis)
    plotar_aproveitamento_geral(aproveitamento_total, top_times, anos_a_mostrar, anos_disponiveis)

def plotar_aproveitamento_mandante(aproveitamento_total, top_times, anos_a_mostrar, anos_disponiveis):
   
    plt.figure(figsize=(16, 10))
    
    diretorio_graficos = garantir_diretorio_graficos()

    
    for time in aproveitamento_total['time'].unique():
        if time not in top_times:
            dados_time = aproveitamento_total[(aproveitamento_total['time'] == time) & 
                                            (aproveitamento_total['tipo'] == 'Mandante')]
            if len(dados_time) > 1:  # Apenas times com dados em mais de um ano
                plt.plot(dados_time['ano'], dados_time['aproveitamento'], color='lightgray', 
                        linewidth=1, alpha=0.3)
    
    # Depois, plotamos os top times com cores destacadas e legendas
    cores = plt.cm.tab20.colors  # Usar tab20 para ter mais cores
    for i, time in enumerate(top_times):
        dados_time = aproveitamento_total[(aproveitamento_total['time'] == time) & 
                                        (aproveitamento_total['tipo'] == 'Mandante')]
        
        # Ordenar por ano para garantir que a linha seja contínua
        dados_time = dados_time.sort_values('ano')
        
        # Somente plotar se tiver dados suficientes
        if len(dados_time) > 1:
            plt.plot(dados_time['ano'], dados_time['aproveitamento'], 
                    marker='o', linewidth=2.5, color=cores[i % len(cores)],
                    label=time)
    
    plt.title(f'Aproveitamento dos Times como Mandante ({min(anos_disponiveis)}-{max(anos_disponiveis)})', fontsize=16)
    plt.xlabel('Ano', fontsize=14)
    plt.ylabel('Taxa de Vitória (%)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Times em destaque:', title_fontsize=12, fontsize=10, 
              loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=6)
    plt.xticks(anos_a_mostrar, rotation=45)
    plt.ylim(0, 100)
    plt.tight_layout()
    caminho_arquivo = os.path.join(diretorio_graficos, 'comparacao_apt_mandante.png')
    plt.savefig(caminho_arquivo, dpi=300, bbox_inches='tight')
    plt.show()

def plotar_aproveitamento_visitante(aproveitamento_total, top_times, anos_a_mostrar, anos_disponiveis):
    
    plt.figure(figsize=(16, 10))
    
    diretorio_graficos = garantir_diretorio_graficos()

    
    for time in aproveitamento_total['time'].unique():
        if time not in top_times:
            dados_time = aproveitamento_total[(aproveitamento_total['time'] == time) & 
                                            (aproveitamento_total['tipo'] == 'Visitante')]
            if len(dados_time) > 1:  
                plt.plot(dados_time['ano'], dados_time['aproveitamento'], color='lightgray', 
                        linewidth=1, alpha=0.3)
    
    
    cores = plt.cm.tab20.colors
    for i, time in enumerate(top_times):
        dados_time = aproveitamento_total[(aproveitamento_total['time'] == time) & 
                                        (aproveitamento_total['tipo'] == 'Visitante')]
        
        
        dados_time = dados_time.sort_values('ano')
        
       
        if len(dados_time) > 1:
            plt.plot(dados_time['ano'], dados_time['aproveitamento'], 
                    marker='s', linewidth=2.5, color=cores[i % len(cores)],
                    label=time)
    
    plt.title(f'Aproveitamento dos Times como Visitante ({min(anos_disponiveis)}-{max(anos_disponiveis)})', fontsize=16)
    plt.xlabel('Ano', fontsize=14)
    plt.ylabel('Taxa de Vitória (%)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Times em destaque:', title_fontsize=12, fontsize=10, 
              loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=6)
    plt.xticks(anos_a_mostrar, rotation=45)
    plt.ylim(0, 100)
    plt.tight_layout()
    caminho_arquivo = os.path.join(diretorio_graficos, 'comparacao_apt_visitante.png')
    plt.savefig(caminho_arquivo, dpi=300, bbox_inches='tight')
    plt.show()

def plotar_aproveitamento_geral(aproveitamento_total, top_times, anos_a_mostrar, anos_disponiveis):
    
    plt.figure(figsize=(16, 10))

    diretorio_graficos = garantir_diretorio_graficos()
    
    
    aproveitamento_geral = aproveitamento_total.groupby(['time', 'ano'])['aproveitamento'].mean().reset_index()
    
    
    for time in aproveitamento_geral['time'].unique():
        if time not in top_times:
            dados_time = aproveitamento_geral[aproveitamento_geral['time'] == time]
            if len(dados_time) > 1:
                plt.plot(dados_time['ano'], dados_time['aproveitamento'], color='lightgray', 
                        linewidth=1, alpha=0.3)
    
    
    cores = plt.cm.tab20.colors
    for i, time in enumerate(top_times):
        dados_time = aproveitamento_geral[aproveitamento_geral['time'] == time]
        
        
        dados_time = dados_time.sort_values('ano')
        
        if len(dados_time) > 1:
            plt.plot(dados_time['ano'], dados_time['aproveitamento'], 
                    marker='d', linewidth=2.5, color=cores[i % len(cores)],
                    label=time)
    
    plt.title(f'Aproveitamento Geral dos Times ({min(anos_disponiveis)}-{max(anos_disponiveis)})', fontsize=16)
    plt.xlabel('Ano', fontsize=14)
    plt.ylabel('Taxa de Vitória (%)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Times em destaque:', title_fontsize=12, fontsize=10, 
              loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=6)
    plt.xticks(anos_a_mostrar, rotation=45)
    plt.ylim(0, 100)
    plt.tight_layout()
    caminho_arquivo = os.path.join(diretorio_graficos, 'comparacao_apt_geral.png')
    plt.savefig(caminho_arquivo, dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    plotar_graficos_aproveitamento()
