import matplotlib.pyplot as plt
import numpy as np
import os
from utils_futebol import load_data, get_regioes_dict

def garantir_diretorio_graficos():
    
    diretorio_graficos = os.path.join(os.path.dirname(__file__), 'graficos')
    
    
    if not os.path.exists(diretorio_graficos):
        os.makedirs(diretorio_graficos)
    
    return diretorio_graficos

def plotar_comparacao_regioes():
    # Carrega os dados
    df = load_data()
    
    
    diretorio_graficos = garantir_diretorio_graficos()
    
    # Adicionar colunas de região usando o dicionário de regiões
    regioes = get_regioes_dict()
    df['regiao_mandante'] = df['mandante_estado'].map(regioes)
    df['regiao_visitante'] = df['visitante_estado'].map(regioes)
    
    # Calcular vitórias por estado (como mandante)
    vitorias_mandante_estado = df[df['resultado'] == 'vitória_mandante'].groupby('mandante_estado').size()
    jogos_mandante_estado = df.groupby('mandante_estado').size()
    aproveitamento_mandante = (vitorias_mandante_estado / jogos_mandante_estado * 100).sort_values(ascending=False)
    
    # Calcular vitórias por região (como mandante)
    vitorias_mandante_regiao = df[df['resultado'] == 'vitória_mandante'].groupby('regiao_mandante').size()
    jogos_mandante_regiao = df.groupby('regiao_mandante').size()
    aproveitamento_regiao = (vitorias_mandante_regiao / jogos_mandante_regiao * 100).sort_values(ascending=False)
    
    # Criar subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Gráfico de aproveitamento por estado
    colors_estados = plt.cm.viridis(np.linspace(0, 1, len(aproveitamento_mandante)))
    bars_estados = ax1.bar(aproveitamento_mandante.index, aproveitamento_mandante.values, color=colors_estados)
    ax1.set_title('Aproveitamento como Mandante por Estado', fontsize=14)
    ax1.set_xlabel('Estado', fontsize=12)
    ax1.set_ylabel('Taxa de Vitória (%)', fontsize=12)
    ax1.set_ylim(0, 100)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    ax1.tick_params(axis='x', rotation=45)
    
    # Adicionar valores sobre as barras
    for bar in bars_estados:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # Gráfico de aproveitamento por região
    colors_regioes = plt.cm.plasma(np.linspace(0, 1, len(aproveitamento_regiao)))
    bars_regioes = ax2.bar(aproveitamento_regiao.index, aproveitamento_regiao.values, color=colors_regioes)
    ax2.set_title('Aproveitamento como Mandante por Região', fontsize=14)
    ax2.set_xlabel('Região', fontsize=12)
    ax2.set_ylabel('Taxa de Vitória (%)', fontsize=12)
    ax2.set_ylim(0, 100)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Adicionar valores sobre as barras
    for bar in bars_regioes:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    caminho_arquivo = os.path.join(diretorio_graficos, 'comparacao_regioes.png')
    plt.savefig(caminho_arquivo, dpi=300, bbox_inches='tight')
    plt.show()  # Exibe o gráfico
    
    # gráfico extra: Número total de jogos por região
    plt.figure(figsize=(12, 6))
    jogos_por_regiao = jogos_mandante_regiao.sort_values(ascending=False)
    
    colors_total = plt.cm.Blues(np.linspace(0.4, 0.8, len(jogos_por_regiao)))
    bars_total = plt.bar(jogos_por_regiao.index, jogos_por_regiao.values, color=colors_total)
    
    plt.title('Número Total de Jogos como Mandante por Região', fontsize=14)
    plt.xlabel('Região', fontsize=12)
    plt.ylabel('Quantidade de Jogos', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Adicionar valores sobre as barras
    for bar in bars_total:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{int(height)}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    caminho_arquivo = os.path.join(diretorio_graficos, 'jogos_por_regiao.png')
    plt.savefig(caminho_arquivo, dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    plotar_comparacao_regioes()
