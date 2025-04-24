# Previsão de Resultados do Brasileirão com Machine Learning  
## Mini Trabalho 3 — Exploração dos Dados  

## Membros do Grupo:
- Harryson Campos Martins — 211039466  
- Pedro Henrique Muniz de Oliveira — 200059947  
- Flávio Gustavo Araújo de Melo — 211030602  
- Leandro de Almeida Oliveira — 211030827  
- Jefferson Sena Oliveira — 200020323  
- José André Rabelo Rocha — 211062016  

---

## Fontes dos Dados

- **Base 1:** Brazilian Soccer Database (Kaggle)  
  - Link: [Kaggle - ricardomattos05](https://www.kaggle.com/datasets/ricardomattos05/jogos-do-campeonato-brasileiro)  
  - Licença: CC BY-SA 4.0

- **Base 2:** Campeonato Brasileiro de Futebol (GitHub - adaoduque)  
  - Link: [Kaggle - adaoduque](https://www.kaggle.com/datasets/adaoduque/campeonato-brasileiro-de-futebol)  
  - Licença: Código aberto para fins educacionais e analíticos.

- **Base 3:** Brazilian Soccer - Brasileirão (gustavomartino)  
  - Link: [Kaggle - gustavomartino](https://www.kaggle.com/datasets/gustavomartino/brasileirao?resource=download)  
  - Licença: Uso educacional e não comercial.

---

## O que foi feito

### 1. Limpeza e Padronização dos Dados

O script `limpeza_e_padronizacao_dos_dados.py` realizou:
- Tratamento de valores ausentes;
- Padronização dos nomes das colunas;
- Unificação dos formatos de data;
- Normalização dos nomes dos clubes entre as três bases.

### 2. Exploração de Dados e Visualizações

Foram desenvolvidos três conjuntos principais de gráficos:

#### a. Gráficos Iniciais (`graficos_iniciais.py`)
- Frequência de resultados;
- Distribuição de gols por partida;
- Desempenho mandante vs visitante.

#### b. Comparação Regional (`grafico_comparacao_regioes.py`)
- Aproveitamento como mandante por estado;
- Aproveitamento como mandante por região;
- Número total de jogos como mandante por região.

#### c. Aproveitamento dos Times
- Aproveitamento como mandante;
- Aproveitamento como visitante;
- Aproveitamento geral.

### 3. Padrões, Correlações e Anomalias

A análise estatística do Campeonato Brasileiro entre 2003 e 2023 revelou:

- **Vantagem do Mandante:** Cerca de 50% das vitórias foram dos times mandantes — mais que o dobro dos visitantes.
- **Tendência de Gols:** Queda na média de gols por partida a partir de 2014, indicando jogos mais equilibrados ou defensivos.
- **Outliers de Desempenho:**  
  - *Positivos:* Grêmio, Internacional, Palmeiras e São Paulo — alto aproveitamento como mandantes.  
  - *Negativos:* América-RN e Grêmio Prudente — baixo desempenho devido a participações breves na Série A.
- **Mudanças Estruturais:**  
  - Variações no número de participantes ao longo dos anos.  
  - Redução gradual da vantagem do mandante.  
  - Impacto da pandemia em 2020, com menos jogos e queda acentuada na vantagem de jogar em casa.
- **Qualidade dos Dados:** Conjunto robusto, com boa cobertura temporal, poucos valores ausentes e padronização eficiente dos nomes dos clubes.

---

## Como Rodar o Projeto

1. Certifique-se de que os três arquivos originais das bases estão na pasta `conjunto_de_dados/`.
2. Execute o script `limpeza_e_padronizacao_dos_dados.py` para gerar as bases tratadas.
3. Com as bases limpas, execute os scripts de visualização conforme desejado:
   - `graficos_iniciais.py`
   - `grafico_comparacao_regioes.py`
   - Outros scripts de análise ou visualização

Todos os gráficos e análises dependem da execução prévia da limpeza e padronização para funcionarem corretamente.

---
