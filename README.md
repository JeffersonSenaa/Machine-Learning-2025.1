# 📄 Machine-Learning-2025.1

## 🧩 1. Introdução

**Contextualização:**  
> Este trabalho visa explorar dados históricos do Campeonato Brasileiro com o objetivo de extrair padrões e insights que auxiliem na construção de um modelo preditivo de resultados de partidas.

**Importância da EDA:**  
> A análise exploratória é essencial para compreender os dados, identificar inconsistências, padrões relevantes e preparar o terreno para modelagem de aprendizado de máquina.

---

## 🎯 2. Objetivos

- Realizar análise estatística e visual dos dados históricos.  
- Entender a distribuição e variações de resultados (vitórias, empates, derrotas, gols).  
- Investigar correlações e padrões relevantes para modelagem preditiva.  
- Preparar os dados para as etapas seguintes de aprendizado de máquina.  

---

## 📦 3. Descrição e Carregamento dos Dados

1. **Arquivos utilizados (exemplos):**  
   - `brasileirao.csv`  
   - `times.csv`  
   - `estadios.csv`  

2. **Breve descrição de cada dataset:**  
   - **`brasileirao.csv`**: registros de partidas com datas, times, placar, estádio e rodada.  
   - **`times.csv`**: informações sobre os clubes.  
   - **`estadios.csv`**: localização e capacidade dos estádios.  

3. **Limpeza básica:**  
   - Checagem e tratamento de valores ausentes.  
   - Conversão de tipos de dados (datas, inteiros, strings).  
   - Padronização de nomes de times e colunas.  

---

## 📊 4. Exploração dos Dados

- Utilize gráficos e tabelas com comentários interpretativos.  
- **Frequência de resultados:** vitórias, empates e derrotas.  
- **Distribuição de gols por partida:** histogramas, boxplots.  
- **Desempenho dos times:** mandante vs visitante.  
- **Aproveitamento:** por rodada ou temporada.  
- **Comparações geográficas:** estados ou regiões.  
- **Correlações:** heatmap entre variáveis quantitativas.  
- **Tendências sazonais** e comportamentos fora da curva.

### 📌 Ferramentas visuais

- Gráficos de barras e linhas (Matplotlib, Seaborn, Plotly).  
- Boxplots para variação de gols.  
- Heatmaps para correlações.  

---

## 🧠 5. Padrões, Correlações e Anomalias

- Existe vantagem de jogar em casa?  
- Tendência de aumento/diminuição de gols ao longo dos anos?  
- Clubes com desempenho fora da média (outliers).  
- Presença de mudanças estruturais (ex: alterações de regras no campeonato).  
- Qualidade do dado: erros ou inconsistências impactantes?  

---

## 🧪 6. Preparação Inicial para Modelagem

1. **Variável‑alvo:**  
   - Resultado da partida (vitória / empate / derrota)

2. **Sugestão de variáveis preditoras:**  
   - Time mandante  
   - Time visitante  
   - Rodada  
   - Mando de campo  
   - Saldo de gols anterior  
   - Estado da partida  
   - Últimos resultados (forma recente)

3. **Próximos passos:**  
   - Encoding (One‑Hot, Label Encoding para times, estados, mando).  
   - Normalização de variáveis numéricas.  
   - Criação de variáveis derivadas (ex: streaks, média de gols recentes).  

---

## 🧾 7. Conclusão

- **Principais insights da EDA:**  
  - Padrões observados (ex: mando influencia resultados).  
  - Comportamento dos gols e desempenhos dos times.

- **Como isso auxilia a modelagem:**  
  - Escolha consciente das features.  
  - Evita variáveis com ruído ou irrelevantes.  
  - Permite estratégias melhores de validação.

- **Melhorias futuras:**  
  - Incluir dados de escalações, lesões ou clima.  
  - Agregar dados de outras fontes (rankings internacionais, odds de apostas, etc.).  

---