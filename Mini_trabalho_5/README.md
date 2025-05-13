# Mini Trabalho 5 — Análise Comparativa de Modelos de Classificação

### MEMBROS DO GRUPO

* Harryson Campos Martins — 211039466
* Pedro Henrique Muniz de Oliveira — 200059947
* Flávio Gustavo Araújo de Melo — 211030602
* Leandro de Almeida Oliveira — 211030827
* Jefferson Sena Oliveira — 200020323
* José André Rabelo Rocha — 211062016


## 1. Objetivos

O objetivo deste trabalho é desenvolver e comparar modelos de aprendizado de máquina para prever os resultados das partidas do Campeonato Brasileiro de Futebol.
Trata-se de um problema de **classificação multiclasse**, cujas possíveis saídas são:

* Vitória do time mandante
* Empate
* Derrota do time mandante


## 2. Modelos Implementados

### 2.1 Regressão Logística

**Características:**

* Modelo linear simples e interpretável
* Rápido para treinar e fazer previsões
* Boa base para comparação com modelos mais complexos

### 2.2 Random Forest

**Características:**

* Ensemble de árvores de decisão
* Robusto a overfitting
* Capaz de capturar relações não-lineares
* Lida bem com features numéricas e categóricas

### 2.3 Support Vector Machine (SVM)

**Características:**

* Kernel RBF para capturar relações não-lineares
* `class_weight='balanced'` para lidar com desbalanceamento
* Normalização dos dados para melhor performance
* Boa capacidade de generalização


## 3. Métricas de Avaliação

### 3.1 Accuracy (Precisão Geral)

* Mede a proporção total de previsões corretas
* Útil para uma visão geral do desempenho

### 3.2 F1-Score (Macro Average)

* Equilibra precisão e recall
* Importante para lidar com desbalanceamento entre classes
* Considera igualmente todas as classes

### 3.3 Matriz de Confusão

* Visualiza os erros do modelo
* Permite identificar padrões de erro específicos
* Ajuda a entender como o modelo confunde as classes


## 4. Análise Crítica

### 4.1 Vantagens e Desvantagens

#### Regressão Logística

**Vantagens:**

* Simples e interpretável
* Rápido para treinar

**Desvantagens:**

* Pode não capturar relações complexas
* Performance limitada em problemas não-lineares

#### Random Forest

**Vantagens:**

* Alta capacidade de capturar padrões complexos
* Robusto a overfitting
* Lida bem com diversos tipos de features

**Desvantagens:**

* Pode ser mais lento para treinar
* Menos interpretável

#### SVM

**Vantagens:**

* Boa capacidade de generalização
* Efetivo em espaços de alta dimensionalidade
* Tratamento adequado de classes desbalanceadas

**Desvantagens:**

* Sensível à escala dos dados
* Pode ser computacionalmente intensivo


## 5. Instruções de Uso

### Estrutura de Pastas

* `conjunto_de_dados/`: Contém os arquivos CSV originais das três bases utilizadas
* `conjunto_de_dados_limpos/`: Armazena os arquivos CSV processados e prontos para uso nos modelos
* Scripts principais:

  * `limpeza_e_padronizacao_dos_dados.py`
  * `RandomForest.py`
  * `RegressaoLogistica.py`
  * `SVM.py`

### Requisitos

Este projeto requer as bibliotecas:

* `pandas`, `numpy`, `scikit-learn`

Para instalar as dependências, execute:

```bash
pip install pandas numpy scikit-learn
```

### Etapas de Execução

1. **Pré-processamento dos Dados**
   Execute o script de limpeza e padronização para gerar os dados tratados:

   ```bash
   python limpeza_e_padronizacao_dos_dados.py
   ```

2. **Execução dos Modelos**
   Após a geração dos dados limpos, execute os scripts dos modelos individualmente:

   ```bash
   python RegressaoLogistica.py  
   python RandomForest.py  
   python SVM.py
   ```

## 6. Considerações Finais

Cada modelo oferece vantagens específicas dependendo da complexidade do problema e dos requisitos do projeto.
A Regressão Logística é útil como baseline, enquanto Random Forest e SVM mostram-se mais robustos em capturar padrões não-lineares.
A escolha do modelo ideal deve considerar não apenas a acurácia, mas também a interpretabilidade, o tempo de execução e a robustez frente a desequilíbrios nos dados.
