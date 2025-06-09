# Mini Trabalho 7 — Classificação de Resultados do Campeonato Brasileiro

## MEMBROS DO GRUPO

- Harryson Campos Martins — 211039466
- Pedro Henrique Muniz de Oliveira — 200059947
- Flávio Gustavo Araújo de Melo — 211030602
- Leandro de Almeida Oliveira — 211030827
- Jefferson Sena Oliveira — 200020323
- José André Rabelo Rocha — 211062016

---

## INSTRUÇÕES DE USO

Este diretório contém os scripts e dados necessários para executar e comparar modelos de classificação para prever resultados de partidas do Campeonato Brasileiro de Futebol.

### 1. Estrutura dos Arquivos

- `conjunto_de_dados_limpos/`: Dados CSV tratados e prontos para uso.
- `RandomForest.py`: Script para classificação usando Random Forest.
- `RegressaoLogistica_otimizado.py`: Script para classificação usando Regressão Logística (com otimização).
- `svm.py`: Script para classificação usando SVM (com otimização).
- `limpeza_e_padronizacao_dos_dados.py`: Script para limpeza e padronização dos dados brutos.
- Outros arquivos gerados: matrizes de confusão, relatórios de métricas, etc.

### 2. Requisitos

Certifique-se de ter instalado:

- Bibliotecas: pandas, numpy, scikit-learn, matplotlib, seaborn

Para instalar as dependências, execute:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### 3. Execução dos Scripts

1. **(Opcional) Pré-processamento dos Dados**
   Se necessário, execute o script de limpeza para gerar os dados tratados:

   ```bash
   python limpeza_e_padronizacao_dos_dados.py
   ```

2. **Execução dos Modelos**
   Execute cada script separadamente para treinar, avaliar e comparar os modelos:

   ```bash
   python RandomForest.py
   python RegressaoLogistica_otimizado.py
   python svm.py
   ```

3. **Saída dos Resultados**
   Os scripts geram:
   - Relatórios no terminal
   - Arquivos de imagem (matrizes de confusão)
   - Arquivos de texto (relatórios de métricas)
   - Todos os arquivos são salvos no diretório do projeto

### 4. Observações

- Certifique-se de que o caminho para os arquivos CSV está correto nos scripts
- Os scripts estão preparados para rodar em ambiente local, sem necessidade de configurações adicionais
- Para dúvidas sobre a estrutura dos dados ou funcionamento dos scripts, consulte os comentários nos próprios arquivos `.py`

---
