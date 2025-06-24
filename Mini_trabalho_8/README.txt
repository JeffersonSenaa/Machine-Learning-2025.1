# Mini Trabalho 8 — Lançamento, monitoramento e manutenção do sistema

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

### 2. Requisitos

Certifique-se de ter instalado:

- Bibliotecas: pandas, numpy, scikit-learn, matplotlib, seaborn

Para instalar as dependências, execute:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### 3. Execução dos Scripts

1. **Execução do Modelo**

   ```bash
   python RandomForest.py
   ```

3. **Saída dos Resultados**
   Os scripts geram:
   - Relatórios no terminal
   - Arquivos de imagem das matrizes de confusão:
     - matriz_confusao.png
     - matriz_confusao_calibrado.png
   - Todos os arquivos são salvos no diretório do projeto

### 4. Observações

- Certifique-se de que o caminho para os arquivos CSV está correto nos scripts
- Os scripts estão preparados para rodar em ambiente local, sem necessidade de configurações adicionais
- Para dúvidas sobre a estrutura dos dados ou funcionamento dos scripts, consulte os comentários nos próprios arquivos `.py`

---