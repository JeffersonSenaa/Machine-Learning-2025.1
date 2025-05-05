# PREVISÃO DE RESULTADOS DO BRASILEIRÃO COM MACHINE LEARNING
## Mini Trabalho 4 — Preparação para Modelagem

## MEMBROS DO GRUPO:
- Harryson Campos Martins — 211039466  
- Pedro Henrique Muniz de Oliveira — 200059947  
- Flávio Gustavo Araújo de Melo — 211030602  
- Leandro de Almeida Oliveira — 211030827  
- Jefferson Sena Oliveira — 200020323  
- José André Rabelo Rocha — 211062016  

## INSTRUÇÕES DE USO

### Organização do Material
Este Mini Trabalho 4 está organizado da seguinte forma:
- `conjunto_de_dados/`: Contém os arquivos CSV originais das três bases utilizadas
- `conjunto_de_dados_limpos/`: Pasta onde serão salvos os arquivos CSV processados
- `limpeza_e_padronizacao_dos_dados.py`: Script para tratamento e uniformização das bases de dados

### Pré-requisitos
Para executar os scripts deste projeto, você precisará ter instalado:
- Bibliotecas: pandas, numpy, scikit-learn

Se necessário, instale as dependências com o comando:
```
pip install pandas numpy scikit-learn
```

### Execução dos Scripts
1. **Processamento dos dados brutos:**
   - Execute o script de limpeza e padronização para gerar as bases tratadas:
   ```
   python limpeza_e_padronizacao_dos_dados.py
   ```
   - Este passo é essencial pois os dados serão utilizados nas análises.

### Notas Importantes
- Todos os scripts foram desenvolvidos e testados em um ambiente Python
- O processamento dos dados mantém a consistência com a limpeza realizada no Mini Trabalho 3
- A padronização dos nomes dos times é fundamental para a correta identificação dos padrões nos dados