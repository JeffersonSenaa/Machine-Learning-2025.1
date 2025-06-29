# Preditor de Futebol Brasileirão

Uma aplicação web moderna para predição de resultados de jogos do Campeonato Brasileiro usando Machine Learning.

## Características

- **Interface Moderna**: Design responsivo com tema de futebol
- **Machine Learning**: Modelo Random Forest treinado com dados históricos
- **Análise Preditiva**: Probabilidades de vitória, empate e derrota
- **API REST**: Backend Flask com endpoints para predições
- **Experiência Intuitiva**: Seleção fácil de times e rodadas
- **Exportação de Modelo**: Sistema de salvamento e carregamento de modelos pré-treinados com joblib

## Pré-requisitos

- Python 3.7 ou superior
- pip3
- Dados do Campeonato Brasileiro (arquivo CSV)

## Instalação e Execução

### Inicialização com Modelo Pré-treinado

Se você já possui um modelo salvo (`modelo_futebol.pkl`), a aplicação será iniciada instantaneamente:

```bash
# Primeira execução (treina e salva o modelo)
python3 app.py
# Tempo: aproximadamente 2-3 minutos (treinamento completo)

# Execuções seguintes (carrega modelo salvo)
python3 app.py
# Tempo: aproximadamente 5-10 segundos (carregamento instantâneo)
```

### Método Manual

```bash
# Instala as dependências
pip3 install -r requirements.txt

# Executa o servidor
python3 app.py
```

## Acesso

Após iniciar o servidor, acesse:
- **URL**: http://localhost:5000
- **Porta**: 5000

## Estrutura do Projeto

```
Mini_trabalho_8/
├── app.py                      # Servidor Flask principal
├── RandomForest.py            # Script de treinamento do modelo
├── requirements.txt            # Dependências Python
├── README.md                  # Documentação do projeto
├── conjunto_de_dados_limpos/  # Dados do campeonato
│   └── Campeonato_Brasileiro_de_futebol_limpo.csv
├── templates/                 # Templates HTML
│   └── index.html
└── static/                    # Arquivos estáticos
    ├── css/
    │   └── style.css
    └── js/
        └── script.js
```

## API Endpoints

### GET /api/times
Retorna lista de todos os times disponíveis.

**Resposta:**
```json
{
  "times": ["Flamengo", "Palmeiras", "São Paulo", ...]
}
```

### GET /api/rodadas
Retorna lista de rodadas disponíveis.

**Resposta:**
```json
{
  "rodadas": [1, 2, 3, ..., 38]
}
```

### POST /api/prever
Faz predição para um jogo específico.

**Requisição:**
```json
{
  "time_mandante": "Flamengo",
  "time_visitante": "Palmeiras",
  "rodada": 15
}
```

**Resposta:**
```json
{
  "time_mandante": "Flamengo",
  "time_visitante": "Palmeiras",
  "rodada": 15,
  "probabilidades": {
    "vitoria": 0.45,
    "empate": 0.25,
    "derrota": 0.30
  }
}
```

