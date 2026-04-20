# 📈 Projeto LSTM para Treinamento e Previsão de Preço das Ações (Tesla, BYD e Toyota)
**Powered by Group 9, 6MLET**

### Projeto de Machine Learning e MLOps para previsão de **preços de ações (Tesla, BYD e Toyota)** usando **redes neurais LSTM**, incluindo implantação de API, monitoramento, detecção de desvio de dados, automação de retreinamento, visualização em painel e infraestrutura Dockerizada.



## 🚀 Visão Geral

O objetivo deste projeto é prever o **preço de fechamento** das ações das empresas, utilizando dados históricos obtidos automaticamente via `yfinance`:
- Tesla (TSLA)
- BYD (BYDDF)
- Toyota (TM)

Utilizando **Redes Neurais LSTM (Long Short-Term Memory)**, adequadas para séries temporais financeiras.
O projeto atende **toda a pipeline de Machine Learning**:

1. Coleta de dados históricos
2. Pré-processamento e feature engineering
3. Treinamento e avaliação do modelo
4. Salvamento e versionamento
5. Deploy via API REST
6. Monitoramento de performance
7. Detecção de Data Drift
8. Retraining automático
9. Visualização via Dashboard
10. Dockerização completa



## 📦 Requisitos

- Python 3.11+
- pip
- Docker
- WSL



## 🧠 Tecnologias Utilizadas

- **TensorFlow** - Biblioteca open source criada pelo Google para Machine Learning e Deep Learning, serve para criar, treinar, avaliar e colocar modelos de ML em produção.
- **Keras** - Biblioteca de alto nível para Deep Learning, focada em simplicidade e produtividade, serve para criar, treinar e testar redes neurais de forma rápida.
- **Pandas** - Biblioteca para análise e manipulação de dados estruturados (como planilhas, CSV, tabelas, Parquet).
- **NumPy** - Biblioteca base para computação numérica em python, operações matemáticas de alta performance.
- **Scikit-learn** - Biblioteca Python para Machine Learning “clássico”, focada em modelos estatísticos, simplicidade e produtividade, usada para treinar, avaliar e aplicar modelos de ML sem Deep Learning.
- **FastAPI** - Biblioteca Python para criar APIs REST modernas, com foco em performance, simplicidade e tipagem forte, usado para modelos de Machine Learning, microserviços e backends rápidos.
- **Streamlit** - Biblioteca Python para criar aplicações web interativas de forma rápida e simples, focado em visualização de dados e projetos de Data Science / ML.
- **matplotlib** - Biblioteca Python para criação de gráficos e visualizações de dados, transforma números e tabelas em gráficos visuais: linhas, barras, dispersão e histogramas.
- **yfinance** - Biblioteca Python para baixar dados financeiros do Yahoo Finance, usado para análise financeira e projetos Machine Learning mercado financeiro.



## 🧱 (ML / MLOps) Arquitetura

```bash
┌────────────┐       ┌──────────────┐       ┌───────────────┐
│ Yahoo API  │ ───▶ │ Data Pipeline│ ───▶  │ LSTM Models   │
└────────────┘       └──────────────┘       └───────────────┘
                                                   │
                            ┌──────────────────────┴───────────────┐
                            │ FastAPI REST API                     │
                            │ - Training                           │
                            │ - Inference                          │
                            │ - Metrics                            │
                            │ - Retraining                         │
                            └──────────────────────┬───────────────┘
                                                   │
                                       ┌───────────▼───────────┐
                                       │ Streamlit Dashboard   │
                                       │ (Visualization & BI)  │
                                       └───────────────────────┘

```



## 🧱 Arquitetura do Projeto

```bash
tech-challenge-fase4-lstm/
│
├── api/ # FastAPI (deploy do modelo)
│   └── main.py
│
├── src/ # Pipeline de ML
│   ├── data_loader.py
│   ├── inferencia.py
│   ├── preprocessing.py
│   └── train.py
│   └── utils/
│       └── prediction_saver.py
│
├── data/
│   └── models/ # Modelos
│   │   ├── lstm_BYD.keras
│   │   ├── lstm_TOYOTA.keras
│   │   └── lstm_TSLA.keras
│   │
│   └── predictions/ # Inferência - Usar modelo já treinado para prever
│       ├── TSLA_predictions_v1_YYYYMMDD_HHMMSS.csv
│       ├── BYD_predictions_v1_YYYYMMDD_HHMMSS.csv
│       └── TOYOTA_predictions_v1_YYYYMMDD_HHMMSS.csv
│
├── dashboard/ # Streamlit Dashboard
│   └── app.py
│
├── video/
│   └── presentation.mp4 # Apresentação do projeto desenvolvido
│
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
│
└── README.md
```


## ⚙️ Como Executar o Projeto

### 1. Clonar o repositório

```bash
git clone <https://github.com/fernandotedokon/tech_challenge_fase4_lstm.git>

cd tech_challenge_fase4_lstm
```

### 2. Criar e ativar um ambiente virtual

```bash
python -m venv .venv
source .env/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows
```

### 3. Instalar as dependências

```bash
pip install -r requirements.txt
```

### 4. Inicializar API FastAPI - (Training + Inference)

```bash
uvicorn api.main:app --reload
```

### 5. Swagger - Verificar todas as rotas criadas para treinamento e predição

- **Ambiente Local**
```bash
http://127.0.0.1:8000/docs

http://localhost:8000/docs
```



## 📡 Endpoints Core

| Método | Rota                                    | Descrição |
|--------|-----------------------------------------|-----------|
| POST   | /api/train                              | Treina modelo conforme parametros informados, realizando load das ações yFinance. |
| GET    | /api/train/status/{symbol}              | Verifica status do processamento do treinamento |
| POST   | /api/predict                            | Verifica se existe treinamento e faz a predição conforme parametros informados |


### 6. Iniciar o Streamlit para exibir o Dashboard

```bash
streamlit run dashboard/app.py
```

- **Ambiente Local**
```bash
http://localhost:8501

```



## 🐳 Como Executar o Projeto via docker

1. Dockerfile único (único para API e Streamlit)
2. docker-compose.yml 
3. Estrutura de pastas recomendada
4. Como subir tudo com um comando

#### 📦 Requisitos

- Ter Docker instalado e inicializado
- Docker estar integrado WSL

▶️ Como subir tudo, estando  na raiz do projeto:
```bash
docker-compose build
docker-compose up
```

Ou em modo background:
```bash
docker-compose up -d
```

🌐 Acessos depois de subir
| Serviço   | URL                                                      |
| --------- | -------------------------------------------------------- |
| FastAPI   | [http://localhost:8000/docs](http://localhost:8000/docs) |
| Streamlit | [http://localhost:8501](http://localhost:8501)           |



## ✅ Como testar e validar cada etapa do projeto

### 1️⃣ Coleta de Dados (yfinance)

🔎 Teste manual
```bash
python -m src.data_loader
```

### 2️⃣ Treinamento e Salvar Modelo

🔎 Execução
```bash
curl -X POST http://localhost:8000/train \
-H "Content-Type: application/json" \
-d "{\"symbol\":\"TSLA\",\"start_date\":\"2015-01-01\",\"end_date\":\"2025-12-01\",\"epochs\":\"5\"}"
```

### 3️⃣ Gerar predicão
🔮 Teste /predict
```bash
curl -X POST http://localhost:8000/predict \
-H "Content-Type: application/json" \
-d "{\"symbol\":\"TSLA\",\"days_ahead\":\"5\",\"start_date\":\"2015-01-01\",\"end_date\":\"2025-12-01\"}"
```

## ✅ Validação da Pipeline (Qualidade e Confiabilidade)
> Esta validação garante confiabilidade, reprodutibilidade e aderência a boas práticas de MLOps.



## 🎬 Apresentação

#### O projeto implementa uma solução completa de **Machine Learning em produção**, utilizando modelos **LSTM para previsão de preços de ações**. Foi desenvolvida uma **API REST com FastAPI para treinamento, inferência e monitoramento**, garantindo escalabilidade e versionamento dos artefatos. **As previsões são persistidas em arquivos CSV**, permitindo auditoria, análise histórica e detecção de **data drift**.  Um **dashboard em Streamlit viabiliza a visualização interativa dos resultados e métricas do modelo**. A solução foi **containerizada com Docker**, assegurando reprodutibilidade e facilidade de deploy, alinhada às boas práticas de **MLOps**.

