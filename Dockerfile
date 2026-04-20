FROM python:3.11-slim

# Evita arquivos .pyc e buffer
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Diretório de trabalho
WORKDIR /app

# Dependências de sistema (para numpy, pandas, matplotlib, tensorflow)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copia requirements
COPY requirements.txt .

# Instala dependências Python
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copia o projeto inteiro
COPY . .

# Expõe portas
EXPOSE 8000 8501
