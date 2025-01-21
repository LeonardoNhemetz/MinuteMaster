# Use uma imagem oficial do Python como base
FROM python:3.10-slim

# Defina o diretório de trabalho dentro do contêiner
WORKDIR /app

# Copie os arquivos do projeto para o contêiner
COPY . /app

# Instale as dependências do sistema
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Instale as dependências do Python
RUN pip install --no-cache-dir -r requirements.txt

# Defina a variável de ambiente para carregar o .env
ENV PYTHONUNBUFFERED 1

# Exponha a porta que o contêiner pode usar (caso seja necessário)
# EXPOSE 5000

# Comando para rodar o programa
CMD ["python", "app.py"]
