# Dockerfile
FROM python:3.12-slim

# Dependencias del sistema
RUN apt-get update && apt-get install -y \
    git \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Instalar dependencias Python primero (caché de Docker)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código
COPY . .

# Crear directorios de almacenamiento
RUN mkdir -p repository/storage/images repository/storage/embeddings

# Puerto por defecto (se sobreescribe por servicio en docker-compose)
EXPOSE 8001 8501

# El comando real lo define docker-compose por servicio
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8001"]
