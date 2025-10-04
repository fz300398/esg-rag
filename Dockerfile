# Dockerfile
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
PYTHONUNBUFFERED=1 \
PIP_NO_CACHE_DIR=1


WORKDIR /app


# Systemabhängigkeiten minimal halten; faiss-cpu & torch kommen als Wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
git \
&& rm -rf /var/lib/apt/lists/*


COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt


# Code
COPY src ./src
COPY .env ./.env


EXPOSE 8000


CMD ["uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "8000"]
