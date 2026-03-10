FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
ffmpeg \
curl \
&& rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY api ./api
COPY worker ./worker
COPY docs ./docs

EXPOSE 8080

CMD ["uvicorn","api.app:app","--host","0.0.0.0","--port","8080"]
