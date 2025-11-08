FROM python:3.11-slim AS builder
WORKDIR /app
COPY Project/requirements.txt Project/requirements.txt
RUN pip install --no-cache-dir -r Project/requirements.txt
COPY . .

FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
RUN apt-get update \ 
 && apt-get install -y --no-install-recommends wget \ 
 && rm -rf /var/lib/apt/lists/*
COPY --from=builder /app /app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD wget -qO- http://localhost:8000/readyz || exit 1
CMD ["gunicorn","-k","uvicorn.workers.UvicornWorker","-w","2","-b","0.0.0.0:8000","Deploy.api.serve.app:app"]
