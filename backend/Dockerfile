# Imagen única para la API (uvicorn) y el worker (celery); cambia solo el comando.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
    MODELO_DIR=/app/modelo

WORKDIR /app
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.4,<3"
COPY backend/requirements.txt .
RUN pip install -r requirements.txt

COPY backend/app ./app
# Pesos afinados en el notebook (no están en git; deploy.sh los copia al contexto de build)
COPY modelo ./modelo

RUN useradd --uid 10001 --no-create-home app && chown -R app /app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4)" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2", "--proxy-headers"]
