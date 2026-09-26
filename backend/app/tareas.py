"""Tareas de Celery."""
from . import consultas, modelo
from .celery_app import celery


@celery.task(name="distilroberta.clasificar")
def clasificar(inferencia_id: str, texto: str) -> dict:
    """Clasifica una consulta y deja el resultado en DistilRoBERTa_inferencias."""
    try:
        r = modelo.clasificar(texto)
    except Exception as e:
        consultas.fallar_inferencia(inferencia_id, f"{type(e).__name__}: {e}")
        raise
    consultas.completar_inferencia(inferencia_id, r["clase_id"], r["confianza"], r["top5"], r["duracion_ms"])
    return r
