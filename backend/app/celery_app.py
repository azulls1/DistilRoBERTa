"""Instancia de Celery: Redis como broker y como backend de resultados."""
from celery import Celery
from celery.signals import worker_process_init

from .config import config

celery = Celery("distilroberta", broker=config().redis_url, backend=config().redis_resultados,
                include=["app.tareas"])
celery.conf.update(
    task_serializer="json", result_serializer="json", accept_content=["json"],
    task_acks_late=True, worker_prefetch_multiplier=1,
    task_time_limit=60, result_expires=3600, broker_connection_retry_on_startup=True,
)


@worker_process_init.connect
def _precargar_modelo(**_):
    """Carga el modelo al arrancar cada proceso del worker, no en la primera petición."""
    from .modelo import precargar
    precargar()
