"""Tareas de Celery."""
import hashlib
import zipfile
from datetime import datetime

from . import consultas, modelo
from .celery_app import celery
from .config import config


@celery.task(name="distilroberta.clasificar")
def clasificar(inferencia_id: str, texto: str) -> dict:
    """Clasifica una consulta y deja el resultado (con tokens y texto limpio) en DistilRoBERTa_inferencias."""
    try:
        r = modelo.clasificar(texto)
    except Exception as e:
        consultas.fallar_inferencia(inferencia_id, f"{type(e).__name__}: {e}")
        raise
    consultas.completar_inferencia(inferencia_id, r)
    return {k: r[k] for k in ("clase", "confianza", "duracion_ms")}


@celery.task(name="distilroberta.generar_entregable")
def generar_entregable(paquete_id: str) -> dict:
    """Empaqueta todos los entregables del catálogo en un ZIP con su SHA-256."""
    try:
        cfg = config()
        corrida = consultas.corrida_activa()
        archivos = consultas.entregables(corrida["id"])
        cfg.dir_paquetes.mkdir(parents=True, exist_ok=True)
        nombre = f"DistilRoBERTa_SCA_Actividad2_{datetime.now():%Y%m%d-%H%M%S}.zip"
        destino = cfg.dir_paquetes / nombre
        with zipfile.ZipFile(destino, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for a in archivos:
                z.write(cfg.dir_entregables / a["ruta"], arcname=f"DistilRoBERTa_Actividad2/{a['ruta']}")
            z.writestr("DistilRoBERTa_Actividad2/MANIFIESTO_SHA256.txt",
                       "".join(f"{a['sha256']}  {a['ruta']}\n" for a in archivos))
        sha = hashlib.sha256(destino.read_bytes()).hexdigest()
        consultas.completar_paquete(paquete_id, nombre, len(archivos) + 1, destino.stat().st_size, sha)
        return {"archivo": nombre, "sha256": sha}
    except Exception as e:
        consultas.fallar_paquete(paquete_id, f"{type(e).__name__}: {e}")
        raise
