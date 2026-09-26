"""Tareas de Celery."""
import hashlib
import logging
import zipfile
from datetime import datetime

from . import consultas, modelo
from .celery_app import celery
from .config import config

log = logging.getLogger("distilroberta.tareas")
PAQUETES_CONSERVADOS = 3


@celery.task(name="distilroberta.clasificar")
def clasificar(inferencia_id: str, texto: str) -> dict:
    """Clasifica una consulta y deja el resultado (con tokens y texto limpio) en DistilRoBERTa_inferencias."""
    try:
        r = modelo.clasificar(texto)
    except Exception:
        # El detalle va al log del worker; al visitante solo un mensaje genérico
        log.exception("Fallo al clasificar la inferencia %s", inferencia_id)
        consultas.fallar_inferencia(inferencia_id, "No se pudo clasificar la consulta.")
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
        nombre = f"DistilRoBERTa_SCA_Actividad2_{datetime.now():%Y%m%d-%H%M%S}_{paquete_id[:8]}.zip"
        destino = cfg.dir_paquetes / nombre
        with zipfile.ZipFile(destino, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for a in archivos:
                z.write(cfg.dir_entregables / a["ruta"], arcname=f"DistilRoBERTa_Actividad2/{a['ruta']}")
            z.writestr("DistilRoBERTa_Actividad2/MANIFIESTO_SHA256.txt",
                       "".join(f"{a['sha256']}  {a['ruta']}\n" for a in archivos))
        sha = hashlib.sha256(destino.read_bytes()).hexdigest()
        consultas.completar_paquete(paquete_id, nombre, len(archivos) + 1, destino.stat().st_size, sha)
        _purgar(cfg.dir_paquetes)
        return {"archivo": nombre, "sha256": sha}
    except Exception:
        log.exception("Fallo al generar el paquete %s", paquete_id)
        consultas.fallar_paquete(paquete_id, "No se pudo generar el paquete.")
        raise


def _purgar(carpeta) -> None:
    """Conserva solo los últimos paquetes: borra registros viejos y cualquier ZIP que ya no esté registrado."""
    consultas.purgar_paquetes(PAQUETES_CONSERVADOS)
    vigentes = {p["archivo"] for p in consultas.paquetes_recientes(PAQUETES_CONSERVADOS + 5) if p.get("archivo")}
    for f in carpeta.glob("*.zip"):
        if f.name not in vigentes:
            f.unlink(missing_ok=True)
