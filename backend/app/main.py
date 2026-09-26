"""API REST — contrato en espesificaciones/specs/001-clasificador-intenciones-banking77/contracts/api.md."""
import logging
from contextlib import asynccontextmanager

import redis
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from . import consultas, db
from .config import config

log = logging.getLogger("distilroberta")


@asynccontextmanager
async def ciclo_vida(_: FastAPI):
    yield
    db.cerrar()


app = FastAPI(title="DistilRoBERTa · Banking77", version="1.0.0", lifespan=ciclo_vida,
              docs_url="/api/docs", openapi_url="/api/openapi.json", redoc_url=None)
app.add_middleware(CORSMiddleware, allow_origins=config().cors, allow_methods=["GET", "POST"],
                   allow_headers=["content-type"])


def _corrida(corrida: str | None) -> str:
    c = consultas.corrida_activa(corrida)
    if not c:
        raise HTTPException(404, "No hay una corrida cargada" if not corrida else f"Corrida {corrida} no existe")
    return c["id"]


# ── Salud ───────────────────────────────────────────────────────────────────
@app.get("/api/health")
def health():
    estado = {"db": False, "redis": False}
    try:
        estado["db"] = db.uno("select 1 as ok")["ok"] == 1
    except Exception as e:
        log.warning("db no responde: %s", e)
    try:
        estado["redis"] = bool(redis.Redis.from_url(config().redis_url, socket_timeout=2).ping())
    except Exception as e:
        log.warning("redis no responde: %s", e)
    ok = all(estado.values())
    return JSONResponse({"status": "ok" if ok else "degradado", **estado}, status_code=200 if ok else 503)


# ── Lecturas ────────────────────────────────────────────────────────────────
@app.get("/api/resumen")
def resumen(corrida: str | None = None):
    c = consultas.corrida_activa(corrida)
    if not c:
        raise HTTPException(404, "No hay una corrida cargada")
    extra = consultas.resumen(c["id"])
    for k in ("accuracy", "f1_macro", "duracion_entrenamiento_s"):
        c[k] = float(c[k]) if c[k] is not None else None
    return {"corrida": c, **extra}


@app.get("/api/eda")
def eda(corrida: str | None = None):
    return consultas.eda(_corrida(corrida))


@app.get("/api/clases")
def clases(corrida: str | None = None):
    return consultas.clases(_corrida(corrida))


@app.get("/api/entrenamiento")
def entrenamiento(corrida: str | None = None):
    return consultas.entrenamiento(_corrida(corrida))


@app.get("/api/confusion")
def confusion(corrida: str | None = None):
    return consultas.confusion(_corrida(corrida))


@app.get("/api/confusion/pares")
def confusion_pares(corrida: str | None = None, limit: int = Query(15, ge=1, le=100)):
    return consultas.pares_confundidos(_corrida(corrida), limit)


@app.get("/api/errores")
def errores(corrida: str | None = None, clase_real: int | None = Query(None, ge=0, le=76),
            limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0)):
    return consultas.errores(_corrida(corrida), clase_real, limit, offset)


@app.get("/api/explicaciones")
def explicaciones(corrida: str | None = None):
    return consultas.explicaciones(_corrida(corrida))


@app.get("/api/calibracion")
def calibracion(corrida: str | None = None):
    return consultas.calibracion(_corrida(corrida))


# ── Clasificación en vivo (Celery) ─────────────────────────────────────────
class PeticionClasificar(BaseModel):
    texto: str = Field(..., min_length=1, max_length=512)

    @field_validator("texto")
    @classmethod
    def no_vacio(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("El texto no puede estar vacío")
        return v


@app.post("/api/clasificar", status_code=202)
def clasificar(peticion: PeticionClasificar):
    from .tareas import clasificar as tarea

    inferencia = consultas.crear_inferencia(peticion.texto)
    try:
        r = tarea.apply_async(args=[inferencia["id"], peticion.texto], task_id=inferencia["id"])
    except Exception as e:
        consultas.fallar_inferencia(inferencia["id"], f"No se pudo encolar: {e}")
        raise HTTPException(503, "La cola de clasificación no está disponible") from e
    consultas.asignar_tarea(inferencia["id"], r.id)
    return {"task_id": r.id, "inferencia_id": inferencia["id"]}


@app.get("/api/tareas/{task_id}")
def tarea(task_id: str):
    i = consultas.inferencia_por_tarea(task_id)
    if not i:
        raise HTTPException(404, "Tarea no encontrada")
    respuesta = {"estado": i["estado"]}
    if i["estado"] == "completada":
        respuesta["resultado"] = {"clase": i["clase"], "nombre_legible": i["nombre_legible"],
                                  "confianza": i["confianza"], "top5": i["top5"], "duracion_ms": i["duracion_ms"]}
    elif i["estado"] == "error":
        respuesta["error"] = i["error"]
    return respuesta


@app.get("/api/inferencias")
def inferencias(limit: int = Query(20, ge=1, le=100)):
    return consultas.inferencias_recientes(limit)
