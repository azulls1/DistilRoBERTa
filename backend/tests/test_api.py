"""Pruebas de contrato de la API sin base de datos ni Redis reales (se sustituyen por dobles)."""
import pytest
from fastapi.testclient import TestClient

from app import consultas, main

CORRIDA = {"id": "2026-09-25T23-50", "modelo_base": "distilroberta-base", "llm": "tiiuae/falcon-7b-instruct",
           "accuracy": 0.93, "f1_macro": 0.93, "duracion_entrenamiento_s": 600.0, "activa": True}


@pytest.fixture
def cliente(monkeypatch):
    monkeypatch.setattr(consultas, "corrida_activa", lambda corrida=None: dict(CORRIDA) if corrida in (None, CORRIDA["id"]) else None)
    monkeypatch.setattr(consultas, "resumen", lambda cid: {"n_clases": 77, "n_errores": 210})
    monkeypatch.setattr(consultas, "clases", lambda cid: [{"id": 0, "nombre": "activate_my_card", "f1": 0.97}])
    return TestClient(main.app)


def test_resumen(cliente):
    r = cliente.get("/api/resumen")
    assert r.status_code == 200
    assert r.json()["corrida"]["accuracy"] == 0.93 and r.json()["n_clases"] == 77


def test_corrida_inexistente(cliente):
    assert cliente.get("/api/clases?corrida=otra").status_code == 404


def test_clases(cliente):
    assert cliente.get("/api/clases").json()[0]["nombre"] == "activate_my_card"


@pytest.mark.parametrize("texto", ["", "   ", "x" * 513])
def test_clasificar_valida_longitud(cliente, texto):
    assert cliente.post("/api/clasificar", json={"texto": texto}).status_code == 422


def test_clasificar_encola(cliente, monkeypatch):
    from app import tareas

    monkeypatch.setattr(consultas, "crear_inferencia", lambda t: {"id": "abc"})
    monkeypatch.setattr(consultas, "asignar_tarea", lambda i, t: None)

    class Resultado:
        id = "abc"

    monkeypatch.setattr(tareas.clasificar, "apply_async", lambda **kw: Resultado())
    r = cliente.post("/api/clasificar", json={"texto": "I still have not received my card"})
    assert r.status_code == 202 and r.json() == {"task_id": "abc", "inferencia_id": "abc"}


def test_tarea_completada(cliente, monkeypatch):
    monkeypatch.setattr(consultas, "inferencia_por_tarea", lambda t: {
        "estado": "completada", "clase": "card_arrival", "nombre_legible": "card arrival", "confianza": 0.98,
        "top5": [{"clase": "card_arrival", "prob": 0.98}], "duracion_ms": 30, "error": None})
    r = cliente.get("/api/tareas/abc").json()
    assert r["estado"] == "completada" and r["resultado"]["clase"] == "card_arrival"


def test_tarea_inexistente(cliente, monkeypatch):
    monkeypatch.setattr(consultas, "inferencia_por_tarea", lambda t: None)
    assert cliente.get("/api/tareas/nada").status_code == 404
