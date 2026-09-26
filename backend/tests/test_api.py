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
    return TestClient(main.app, headers={"X-Requested-With": main.CABECERA_PORTAL})


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
        "top5": [{"clase": "card_arrival", "prob": 0.98}], "duracion_ms": 30, "error": None,
        "tokens": [{"id": 0, "token": "<s>"}], "texto_limpio": "still received card"})
    r = cliente.get("/api/tareas/abc").json()
    assert r["estado"] == "completada" and r["resultado"]["clase"] == "card_arrival"


def test_tarea_inexistente(cliente, monkeypatch):
    monkeypatch.setattr(consultas, "inferencia_por_tarea", lambda t: None)
    assert cliente.get("/api/tareas/nada").status_code == 404


def test_muestra_tipo_invalido(cliente):
    assert cliente.get("/api/simulacion/muestra?tipo=otro").status_code == 422


@pytest.mark.parametrize("ruta", ["../../etc/passwd", "no_registrado.txt"])
def test_archivo_no_registrado_no_se_sirve(cliente, monkeypatch, ruta):
    monkeypatch.setattr(consultas, "entregable_por_ruta", lambda cid, r: None)
    assert cliente.get("/api/entregables/archivo", params={"ruta": ruta}).status_code == 404


def test_archivo_registrado_fuera_de_la_carpeta_no_se_sirve(cliente, monkeypatch, tmp_path):
    from app import config as cfg
    monkeypatch.setattr(consultas, "entregable_por_ruta", lambda cid, r: {"ruta": r, "nombre": "x"})
    monkeypatch.setattr(cfg.config(), "dir_entregables", tmp_path)
    assert cliente.get("/api/entregables/archivo", params={"ruta": "../secreto"}).status_code == 404


def test_archivo_registrado_se_descarga(cliente, monkeypatch, tmp_path):
    from app import config as cfg
    (tmp_path / "figuras").mkdir()
    (tmp_path / "figuras" / "a.png").write_bytes(b"png")
    monkeypatch.setattr(consultas, "entregable_por_ruta", lambda cid, r: {"ruta": r, "nombre": "a"})
    monkeypatch.setattr(cfg.config(), "dir_entregables", tmp_path)
    r = cliente.get("/api/entregables/archivo", params={"ruta": "figuras/a.png"})
    assert r.status_code == 200 and r.content == b"png"


@pytest.mark.parametrize("ruta", ["/api/clasificar", "/api/entregables/generar"])
def test_escrituras_exigen_cabecera_del_portal(cliente, ruta):
    """Sin la cabecera propia (lo que enviaría otra web desde el navegador de un visitante) → 403."""
    r = cliente.post(ruta, json={"texto": "hola"}, headers={"X-Requested-With": ""})
    assert r.status_code == 403


@pytest.mark.parametrize("ruta", ["/api/docs", "/api/openapi.json", "/api/redoc", "/api/inferencias"])
def test_sin_documentacion_ni_historial_publico(cliente, ruta):
    assert cliente.get(ruta).status_code == 404


def test_generar_reutiliza_el_paquete_en_curso(cliente, monkeypatch):
    monkeypatch.setattr(consultas, "paquete_en_curso", lambda: {"id": "p-1", "estado": "pendiente"})
    monkeypatch.setattr(consultas, "crear_paquete", lambda: pytest.fail("no debe crear otro paquete"))
    r = cliente.post("/api/entregables/generar")
    assert r.status_code == 202 and r.json() == {"paquete_id": "p-1", "task_id": "p-1", "reutilizado": True}
