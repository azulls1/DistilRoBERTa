"""SQL de lectura. Todas las consultas trabajan sobre la corrida activa salvo que se pida otra."""
from . import db

T = {  # nombres de tabla con la nomenclatura del proyecto
    n: f'public."DistilRoBERTa_{n}"' for n in (
        "corridas", "clases", "consultas", "eda_estadisticas", "eda_balance", "ngramas", "entrenamiento",
        "metricas_clase", "confusion", "predicciones", "calibracion_llm", "explicaciones_llm", "inferencias",
        "cumplimiento", "simulacion_llm", "entregables", "entregables_generados")
}


def corrida_activa(corrida: str | None = None) -> dict | None:
    if corrida:
        return db.uno(f"select * from {T['corridas']} where id = %s", (corrida,))
    return db.uno(f"select * from {T['corridas']} where activa")


def resumen(cid: str) -> dict:
    return db.uno(
        f"""select (select count(*) from {T['clases']}) as n_clases,
                   (select count(*) from {T['predicciones']} where corrida_id = %(c)s and not correcta) as n_errores""",
        {"c": cid})


def eda(cid: str) -> dict:
    estadisticas = db.todos(
        f"""select metrica, particion, count, mean::float, std::float, min::float, q1::float,
                   mediana::float, q3::float, max::float
            from {T['eda_estadisticas']} where corrida_id = %s order by metrica, particion desc""", (cid,))
    ngramas = {"palabra": [], "bigrama": [], "trigrama": []}
    for f in db.todos(f"select tipo, rango, ngrama, frecuencia from {T['ngramas']} "
                      f"where corrida_id = %s order by tipo, rango", (cid,)):
        ngramas[f.pop("tipo")].append(f)
    balance = db.uno(f"select datos from {T['eda_balance']} where corrida_id = %s", (cid,))
    return {"estadisticas": estadisticas, "ngramas": ngramas, "balance": balance["datos"] if balance else {}}


def clases(cid: str) -> list[dict]:
    return db.todos(
        f"""select c.id, c.nombre, c.nombre_legible, c.n_train, c.n_test,
                   m.precision::float, m.recall::float, m.f1::float, m.soporte, m.errores, m.categoria
            from {T['clases']} c left join {T['metricas_clase']} m on m.clase_id = c.id and m.corrida_id = %s
            order by c.id""", (cid,))


def entrenamiento(cid: str) -> list[dict]:
    return db.todos(
        f"""select epoca, train_loss::float, eval_loss::float, eval_accuracy::float, eval_f1_macro::float
            from {T['entrenamiento']} where corrida_id = %s order by epoca""", (cid,))


def confusion(cid: str) -> dict:
    nombres = [f["nombre"] for f in db.todos(f"select nombre from {T['clases']} order by id")]
    celdas = [[f["clase_real_id"], f["clase_pred_id"], f["conteo"]] for f in db.todos(
        f"select clase_real_id, clase_pred_id, conteo from {T['confusion']} where corrida_id = %s", (cid,))]
    return {"clases": nombres, "celdas": celdas}


def pares_confundidos(cid: str, limite: int = 15) -> list[dict]:
    return db.todos(
        f"""select r.nombre as real, p.nombre as pred, x.conteo
            from {T['confusion']} x join {T['clases']} r on r.id = x.clase_real_id
                                    join {T['clases']} p on p.id = x.clase_pred_id
            where x.corrida_id = %s and x.clase_real_id <> x.clase_pred_id
            order by x.conteo desc, r.nombre limit %s""", (cid, limite))


def errores(cid: str, clase_real: int | None, limite: int, desde: int) -> dict:
    filtro = "and q.clase_id = %(clase)s" if clase_real is not None else ""
    params = {"c": cid, "clase": clase_real, "lim": limite, "off": desde}
    base = f"""from {T['predicciones']} p join {T['consultas']} q on q.id = p.consulta_id
               where p.corrida_id = %(c)s and not p.correcta {filtro}"""
    total = db.uno(f"select count(*) as n {base}", params)["n"]
    items = db.todos(
        f"""select p.consulta_id, q.texto,
                   (select nombre from {T['clases']} where id = q.clase_id) as real,
                   (select nombre from {T['clases']} where id = p.clase_pred_id) as pred,
                   p.confianza::float
            {base} order by p.confianza desc limit %(lim)s offset %(off)s""", params)
    return {"total": total, "items": items}


def explicaciones(cid: str) -> list[dict]:
    return db.todos(
        f"""select e.orden, q.texto, r.nombre as real, p.nombre as pred, e.confianza::float, e.explicacion,
                   e.salida_cruda, e.razon_categoria, e.veredicto, e.nota_revision, e.parametros, e.prompt
            from {T['explicaciones_llm']} e join {T['consultas']} q on q.id = e.consulta_id
                 join {T['clases']} r on r.id = e.clase_real_id join {T['clases']} p on p.id = e.clase_pred_id
            where e.corrida_id = %s order by e.orden""", (cid,))


def calibracion(cid: str) -> list[dict]:
    return db.todos(
        f"""select k.config, k.prompt, k.parametros, q.texto as consulta, k.salida, k.n_oraciones, k.n_tokens,
                   k.segundos::float, k.formato_ok, k.cita_falsa, k.cita_verificable, k.elegida
            from {T['calibracion_llm']} k join {T['consultas']} q on q.id = k.consulta_id
            where k.corrida_id = %s order by k.prompt, k.config, k.consulta_id""", (cid,))


# ── Inferencias en vivo ─────────────────────────────────────────────────────
def crear_inferencia(texto: str) -> dict:
    return db.uno(f"insert into {T['inferencias']} (texto) values (%s) returning id::text, creado_en", (texto,))


def asignar_tarea(inferencia_id: str, task_id: str) -> None:
    db.uno(f"update {T['inferencias']} set task_id = %s where id = %s returning id", (task_id, inferencia_id))


def inferencia_por_tarea(task_id: str) -> dict | None:
    return db.uno(
        f"""select i.id::text, i.estado, i.texto, i.confianza::float, i.top5, i.error, i.duracion_ms,
                   i.tokens, i.texto_limpio,
                   c.nombre as clase, c.nombre_legible
            from {T['inferencias']} i left join {T['clases']} c on c.id = i.clase_pred_id
            where i.task_id = %s""", (task_id,))


def completar_inferencia(inferencia_id: str, r: dict) -> None:
    from psycopg.types.json import Jsonb
    db.uno(f"""update {T['inferencias']} set estado = 'completada', clase_pred_id = %s, confianza = %s,
                      top5 = %s, tokens = %s, texto_limpio = %s, duracion_ms = %s, completado_en = now()
               where id = %s returning id""",
           (r["clase_id"], r["confianza"], Jsonb(r["top5"]), Jsonb(r["tokens"]), r["texto_limpio"],
            r["duracion_ms"], inferencia_id))


def fallar_inferencia(inferencia_id: str, error: str) -> None:
    db.uno(f"""update {T['inferencias']} set estado = 'error', error = %s, completado_en = now()
               where id = %s returning id""", (error[:500], inferencia_id))


# ── Simulación ──────────────────────────────────────────────────────────────
def muestra(cid: str, tipo: str) -> dict | None:
    """Una consulta real del conjunto de prueba: 'error' (el modelo falló), 'acierto' o 'aleatoria'."""
    filtro = {"error": "and not p.correcta", "acierto": "and p.correcta", "aleatoria": ""}[tipo]
    return db.uno(
        f"""select q.id as consulta_id, q.texto, c.nombre as real, c.nombre_legible as real_legible,
                   pc.nombre as pred_guardada, p.correcta,
                   exists(select 1 from {T['simulacion_llm']} s where s.corrida_id = p.corrida_id
                          and s.consulta_id = q.id) as tiene_llm
            from {T['predicciones']} p join {T['consultas']} q on q.id = p.consulta_id
                 join {T['clases']} c on c.id = q.clase_id join {T['clases']} pc on pc.id = p.clase_pred_id
            where p.corrida_id = %s {filtro} order by random() limit 1""", (cid,))


def simulacion_llm(cid: str, consulta_id: int) -> list[dict]:
    return db.todos(
        f"""select config, prompt, salida_cruda, explicacion, n_oraciones, segundos::float, revisada
            from {T['simulacion_llm']} where corrida_id = %s and consulta_id = %s order by config""",
        (cid, consulta_id))


def calibracion_completa(cid: str) -> list[dict]:
    """Las consultas revisadas con todas sus configuraciones de decodificación y el veredicto de la elegida."""
    return db.todos(
        f"""select e.orden, s.consulta_id, q.texto, r.nombre as real, p.nombre as pred, e.veredicto,
                   json_agg(json_build_object('config', s.config, 'prompt', s.prompt, 'explicacion', s.explicacion,
                            'salida_cruda', s.salida_cruda, 'n_oraciones', s.n_oraciones, 'segundos', s.segundos)
                            order by s.config) as configs
            from {T['explicaciones_llm']} e
                 join {T['simulacion_llm']} s on s.corrida_id = e.corrida_id and s.consulta_id = e.consulta_id
                 join {T['consultas']} q on q.id = e.consulta_id
                 join {T['clases']} r on r.id = e.clase_real_id join {T['clases']} p on p.id = e.clase_pred_id
            where e.corrida_id = %s group by e.orden, s.consulta_id, q.texto, r.nombre, p.nombre, e.veredicto
            order by e.orden""", (cid,))


# ── Cumplimiento y entregables ─────────────────────────────────────────────
def cumplimiento() -> list[dict]:
    return db.todos(f"select *, puntos::float as puntos from {T['cumplimiento']} order by orden")


def entregables(cid: str) -> list[dict]:
    return db.todos(f"select criterio, tipo, nombre, detalle, ruta, bytes, sha256 from {T['entregables']} "
                    f"where corrida_id = %s order by id", (cid,))


def entregable_por_ruta(cid: str, ruta: str) -> dict | None:
    return db.uno(f"select ruta, nombre from {T['entregables']} where corrida_id = %s and ruta = %s", (cid, ruta))


def crear_paquete() -> dict:
    return db.uno(f"insert into {T['entregables_generados']} default values returning id::text")


def asignar_tarea_paquete(pid: str, task_id: str) -> None:
    db.uno(f"update {T['entregables_generados']} set task_id = %s where id = %s returning id", (task_id, pid))


def completar_paquete(pid: str, archivo: str, n: int, bytes_: int, sha: str) -> None:
    db.uno(f"""update {T['entregables_generados']} set estado = 'completada', archivo = %s, n_archivos = %s,
                      bytes = %s, sha256 = %s, completado_en = now() where id = %s returning id""",
           (archivo, n, bytes_, sha, pid))


def fallar_paquete(pid: str, error: str) -> None:
    db.uno(f"""update {T['entregables_generados']} set estado = 'error', error = %s, completado_en = now()
               where id = %s returning id""", (error[:500], pid))


def paquete(pid: str) -> dict | None:
    return db.uno(f"select id::text, estado, archivo, bytes, sha256, n_archivos, error, creado_en, completado_en "
                  f"from {T['entregables_generados']} where id::text = %s or task_id = %s", (pid, pid))


def ultimo_paquete() -> dict | None:
    return db.uno(f"select id::text, estado, archivo, bytes, sha256, n_archivos, creado_en, completado_en "
                  f"from {T['entregables_generados']} where estado = 'completada' order by completado_en desc limit 1")


def paquete_en_curso(max_minutos: int = 5) -> dict | None:
    """Un paquete que se está generando ahora mismo (para no encolar otro igual)."""
    return db.uno(f"select id::text, estado from {T['entregables_generados']} where estado = 'pendiente' "
                  f"and creado_en > now() - make_interval(mins => %s) order by creado_en desc limit 1", (max_minutos,))


def purgar_paquetes(conservar: int) -> list[str]:
    """Borra los registros de paquetes salvo los `conservar` completados más recientes; devuelve sus archivos."""
    filas = db.todos(
        f"""delete from {T['entregables_generados']} where id not in (
                select id from {T['entregables_generados']} where estado = 'completada'
                order by completado_en desc limit %s)
            and (estado <> 'pendiente' or creado_en < now() - interval '1 hour')
            returning archivo""", (conservar,))
    return [f["archivo"] for f in filas if f.get("archivo")]


def paquetes_recientes(limite: int = 5) -> list[dict]:
    return db.todos(f"select id::text, estado, archivo, bytes, sha256, n_archivos, creado_en, completado_en "
                    f"from {T['entregables_generados']} order by creado_en desc limit %s", (limite,))
