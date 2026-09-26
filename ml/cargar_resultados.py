"""Carga los artefactos del notebook (artefactos/*.json|csv) en las tablas DistilRoBERTa_* y marca
la corrida como activa. Idempotente: volver a cargar la misma corrida reemplaza sus filas.

Uso:  python ml/cargar_resultados.py --dsn "postgresql://distilroberta_app:…@host:5432/postgres"
"""
import argparse
import csv
import json
from pathlib import Path

import psycopg
from psycopg.types.json import Jsonb

ART = Path(__file__).resolve().parents[1] / "artefactos"
T = lambda n: f'public."DistilRoBERTa_{n}"'  # noqa: E731


def leer(nombre):
    return json.load(open(ART / nombre, encoding="utf-8"))


def main(dsn: str) -> None:
    corrida = leer("corrida.json")
    cid = corrida["id"]
    clases = leer("clases.json")
    eda = leer("eda.json")

    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        # Catálogo estable: clases y consultas (no dependen de la corrida)
        cur.executemany(
            f"""insert into {T('clases')} (id, nombre, nombre_legible, n_train, n_test)
                values (%(id)s, %(nombre)s, %(nombre_legible)s, %(n_train)s, %(n_test)s)
                on conflict (id) do update set nombre = excluded.nombre, nombre_legible = excluded.nombre_legible,
                    n_train = excluded.n_train, n_test = excluded.n_test""", clases)
        with open(ART / "consultas.csv", encoding="utf-8") as f:
            filas = [(int(r["id"]), r["particion"], r["text"], r["texto_limpio"] or "", int(r["label"]),
                      int(r["n_caracteres"]), int(r["n_palabras"]), int(r["n_tokens"])) for r in csv.DictReader(f)]
        cur.executemany(
            f"""insert into {T('consultas')} (id, particion, texto, texto_limpio, clase_id, n_caracteres, n_palabras, n_tokens)
                values (%s, %s, %s, %s, %s, %s, %s, %s)
                on conflict (id) do update set texto = excluded.texto, texto_limpio = excluded.texto_limpio,
                    clase_id = excluded.clase_id, n_tokens = excluded.n_tokens""", filas)

        # La corrida: borrar la anterior con el mismo id (cascade) y volver a insertar
        cur.execute(f"delete from {T('corridas')} where id = %s", (cid,))
        cur.execute(f"update {T('corridas')} set activa = false where activa")
        cur.execute(
            f"""insert into {T('corridas')} (id, modelo_base, llm, dispositivo, hiperparametros, accuracy, f1_macro,
                    n_train, n_val, n_test, duracion_entrenamiento_s, activa)
                values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, true)""",
            (cid, corrida["modelo_base"], corrida["llm"], corrida["dispositivo"], Jsonb(corrida["hiperparametros"]),
             corrida["accuracy"], corrida["f1_macro"], corrida["n_train"], corrida["n_val"], corrida["n_test"],
             corrida["duracion_entrenamiento_s"]))

        cur.executemany(
            f"""insert into {T('eda_estadisticas')} (corrida_id, metrica, particion, count, mean, std, min, q1, mediana, q3, max)
                values (%(c)s, %(metrica)s, %(particion)s, %(count)s, %(mean)s, %(std)s, %(min)s, %(q1)s, %(mediana)s, %(q3)s, %(max)s)""",
            [{**e, "c": cid} for e in eda["estadisticas"]])
        cur.execute(f"insert into {T('eda_balance')} (corrida_id, datos) values (%s, %s)", (cid, Jsonb(eda["balance"])))
        cur.executemany(
            f"insert into {T('ngramas')} (corrida_id, tipo, rango, ngrama, frecuencia) values (%(c)s, %(tipo)s, %(rango)s, %(ngrama)s, %(frecuencia)s)",
            [{**n, "c": cid} for n in eda["ngramas"]])
        cur.executemany(
            f"""insert into {T('entrenamiento')} (corrida_id, epoca, train_loss, eval_loss, eval_accuracy, eval_f1_macro)
                values (%(c)s, %(epoch)s, %(train_loss)s, %(eval_loss)s, %(eval_accuracy)s, %(eval_f1_macro)s)""",
            [{**e, "c": cid} for e in leer("entrenamiento.json")])
        cur.executemany(
            f"""insert into {T('metricas_clase')} (corrida_id, clase_id, precision, recall, f1, soporte, errores, categoria)
                values (%(c)s, %(clase_id)s, %(precision)s, %(recall)s, %(f1)s, %(soporte)s, %(errores)s, %(categoria)s)""",
            [{**m, "c": cid} for m in leer("metricas_clase.json")])
        cur.executemany(
            f"insert into {T('confusion')} (corrida_id, clase_real_id, clase_pred_id, conteo) values (%s, %s, %s, %s)",
            [(cid, i, j, n) for i, j, n in leer("confusion.json")])

        real = {r[0]: r[4] for r in filas}
        with open(ART / "predicciones.csv", encoding="utf-8") as f:
            preds = [(cid, int(r["consulta_id"]), int(r["clase_pred_id"]), float(r["confianza"]),
                      int(r["clase_pred_id"]) == real[int(r["consulta_id"])], Jsonb(json.loads(r["top5"])))
                     for r in csv.DictReader(f)]
        cur.executemany(
            f"""insert into {T('predicciones')} (corrida_id, consulta_id, clase_pred_id, confianza, correcta, top5)
                values (%s, %s, %s, %s, %s, %s)""", preds)

        cur.executemany(
            f"""insert into {T('calibracion_llm')} (corrida_id, config, parametros, consulta_id, salida, n_oraciones,
                    n_tokens, segundos, palabras_ajenas, elegida)
                values (%(c)s, %(config)s, %(p)s, %(consulta_id)s, %(salida)s, %(n_oraciones)s, %(n_tokens)s,
                        %(segundos)s, %(palabras_ajenas)s, %(elegida)s)""",
            [{**k, "c": cid, "p": Jsonb(k["parametros"])} for k in leer("calibracion.json")])

        id_clase = {c["nombre"]: c["id"] for c in clases}
        cur.executemany(
            f"""insert into {T('explicaciones_llm')} (corrida_id, orden, consulta_id, clase_real_id, clase_pred_id,
                    confianza, prompt, parametros, salida_cruda, explicacion, razon_categoria, veredicto, nota_revision)
                values (%(c)s, %(orden)s, %(consulta_id)s, %(r)s, %(p)s, %(confianza)s, %(prompt)s, %(params)s,
                        %(salida_cruda)s, %(explicacion)s, %(razon_categoria)s, %(veredicto)s, %(nota_revision)s)""",
            [{**e, "c": cid, "r": id_clase[e["real"]], "p": id_clase[e["predicha"]], "params": Jsonb(e["parametros"])}
             for e in leer("explicaciones.json")])

        # Matriz de cumplimiento del enunciado (catálogo estable: se reemplaza completa)
        if (ART / "cumplimiento.json").exists():
            cur.execute(f"delete from {T('cumplimiento')}")
            cur.executemany(
                f"""insert into {T('cumplimiento')} (orden, criterio, criterio_nombre, puntos, peso, requisito,
                        seccion_notebook, ruta_web, evidencia, cumplido)
                    values (%(orden)s, %(criterio)s, %(criterio_nombre)s, %(puntos)s, %(peso)s, %(requisito)s,
                            %(seccion_notebook)s, %(ruta_web)s, %(evidencia)s, %(cumplido)s)""",
                leer("cumplimiento.json"))
        # Salidas reales de Falcon para la simulación
        if (ART / "simulacion_llm.json").exists():
            cur.executemany(
                f"""insert into {T('simulacion_llm')} (corrida_id, consulta_id, config, salida_cruda, explicacion,
                        n_oraciones, segundos, revisada)
                    values (%(c)s, %(consulta_id)s, %(config)s, %(salida_cruda)s, %(explicacion)s, %(n_oraciones)s,
                            %(segundos)s, %(revisada)s)""",
                [{**s, "c": cid} for s in leer("simulacion_llm.json")])
        # Catálogo de entregables con su hash
        if (ART / "entregables.json").exists():
            cur.executemany(
                f"""insert into {T('entregables')} (corrida_id, criterio, tipo, nombre, detalle, ruta, bytes, sha256)
                    values (%(c)s, %(criterio)s, %(tipo)s, %(nombre)s, %(detalle)s, %(ruta)s, %(bytes)s, %(sha256)s)""",
                [{**a, "c": cid} for a in leer("entregables.json")["archivos"]])
        conn.commit()

        for tabla in ("clases", "consultas", "predicciones", "confusion", "explicaciones_llm", "calibracion_llm", "simulacion_llm", "entregables"):
            cur.execute(f"select count(*) from {T(tabla)}" + ("" if tabla in ("clases", "consultas") else " where corrida_id = %s"),
                        () if tabla in ("clases", "consultas") else (cid,))
            print(f"DistilRoBERTa_{tabla}: {cur.fetchone()[0]:,}")
    print(f"Corrida {cid} cargada y activa · accuracy {corrida['accuracy']:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", required=True)
    main(ap.parse_args().dsn)
