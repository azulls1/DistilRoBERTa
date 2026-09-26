"""Arma la carpeta entregables/ (lo que la web ofrece para descargar y empaqueta en el ZIP) y su
manifiesto artefactos/entregables.json con criterio de la rúbrica, tamaño y SHA-256 de cada archivo.

Incluye: notebook y PDF, las figuras extraídas del notebook ejecutado, los datos de resultados,
el código fuente del análisis y las especificaciones de Spec Kit (carpeta hermana espesificaciones/).

Uso:  python ml/preparar_entregables.py
"""
import base64
import hashlib
import json
import re
import shutil
from pathlib import Path

import nbformat

RAIZ = Path(__file__).resolve().parents[1]
DEST = RAIZ / "entregables"
ESPEC = RAIZ.parent / "espesificaciones"
NB = RAIZ / "notebooks" / "SCA_Actividad2_DistilRoBERTa_Banking77.ipynb"
CRIT_SECCION = {"2": "C1", "3": "C2", "4": "C3"}

FIGURAS = {  # título legible por sección del notebook
    "2.1": "Distribución de longitud y número de palabras",
    "2.3": "Palabras, bigramas y trigramas más frecuentes",
    "2.4": "Nube de palabras",
    "2.5": "Balance de clases en entrenamiento",
    "3.3": "Curvas de entrenamiento y validación",
    "3.4": "Matriz de confusión (conteos y solo errores)",
    "3.5": "F1 por clase con las 7 mejores y 7 peores",
    "4.6": "Veredicto manual y razones del LLM",
}
DATOS = [
    ("corrida.json", "C2", "Hiperparámetros, accuracy, F1 macro y duración de la corrida"),
    ("eda.json", "C1", "Estadísticas descriptivas, n-gramas y balance"),
    ("entrenamiento.json", "C2", "Pérdida y métricas por época"),
    ("metricas_clase.json", "C2", "Precision, recall, F1 y errores de las 77 clases"),
    ("confusion.json", "C2", "Matriz de confusión dispersa"),
    ("predicciones.csv", "C2", "Predicción, confianza y top-5 de las 3 080 consultas de prueba"),
    ("calibracion.json", "C3", "Calibración del prompt: 3 configuraciones × 5 consultas"),
    ("explicaciones.json", "C3", "Las 20 explicaciones de Falcon-7b-instruct con su prompt"),
    ("revision_manual.json", "C3", "Veredicto manual y razón de cada explicación"),
    ("simulacion_llm.json", "C3", "Salidas reales de Falcon para los 208 errores y A/B/C sobre las 20"),
    ("cumplimiento.json", "C5", "Matriz de cumplimiento del enunciado (33 requisitos)"),
]
CODIGO = [
    ("ml/construir_notebook.py", "Genera el notebook celda a celda"),
    ("ml/ejecutar_notebook.py", "Ejecuta el notebook en un kernel vivo con pausa para la revisión manual"),
    ("ml/interpretar.py", "Inserta las interpretaciones escritas con las cifras reales"),
    ("ml/exportar_pdf.py", "Exporta el notebook a PDF"),
    ("ml/simulacion_llm.py", "Genera las salidas de Falcon para la simulación"),
    ("ml/cargar_resultados.py", "Carga los resultados en la base de datos"),
]
ESPECS = [
    (".specify/memory/constitution.md", "Constitución del proyecto (Spec Kit)"),
    ("specs/001-clasificador-intenciones-banking77/spec.md", "Especificación funcional (Spec Kit)"),
    ("specs/001-clasificador-intenciones-banking77/plan.md", "Plan técnico (Spec Kit)"),
    ("specs/001-clasificador-intenciones-banking77/tasks.md", "Tareas y criterios de éxito medidos (Spec Kit)"),
]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


ZIP_ENTREGA = "entregable-actividad-2-adonai-hernandez.zip"
CARPETAS_ZIP = {"Notebook": "01-notebook", "PDF": "02-pdf", "Figura": "03-figuras"}
RESULTADOS_ZIP = {"corrida.json", "metricas_clase.json", "explicaciones.json", "revision_manual.json",
                  "calibracion.json", "cumplimiento.json"}


def construir_zip_entrega(items: list[dict]) -> str:
    """ZIP que se sube a Moodle. El LEEME se escribe con las cifras de los artefactos, no a mano."""
    import zipfile
    from collections import defaultdict
    from datetime import date

    art = RAIZ / "artefactos"
    c = json.load(open(art / "corrida.json"))
    cumpl = json.load(open(art / "cumplimiento.json"))
    rev = json.load(open(art / "revision_manual.json"))
    ver = {v: sum(r["veredicto"] == v for r in rev) for v in ("pertinente", "parcial", "alucinada")}

    elegidos = []  # (ruta en catálogo, ruta dentro del zip)
    for i in items:
        if i["tipo"] in CARPETAS_ZIP:
            elegidos.append((i["ruta"], f"{CARPETAS_ZIP[i['tipo']]}/{Path(i['ruta']).name}"))
        elif i["tipo"] == "Datos" and Path(i["ruta"]).name in RESULTADOS_ZIP:
            elegidos.append((i["ruta"], f"04-resultados/{Path(i['ruta']).name}"))

    por_crit = defaultdict(list)
    for q in cumpl:
        por_crit[q["criterio"]].append(q)
    filas = []
    for crit, qs in por_crit.items():
        if qs[0]["puntos"] is None:
            continue
        partes = [x.strip() for q in qs for x in q["seccion_notebook"].split("·") if x.strip() not in ("Todo", "—")]
        secciones = ", ".join(sorted(dict.fromkeys(partes), key=lambda x: [int(n) for n in x.split(".")]))
        filas.append(f"| **{crit}** {qs[0]['criterio_nombre']} | {qs[0]['puntos']:g} | "
                     f"{sum(q['cumplido'] for q in qs)}/{len(qs)} | Notebook § {secciones or 'todo'} |")
    leeme = f"""# Entregable — Actividad 2: Transformers y Modelos de Lenguaje Grande (LLM)

Autor: **Adonai Samael Hernández Mata**
Asignatura: Sistemas Cognitivos Artificiales — Maestría en Inteligencia Artificial, UNIR México
Fecha de generación: {date.today():%d/%m/%Y}

---

## Resultados principales

- DistilRoBERTa afinado: **accuracy {c['accuracy'] * 100:.2f} %** y **F1 macro {c['f1_macro']:.3f}** en {c['n_test']:,} consultas de prueba.
- Prompt calibrado para Falcon-7b-instruct (configuración {c['hiperparametros']['config_llm']}):
  {len(rev)} errores explicados → {ver['pertinente']} pertinentes, {ver['parcial']} parciales, {ver['alucinada']} alucinadas (revisión manual).

## Mapeo a la rúbrica ({len(filas)} criterios · 10 pts)

| Criterio | Pts | Exigencias cubiertas | Dónde |
|---|---|---|---|
{chr(10).join(filas)}

## Contenido

| Carpeta | Qué contiene |
|---|---|
| `01-notebook/` | Notebook ejecutado (.ipynb): código comentado, análisis, visualizaciones y conclusiones |
| `02-pdf/` | El mismo notebook exportado a PDF |
| `03-figuras/` | Las figuras del notebook, extraídas de sus salidas |
| `04-resultados/` | Métricas, explicaciones del LLM, revisión manual y matriz de cumplimiento (JSON) |

`MANIFIESTO_SHA256.txt` trae la huella de cada archivo. Portal con todos los resultados y la simulación:
https://distilroberta.iagentek.com.mx · Código: https://github.com/azulls1/DistilRoBERTa
"""
    destino = DEST / ZIP_ENTREGA
    raiz = Path(ZIP_ENTREGA).stem
    manifiesto = []
    with zipfile.ZipFile(destino, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        z.writestr(f"{raiz}/LEEME.md", leeme)
        for origen, dentro in elegidos:
            z.write(DEST / origen, arcname=f"{raiz}/{dentro}")
            manifiesto.append(f"{sha(DEST / origen)}  {dentro}")
        z.writestr(f"{raiz}/MANIFIESTO_SHA256.txt", "\n".join(manifiesto) + "\n")
    return ZIP_ENTREGA


def main():
    if DEST.exists():
        shutil.rmtree(DEST)
    for d in ("figuras", "datos", "codigo", "especificaciones"):
        (DEST / d).mkdir(parents=True)
    corrida = json.load(open(RAIZ / "artefactos" / "corrida.json"))["id"]
    items = []

    def agregar(origen: Path, ruta: str, criterio: str, tipo: str, nombre: str, detalle: str):
        destino = DEST / ruta
        destino.parent.mkdir(parents=True, exist_ok=True)
        if origen != destino:
            shutil.copy2(origen, destino)
        items.append({"criterio": criterio, "tipo": tipo, "nombre": nombre, "detalle": detalle, "ruta": ruta,
                      "bytes": destino.stat().st_size, "sha256": sha(destino)})

    agregar(NB, NB.name, "Entrega", "Notebook", "Notebook completo (.ipynb)",
            "Código comentado, análisis, visualizaciones y conclusiones; ejecutado de punta a punta.")
    agregar(NB.with_suffix(".pdf"), NB.with_suffix(".pdf").name, "Entrega", "PDF", "Notebook exportado a PDF",
            "Versión PDF exigida por el enunciado.")

    # Figuras: se extraen de las salidas del notebook ejecutado (no se regeneran)
    nb = nbformat.read(NB, as_version=4)
    seccion = "0"
    for celda in nb.cells:
        if celda.cell_type == "markdown":
            m = re.findall(r"^###? (\d+(?:\.\d+)?)", celda.source, re.M)
            if m:
                seccion = m[-1]  # la subsección más específica de la celda
            continue
        for k, salida in enumerate(o for o in celda.outputs if "image/png" in o.get("data", {})):
            titulo = FIGURAS.get(seccion, f"Figura de la sección {seccion}")
            ruta = f"figuras/seccion_{seccion.replace('.', '_')}{'_' + str(k + 1) if k else ''}.png"
            (DEST / ruta).write_bytes(base64.b64decode(salida["data"]["image/png"]))
            agregar(DEST / ruta, ruta, CRIT_SECCION.get(seccion.split(".")[0], "C5"), "Figura", titulo,
                    f"Sección {seccion} del notebook.")

    for archivo, criterio, detalle in DATOS:
        agregar(RAIZ / "artefactos" / archivo, f"datos/{archivo}", criterio, "Datos", archivo, detalle)
    for archivo, detalle in CODIGO:
        agregar(RAIZ / archivo, f"codigo/{Path(archivo).name}", "C5", "Código", Path(archivo).name, detalle)
    agregar(RAIZ / "README.md", "README.md", "C5", "Documento", "README del repositorio", "Cómo reproducir todo.")
    if ESPEC.exists():
        for archivo, detalle in ESPECS:
            agregar(ESPEC / archivo, f"especificaciones/{Path(archivo).name}", "C5", "Especificación",
                    Path(archivo).name, detalle)

    # ZIP oficial para el profesor (mismo patrón que las otras actividades: LEEME + carpetas numeradas)
    zip_ruta = construir_zip_entrega(items)
    agregar(DEST / zip_ruta, zip_ruta, "Entrega", "Paquete", "Entregable para el profesor (.zip)",
            "Notebook, PDF, figuras y resultados, con LEEME que mapea cada criterio de la rúbrica a su evidencia.")
    items.insert(0, items.pop())  # el paquete encabeza el catálogo

    json.dump({"corrida_id": corrida, "archivos": items},
              open(RAIZ / "artefactos" / "entregables.json", "w"), indent=1, ensure_ascii=False)
    total = sum(i["bytes"] for i in items)
    print(f"{len(items)} archivos · {total / 1e6:.1f} MB → {DEST.relative_to(RAIZ)}/")


if __name__ == "__main__":
    main()
