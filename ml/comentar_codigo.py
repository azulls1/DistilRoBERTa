"""Agrega comentarios explicativos al código del notebook sin tocar una sola instrucción.

Solo inserta líneas que empiezan con «#», por eso las salidas ya ejecutadas siguen siendo válidas.
Aplica los mismos comentarios al notebook ejecutado y a su generador (ml/construir_notebook.py) y
comprueba al final que, quitando comentarios, el código quedó idéntico.

Uso:  python ml/comentar_codigo.py
"""
import re
from pathlib import Path

import nbformat

RAIZ = Path(__file__).resolve().parents[1]
NB = RAIZ / "notebooks" / "SCA_Actividad2_DistilRoBERTa_Banking77.ipynb"
GEN = RAIZ / "ml" / "construir_notebook.py"

# (línea ancla que identifica la celda, [(inicio de la línea a comentar, comentario), ...])
COMENTARIOS = [
    ("# ── Librerías ─", [
        ("import numpy as np", "Cálculo numérico, tablas, gráficas y modelos"),
        ("warnings.filterwarnings", "Salida limpia: sin avisos de versiones, columnas de texto anchas y figuras sin bordes superiores"),
        ("RAIZ = Path.cwd()", "Funciona igual si el notebook se abre desde la raíz del proyecto o desde notebooks/"),
        ("print(f\"Dispositivo", "Dejar constancia del entorno en el que se ejecutó"),
    ]),
    ("URL_BASE = ", [
        ("URL_BASE = ", "CSV oficiales de PolyAI (los mismos que descarga el script banking77.py de Hugging Face)"),
        ("train_df, test_df = cargar_particion", "Cargar las dos particiones y el orden oficial de las 77 etiquetas"),
        ("ID = {n: i", "Diccionario nombre de etiqueta → índice (0…76), igual que la ClassLabel del dataset en Hugging Face"),
        ("for df in (train_df, test_df):", "Convertir la categoría de texto a su índice numérico y comprobar que ninguna quede fuera"),
        ("print(f\"train:", "Tamaños por partición y una muestra aleatoria reproducible"),
    ]),
    ("fig, ejes = plt.subplots(1, 2, figsize=(13, 4))", [
        ("fig, ejes = plt.subplots(1, 2", "Histogramas de densidad de train y test superpuestos, con la mediana de train como referencia"),
        ("plt.tight_layout(); plt.show()", "Mostrar la figura"),
    ]),
    ("import nltk", [
        ("import nltk", "Lista de stopwords del inglés de NLTK (198 palabras); si no hay red, la equivalente de scikit-learn"),
        ("for df in (train_df, test_df):", "Aplicar la limpieza a las dos particiones (columna nueva: el texto original se conserva)"),
        ("todas = pd.concat", "Corpus completo (train + test) para las frecuencias del análisis exploratorio"),
    ]),
    ("from sklearn.feature_extraction.text import CountVectorizer", [
        ("from sklearn.feature_extraction.text import CountVectorizer", "CountVectorizer cuenta n-gramas; el patrón de token acepta también palabras de una letra"),
        ("corpus_limpio = todas", "Frecuencias sobre el texto limpio: 100 palabras (15 para la tabla, 100 para la nube), 10 bigramas y 10 trigramas"),
        ("fig, ejes = plt.subplots(1, 3", "Tres gráficas de barras horizontales, de mayor a menor frecuencia"),
        ("display(pd.concat([top_palabras", "Las tres listas lado a lado en una sola tabla"),
    ]),
    ("from wordcloud import WordCloud", [
        ("from wordcloud import WordCloud", "Nube de palabras: el tamaño de cada palabra es proporcional a su frecuencia en el texto limpio"),
        ("nube = WordCloud(", "150 palabras como máximo; semilla fija para que la disposición sea reproducible"),
        ("plt.figure(figsize=(14, 6))", "Dibujar la nube sin ejes"),
    ]),
    ("conteo = pd.DataFrame({", [
        ("conteo = pd.DataFrame({", "Número de consultas por clase en cada partición, en el orden oficial de las etiquetas"),
        ("c = conteo[\"train\"]", "Medidas de balance sobre train: extremos, razón máx/mín, coeficiente de variación y entropía"),
        ("p = c / c.sum()", "Proporción de cada clase (para la entropía normalizada: 1 = perfectamente balanceado)"),
        ("display(pd.Series(balance", "Tabla de medidas de balance"),
        ("fig, eje = plt.subplots(figsize=(17, 4.5))", "Barras de las 77 clases ordenadas de mayor a menor, con la media como línea de referencia"),
    ]),
    ("from sklearn.model_selection import train_test_split", [
        ("from sklearn.model_selection import train_test_split", "Partición de validación y conversión a Dataset de Hugging Face"),
        ("tr_df, val_df = train_test_split", "10 % de train como validación, estratificado por clase; el conjunto de prueba no se toca"),
        ("def a_dataset(df):", "Pasar un DataFrame a Dataset y tokenizarlo por lotes (el relleno se hace después, por lote)"),
        ("ds_tr, ds_val, ds_test = a_dataset", "Las tres particiones tokenizadas"),
    ]),
    ("from transformers import (AutoModelForSequenceClassification", [
        ("from transformers import (AutoModelForSequenceClassification", "Clases de Hugging Face para el fine-tuning y métricas de scikit-learn"),
        ("modelo = AutoModelForSequenceClassification.from_pretrained(", "DistilRoBERTa preentrenado + cabeza de clasificación nueva de 77 salidas (se inicializa al azar)"),
        ("def metricas(pred):", "Métricas que el Trainer calcula en validación al final de cada época"),
        ("args = TrainingArguments(", "Hiperparámetros del fine-tuning (ver la tabla anterior); se guarda y restaura la mejor época por F1 macro"),
        ("trainer = Trainer(", "El Trainer une modelo, datos, relleno dinámico, métricas y early stopping (paciencia de 2 épocas)"),
        ("t0 = time.time()", "Entrenar midiendo el tiempo total"),
    ]),
    ("# Curvas de aprendizaje a partir del historial del Trainer", [
        ("fig, ejes = plt.subplots(1, 2, figsize=(13, 4))", "Izquierda: pérdida de train y validación. Derecha: accuracy y F1 macro en validación"),
        ("print(f\"Mejor checkpoint", "La época que se conserva (y que se evalúa en prueba) es la de mayor F1 macro en validación"),
    ]),
    ("from sklearn.metrics import classification_report, confusion_matrix", [
        ("from sklearn.metrics import classification_report, confusion_matrix", "Evaluación final sobre las consultas de prueba, que el modelo no vio en el entrenamiento"),
        ("salida = trainer.predict(ds_test)", "Logits → probabilidades con softmax; la clase predicha es la de mayor probabilidad"),
        ("ACCURACY = accuracy_score", "Métricas globales"),
        ("reporte = pd.DataFrame(classification_report(", "Reporte por clase: precision, recall, F1 y soporte de las 77 intenciones"),
    ]),
    ("cm = confusion_matrix(y_true, y_pred, labels=range(N_CLASES))", [
        ("cm = confusion_matrix(", "Matriz de confusión 77 × 77 (filas: clase real; columnas: clase predicha)"),
        ("fig, ejes = plt.subplots(1, 2, figsize=(26, 12))", "Izquierda: todos los conteos en escala logarítmica. Derecha: solo los errores (diagonal a cero)"),
        ("for e in ejes:", "Etiquetas de ejes y tamaño de letra para que quepan los 77 nombres"),
    ]),
    ("por_clase = reporte.loc[ETIQUETAS", [
        ("por_clase = reporte.loc[ETIQUETAS", "Tabla por clase con el número de errores (consultas de la clase que no se predijeron bien)"),
        ("print(\"Siete clases mejor clasificadas\")", "Las dos tablas que pide el enunciado"),
        ("fig, eje = plt.subplots(figsize=(17, 4.5))", "F1 de las 77 clases ordenado, resaltando las 7 mejores y las 7 con más errores"),
    ]),
    ("from sklearn.feature_extraction.text import TfidfVectorizer", [
        ("test_df[\"pred\"] = y_pred", "Guardar la predicción y la confianza de cada consulta de prueba y quedarse con los errores"),
        ("filas = []", "Para cada clase problemática: a qué clase se va con más frecuencia y cuán parecidas son léxicamente"),
        ("print(\"\\nEjemplos", "Tres errores reales por clase problemática, de mayor a menor confianza"),
    ]),
    ("from transformers import AutoModelForCausalLM", [
        ("LLM_ID = \"tiiuae/falcon-7b-instruct\"", "Carga de Falcon-7b-instruct según el acelerador disponible"),
        ("if DEVICE == \"cuda\":\n", "Colab (GPU T4): cuantización NF4 de 4 bits con bitsandbytes (≈ 4 GB de VRAM)"),
        ("llm.eval()", "Modo inferencia: sin dropout"),
    ]),
    ("from transformers import AutoTokenizer", [
        ("MODELO_BASE = ", "Tokenizador BPE de DistilRoBERTa (el mismo vocabulario de 50 265 sub-palabras que RoBERTa)"),
        ("ejemplo = train_df", "Ejemplo: texto → tokens (sub-palabras; «Ġ» marca un espacio antes) → identificadores numéricos"),
        ("MAX_LEN = 64 if", "Regla: 64 tokens si ninguna consulta los supera; si alguna los supera, 128 (así nada se trunca)"),
    ]),
    ("def legible(etiqueta: str) -> str:", [
        ("def legible(etiqueta: str) -> str:", "Nombre de intención legible para el LLM: card_arrival → «card arrival»"),
        ("PLANTILLA = (", "Plantilla del prompt (en inglés, el idioma del dataset y del entrenamiento de Falcon). Traducción al español:\n"
         "# «Eres un analista de atención a clientes bancarios que audita un clasificador automático de intenciones.\n"
         "#  El clasificador leyó la consulta del cliente de abajo y predijo una intención que es INCORRECTA.\n"
         "#  En máximo dos oraciones cortas, explica por qué el clasificador probablemente eligió la intención\n"
         "#  predicha en lugar de la correcta. Refiérete solo a palabras que aparecen en la consulta.\n"
         "#  No inventes hechos y no des consejos al cliente.\n"
         "#  Consulta del cliente: «{texto}» · Intención predicha: {pred} · Intención correcta: {real}\n"
         "#  Explicación:»"),
        ("def construir_prompt(", "Rellenar la plantilla con una consulta y sus dos intenciones"),
        ("def contar_oraciones(", "Contar oraciones: se corta después de «.», «!» o «?» seguidos de espacio"),
    ]),
    ("seleccion = (errores_df.sort_values(\"confianza\", ascending=False)", [
        ("seleccion = (errores_df.sort_values(", "Los 20 errores de mayor confianza, con máximo 2 por clase real para tener variedad"),
        ("seleccion[\"orden\"] = range(", "Numeración 1…20 que se usa en el resto de la sección"),
    ]),
    ("CONFIGS = {", [
        ("CONFIGS = {", "Las tres configuraciones de generación que se comparan (A: codiciosa, B: T = 0.3, C: T = 1.0)"),
        ("calibracion = []", "Generar con las tres configuraciones sobre las mismas 5 consultas y medir cada salida"),
        ("cal_df = pd.DataFrame(calibracion)", "Tabla con todas las salidas de la calibración"),
    ]),
    ("resumen_cal = cal_df.groupby(\"config\").agg(", [
        ("resumen_cal = cal_df.groupby(", "Resumen por configuración: cuántas respuestas respetan 1–2 oraciones, longitud y palabras ajenas"),
        ("temp = {\"A\": 0.0", "Temperatura efectiva de cada configuración (la codiciosa equivale a T → 0)"),
        ("CONFIG_ELEGIDA = (", "Regla fijada antes de ver los resultados: más respuestas de 1–2 oraciones, luego menos palabras ajenas, luego menor temperatura"),
    ]),
    ("explicaciones = []", [
        ("explicaciones = []", "Pedir a Falcon, con la configuración elegida, una explicación de cada uno de los 20 errores"),
        ("exp_df = pd.DataFrame(explicaciones)", "Cumplimiento del formato y tiempo medio por explicación"),
        ("for _, e in exp_df.iterrows():", "Mostrar cada explicación junto a su consulta y sus dos intenciones"),
        ("json.dump(explicaciones", "Guardar las salidas crudas para la revisión manual"),
    ]),
    ("revision = json.load(open(ART / \"revision_manual.json\"))", [
        ("revision = json.load(", "Revisión manual: veredicto y razón de cada explicación, guardados en un archivo auditable"),
        ("display(exp_df[[\"orden\", \"texto\"", "Tabla con cada explicación y su veredicto"),
        ("fig, ejes = plt.subplots(1, 2, figsize=(12, 3.5))", "Conteo de veredictos y de categorías de razón"),
    ]),
    ("from datetime import datetime", [
        ("CORRIDA_ID = datetime.now()", "Identificador de esta ejecución: fecha y hora"),
        ("guardar(\"corrida.json\"", "Hiperparámetros y métricas globales de la corrida"),
        ("guardar(\"clases.json\"", "Las 77 clases con su conteo por partición"),
        ("consultas = pd.concat([", "Las 13 083 consultas con su texto limpio, longitudes y tokens (id de prueba = 100000 + índice)"),
        ("est = []", "Estadísticas descriptivas y n-gramas del análisis exploratorio"),
        ("guardar(\"entrenamiento.json\"", "Curvas de entrenamiento, métricas por clase y matriz de confusión dispersa"),
        ("top5 = np.argsort(", "Predicción, confianza y top-5 de cada consulta de prueba"),
        ("guardar(\"calibracion.json\"", "Calibración y explicaciones del LLM"),
    ]),
]


def sin_comentarios(texto: str) -> str:
    return "\n".join(l for l in texto.splitlines() if not l.strip().startswith("#"))


def comentar(texto: str, desde: int, fin: int, pares, escapado: bool = False) -> tuple[str, int]:
    """Inserta cada comentario antes de la primera línea que empieza con el prefijo, dentro de [desde, fin).
    escapado=True: el texto es código fuente de Python con las barras invertidas escapadas (el generador)."""
    pos = desde
    for prefijo, comentario in pares:
        buscar = prefijo.rstrip("\n")
        if escapado:
            buscar = buscar.replace("\\", "\\\\")
        m = re.search(r"^([ \t]*)" + re.escape(buscar), texto[pos:fin], re.M)
        if not m:
            raise SystemExit(f"No encontré «{buscar}»")
        inicio = pos + m.start()
        sangria = m.group(1)
        linea = f"{sangria}# {comentario}\n"
        primera = f"# {comentario}".split("\n")[0]
        if primera not in texto[max(0, inicio - len(linea) - 200):inicio]:  # ya estaba: no duplicar
            texto = texto[:inicio] + linea + texto[inicio:]
            fin += len(linea)
            inicio += len(linea)
        pos = inicio + len(sangria) + 1
    return texto, fin


def main():
    nb = nbformat.read(NB, as_version=4)
    antes = [sin_comentarios(c.source) for c in nb.cells if c.cell_type == "code"]
    for ancla, pares in COMENTARIOS:
        celda = next(c for c in nb.cells if c.cell_type == "code" and ancla.split("\n")[0] in c.source)
        celda.source, _ = comentar(celda.source, 0, len(celda.source), pares)
    despues = [sin_comentarios(c.source) for c in nb.cells if c.cell_type == "code"]
    assert antes == despues, "Cambió una instrucción: se aborta"
    nbformat.write(nb, NB)

    gen = GEN.read_text()
    for ancla, pares in COMENTARIOS:
        a = gen.rindex('code("""', 0, gen.index(ancla.split("\n")[0]))
        fin = gen.index('""")', a)
        gen, _ = comentar(gen, a, fin, pares, escapado=True)
    GEN.write_text(gen)

    for i, c in enumerate(c for c in nb.cells if c.cell_type == "code"):
        lineas = [l for l in c.source.splitlines() if l.strip()]
        n = sum(1 for l in lineas if l.strip().startswith("#") or '"""' in l)
        print(f"celda de código {i:2d}: {n:2d} comentarios / {len(lineas):2d} líneas")


if __name__ == "__main__":
    main()
