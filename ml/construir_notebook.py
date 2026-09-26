"""Construye el notebook de la Actividad 2 (SCA · UNIR) a partir de celdas definidas aquí.

El notebook se genera con código para poder versionarlo con diffs legibles y para que las
celdas de interpretación (Markdown) se completen después de ejecutar, con las cifras reales.

Uso:  python ml/construir_notebook.py
"""
from pathlib import Path

import nbformat as nbf

RAIZ = Path(__file__).resolve().parents[1]
SALIDA = RAIZ / "notebooks" / "SCA_Actividad2_DistilRoBERTa_Banking77.ipynb"

celdas = []


def md(texto, marca=None):
    c = nbf.v4.new_markdown_cell(texto.strip("\n"))
    if marca:
        c.metadata["interpretacion"] = marca
    celdas.append(c)


def code(texto, tags=None):
    c = nbf.v4.new_code_cell(texto.strip("\n"))
    if tags:
        c.metadata["tags"] = tags
    celdas.append(c)


def interpretacion(marca):
    """Celda Markdown que se completa tras la ejecución con las cifras reales."""
    md(f"<!-- INTERPRETACION:{marca} -->", marca=marca)


# ════════════════════════════════════════════════════════════════════════════
# 0. Portada
# ════════════════════════════════════════════════════════════════════════════
md("""
# Transformers y Modelos de Lenguaje Grande (LLM)
## Clasificación de intenciones bancarias con DistilRoBERTa y explicación de errores con Falcon-7b-instruct

**Asignatura:** Sistemas Cognitivos Artificiales · Maestría en Inteligencia Artificial · UNIR México  
**Actividad:** 2 (individual)  
**Autor:** Adonai Hernández  
**Fecha:** septiembre de 2026

---

### Objetivo

1. Hacer un **análisis exploratorio** de las 13 083 consultas bancarias de `PolyAI/banking77`.
2. Afinar (*fine-tuning*) el Transformer preentrenado **DistilRoBERTa** para clasificar cada
   consulta en una de **77 intenciones**, y evaluarlo con accuracy, matriz de confusión y
   precision/recall/F1 por clase.
3. Diseñar y **calibrar un prompt** para `tiiuae/falcon-7b-instruct` que explique, en máximo dos
   oraciones, por qué el clasificador se equivocó en 20 consultas, controlando las
   alucinaciones con la temperatura, la longitud de respuesta y la estructura del prompt.

### Índice

0. Entorno de ejecución
1. Carga del dataset
2. Análisis exploratorio de datos
3. Clasificación con DistilRoBERTa
4. Explicación de errores con Falcon-7b-instruct
5. Conclusiones generales
6. Referencias
7. Anexo: exportación de resultados
""")

# ════════════════════════════════════════════════════════════════════════════
# 0. Entorno
# ════════════════════════════════════════════════════════════════════════════
md("""
## 0. Entorno de ejecución

El notebook detecta el acelerador disponible y se adapta:

| Entorno | Clasificador | Falcon-7b-instruct |
|---|---|---|
| Google Colab (GPU T4) | CUDA | cuantizado a 4 bits con `bitsandbytes` (≈ 4 GB de VRAM) |
| Apple Silicon (esta ejecución) | MPS | `float16` en memoria unificada (≈ 14 GB) |
| Solo CPU | CPU | `bfloat16` en CPU (lento, pero funciona) |

Para Colab basta con descomentar la celda de instalación. Se fija la semilla `42` en todas las
librerías para que los resultados sean reproducibles.
""")
code("""
# En Google Colab, descomentar para instalar las dependencias y verificar la GPU:
# !pip install -q "transformers>=4.46" accelerate datasets scikit-learn nltk wordcloud seaborn bitsandbytes
# !nvidia-smi          # comprobar GPU T4 y ≥ 15 GB de VRAM libres antes de cargar Falcon-7b
# Solo si aparece "ValueError: Unable to avoid copy while creating an array":
# !pip install -q "numpy<=1.24.3"
""")
code("""
# ── Librerías ────────────────────────────────────────────────────────────────
import gc, json, os, re, time, random, warnings
from collections import Counter
from pathlib import Path

# Cálculo numérico, tablas, gráficas y modelos
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import transformers
from IPython.display import display, Markdown

# Salida limpia: sin avisos de versiones, columnas de texto anchas y figuras sin bordes superiores
warnings.filterwarnings("ignore")
pd.set_option("display.max_colwidth", 120)
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})

# ── Reproducibilidad ────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
transformers.set_seed(SEED)

# ── Dispositivo ─────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

# ── Rutas del proyecto ──────────────────────────────────────────────────────
# Funciona igual si el notebook se abre desde la raíz del proyecto o desde notebooks/
RAIZ = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
DATA, ART, MODELO = RAIZ / "data", RAIZ / "artefactos", RAIZ / "modelo"
for p in (DATA, ART, MODELO):
    p.mkdir(exist_ok=True)

# Dejar constancia del entorno en el que se ejecutó
print(f"Dispositivo: {DEVICE}")
print(f"torch {torch.__version__} · transformers {transformers.__version__} · numpy {np.__version__}")
if DEVICE == "cuda":
    print(torch.cuda.get_device_name(0))
""")

# ════════════════════════════════════════════════════════════════════════════
# 1. Carga
# ════════════════════════════════════════════════════════════════════════════
md("""
## 1. Carga del dataset

`PolyAI/banking77` se publica en Hugging Face con un *script* de carga (`banking77.py`). Las
versiones recientes de la librería `datasets` (≥ 4.0) **ya no ejecutan scripts** y fallan con
*"Dataset scripts are no longer supported"*. Por eso se leen directamente los **dos CSV
oficiales** a los que apunta ese script, y se toma de él el **orden oficial de las 77
etiquetas**, de modo que los identificadores de clase coinciden con los del dataset de Hugging
Face.
""")
code("""
# CSV oficiales de PolyAI (los mismos que descarga el script banking77.py de Hugging Face)
URL_BASE = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data"

def cargar_particion(nombre):
    \"\"\"Lee el CSV local si existe; si no, lo descarga del repositorio oficial de PolyAI.\"\"\"
    ruta = DATA / f"{nombre}.csv"
    if not ruta.exists():
        pd.read_csv(f"{URL_BASE}/{nombre}.csv").to_csv(ruta, index=False)
    return pd.read_csv(ruta)

def cargar_nombres_etiquetas():
    \"\"\"Orden oficial de las 77 etiquetas, tomado del script banking77.py de Hugging Face.\"\"\"
    ruta = DATA / "label_names.json"
    if not ruta.exists():
        from huggingface_hub import hf_hub_download
        script = open(hf_hub_download("PolyAI/banking77", "banking77.py", repo_type="dataset")).read()
        bloque = script.split("names=[")[1].split("]")[0]
        json.dump(re.findall(r'"([^"]+)"', bloque), open(ruta, "w"), indent=1)
    return json.load(open(ruta))

# Cargar las dos particiones y el orden oficial de las 77 etiquetas
train_df, test_df = cargar_particion("train"), cargar_particion("test")
ETIQUETAS = cargar_nombres_etiquetas()
# Diccionario nombre de etiqueta → índice (0…76), igual que la ClassLabel del dataset en Hugging Face
ID = {n: i for i, n in enumerate(ETIQUETAS)}
N_CLASES = len(ETIQUETAS)

# Convertir la categoría de texto a su índice numérico y comprobar que ninguna quede fuera
for df in (train_df, test_df):
    df["label"] = df["category"].map(ID)
    assert df["label"].notna().all(), "Hay categorías fuera del listado oficial"

# Tamaños por partición y una muestra aleatoria reproducible
print(f"train: {len(train_df):,} consultas · test: {len(test_df):,} consultas · clases: {N_CLASES}")
display(train_df.sample(8, random_state=SEED))
""")

# ════════════════════════════════════════════════════════════════════════════
# 2. EDA
# ════════════════════════════════════════════════════════════════════════════
md("""
## 2. Análisis exploratorio de datos

### 2.1 Longitud y número de palabras por consulta
""")
code("""
# Longitud en caracteres y número de palabras (separadas por espacios) de cada consulta
for df in (train_df, test_df):
    df["n_caracteres"] = df["text"].str.len()
    df["n_palabras"] = df["text"].str.split().str.len()

# Estadísticas descriptivas de ambas métricas, por partición
estadisticas = pd.concat({
    "train": train_df[["n_caracteres", "n_palabras"]].describe(),
    "test": test_df[["n_caracteres", "n_palabras"]].describe(),
}, axis=1).round(2)
display(estadisticas)
""")
code("""
# Histogramas de densidad de train y test superpuestos, con la mediana de train como referencia
fig, ejes = plt.subplots(1, 2, figsize=(13, 4))
for eje, col, titulo in zip(ejes, ["n_caracteres", "n_palabras"],
                            ["Longitud en caracteres", "Número de palabras"]):
    sns.histplot(train_df[col], bins=40, ax=eje, color="#4C72B0", label="train", stat="density")
    sns.histplot(test_df[col], bins=40, ax=eje, color="#DD8452", label="test", stat="density", alpha=.5)
    eje.axvline(train_df[col].median(), ls="--", c="k", lw=1, label="mediana train")
    eje.set_title(titulo); eje.legend()
# Mostrar la figura
plt.tight_layout(); plt.show()

# Longitud media por clase: ¿hay intenciones que se expresan con frases más largas?
largo_clase = train_df.groupby("category")["n_palabras"].mean().sort_values()
print("Clases con consultas más cortas (media de palabras):")
display(largo_clase.head(5).round(1).to_frame())
print("Clases con consultas más largas:")
display(largo_clase.tail(5).round(1).to_frame())
""")
md("### 2.2 Limpieza básica del texto")
md("""
Se aplica: (1) conversión a minúsculas, (2) eliminación de caracteres especiales (todo lo que
no sea letra, dígito o espacio) y (3) eliminación de *stopwords* del inglés (lista de NLTK).

> **Importante:** esta limpieza se usa **solo para el análisis exploratorio** (frecuencias,
> n-gramas y nube de palabras). El Transformer recibe el **texto original**, porque su
> tokenizador y su preentrenamiento aprovechan la puntuación y, sobre todo, palabras que NLTK
> considera *stopwords* pero que cambian la intención — por ejemplo *not*, *why*, *my*,
> *can't*: "*my card is **not** working*" frente a "*my card is working*".
""")
code("""
# Lista de stopwords del inglés de NLTK (198 palabras); si no hay red, la equivalente de scikit-learn
import nltk
from nltk.corpus import stopwords
try:
    STOPWORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords", quiet=True)
    try:
        STOPWORDS = set(stopwords.words("english"))
    except LookupError:  # sin red o sin certificados: lista equivalente de scikit-learn
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as STOPWORDS

def limpiar(texto: str) -> str:
    \"\"\"Minúsculas → quitar caracteres especiales → quitar stopwords y tokens de 1 carácter.\"\"\"
    texto = texto.lower()
    texto = re.sub(r"[^a-z0-9\\s]", " ", texto)
    return " ".join(t for t in texto.split() if t not in STOPWORDS and len(t) > 1)

# Aplicar la limpieza a las dos particiones (columna nueva: el texto original se conserva)
for df in (train_df, test_df):
    df["texto_limpio"] = df["text"].apply(limpiar)

# Corpus completo (train + test) para las frecuencias del análisis exploratorio
todas = pd.concat([train_df, test_df], ignore_index=True)
print(f"Stopwords usadas: {len(STOPWORDS)}")
display(todas[["text", "texto_limpio"]].sample(6, random_state=1))
""")
md("### 2.3 Palabras, bigramas y trigramas más frecuentes")
code("""
# CountVectorizer cuenta n-gramas; el patrón de token acepta también palabras de una letra
from sklearn.feature_extraction.text import CountVectorizer

def top_ngramas(textos, n, k):
    \"\"\"Devuelve los k n-gramas más frecuentes (sobre el texto ya limpio).\"\"\"
    vec = CountVectorizer(ngram_range=(n, n), token_pattern=r"(?u)\\b\\w+\\b")
    X = vec.fit_transform(textos)
    frec = np.asarray(X.sum(axis=0)).ravel()
    orden = frec.argsort()[::-1][:k]
    return pd.DataFrame({"ngrama": vec.get_feature_names_out()[orden], "frecuencia": frec[orden]})

# Frecuencias sobre el texto limpio: 100 palabras (15 para la tabla, 100 para la nube), 10 bigramas y 10 trigramas
corpus_limpio = todas["texto_limpio"]
top_palabras = top_ngramas(corpus_limpio, 1, 100)   # 100 para la nube; se muestran 15
top_bigramas = top_ngramas(corpus_limpio, 2, 10)
top_trigramas = top_ngramas(corpus_limpio, 3, 10)

# Tres gráficas de barras horizontales, de mayor a menor frecuencia
fig, ejes = plt.subplots(1, 3, figsize=(17, 5))
for eje, df_, titulo, color in zip(
        ejes, [top_palabras.head(15), top_bigramas, top_trigramas],
        ["15 palabras más frecuentes", "10 bigramas más frecuentes", "10 trigramas más frecuentes"],
        ["#4C72B0", "#55A868", "#C44E52"]):
    eje.barh(df_["ngrama"][::-1], df_["frecuencia"][::-1], color=color)
    eje.set_title(titulo)
plt.tight_layout(); plt.show()

# Las tres listas lado a lado en una sola tabla
display(pd.concat([top_palabras.head(15).reset_index(drop=True),
                   top_bigramas, top_trigramas], axis=1,
                  keys=["Palabras", "Bigramas", "Trigramas"]).fillna(""))
""")
md("### 2.4 Nube de palabras")
code("""
# Nube de palabras: el tamaño de cada palabra es proporcional a su frecuencia en el texto limpio
from wordcloud import WordCloud

# 150 palabras como máximo; semilla fija para que la disposición sea reproducible
nube = WordCloud(width=1400, height=600, background_color="white", colormap="viridis",
                 max_words=150, random_state=SEED).generate(" ".join(corpus_limpio))
# Dibujar la nube sin ejes
plt.figure(figsize=(14, 6)); plt.imshow(nube, interpolation="bilinear"); plt.axis("off")
plt.title("Nube de palabras (train + test, texto limpio)"); plt.show()
""")
md("### 2.5 ¿Está balanceado el conjunto de datos?")
code("""
# Número de consultas por clase en cada partición, en el orden oficial de las etiquetas
conteo = pd.DataFrame({
    "train": train_df["category"].value_counts(),
    "test": test_df["category"].value_counts(),
}).reindex(ETIQUETAS)

# Medidas de balance sobre train: extremos, razón máx/mín, coeficiente de variación y entropía
c = conteo["train"]
# Proporción de cada clase (para la entropía normalizada: 1 = perfectamente balanceado)
p = c / c.sum()
balance = {
    "mínimo (train)": int(c.min()), "clase mínima": c.idxmin(),
    "máximo (train)": int(c.max()), "clase máxima": c.idxmax(),
    "razón máx/mín": round(c.max() / c.min(), 2),
    "coeficiente de variación": round(c.std() / c.mean(), 3),
    "entropía normalizada": round(float(-(p * np.log(p)).sum() / np.log(N_CLASES)), 4),
    "test: valores distintos": sorted(conteo["test"].unique().tolist()),
}
# Tabla de medidas de balance
display(pd.Series(balance, name="valor").to_frame())

# Barras de las 77 clases ordenadas de mayor a menor, con la media como línea de referencia
fig, eje = plt.subplots(figsize=(17, 4.5))
orden = c.sort_values(ascending=False)
eje.bar(range(N_CLASES), orden.values, color="#4C72B0")
eje.axhline(c.mean(), c="k", ls="--", lw=1, label=f"media = {c.mean():.0f}")
eje.set_xticks(range(N_CLASES)); eje.set_xticklabels(orden.index, rotation=90, fontsize=7)
eje.set_title("Consultas por clase en train (ordenadas)"); eje.legend()
plt.tight_layout(); plt.show()
""")
md("### 2.6 Conclusiones del análisis exploratorio")
interpretacion("eda")

# ════════════════════════════════════════════════════════════════════════════
# 3. DistilRoBERTa
# ════════════════════════════════════════════════════════════════════════════
md("""
## 3. Clasificación con DistilRoBERTa

**DistilRoBERTa** (`distilroberta-base`) es una versión destilada de RoBERTa: 6 capas de
Transformer en lugar de 12, 82 M de parámetros, y aproximadamente el doble de rápida que
RoBERTa-base conservando la mayor parte de su desempeño. Se usa *transfer learning*: se parte
de los pesos preentrenados en inglés y se añade una cabeza de clasificación de 77 salidas sobre
el token `<s>`; después se afina **todo** el modelo con las consultas etiquetadas.

### 3.1 Tokenización
""")
code("""
from transformers import AutoTokenizer

# Tokenizador BPE de DistilRoBERTa (el mismo vocabulario de 50 265 sub-palabras que RoBERTa)
MODELO_BASE = "distilbert/distilroberta-base"   # alias oficial de "distilroberta-base"
tokenizer = AutoTokenizer.from_pretrained(MODELO_BASE)

# Ejemplo: texto → tokens (sub-palabras; «Ġ» marca un espacio antes) → identificadores numéricos
ejemplo = train_df["text"].iloc[0]
enc = tokenizer(ejemplo)
print("Texto   :", ejemplo)
print("Tokens  :", tokenizer.convert_ids_to_tokens(enc["input_ids"]))
print("IDs     :", enc["input_ids"])

# Distribución de la longitud en tokens (incluye <s> y </s>) para fijar max_length
for df in (train_df, test_df):
    df["n_tokens"] = [len(x) for x in tokenizer(df["text"].tolist())["input_ids"]]
q = train_df["n_tokens"].quantile([.5, .95, .99, 1]).astype(int)
print(f"\\nTokens por consulta (train) → mediana {q[.5]}, p95 {q[.95]}, p99 {q[.99]}, máximo {q[1.0]}")
# Regla: 64 tokens si ninguna consulta los supera; si alguna los supera, 128 (así nada se trunca)
MAX_LEN = 64 if max(train_df.n_tokens.max(), test_df.n_tokens.max()) <= 64 else 128
print(f"max_length elegido = {MAX_LEN}  (consultas truncadas en test: {(test_df.n_tokens > MAX_LEN).sum()})")
""")
md("""
### 3.2 Partición de validación y preparación de los datos

El conjunto de **prueba (3 080) no se toca** hasta la evaluación final. Del entrenamiento se
separa un **10 % estratificado** como validación, que se usa para elegir la mejor época
(*early stopping*).
""")
code("""
# Partición de validación y conversión a Dataset de Hugging Face
from sklearn.model_selection import train_test_split
from datasets import Dataset

# 10 % de train como validación, estratificado por clase; el conjunto de prueba no se toca
tr_df, val_df = train_test_split(train_df, test_size=0.10, stratify=train_df["label"], random_state=SEED)
print(f"entrenamiento: {len(tr_df):,} · validación: {len(val_df):,} · prueba: {len(test_df):,}")

# Pasar un DataFrame a Dataset y tokenizarlo por lotes (el relleno se hace después, por lote)
def a_dataset(df):
    ds = Dataset.from_pandas(df[["text", "label"]].reset_index(drop=True))
    return ds.map(lambda b: tokenizer(b["text"], truncation=True, max_length=MAX_LEN), batched=True,
                  remove_columns=["text"])

# Las tres particiones tokenizadas
ds_tr, ds_val, ds_test = a_dataset(tr_df), a_dataset(val_df), a_dataset(test_df)
ds_tr
""")
md("""
### 3.3 Fine-tuning

| Hiperparámetro | Valor | Motivo |
|---|---|---|
| Tasa de aprendizaje | 5e-5 con *warmup* del 10 % y decaimiento lineal | valor estándar para afinar modelos BERT |
| Tamaño de lote | 32 | cabe holgado en memoria: las consultas son cortas (mediana de 13 tokens) |
| Épocas | hasta 8, *early stopping* con paciencia 2 | evita sobreajuste: se queda la mejor época en validación |
| Métrica de selección | F1 macro en validación | trata igual a las 77 clases |
| *Weight decay* | 0.01 | regularización |
| Relleno | dinámico por lote | no se desperdicia cómputo en `<pad>` |
""")
code("""
# Clases de Hugging Face para el fine-tuning y métricas de scikit-learn
from transformers import (AutoModelForSequenceClassification, TrainingArguments, Trainer,
                          DataCollatorWithPadding, EarlyStoppingCallback)
from sklearn.metrics import accuracy_score, f1_score

# DistilRoBERTa preentrenado + cabeza de clasificación nueva de 77 salidas (se inicializa al azar)
modelo = AutoModelForSequenceClassification.from_pretrained(
    MODELO_BASE, num_labels=N_CLASES,
    id2label=dict(enumerate(ETIQUETAS)), label2id=ID)
print(f"Parámetros entrenables: {sum(p.numel() for p in modelo.parameters() if p.requires_grad)/1e6:.1f} M")

# Métricas que el Trainer calcula en validación al final de cada época
def metricas(pred):
    y_pred = pred.predictions.argmax(-1)
    return {"accuracy": accuracy_score(pred.label_ids, y_pred),
            "f1_macro": f1_score(pred.label_ids, y_pred, average="macro")}

# Warmup del 10 % expresado en pasos (compatible con transformers 4.x y 5.x)
PASOS_WARMUP = int(0.10 * 8 * np.ceil(len(ds_tr) / 32))
# Hiperparámetros del fine-tuning (ver la tabla anterior); se guarda y restaura la mejor época por F1 macro
args = TrainingArguments(
    output_dir=str(RAIZ / "salidas_entrenamiento"),
    num_train_epochs=8, learning_rate=5e-5, warmup_steps=PASOS_WARMUP, weight_decay=0.01,
    per_device_train_batch_size=32, per_device_eval_batch_size=64,
    eval_strategy="epoch", save_strategy="epoch", logging_strategy="epoch",
    load_best_model_at_end=True, metric_for_best_model="f1_macro", greater_is_better=True,
    save_total_limit=2, seed=SEED, report_to="none", dataloader_num_workers=0,
    fp16=(DEVICE == "cuda"),
)
# El Trainer une modelo, datos, relleno dinámico, métricas y early stopping (paciencia de 2 épocas)
trainer = Trainer(model=modelo, args=args, train_dataset=ds_tr, eval_dataset=ds_val,
                  processing_class=tokenizer, data_collator=DataCollatorWithPadding(tokenizer),
                  compute_metrics=metricas, callbacks=[EarlyStoppingCallback(early_stopping_patience=2)])

# Entrenar midiendo el tiempo total
t0 = time.time()
trainer.train()
DURACION_ENTRENAMIENTO = time.time() - t0
print(f"Entrenamiento: {DURACION_ENTRENAMIENTO/60:.1f} min en {DEVICE}")
""")
code("""
# Curvas de aprendizaje a partir del historial del Trainer
hist = pd.DataFrame(trainer.state.log_history)
curva = (hist.dropna(subset=["loss"])[["epoch", "loss"]].rename(columns={"loss": "train_loss"})
         .merge(hist.dropna(subset=["eval_loss"])[["epoch", "eval_loss", "eval_accuracy", "eval_f1_macro"]],
                on="epoch"))
curva["epoch"] = curva["epoch"].round().astype(int)
display(curva.round(4))

# Izquierda: pérdida de train y validación. Derecha: accuracy y F1 macro en validación
fig, ejes = plt.subplots(1, 2, figsize=(13, 4))
ejes[0].plot(curva.epoch, curva.train_loss, "o-", label="train"); ejes[0].plot(curva.epoch, curva.eval_loss, "o-", label="validación")
ejes[0].set_title("Pérdida (entropía cruzada)"); ejes[0].set_xlabel("época"); ejes[0].legend()
ejes[1].plot(curva.epoch, curva.eval_accuracy, "o-", label="accuracy"); ejes[1].plot(curva.epoch, curva.eval_f1_macro, "o-", label="F1 macro")
ejes[1].set_title("Validación"); ejes[1].set_xlabel("época"); ejes[1].legend()
plt.tight_layout(); plt.show()
# La época que se conserva (y que se evalúa en prueba) es la de mayor F1 macro en validación
print(f"Mejor checkpoint: {trainer.state.best_model_checkpoint} (F1 macro val = {trainer.state.best_metric:.4f})")
""")
md("### 3.4 Evaluación en el conjunto de prueba")
code("""
# Evaluación final sobre las consultas de prueba, que el modelo no vio en el entrenamiento
from sklearn.metrics import classification_report, confusion_matrix

# Logits → probabilidades con softmax; la clase predicha es la de mayor probabilidad
salida = trainer.predict(ds_test)
logits = torch.tensor(salida.predictions)
probs = torch.softmax(logits, dim=-1).numpy()
y_true = salida.label_ids
y_pred = probs.argmax(-1)
confianza = probs.max(-1)

# Métricas globales
ACCURACY = accuracy_score(y_true, y_pred)
F1_MACRO = f1_score(y_true, y_pred, average="macro")
print(f"Accuracy (test): {ACCURACY:.4f}   ·   F1 macro (test): {F1_MACRO:.4f}")
print(f"Errores: {(y_true != y_pred).sum()} de {len(y_true)}")

# Reporte por clase: precision, recall, F1 y soporte de las 77 intenciones
reporte = pd.DataFrame(classification_report(y_true, y_pred, target_names=ETIQUETAS,
                                             output_dict=True, zero_division=0)).T
display(reporte.round(3))
""")
code("""
# Matriz de confusión 77 × 77 (filas: clase real; columnas: clase predicha)
cm = confusion_matrix(y_true, y_pred, labels=range(N_CLASES))
# Izquierda: todos los conteos en escala logarítmica. Derecha: solo los errores (diagonal a cero)
fig, ejes = plt.subplots(1, 2, figsize=(26, 12))
sns.heatmap(cm, ax=ejes[0], cmap="Blues", norm=__import__("matplotlib.colors", fromlist=["LogNorm"]).LogNorm(vmin=1),
            xticklabels=ETIQUETAS, yticklabels=ETIQUETAS, cbar_kws={"label": "consultas (escala log)"})
ejes[0].set_title("Matriz de confusión (conteos, escala logarítmica)")
fuera = cm.copy(); np.fill_diagonal(fuera, 0)
sns.heatmap(fuera, ax=ejes[1], cmap="Reds", xticklabels=ETIQUETAS, yticklabels=ETIQUETAS,
            cbar_kws={"label": "errores"})
ejes[1].set_title("Solo errores (diagonal a cero)")
# Etiquetas de ejes y tamaño de letra para que quepan los 77 nombres
for e in ejes:
    e.set_xlabel("clase predicha"); e.set_ylabel("clase real")
    e.tick_params(labelsize=6)
plt.tight_layout(); plt.show()

# Pares de clases más confundidos (real → predicha)
pares = (pd.DataFrame([(ETIQUETAS[i], ETIQUETAS[j], fuera[i, j]) for i in range(N_CLASES)
                       for j in range(N_CLASES) if fuera[i, j] > 0], columns=["real", "predicha", "errores"])
         .sort_values("errores", ascending=False).reset_index(drop=True))
print("15 pares más confundidos:")
display(pares.head(15))
""")
md("### 3.5 Las siete clases mejor clasificadas y las siete con más errores")
code("""
# Tabla por clase con el número de errores (consultas de la clase que no se predijeron bien)
por_clase = reporte.loc[ETIQUETAS, ["precision", "recall", "f1-score", "support"]].copy()
por_clase["support"] = por_clase["support"].astype(int)
por_clase["errores"] = [int(cm[i].sum() - cm[i, i]) for i in range(N_CLASES)]

# Mejores: mayor F1; desempate por menos errores. Peores: más errores; desempate por menor F1.
mejores = por_clase.sort_values(["f1-score", "errores"], ascending=[False, True]).head(7)
peores = por_clase.sort_values(["errores", "f1-score"], ascending=[False, True]).head(7)

# Las dos tablas que pide el enunciado
print("Siete clases mejor clasificadas"); display(mejores.round(3))
print("Siete clases con más errores de clasificación"); display(peores.round(3))

# F1 de las 77 clases ordenado, resaltando las 7 mejores y las 7 con más errores
fig, eje = plt.subplots(figsize=(17, 4.5))
orden = por_clase["f1-score"].sort_values()
colores = ["#C44E52" if n in peores.index else "#55A868" if n in mejores.index else "#8c8c8c" for n in orden.index]
eje.bar(range(N_CLASES), orden.values, color=colores)
eje.set_xticks(range(N_CLASES)); eje.set_xticklabels(orden.index, rotation=90, fontsize=7)
eje.set_ylim(orden.min() - .05, 1.01); eje.set_title("F1 por clase (rojo: 7 con más errores · verde: 7 mejores)")
plt.tight_layout(); plt.show()
""")
md("""
### 3.6 Análisis de causas en las clases problemáticas

Para no quedarnos en la intuición se miden tres cosas en cada clase problemática:
(1) **a qué clases se va** cuando falla, (2) **ejemplos reales** de sus errores y (3) la
**similitud léxica** entre la clase real y la predicha, calculada como el coseno entre los
vectores TF-IDF de todas las consultas de entrenamiento de cada clase. Si los errores se
concentran en pares con vocabulario muy parecido, la causa es de solapamiento semántico y no un
defecto del entrenamiento.
""")
code("""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr

# Similitud léxica entre clases (TF-IDF de las consultas limpias de train, agregadas por clase)
docs_clase = train_df.groupby("label")["texto_limpio"].apply(" ".join).reindex(range(N_CLASES))
sim = cosine_similarity(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit_transform(docs_clase))
triu = np.triu_indices(N_CLASES, 1)
sim_pares = sim[triu]

# Guardar la predicción y la confianza de cada consulta de prueba y quedarse con los errores
test_df["pred"] = y_pred
test_df["confianza"] = confianza
errores_df = test_df[test_df["label"] != test_df["pred"]].copy()
errores_df["predicha"] = errores_df["pred"].map(lambda i: ETIQUETAS[i])

# Para cada clase problemática: a qué clase se va con más frecuencia y cuán parecidas son léxicamente
filas = []
for clase in peores.index:
    i = ID[clase]
    destinos = errores_df[errores_df.label == i]["predicha"].value_counts()
    dest = destinos.index[0]
    s = sim[i, ID[dest]]
    filas.append({"clase": clase, "errores": int(peores.loc[clase, "errores"]),
                  "principal destino": dest, "errores hacia ese destino": int(destinos.iloc[0]),
                  "similitud TF-IDF": round(float(s), 3),
                  "percentil de similitud": round(float((sim_pares < s).mean() * 100), 1)})
display(pd.DataFrame(filas))

# ¿Los pares más parecidos léxicamente se confunden más?  (correlación de Spearman)
conf_sim = fuera + fuera.T
rho, pval = spearmanr(sim_pares, conf_sim[triu])
print(f"Spearman(similitud léxica, errores entre el par) = {rho:.3f} (p = {pval:.1e}) sobre {len(sim_pares)} pares")

# Tres errores reales por clase problemática, de mayor a menor confianza
print("\\nEjemplos de errores en las clases problemáticas:")
display(errores_df[errores_df["category"].isin(peores.index)]
        .sort_values(["category", "confianza"], ascending=[True, False])
        .groupby("category").head(3)[["text", "category", "predicha", "confianza"]].round(3))
""")
code(r'''
# ── ¿Explica el EDA los errores? Dos hipótesis que salen del análisis exploratorio ──────────
# Celda autocontenida: lee los artefactos que guardó esta misma corrida (artefactos/), así se puede
# volver a ejecutar sin reentrenar. H1: el desbalance de clases (razón máx/mín 5.3×) causa errores.
# H2: las clases que comparten n-gramas (bigramas del EDA) se confunden entre sí.
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import CountVectorizer
from IPython.display import display

ART = (Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()) / "artefactos"
clases = pd.DataFrame(json.load(open(ART / "clases.json"))).set_index("id")
metr = pd.DataFrame(json.load(open(ART / "metricas_clase.json"))).set_index("clase_id")
tabla = clases.join(metr)

# H1 · Tamaño de la clase en train frente a su F1 en prueba
rho1, p1 = spearmanr(tabla["n_train"], tabla["f1"])
peores_t = tabla[tabla["categoria"] == "peor"]
pequenas = tabla.nsmallest(7, "n_train")
print(f"H1 · Spearman(ejemplos de entrenamiento, F1) = {rho1:.3f} (p = {p1:.2f}) sobre {len(tabla)} clases")
print(f"     Las 7 clases con más errores tienen {peores_t['n_train'].mean():.0f} ejemplos de media; "
      f"el promedio general es {tabla['n_train'].mean():.0f}")
print("     Las 7 clases más pequeñas y su F1:")
display(pequenas[["nombre", "n_train", "f1", "errores"]].round(3).reset_index(drop=True))

# H2 · Bigramas frecuentes compartidos entre cada par de clases (los 15 más frecuentes de cada clase)
consultas = pd.read_csv(ART / "consultas.csv")
train_q = consultas[consultas["particion"] == "train"]
docs = train_q.groupby("label")["texto_limpio"].apply(lambda s: "\n".join(s.fillna(""))).reindex(range(len(tabla)))
vec = CountVectorizer(ngram_range=(2, 2), token_pattern=r"(?u)\b\w+\b")
X = vec.fit_transform(docs.fillna(""))
vocab = np.array(vec.get_feature_names_out())
top_bi = {i: set(vocab[np.asarray(X[i].todense()).ravel().argsort()[::-1][:15]]) for i in range(len(tabla))}

conf = np.zeros((len(tabla), len(tabla)), int)
for i, j, c in json.load(open(ART / "confusion.json")):
    conf[i, j] = c
np.fill_diagonal(conf, 0)
errores_par = conf + conf.T
pares = [(i, j) for i in range(len(tabla)) for j in range(i + 1, len(tabla))]
jaccard = np.array([len(top_bi[i] & top_bi[j]) / len(top_bi[i] | top_bi[j]) for i, j in pares])
err = np.array([errores_par[i, j] for i, j in pares])
rho2, p2 = spearmanr(jaccard, err)
print(f"\nH2 · Bigramas compartidos (Jaccard): pares que se confunden {jaccard[err > 0].mean():.3f} · "
      f"pares que nunca se confunden {jaccard[err == 0].mean():.3f} "
      f"({jaccard[err > 0].mean() / jaccard[err == 0].mean():.0f}× más)")
print(f"     Spearman(bigramas compartidos, errores entre el par) = {rho2:.3f} (p = {p2:.1e})")
filas = [{"par": f"{clases.nombre[i]} ↔ {clases.nombre[j]}", "errores": int(errores_par[i, j]),
          "bigramas compartidos": ", ".join(sorted(top_bi[i] & top_bi[j])) or "—"}
         for i, j in sorted(pares, key=lambda x: -errores_par[x])[:8]]
display(pd.DataFrame(filas))
''')
interpretacion("causas")
md("### 3.7 Conclusiones del uso del Transformer")
interpretacion("transformer")
code("""
# Guardar el modelo afinado (lo usa la aplicación web para clasificar en vivo)
trainer.save_model(str(MODELO)); tokenizer.save_pretrained(str(MODELO))
print("Modelo guardado en", MODELO, "·", sorted(p.name for p in MODELO.iterdir()))
""")

# ════════════════════════════════════════════════════════════════════════════
# 4. Falcon
# ════════════════════════════════════════════════════════════════════════════
md("""
## 4. Explicación de errores con Falcon-7b-instruct

### 4.1 Carga del modelo

`tiiuae/falcon-7b-instruct` tiene 7 mil millones de parámetros: ≈ 14 GB en `float16`. Antes de
cargarlo se libera el clasificador. En Colab se cuantiza a 4 bits (NF4) para que entre en la
T4; en Apple Silicon cabe en `float16` en la memoria unificada.
""")
code("""
from transformers import AutoModelForCausalLM

# Liberar memoria del clasificador antes de cargar el LLM
del trainer, modelo
gc.collect()
if DEVICE == "cuda": torch.cuda.empty_cache()
if DEVICE == "mps": torch.mps.empty_cache()

# Carga de Falcon-7b-instruct según el acelerador disponible
LLM_ID = "tiiuae/falcon-7b-instruct"
t0 = time.time()
tok_llm = AutoTokenizer.from_pretrained(LLM_ID)
# Colab (GPU T4): cuantización NF4 de 4 bits con bitsandbytes (≈ 4 GB de VRAM)
if DEVICE == "cuda":
    from transformers import BitsAndBytesConfig
    llm = AutoModelForCausalLM.from_pretrained(LLM_ID, device_map="auto", quantization_config=BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16))
elif DEVICE == "mps":
    # Cargar directo en MPS con device_map aborta (SIGBUS) en macOS; en CPU y luego .to("mps") funciona
    llm = AutoModelForCausalLM.from_pretrained(LLM_ID, dtype=torch.float16).to("mps")
else:
    llm = AutoModelForCausalLM.from_pretrained(LLM_ID, dtype=torch.bfloat16)
# Modo inferencia: sin dropout
llm.eval()
print(f"{LLM_ID} cargado en {time.time()-t0:.0f} s · {llm.num_parameters()/1e9:.2f} B parámetros · {llm.dtype}")
""")
md("""
### 4.2 Diseño del prompt

El prompt se construye con cinco piezas, cada una pensada contra un tipo de alucinación:

| Pieza | Para qué |
|---|---|
| **Rol** («analista de soporte bancario que audita un clasificador») | fija el dominio y el tono |
| **Tarea explícita**: explicar por qué el clasificador eligió la intención *predicha* | evita que el modelo «corrija» o responda al cliente |
| **Restricciones**: máximo dos oraciones, citar palabras de la consulta, no inventar | ataca directamente la alucinación y la longitud |
| **Datos**: consulta, intención predicha e intención correcta, con nombres legibles | da al modelo todo lo que necesita y nada más |
| **Ancla de salida** `Explanation:` | el modelo empieza a escribir la explicación directamente |

Se pasan las dos intenciones porque se explican **errores**: la explicación útil es la que dice
qué palabras de la consulta empujaron hacia la clase equivocada en lugar de la correcta.
""")
code("""
# Nombre de intención legible para el LLM: card_arrival → «card arrival»
def legible(etiqueta: str) -> str:
    return etiqueta.replace("_", " ").replace("?", "").lower()

# Plantilla del prompt (en inglés, el idioma del dataset y del entrenamiento de Falcon). Traducción al español:
# «Eres un analista de atención a clientes bancarios que audita un clasificador automático de intenciones.
#  El clasificador leyó la consulta del cliente de abajo y predijo una intención que es INCORRECTA.
#  En máximo dos oraciones cortas, explica por qué el clasificador probablemente eligió la intención
#  predicha en lugar de la correcta. Refiérete solo a palabras que aparecen en la consulta.
#  No inventes hechos y no des consejos al cliente.
#  Consulta del cliente: «{texto}» · Intención predicha: {pred} · Intención correcta: {real}
#  Explicación:»
PLANTILLA = (
    "You are a banking customer-support analyst who audits an automatic intent classifier.\\n"
    "The classifier read the customer query below and predicted an intent that is WRONG.\\n"
    "In at most two short sentences, explain why the classifier probably chose the predicted "
    "intent instead of the correct one. Refer only to words that appear in the query. "
    "Do not invent facts, do not give advice to the customer.\\n\\n"
    'Customer query: "{texto}"\\n'
    "Predicted intent: {pred}\\n"
    "Correct intent: {real}\\n\\n"
    "Explanation:"
)

# Rellenar la plantilla con una consulta y sus dos intenciones
def construir_prompt(texto, pred, real):
    return PLANTILLA.format(texto=texto.strip(), pred=legible(pred), real=legible(real))

# Contar oraciones: se corta después de «.», «!» o «?» seguidos de espacio
def contar_oraciones(texto):
    return len([s for s in re.split(r"(?<=[.!?])\\s+", texto.strip()) if s])

def recortar(texto, max_oraciones=2):
    \"\"\"Postproceso: primera línea no vacía y como máximo dos oraciones.\"\"\"
    texto = texto.strip().split("\\n\\n")[0].strip()
    partes = [s for s in re.split(r"(?<=[.!?])\\s+", texto) if s]
    return " ".join(partes[:max_oraciones])

@torch.no_grad()
def generar(prompt, max_new_tokens=60, temperature=None, top_p=None):
    \"\"\"Genera con Falcon. temperature=None → decodificación codiciosa (determinista).\"\"\"
    torch.manual_seed(SEED)
    entrada = tok_llm(prompt, return_tensors="pt").to(llm.device)
    muestreo = temperature is not None
    t0 = time.time()
    out = llm.generate(**entrada, max_new_tokens=max_new_tokens, do_sample=muestreo,
                       temperature=temperature if muestreo else None, top_p=top_p if muestreo else None,
                       repetition_penalty=1.15, pad_token_id=tok_llm.eos_token_id,
                       eos_token_id=tok_llm.eos_token_id, stop_strings=["\\n\\n", "Customer query:"],
                       tokenizer=tok_llm)
    nuevos = out[0, entrada["input_ids"].shape[1]:]
    return tok_llm.decode(nuevos, skip_special_tokens=True).strip(), len(nuevos), time.time() - t0

print(construir_prompt("I still haven't received my new card", "card_delivery_estimate", "card_arrival"))
""")
md("""
### 4.3 Selección de las 20 consultas mal clasificadas

Criterio reproducible: de todos los errores en prueba se toman los **20 con mayor confianza en
la clase equivocada**, con un máximo de **2 por clase real** para que haya variedad. Son los
errores más interesantes: el modelo se equivoca *con seguridad*.
""")
code("""
# Los 20 errores de mayor confianza, con máximo 2 por clase real para tener variedad
seleccion = (errores_df.sort_values("confianza", ascending=False)
             .groupby("category").head(2).head(20).reset_index().rename(columns={"index": "idx_test"}))
# Numeración 1…20 que se usa en el resto de la sección
seleccion["orden"] = range(1, len(seleccion) + 1)
display(seleccion[["orden", "text", "category", "predicha", "confianza"]].round(3))
""")
md("""
### 4.4 Calibración: temperatura, longitud y estructura

Se comparan tres configuraciones sobre las mismas **5 consultas** (las 5 primeras de la
selección):

| Config. | Decodificación | `max_new_tokens` | Hipótesis |
|---|---|---|---|
| **A** | codiciosa (`do_sample=False`, equivale a temperatura → 0) | 60 | la más estable y fiel |
| **B** | muestreo, `temperature=0.3`, `top_p=0.9` | 60 | algo de variedad sin perder foco |
| **C** | muestreo, `temperature=1.0`, `top_p=0.95` | 150 | «creativa»: esperable que divague e invente |

En todas: `repetition_penalty=1.15` y parada en línea en blanco. Nota: el enunciado habla de
`max_length`; en `transformers` ese parámetro cuenta también los tokens del prompt, así que el
control correcto de la longitud de la **respuesta** es `max_new_tokens`.

**Regla de elección fijada de antemano**: gana la configuración con más respuestas de 1–2
oraciones *sin* recorte; desempata la que menos palabras inventa (palabras de contenido de la
respuesta que no están ni en la consulta ni en los nombres de las intenciones) y, si persiste el
empate, la de menor temperatura.
""")
code("""
# Las tres configuraciones de generación que se comparan (A: codiciosa, B: T = 0.3, C: T = 1.0)
CONFIGS = {
    "A": dict(max_new_tokens=60, temperature=None, top_p=None),
    "B": dict(max_new_tokens=60, temperature=0.3, top_p=0.9),
    "C": dict(max_new_tokens=150, temperature=1.0, top_p=0.95),
}

def palabras_ajenas(salida, texto, pred, real):
    \"\"\"Proporción de palabras de contenido de la salida que no aparecen en la consulta ni en las etiquetas.\"\"\"
    base = set(limpiar(f"{texto} {legible(pred)} {legible(real)}").split())
    vocab_tarea = {"query", "customer", "classifier", "intent", "predicted", "correct", "word", "words",
                   "mention", "mentions", "mentioned", "chose", "choose", "likely", "probably", "because",
                   "related", "refers", "instead", "classified", "suggests", "indicates"}
    cont = [w for w in limpiar(salida).split() if w not in vocab_tarea]
    return 0.0 if not cont else sum(w not in base for w in cont) / len(cont)

# Generar con las tres configuraciones sobre las mismas 5 consultas y medir cada salida
calibracion = []
for _, fila in seleccion.head(5).iterrows():
    prompt = construir_prompt(fila.text, fila.predicha, fila.category)
    for nombre, cfg in CONFIGS.items():
        salida_llm, n_tok, seg = generar(prompt, **cfg)
        calibracion.append({"config": nombre, "orden": fila.orden, "idx_test": fila.idx_test,
                            "salida": salida_llm, "n_oraciones": contar_oraciones(salida_llm),
                            "n_tokens": n_tok, "segundos": round(seg, 1),
                            "palabras_ajenas": round(palabras_ajenas(salida_llm, fila.text, fila.predicha, fila.category), 3)})
# Tabla con todas las salidas de la calibración
cal_df = pd.DataFrame(calibracion)
display(cal_df[["config", "orden", "n_oraciones", "n_tokens", "segundos", "palabras_ajenas", "salida"]])
""")
code("""
# Resumen por configuración: cuántas respuestas respetan 1–2 oraciones, longitud y palabras ajenas
resumen_cal = cal_df.groupby("config").agg(
    respuestas_1_2_oraciones=("n_oraciones", lambda s: int(s.between(1, 2).sum())),
    oraciones_media=("n_oraciones", "mean"), tokens_media=("n_tokens", "mean"),
    palabras_ajenas_media=("palabras_ajenas", "mean"), segundos_media=("segundos", "mean")).round(3)
# Temperatura efectiva de cada configuración (la codiciosa equivale a T → 0)
temp = {"A": 0.0, "B": 0.3, "C": 1.0}
resumen_cal["temperatura"] = [temp[c] for c in resumen_cal.index]
display(resumen_cal)

# Regla fijada antes de ver los resultados: más respuestas de 1–2 oraciones, luego menos palabras ajenas, luego menor temperatura
CONFIG_ELEGIDA = (resumen_cal.sort_values(["respuestas_1_2_oraciones", "palabras_ajenas_media", "temperatura"],
                                          ascending=[False, True, True]).index[0])
print(f"Configuración elegida por la regla: {CONFIG_ELEGIDA} → {CONFIGS[CONFIG_ELEGIDA]}")
cal_df["elegida"] = cal_df["config"] == CONFIG_ELEGIDA
""")
interpretacion("calibracion")
md("### 4.5 Explicación de las 20 clasificaciones incorrectas")
code("""
# Pedir a Falcon, con la configuración elegida, una explicación de cada uno de los 20 errores
explicaciones = []
for _, fila in seleccion.iterrows():
    prompt = construir_prompt(fila.text, fila.predicha, fila.category)
    cruda, n_tok, seg = generar(prompt, **CONFIGS[CONFIG_ELEGIDA])
    explicaciones.append({"orden": fila.orden, "idx_test": fila.idx_test, "texto": fila.text,
                          "real": fila.category, "predicha": fila.predicha,
                          "confianza": round(float(fila.confianza), 4), "prompt": prompt,
                          "salida_cruda": cruda, "explicacion": recortar(cruda),
                          "n_oraciones_crudas": contar_oraciones(cruda), "segundos": round(seg, 1)})
# Cumplimiento del formato y tiempo medio por explicación
exp_df = pd.DataFrame(explicaciones)
print(f"Respuestas crudas con 1–2 oraciones: {exp_df.n_oraciones_crudas.between(1, 2).sum()} de {len(exp_df)}")
print(f"Tiempo medio por explicación: {exp_df.segundos.mean():.1f} s")
# Mostrar cada explicación junto a su consulta y sus dos intenciones
for _, e in exp_df.iterrows():
    display(Markdown(f"**{e.orden}.** *\\"{e.texto}\\"* — real: `{e.real}` · predicha: `{e.predicha}` "
                     f"({e.confianza:.2f})  \\n→ {e.explicacion}"))
# Guardar las salidas crudas para la revisión manual
json.dump(explicaciones, open(ART / "explicaciones_crudas.json", "w"), indent=1, ensure_ascii=False)
""", tags=["generacion_llm"])
md("""
### 4.6 Validación manual y razones del LLM

Cada explicación se leyó a mano contra la consulta y las dos intenciones, y se le asignó:

- **Veredicto**: `pertinente` (señala palabras reales de la consulta y la razón es plausible),
  `parcial` (algo cierto pero vago, o repite las etiquetas sin explicar) o `alucinada` (afirma
  algo que la consulta no dice, o se contradice).
- **Razón que da el LLM**, en categorías: `solapamiento_lexico` (una palabra de la consulta
  pertenece al vocabulario de la clase predicha), `ambiguedad_real` (la consulta encaja en las
  dos intenciones), `etiqueta_dudosa` (la etiqueta del dataset es discutible), `generica`
  (no da una razón concreta), `frecuencia_supuesta` (atribuye el error a que una clase es «más
  común», dato que el LLM no tiene) y `otra`.

La revisión se guarda en `artefactos/revision_manual.json` para que sea auditable.
""")
code("""
# Revisión manual: veredicto y razón de cada explicación, guardados en un archivo auditable
revision = json.load(open(ART / "revision_manual.json"))
rev_df = pd.DataFrame(revision)
exp_df = exp_df.merge(rev_df[["orden", "veredicto", "razon_categoria", "nota_revision"]], on="orden", how="left")
assert exp_df["veredicto"].notna().all(), "Falta el veredicto de alguna explicación"

# Tabla con cada explicación y su veredicto
display(exp_df[["orden", "texto", "real", "predicha", "explicacion", "veredicto", "razon_categoria", "nota_revision"]])

# Conteo de veredictos y de categorías de razón
fig, ejes = plt.subplots(1, 2, figsize=(12, 3.5))
exp_df["veredicto"].value_counts().reindex(["pertinente", "parcial", "alucinada"]).fillna(0).plot.bar(
    ax=ejes[0], color=["#55A868", "#DD8452", "#C44E52"], rot=0, title="Veredicto manual")
exp_df["razon_categoria"].value_counts().plot.barh(ax=ejes[1], color="#4C72B0", title="Razón que da el LLM")
plt.tight_layout(); plt.show()
print(exp_df["veredicto"].value_counts().to_dict(), "·", exp_df["razon_categoria"].value_counts().to_dict())
""", tags=["pausa_revision"])
md("### 4.7 Razones proporcionadas por el LLM")
interpretacion("razones")
md("### 4.8 Conclusiones sobre el uso del LLM para interpretar el clasificador")
interpretacion("llm")

# ════════════════════════════════════════════════════════════════════════════
# 5. Conclusiones
# ════════════════════════════════════════════════════════════════════════════
md("## 5. Conclusiones generales")
interpretacion("generales")
md("""
## 6. Referencias

- Almazrouei, E., Alobeidli, H., Alshamsi, A., et al. (2023). *The Falcon Series of Open Language
  Models*. arXiv:2311.16867.
- Casanueva, I., Temčinas, T., Gerz, D., Henderson, M., & Vulić, I. (2020). Efficient Intent
  Detection with Dual Sentence Encoders. *Proceedings of the 2nd Workshop on NLP for
  Conversational AI*, 38–45. arXiv:2003.04807. (Origen del dataset BANKING77.)
- Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep
  Bidirectional Transformers for Language Understanding. *NAACL-HLT*.
- Holtzman, A., Buys, J., Du, L., Forbes, M., & Choi, Y. (2020). The Curious Case of Neural Text
  Degeneration. *ICLR*. (Muestreo *nucleus*/top-p y temperatura.)
- Ji, Z., Lee, N., Frieske, R., et al. (2023). Survey of Hallucination in Natural Language
  Generation. *ACM Computing Surveys*, 55(12).
- Liu, Y., Ott, M., Goyal, N., et al. (2019). *RoBERTa: A Robustly Optimized BERT Pretraining
  Approach*. arXiv:1907.11692.
- Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). *DistilBERT, a distilled version of BERT:
  smaller, faster, cheaper and lighter*. arXiv:1910.01108.
- Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). Attention Is All You Need. *NeurIPS*.
- Wolf, T., Debut, L., Sanh, V., et al. (2020). Transformers: State-of-the-Art Natural Language
  Processing. *EMNLP: System Demonstrations*.
- Documentación de Hugging Face: `PolyAI/banking77`, `distilbert/distilroberta-base`,
  `tiiuae/falcon-7b-instruct`, *Text generation strategies*.
""")

# ════════════════════════════════════════════════════════════════════════════
# 7. Exportación
# ════════════════════════════════════════════════════════════════════════════
md("""
## 7. Anexo: exportación de resultados

Todos los resultados se guardan en `artefactos/` para cargarlos en la base de datos del
proyecto (tablas `DistilRoBERTa_*` en Supabase) y publicarlos en
<https://distilroberta.iagentek.com.mx>. Así, cada cifra de la web sale de esta ejecución.
""")
code("""
from datetime import datetime

# Identificador de esta ejecución: fecha y hora
CORRIDA_ID = datetime.now().strftime("%Y-%m-%dT%H-%M")
def guardar(nombre, obj):
    json.dump(obj, open(ART / nombre, "w"), indent=1, ensure_ascii=False, default=float)

# Hiperparámetros y métricas globales de la corrida
guardar("corrida.json", {
    "id": CORRIDA_ID, "modelo_base": MODELO_BASE, "llm": LLM_ID, "dispositivo": DEVICE,
    "hiperparametros": {"learning_rate": 5e-5, "epocas_max": 8, "lote": 32, "max_length": MAX_LEN,
                        "warmup_ratio": 0.1, "weight_decay": 0.01, "seed": SEED,
                        "mejor_epoca": int(curva.loc[curva.eval_f1_macro.idxmax(), "epoch"]),
                        "config_llm": CONFIG_ELEGIDA, **{f"llm_{k}": v for k, v in CONFIGS[CONFIG_ELEGIDA].items()}},
    "accuracy": ACCURACY, "f1_macro": F1_MACRO, "n_train": len(tr_df), "n_val": len(val_df),
    "n_test": len(test_df), "duracion_entrenamiento_s": DURACION_ENTRENAMIENTO})

# Las 77 clases con su conteo por partición
guardar("clases.json", [{"id": i, "nombre": n, "nombre_legible": legible(n),
                         "n_train": int(conteo.loc[n, "train"]), "n_test": int(conteo.loc[n, "test"])}
                        for i, n in enumerate(ETIQUETAS)])

# Las 13 083 consultas con su texto limpio, longitudes y tokens (id de prueba = 100000 + índice)
consultas = pd.concat([
    train_df.assign(particion="train", id=range(len(train_df))),
    test_df.assign(particion="test", id=[100000 + i for i in range(len(test_df))]),
])[["id", "particion", "text", "texto_limpio", "label", "n_caracteres", "n_palabras", "n_tokens"]]
consultas.to_csv(ART / "consultas.csv", index=False)

# Estadísticas descriptivas y n-gramas del análisis exploratorio
est = []
for part, df in (("train", train_df), ("test", test_df)):
    for met in ("n_caracteres", "n_palabras", "n_tokens"):
        d = df[met].describe()
        est.append({"metrica": met, "particion": part, "count": int(d["count"]), "mean": d["mean"], "std": d["std"],
                    "min": d["min"], "q1": d["25%"], "mediana": d["50%"], "q3": d["75%"], "max": d["max"]})
ngr = [{"tipo": t, "rango": r + 1, "ngrama": f.ngrama, "frecuencia": int(f.frecuencia)}
       for t, df in (("palabra", top_palabras), ("bigrama", top_bigramas), ("trigrama", top_trigramas))
       for r, f in enumerate(df.itertuples())]
guardar("eda.json", {"estadisticas": est, "ngramas": ngr, "balance": {k: v for k, v in balance.items()
                                                                        if k != "test: valores distintos"}})

# Curvas de entrenamiento, métricas por clase y matriz de confusión dispersa
guardar("entrenamiento.json", curva.to_dict("records"))
guardar("metricas_clase.json", [{"clase_id": ID[n], "precision": r["precision"], "recall": r["recall"],
                                 "f1": r["f1-score"], "soporte": int(r["support"]), "errores": int(r["errores"]),
                                 "categoria": "mejor" if n in mejores.index else "peor" if n in peores.index else None}
                                for n, r in por_clase.iterrows()])
guardar("confusion.json", [[int(i), int(j), int(cm[i, j])] for i in range(N_CLASES) for j in range(N_CLASES) if cm[i, j]])

# Predicción, confianza y top-5 de cada consulta de prueba
top5 = np.argsort(-probs, axis=1)[:, :5]
pd.DataFrame({"consulta_id": [100000 + i for i in range(len(test_df))], "clase_pred_id": y_pred,
              "confianza": confianza.round(5),
              "top5": [json.dumps([{"clase_id": int(c), "prob": round(float(probs[k, c]), 5)} for c in top5[k]])
                       for k in range(len(test_df))]}).to_csv(ART / "predicciones.csv", index=False)

# Calibración y explicaciones del LLM
guardar("calibracion.json", [{**r, "consulta_id": 100000 + int(r["idx_test"]), "parametros": CONFIGS[r["config"]]}
                             for r in cal_df.to_dict("records")])
guardar("explicaciones.json", [{**r, "consulta_id": 100000 + int(r["idx_test"]), "parametros": CONFIGS[CONFIG_ELEGIDA]}
                               for r in exp_df.to_dict("records")])
print("Corrida", CORRIDA_ID, "→", sorted(p.name for p in ART.iterdir()))
""")

nb = nbf.v4.new_notebook(cells=celdas)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nb.metadata["language_info"] = {"name": "python"}
SALIDA.parent.mkdir(exist_ok=True)
nbf.write(nb, SALIDA)
print(f"{SALIDA.relative_to(RAIZ)}: {len(celdas)} celdas")
