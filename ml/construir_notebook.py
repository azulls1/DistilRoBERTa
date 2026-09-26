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
**Autor:** Adonai Samael Hernández Mata  
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
3. Clasificación con DistilRoBERTa (3.6 causas con base en el EDA · 3.8 mejoras técnicas propuestas)
4. Explicación de errores con Falcon-7b-instruct (4.4 calibración de decodificación · 4.5 calibración de la estructura del prompt)
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

**Requisitos de hardware del enunciado y cómo se atienden:**

- *Colab (GPU T4, ≥ 15 GB libres para Falcon-7b)*: la VRAM se verifica con `!nvidia-smi`; además, la
  celda 4.1 imprime la memoria que ocupa Falcon tras cargarlo, en cualquier acelerador. Con la
  cuantización a 4 bits NF4 (Dettmers et al., 2023) Falcon ocupa bastante menos que en `float16`.
- *Esta ejecución (Apple Silicon, memoria unificada)*: Falcon-7b se carga en `float16`; la celda 4.1
  deja constancia de la memoria medida.
- *Aviso de NumPy del enunciado* (`ValueError: Unable to avoid copy…` → reinstalar `NumPy<=1.24.3`):
  **no se presentó** con las versiones de esta ejecución (las imprime la celda siguiente); si aparece en
  Colab con versiones antiguas de `datasets`, la solución es la del enunciado.
""")
code("""
# En Google Colab, descomentar para instalar las dependencias y verificar la GPU:
# !pip install -q "transformers>=4.56" accelerate datasets scikit-learn nltk wordcloud seaborn bitsandbytes
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
import datasets
from IPython.display import display, Markdown

# Salida limpia: sin avisos ni barras de progreso de Hugging Face (ensucian el PDF), tablas completas
warnings.filterwarnings("ignore")
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
transformers.logging.set_verbosity_error()
transformers.utils.logging.disable_progress_bar()
datasets.disable_progress_bars()
pd.set_option("display.max_colwidth", 120)
pd.set_option("display.max_rows", 120)
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
print(f"torch {torch.__version__} · transformers {transformers.__version__} · datasets {datasets.__version__} · numpy {np.__version__}")
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
interpretacion("fig_longitud")
md("### 2.2 Limpieza básica del texto")
md("""
Se aplica: (1) conversión a minúsculas, (2) eliminación de caracteres especiales (todo lo que
no sea letra, dígito o espacio) y (3) eliminación de *stopwords* del inglés (lista de NLTK).

> **Importante:** esta limpieza se usa **solo para el análisis exploratorio** (frecuencias,
> n-gramas y nube de palabras). El Transformer recibe el **texto original**, porque su
> tokenizador y su preentrenamiento aprovechan la puntuación y, sobre todo, palabras que NLTK
> considera *stopwords* pero que cambian la intención — por ejemplo *not*, *no*, *why*, *how* o
> *up*: «*My card is **not** working*» queda como «card working», y «*How do I top **up** my
> card?*» como «top card». (*can't* también desaparece, pero no por ser stopword: la regla de
> caracteres especiales lo parte en «can» y «t», y esas dos sí lo son.)
""")
code("""
# Lista de stopwords del inglés de NLTK; si no hay red, la equivalente de scikit-learn
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
# Palabras con intención que la lista trata como stopwords (y por eso el Transformer recibe el texto original)
print("Palabras con intención incluidas en la lista:", [w for w in ["not", "no", "why", "how", "up", "don't", "didn't"] if w in STOPWORDS])
for ejemplo in ["My card is not working", "How do I top up my card?", "I can't transfer money"]:
    print(f"   {ejemplo!r:32} → {limpiar(ejemplo)!r}")
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

# Frecuencias sobre el texto limpio: 150 palabras (15 para la tabla, 150 para la nube), 10 bigramas y 10 trigramas
corpus_limpio = todas["texto_limpio"]
top_palabras = top_ngramas(corpus_limpio, 1, 150)   # la nube de 2.4 usa exactamente estas frecuencias
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
interpretacion("fig_ngramas")
md("### 2.4 Nube de palabras")
code("""
# Nube de palabras: el tamaño de cada palabra es proporcional a su frecuencia en el texto limpio
from wordcloud import WordCloud

# Mismas frecuencias que la tabla de 2.3 (150 palabras), sin fusionar bigramas; semilla fija para la disposición
nube = WordCloud(width=1400, height=600, background_color="white", colormap="viridis", max_words=150,
                 random_state=SEED).generate_from_frequencies(dict(zip(top_palabras.ngrama, top_palabras.frecuencia)))
# Dibujar la nube sin ejes
plt.figure(figsize=(14, 6)); plt.imshow(nube, interpolation="bilinear"); plt.axis("off")
plt.title("Nube de palabras (train + test, texto limpio)"); plt.show()
""")
interpretacion("fig_nube")
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
interpretacion("fig_balance")
md("### 2.6 Conclusiones del análisis exploratorio")
interpretacion("eda")

# ════════════════════════════════════════════════════════════════════════════
# 3. DistilRoBERTa
# ════════════════════════════════════════════════════════════════════════════
md("""
## 3. Clasificación con DistilRoBERTa

Un Transformer procesa la frase completa con **autoatención**: cada token se representa en función de
todos los demás (Vaswani et al., 2017). Los codificadores preentrenados como BERT (Devlin et al., 2019) y
RoBERTa (Liu et al., 2019) aprenden esas representaciones sobre grandes corpus sin etiquetar y luego se
**afinan** para una tarea concreta.

**DistilRoBERTa** (`distilroberta-base`) es la versión **destilada** de RoBERTa, con la técnica de
DistilBERT (Sanh et al., 2019): 6 capas de Transformer en lugar de 12 y 82 M de parámetros frente a
125 M; según la ficha oficial del modelo en Hugging Face es, en promedio, el doble de rápida que
RoBERTa-base. Se usa *transfer learning*: se parte de los pesos preentrenados en inglés y se añade una
cabeza de clasificación de 77 salidas sobre el token `<s>`; después se afina **todo** el modelo con las
consultas etiquetadas, con la librería `transformers` (Wolf et al., 2020).

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
    output_dir=str(RAIZ / "salidas_entrenamiento"), disable_tqdm=True,
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
print(f"Mejor checkpoint: {Path(trainer.state.best_model_checkpoint).name} (F1 macro val = {trainer.state.best_metric:.4f})")
""")
interpretacion("fig_curvas")
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

# ¿La confianza del modelo distingue aciertos de errores?
acierto = y_true == y_pred
print(f"Confianza media: aciertos {confianza[acierto].mean():.3f} · errores {confianza[~acierto].mean():.3f}")
print(f"Errores cometidos con confianza > 0.90: {(confianza[~acierto] > 0.90).mean():.1%} "
      f"({(confianza[~acierto] > 0.90).sum()} de {(~acierto).sum()})")

# Reporte por clase COMPLETO: precision, recall, F1 y soporte de las 77 intenciones
print()
print(classification_report(y_true, y_pred, target_names=ETIQUETAS, digits=3, zero_division=0))
# La misma información como tabla, para usarla en las secciones siguientes
reporte = pd.DataFrame(classification_report(y_true, y_pred, target_names=ETIQUETAS,
                                             output_dict=True, zero_division=0)).T
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
         .sort_values(["errores", "real", "predicha"], ascending=[False, True, True], kind="stable")
         .reset_index(drop=True))
print(f"15 pares más confundidos (de {len(pares)} pares con al menos un error; empates en orden alfabético):")
display(pares.head(15))
""")
interpretacion("fig_confusion")
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

# Empates que el orden resuelve (para declararlos en el texto)
print(f"Clases con F1 = 1.000: {int((por_clase['f1-score'] >= 0.9999).sum())} (se muestran las 7 primeras en el orden oficial)")
corte = int(peores["errores"].min())
print(f"Clases con {corte} errores (el corte de las 7 peores): {sorted(por_clase.index[por_clase['errores'] == corte])}")
""")
interpretacion("fig_f1")
md("""
### 3.6 Análisis de causas en las clases problemáticas

Para no quedarse en la intuición se miden cuatro cosas: (1) **a qué clases se va** cada clase
problemática cuando falla, (2) **ejemplos reales** de sus errores, (3) la **similitud léxica** entre la
clase real y la predicha —el coseno entre los vectores TF-IDF de todas las consultas de entrenamiento
de cada clase— y (4) qué parte de los errores ocurre **dentro de la misma familia** de intenciones. Si
los errores se concentran en pares con vocabulario muy parecido, la causa es de solapamiento léxico y no
un defecto del entrenamiento. Después, la celda siguiente contrasta las dos hipótesis que dejó el EDA.
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

# Familias de intenciones: regla por palabra clave del nombre de la clase (la primera que coincide gana)
REGLAS_FAMILIA = [("identidad", ("identity", "verify")), ("recargas", ("top_up", "topping_up")),
                  ("transferencias", ("transfer", "beneficiary")), ("divisas", ("exchange", "currenc", "fiat")),
                  ("efectivo y cajero", ("cash", "atm")),
                  ("tarjetas", ("card", "pin", "passcode", "contactless", "apple_pay", "visa")),
                  ("cargos y pagos", ("payment", "debit", "charge", "fee", "refund", "statement", "transaction"))]
def familia(clase):
    return next((f for f, claves in REGLAS_FAMILIA if any(k in clase for k in claves)), "cuenta y otros")
FAMILIA = {c: familia(c) for c in ETIQUETAS}
misma = errores_df["category"].map(FAMILIA) == errores_df["predicha"].map(FAMILIA)
print(f"Errores dentro de la misma familia: {misma.mean():.1%} ({misma.sum()} de {len(errores_df)})")
display(pd.Series(FAMILIA).value_counts().rename("clases por familia").to_frame().T)

# Tres errores reales por clase problemática, de mayor a menor confianza
print("\\nEjemplos de errores en las clases problemáticas:")
display(errores_df[errores_df["category"].isin(peores.index)]
        .sort_values(["category", "confianza"], ascending=[True, False])
        .groupby("category").head(3)[["text", "category", "predicha", "confianza"]].round(3))
""")
code(r'''
# ── ¿Explica el EDA los errores? Dos hipótesis que salen del análisis exploratorio ──────────
# H1: el desbalance de clases que midió el EDA (sección 2.5) causa los errores.
# H2: las clases que comparten n-gramas frecuentes (sección 2.3) se confunden entre sí.
# Usa los objetos de esta misma ejecución: conteo (2.5), por_clase y peores (3.5), fuera (3.4) y train_df.
tabla = por_clase.assign(n_train=conteo["train"])

# H1 · Tamaño de la clase en train frente a su F1 en prueba
rho1, p1 = spearmanr(tabla["n_train"], tabla["f1-score"])
print(f"H1 · Spearman(ejemplos de entrenamiento, F1) = {rho1:.3f} (p = {p1:.2f}) sobre {len(tabla)} clases")
print(f"     Las 7 clases con más errores tienen {tabla.loc[peores.index, 'n_train'].mean():.0f} ejemplos de media; "
      f"el promedio general es {tabla['n_train'].mean():.0f}")
print("     Las 7 clases más pequeñas y su F1:")
display(tabla.nsmallest(7, "n_train")[["n_train", "f1-score", "errores"]].round(3))

# H2 · Los 15 bigramas más frecuentes de cada clase (texto limpio de train) y cuántos comparte cada par
docs_bi = train_df.groupby("label")["texto_limpio"].apply(lambda t: "\n".join(t)).reindex(range(N_CLASES)).fillna("")
vec_bi = CountVectorizer(ngram_range=(2, 2), token_pattern=r"(?u)\b\w+\b")
X_bi = vec_bi.fit_transform(docs_bi)
vocab_bi = np.array(vec_bi.get_feature_names_out())
top_bi = {i: set(vocab_bi[np.asarray(X_bi[i].todense()).ravel().argsort()[::-1][:15]]) for i in range(N_CLASES)}

# Jaccard de bigramas frecuentes frente a errores entre cada par (en ambos sentidos)
errores_par = fuera + fuera.T
pares_idx = list(zip(*triu))
jaccard = np.array([len(top_bi[i] & top_bi[j]) / len(top_bi[i] | top_bi[j]) for i, j in pares_idx])
err_par = errores_par[triu]
rho2, p2 = spearmanr(jaccard, err_par)
print(f"\nH2 · Bigramas compartidos (Jaccard): pares que se confunden {jaccard[err_par > 0].mean():.3f} · "
      f"pares que nunca se confunden {jaccard[err_par == 0].mean():.3f} "
      f"({jaccard[err_par > 0].mean() / jaccard[err_par == 0].mean():.0f}× más)")
print(f"     Spearman(bigramas compartidos, errores entre el par) = {rho2:.3f} (p = {p2:.1e})")
# Los pares con más errores y los bigramas frecuentes que comparten (orden estable por nombre en empates)
mas_errores = sorted(pares_idx, key=lambda x: (-errores_par[x], ETIQUETAS[x[0]], ETIQUETAS[x[1]]))[:8]
display(pd.DataFrame([{"par": f"{ETIQUETAS[i]} ↔ {ETIQUETAS[j]}", "errores": int(errores_par[i, j]),
                       "similitud TF-IDF (percentil)": f"{sim[i, j]:.3f} ({(sim_pares < sim[i, j]).mean() * 100:.1f})",
                       "bigramas compartidos": ", ".join(sorted(top_bi[i] & top_bi[j])) or "—"}
                      for i, j in mas_errores]))
''')
interpretacion("causas")
md("### 3.7 Conclusiones del uso del Transformer")
interpretacion("transformer")
code("""
# Guardar el modelo afinado (lo usa la aplicación web para clasificar en vivo)
trainer.save_model(str(MODELO)); tokenizer.save_pretrained(str(MODELO))
print("Modelo guardado en", MODELO.relative_to(RAIZ), "·", sorted(p.name for p in MODELO.iterdir()))
""")
md("### 3.8 Mejoras técnicas propuestas: del diagnóstico al ajuste")
interpretacion("mejoras")

# ════════════════════════════════════════════════════════════════════════════
# 4. Falcon
# ════════════════════════════════════════════════════════════════════════════
md("""
## 4. Explicación de errores con Falcon-7b-instruct

### 4.1 Carga del modelo

`tiiuae/falcon-7b-instruct` es la versión ajustada con instrucciones de Falcon-7B (Almazrouei et al.,
2023): 7 mil millones de parámetros, ≈ 14 GB en `float16`. Antes de cargarlo se libera el clasificador.
En Colab se cuantiza a 4 bits NF4 (Dettmers et al., 2023) para que entre en la T4; en Apple Silicon cabe
en `float16` en la memoria unificada. La celda imprime la memoria que ocupa realmente.
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
# Colab (GPU T4): cuantización NF4 de 4 bits con bitsandbytes
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

# Memoria ocupada por el modelo (verificación que pide el enunciado, en cualquier acelerador)
if DEVICE == "cuda":
    print(f"VRAM ocupada: {torch.cuda.memory_allocated() / 1e9:.2f} GB de {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    import subprocess; print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
elif DEVICE == "mps":
    print(f"Memoria MPS ocupada: {torch.mps.current_allocated_memory() / 1e9:.2f} GB")
""")
md("""
### 4.2 Diseño del prompt y sus tres variantes de estructura

La base del diseño son cinco piezas, cada una pensada contra un tipo de alucinación:

| Pieza | Para qué |
|---|---|
| **Rol** («analista de soporte bancario que audita un clasificador») | fija el dominio y el tono |
| **Tarea explícita**: explicar por qué el clasificador eligió la intención *predicha* | evita que el modelo responda al cliente |
| **Restricciones**: máximo dos oraciones, referirse solo a palabras de la consulta, no inventar | ataca la alucinación y la longitud |
| **Datos**: consulta, intención predicha y etiqueta del dataset, con nombres legibles | da al modelo todo lo que necesita y nada más |
| **Ancla de salida** `Explanation:` | el modelo empieza a escribir la explicación directamente |

Para **calibrar la claridad y la estructura** (el tercer eje que pide el enunciado) se comparan tres variantes
que solo difieren en la estructura; todas se envían en inglés, el idioma de las consultas y de las
instrucciones con que se ajustó Falcon (en español se incluye la traducción):

| Variante | Qué cambia | Hipótesis |
|---|---|---|
| **P1 · premisa de error** | afirma que la predicción es INCORRECTA y pide explicar por qué se eligió | el encuadre puede empujar al modelo a racionalizar |
| **P2 · neutral con citas** | no afirma quién tiene razón y obliga a citar entre comillas simples las palabras de la consulta | citas exactas → explicaciones verificables |
| **P3 · P2 + dos ejemplos** (*few-shot*, Brown et al., 2020) | añade dos ejemplos resueltos con consultas del conjunto de **entrenamiento** | los ejemplos fijan el formato y el tipo de razonamiento |

**P1 en inglés (tal como se envía) y en español:**

| Inglés | Español |
|---|---|
| You are a banking customer-support analyst who audits an automatic intent classifier. | Eres un analista de atención a clientes bancarios que audita un clasificador automático de intenciones. |
| The classifier read the customer query below and predicted an intent that is WRONG. | El clasificador leyó la consulta del cliente de abajo y predijo una intención que es INCORRECTA. |
| In at most two short sentences, explain why the classifier probably chose the predicted intent instead of the correct one. | En máximo dos oraciones cortas, explica por qué el clasificador probablemente eligió la intención predicha en lugar de la correcta. |
| Refer only to words that appear in the query. Do not invent facts, do not give advice to the customer. | Refiérete solo a palabras que aparecen en la consulta. No inventes hechos y no des consejos al cliente. |
| Customer query: "{texto}" · Predicted intent: {pred} · Correct intent: {real} · Explanation: | Consulta del cliente: «{texto}» · Intención predicha: {pred} · Intención correcta: {real} · Explicación: |

**P2 en inglés y en español** (lo que cambia respecto de P1):

| Inglés | Español |
|---|---|
| The classifier read the customer query below and predicted an intent that differs from the label in the dataset. | El clasificador leyó la consulta del cliente de abajo y predijo una intención distinta de la etiqueta del dataset. |
| In at most two short sentences, explain which words of the query led the classifier to the predicted intent, quoting those words between single quotes exactly as they appear in the query. | En máximo dos oraciones cortas, explica qué palabras de la consulta llevaron al clasificador a la intención predicha, citándolas entre comillas simples tal como aparecen en la consulta. |
| Dataset label: {real} | Etiqueta del dataset: {real} |

**P3** es P2 precedido de dos ejemplos resueltos (en la celda siguiente, con su traducción en los comentarios).
""")
code('''
# Nombre de intención legible para el LLM: card_arrival → «card arrival»
def legible(etiqueta: str) -> str:
    return etiqueta.replace("_", " ").replace("?", "").lower()

CABECERA = "You are a banking customer-support analyst who audits an automatic intent classifier.\\n"
# P1 · premisa de error. Español: «El clasificador leyó la consulta… y predijo una intención que es INCORRECTA.
#   En máximo dos oraciones cortas, explica por qué… Refiérete solo a palabras de la consulta. No inventes hechos…»
P1 = (CABECERA +
      "The classifier read the customer query below and predicted an intent that is WRONG.\\n"
      "In at most two short sentences, explain why the classifier probably chose the predicted "
      "intent instead of the correct one. Refer only to words that appear in the query. "
      "Do not invent facts, do not give advice to the customer.\\n\\n"
      'Customer query: "{texto}"\\nPredicted intent: {pred}\\nCorrect intent: {real}\\n\\nExplanation:')
# P2 · neutral con citas. Español: «…predijo una intención distinta de la etiqueta del dataset. En máximo dos
#   oraciones cortas, explica qué palabras de la consulta llevaron a la intención predicha, citándolas entre
#   comillas simples tal como aparecen. No inventes hechos…» · «Etiqueta del dataset: {real}»
INSTRUCCION_P2 = (
    "The classifier read the customer query below and predicted an intent that differs from the label in the dataset.\\n"
    "In at most two short sentences, explain which words of the query led the classifier to the predicted intent, "
    "quoting those words between single quotes exactly as they appear in the query. "
    "Do not invent facts, do not give advice to the customer.\\n\\n")
BLOQUE = 'Customer query: "{texto}"\\nPredicted intent: {pred}\\nDataset label: {real}\\n\\nExplanation:'
P2 = CABECERA + INSTRUCCION_P2 + BLOQUE
# P3 · P2 + dos ejemplos resueltos con consultas reales del conjunto de ENTRENAMIENTO (no de prueba).
# Traducción de los ejemplos: 1) «Creo que mi recarga fue revertida» → la consulta dice 'top up' y 'reverted';
#   una recarga revertida se parece a una fallida… 2) «¿Cuánto hay que esperar mi tarjeta?» → 'wait' y 'my card'
#   aparecen en mensajes de tarjetas que no han llegado…
EJEMPLOS = (
    'Customer query: "I think my top up has been reverted"\\nPredicted intent: top up failed\\n'
    "Dataset label: top up reverted\\n\\nExplanation: The query mentions 'top up' and a reverted top-up looks "
    "like a failed one, which pushes toward top up failed. The word 'reverted' is what points to the label.\\n\\n"
    'Customer query: "How long is the wait for my card?"\\nPredicted intent: card arrival\\n'
    "Dataset label: card delivery estimate\\n\\nExplanation: The words 'wait' and 'my card' are typical of "
    "messages about cards that have not arrived. 'How long' asks for a delivery time, which fits the label better.\\n\\n")
P3 = CABECERA + INSTRUCCION_P2 + EJEMPLOS + BLOQUE
PROMPTS = {"P1": P1, "P2": P2, "P3": P3}

# Rellenar una plantilla con una consulta y sus dos intenciones
def construir_prompt(texto, pred, real, variante="P1"):
    return PROMPTS[variante].format(texto=texto.strip(), pred=legible(pred), real=legible(real))

# Contar oraciones: se corta después de «.», «!» o «?» seguidos de espacio
def contar_oraciones(texto):
    return len([s for s in re.split(r"(?<=[.!?])\\s+", texto.strip()) if s])

@torch.no_grad()
def generar(prompt, max_new_tokens=60, temperature=None, top_p=None):
    """Genera con Falcon. temperature=None → decodificación codiciosa (determinista)."""
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

# Métricas automáticas de cada respuesta (se definen ANTES de ver resultados)
CITA = re.compile(r"(?:(?<=\\s)|^)['\\"‘“]([^'\\"‘’“”]{2,60}?)['\\"’”](?=[\\s.,;:!?)]|$)")
def medir(salida, n_tok, max_new, texto, pred, real):
    """formato_ok: 1-2 oraciones completas y sin tocar el tope · citas verificables/falsas frente a la consulta."""
    completa = salida.rstrip().endswith((".", "!", "?")) and n_tok < max_new
    citas = [c.strip().lower() for c in CITA.findall(salida)]
    etiquetas = {legible(pred), legible(real)}
    citas = [c for c in citas if c not in etiquetas]          # citar el nombre de la intención no es evidencia
    falsas = [c for c in citas if c not in texto.lower()]
    return {"n_oraciones": contar_oraciones(salida), "completa": completa,
            "formato_ok": completa and 1 <= contar_oraciones(salida) <= 2,
            "citas": len(citas), "cita_falsa": bool(falsas),
            "cita_verificable": bool(citas) and not falsas, "citas_falsas": ", ".join(falsas)}

print(construir_prompt("How do I top up my card?", "topping_up_by_card", "transfer_into_account", "P2"))
''')
md("""
### 4.3 Selección de las 20 consultas mal clasificadas

Criterio reproducible: de todos los errores en prueba se toman los **20 con mayor confianza en la clase
equivocada**, con un máximo de **2 por clase real** para que haya variedad. No es una muestra aleatoria:
son los errores que el modelo comete *con seguridad*, justo los que más conviene entender. Las primeras 8
se usan para calibrar.
""")
code("""
# Los 20 errores de mayor confianza, con máximo 2 por clase real para tener variedad
seleccion = (errores_df.sort_values("confianza", ascending=False)
             .groupby("category").head(2).head(20).reset_index().rename(columns={"index": "idx_test"}))
# Numeración 1…20 e identificador estable de la consulta (el mismo que usan la base de datos y la web)
seleccion["orden"] = range(1, len(seleccion) + 1)
seleccion["consulta_id"] = 100000 + seleccion["idx_test"]
# ¿Cuántas de las 20 pertenecen a las 7 clases con más errores?
print(f"De las 20, {seleccion['category'].isin(peores.index).sum()} tienen su clase real entre las 7 con más errores")
display(seleccion[["orden", "consulta_id", "text", "category", "predicha", "confianza"]].round(3))
""")
md("""
### 4.4 Calibración de la decodificación: temperatura y longitud

Se comparan cinco configuraciones con la estructura base (P1) sobre las mismas **8 consultas**. El diseño
**separa los dos factores**: A, B y C cambian solo la temperatura con la misma longitud; B frente a E y C
frente a D cambian solo la longitud con la misma temperatura.

| Config. | Decodificación | `max_new_tokens` |
|---|---|---|
| **A** | codiciosa (`do_sample=False`, equivale a temperatura → 0) | 60 |
| **B** | muestreo, `temperature=0.3`, `top_p=0.9` | 60 |
| **C** | muestreo, `temperature=1.0`, `top_p=0.95` | 60 |
| **D** | muestreo, `temperature=1.0`, `top_p=0.95` | 150 |
| **E** | muestreo, `temperature=0.3`, `top_p=0.9` | 150 |

La temperatura reparte más o menos la probabilidad entre palabras poco probables y `top_p` corta la cola
(muestreo *nucleus*, Holtzman et al., 2020). En todas: `repetition_penalty=1.15` y parada en línea en
blanco. El enunciado habla de `max_length`; en `transformers` ese parámetro cuenta también los tokens del
prompt, así que el control correcto de la longitud de la **respuesta** es `max_new_tokens`.

**Métricas** (definidas en 4.2 antes de ver resultados): *formato correcto* = 1–2 oraciones completas que no
tocan el tope de tokens; *cita falsa* = la respuesta pone entre comillas palabras que no están en la
consulta; *cita verificable* = cita al menos una palabra y todas existen en la consulta.

**Regla de elección fijada de antemano:** gana la configuración con más respuestas en formato correcto;
desempata la que tenga menos citas falsas; luego la menor temperatura; luego la menor longitud.
""")
code("""
# Las cinco configuraciones de generación (A-C: efecto de la temperatura; B-E y C-D: efecto de la longitud)
CONFIGS = {
    "A": dict(max_new_tokens=60, temperature=None, top_p=None),
    "B": dict(max_new_tokens=60, temperature=0.3, top_p=0.9),
    "C": dict(max_new_tokens=60, temperature=1.0, top_p=0.95),
    "D": dict(max_new_tokens=150, temperature=1.0, top_p=0.95),
    "E": dict(max_new_tokens=150, temperature=0.3, top_p=0.9),
}
TEMP = {k: (v["temperature"] or 0.0) for k, v in CONFIGS.items()}
calib = seleccion.head(8)

# Generar con cada configuración sobre las mismas 8 consultas y medir cada salida
filas_dec = []
for _, f in calib.iterrows():
    prompt = construir_prompt(f.text, f.predicha, f.category, "P1")
    for nombre, cfg in CONFIGS.items():
        salida_llm, n_tok, seg = generar(prompt, **cfg)
        filas_dec.append({"config": nombre, "prompt": "P1", "orden": f.orden, "consulta_id": f.consulta_id,
                          "salida": salida_llm, "n_tokens": n_tok, "segundos": round(seg, 1),
                          **medir(salida_llm, n_tok, cfg["max_new_tokens"], f.text, f.predicha, f.category)})
dec_df = pd.DataFrame(filas_dec)

# Resumen por configuración
res_dec = dec_df.groupby("config").agg(
    formato_ok=("formato_ok", "sum"), completas=("completa", "sum"), citas_falsas=("cita_falsa", "sum"),
    citas_verificables=("cita_verificable", "sum"), oraciones_media=("n_oraciones", "mean"),
    tokens_media=("n_tokens", "mean"), segundos_media=("segundos", "mean")).round(2)
res_dec["temperatura"] = [TEMP[c] for c in res_dec.index]
res_dec["max_new_tokens"] = [CONFIGS[c]["max_new_tokens"] for c in res_dec.index]
display(res_dec)

# Regla fijada antes de ver los resultados
CONFIG_ELEGIDA = res_dec.sort_values(["formato_ok", "citas_falsas", "temperatura", "max_new_tokens"],
                                     ascending=[False, True, True, True]).index[0]
print(f"Configuración elegida por la regla: {CONFIG_ELEGIDA} → {CONFIGS[CONFIG_ELEGIDA]}")

# Todas las salidas completas, para poder leerlas
with pd.option_context("display.max_colwidth", None):
    display(dec_df[["config", "orden", "n_oraciones", "n_tokens", "formato_ok", "citas_falsas", "salida"]])
""")
interpretacion("calibracion_decodificacion")
md("""
### 4.5 Calibración de la claridad y la estructura del prompt

Con la decodificación elegida fija, se comparan las tres estructuras de 4.2 (P1, P2 y P3) sobre las mismas 8
consultas. **Regla fijada de antemano:** gana la variante con más respuestas de *cita verificable* menos
respuestas con *cita falsa*; desempata la de más respuestas en formato correcto; luego el prompt más corto.
""")
code("""
# Mismas 8 consultas, misma decodificación, tres estructuras de prompt
cfg = CONFIGS[CONFIG_ELEGIDA]
filas_est = [r for r in filas_dec if r["config"] == CONFIG_ELEGIDA]    # P1 ya se generó en 4.4
for _, f in calib.iterrows():
    for variante in ("P2", "P3"):
        salida_llm, n_tok, seg = generar(construir_prompt(f.text, f.predicha, f.category, variante), **cfg)
        filas_est.append({"config": CONFIG_ELEGIDA, "prompt": variante, "orden": f.orden, "consulta_id": f.consulta_id,
                          "salida": salida_llm, "n_tokens": n_tok, "segundos": round(seg, 1),
                          **medir(salida_llm, n_tok, cfg["max_new_tokens"], f.text, f.predicha, f.category)})
est_df = pd.DataFrame(filas_est)

# Resumen por variante de prompt
res_est = est_df.groupby("prompt").agg(
    citas_verificables=("cita_verificable", "sum"), citas_falsas=("cita_falsa", "sum"),
    formato_ok=("formato_ok", "sum"), tokens_media=("n_tokens", "mean")).round(2)
res_est["balance_citas"] = res_est["citas_verificables"] - res_est["citas_falsas"]
res_est["longitud_prompt"] = [len(tok_llm(PROMPTS[p])["input_ids"]) for p in res_est.index]
display(res_est)

# Regla fijada antes de ver los resultados
PROMPT_ELEGIDO = res_est.sort_values(["balance_citas", "formato_ok", "longitud_prompt"],
                                     ascending=[False, False, True]).index[0]
print(f"Estructura elegida por la regla: {PROMPT_ELEGIDO}")

with pd.option_context("display.max_colwidth", None):
    display(est_df[["prompt", "orden", "formato_ok", "cita_verificable", "citas_falsas", "salida"]])
""")
interpretacion("calibracion_estructura")
md("### 4.6 Explicación de las 20 clasificaciones incorrectas")
code("""
# Pedir a Falcon, con la configuración y la estructura elegidas, una explicación de cada uno de los 20 errores
explicaciones = []
for _, f in seleccion.iterrows():
    prompt = construir_prompt(f.text, f.predicha, f.category, PROMPT_ELEGIDO)
    cruda, n_tok, seg = generar(prompt, **CONFIGS[CONFIG_ELEGIDA])
    explicaciones.append({"orden": f.orden, "consulta_id": int(f.consulta_id), "idx_test": int(f.idx_test),
                          "texto": f.text, "real": f.category, "predicha": f.predicha,
                          "confianza": round(float(f.confianza), 4), "prompt": prompt, "salida_cruda": cruda,
                          "explicacion": cruda, "n_tokens": n_tok, "segundos": round(seg, 1),
                          **medir(cruda, n_tok, CONFIGS[CONFIG_ELEGIDA]["max_new_tokens"], f.text, f.predicha, f.category)})
exp_df = pd.DataFrame(explicaciones)
# Métricas automáticas sobre las 20 (la validación de fondo es la manual de 4.7)
print(f"Formato correcto (1-2 oraciones completas): {exp_df.formato_ok.sum()} de {len(exp_df)}")
print(f"Con cita verificable: {exp_df.cita_verificable.sum()} · con cita falsa: {exp_df.cita_falsa.sum()}")
print(f"Tiempo medio por explicación: {exp_df.segundos.mean():.1f} s")
# Mostrar cada explicación junto a su consulta y sus dos intenciones
for _, e in exp_df.iterrows():
    display(Markdown(f"**{e.orden}.** *\\"{e.texto}\\"* — real: `{e.real}` · predicha: `{e.predicha}` "
                     f"({e.confianza:.2f})  \\n→ {e.explicacion}"))
# Guardar las salidas para la revisión manual
json.dump(explicaciones, open(ART / "explicaciones_crudas.json", "w"), indent=1, ensure_ascii=False, default=float)
""", tags=["generacion_llm"])
md("""
### 4.7 Validación manual de veracidad y pertinencia

Cada explicación se leyó a mano contra la consulta y las dos intenciones (la validación que pide el enunciado;
las métricas automáticas de 4.4–4.5 solo detectan citas inexistentes). A cada una se le asignó:

- **Veredicto**: `pertinente` (señala palabras reales de la consulta y la razón es plausible),
  `parcial` (algo cierto pero vago o incompleto) o `alucinada` (afirma algo que la consulta no dice,
  confunde las clases o se contradice). La alucinación se entiende como en Ji et al. (2023): contenido
  no respaldado por la entrada.
- **Razón que da el LLM**, en categorías: `solapamiento_lexico` (una palabra de la consulta pertenece al
  vocabulario de la clase predicha), `ambiguedad_real` (la consulta encaja en las dos intenciones),
  `etiqueta_dudosa` (la explicación sugiere que la etiqueta del dataset es discutible), `generica` (no da
  una razón concreta), `frecuencia_supuesta` (atribuye el error a que una clase es «más común», dato que el
  LLM no tiene) y `otra`.

La revisión es una **entrada manual**: se escribió leyendo las salidas de 4.6 y está en
`artefactos/revision_manual.json` (en la entrega, `04-resultados/`). Cada veredicto va atado al
`consulta_id` y al texto exacto de la explicación revisada, y la celda comprueba que coincidan: si las
explicaciones cambiaran (otra GPU, otra versión), la celda falla en vez de asignar veredictos a salidas
que no se revisaron.
""")
code("""
# Revisión manual (entrada escrita a mano tras leer 4.6), atada a consulta_id y al texto revisado
revision = json.load(open(ART / "revision_manual.json"))
rev_df = pd.DataFrame(revision)
exp_df = exp_df.merge(rev_df[["consulta_id", "explicacion_revisada", "veredicto", "razon_categoria", "nota_revision"]],
                      on="consulta_id", how="left")
assert exp_df["veredicto"].notna().all(), "Falta el veredicto de alguna explicación"
assert (exp_df["explicacion_revisada"] == exp_df["explicacion"]).all(), "La revisión no corresponde a estas salidas"

# Tabla con cada explicación y su veredicto
with pd.option_context("display.max_colwidth", None):
    display(exp_df[["orden", "texto", "real", "predicha", "explicacion", "veredicto", "razon_categoria", "nota_revision"]])

# Conteo de veredictos y de categorías de razón
fig, ejes = plt.subplots(1, 2, figsize=(12, 3.5))
exp_df["veredicto"].value_counts().reindex(["pertinente", "parcial", "alucinada"]).fillna(0).plot.bar(
    ax=ejes[0], color=["#55A868", "#DD8452", "#C44E52"], rot=0, title="Veredicto manual")
exp_df["razon_categoria"].value_counts().plot.barh(ax=ejes[1], color="#4C72B0", title="Razón que da el LLM")
plt.tight_layout(); plt.show()
print(exp_df["veredicto"].value_counts().to_dict(), "·", exp_df["razon_categoria"].value_counts().to_dict())
# ¿Coinciden las métricas automáticas con el juicio humano?
print(pd.crosstab(exp_df["veredicto"], exp_df["cita_falsa"].map({True: "con cita falsa", False: "sin cita falsa"})))
""", tags=["pausa_revision"])
md("### 4.8 Razones proporcionadas por el LLM")
interpretacion("razones")
md("### 4.9 Conclusiones sobre el uso del LLM para interpretar el clasificador")
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
- Brown, T., Mann, B., Ryder, N., et al. (2020). Language Models are Few-Shot Learners. *NeurIPS*.
- Casanueva, I., Temčinas, T., Gerz, D., Henderson, M., & Vulić, I. (2020). Efficient Intent
  Detection with Dual Sentence Encoders. *Proceedings of the 2nd Workshop on NLP for
  Conversational AI*, 38–45. arXiv:2003.04807. (Origen del dataset BANKING77.)
- Cui, Y., Jia, M., Lin, T.-Y., Song, Y., & Belongie, S. (2019). Class-Balanced Loss Based on Effective
  Number of Samples. *CVPR*.
- Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). QLoRA: Efficient Finetuning of
  Quantized LLMs. *NeurIPS*. (Cuantización NF4 de 4 bits.)
- Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep
  Bidirectional Transformers for Language Understanding. *NAACL-HLT*.
- Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On Calibration of Modern Neural Networks.
  *ICML*.
- Holtzman, A., Buys, J., Du, L., Forbes, M., & Choi, Y. (2020). The Curious Case of Neural Text
  Degeneration. *ICLR*. (Muestreo *nucleus*/top-p y temperatura.)
- Ji, Z., Lee, N., Frieske, R., et al. (2023). Survey of Hallucination in Natural Language
  Generation. *ACM Computing Surveys*, 55(12).
- Liu, Y., Ott, M., Goyal, N., et al. (2019). *RoBERTa: A Robustly Optimized BERT Pretraining
  Approach*. arXiv:1907.11692.
- Lundberg, S. M., & Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. *NeurIPS*.
  (SHAP.)
- Northcutt, C., Jiang, L., & Chuang, I. (2021). Confident Learning: Estimating Uncertainty in Dataset
  Labels. *Journal of Artificial Intelligence Research*, 70, 1373–1411.
- Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). *DistilBERT, a distilled version of BERT:
  smaller, faster, cheaper and lighter*. arXiv:1910.01108.
- Sennrich, R., Haddow, B., & Birch, A. (2016). Improving Neural Machine Translation Models with
  Monolingual Data. *ACL*. (Back-translation.)
- Sundararajan, M., Taly, A., & Yan, Q. (2017). Axiomatic Attribution for Deep Networks. *ICML*.
  (Gradientes integrados.)
- Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). Attention Is All You Need. *NeurIPS*.
- Wei, J., & Zou, K. (2019). EDA: Easy Data Augmentation Techniques for Boosting Performance on Text
  Classification Tasks. *EMNLP-IJCNLP*.
- Wolf, T., Debut, L., Sanh, V., et al. (2020). Transformers: State-of-the-Art Natural Language
  Processing. *EMNLP: System Demonstrations*.
- Ying, C., & Thomas, S. (2022). Label Errors in BANKING77. *Proceedings of the Third Workshop on
  Insights from Negative Results in NLP*, ACL.
- Documentación de Hugging Face: `PolyAI/banking77`, ficha de `distilbert/distilroberta-base`,
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
                        "config_llm": CONFIG_ELEGIDA, "prompt_llm": PROMPT_ELEGIDO,
                        **{f"llm_{k}": v for k, v in CONFIGS[CONFIG_ELEGIDA].items()}},
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

# Calibración (decodificación y estructura) y explicaciones del LLM
calibracion = pd.concat([dec_df, est_df[est_df["prompt"] != "P1"]], ignore_index=True)
calibracion["elegida"] = (calibracion["config"] == CONFIG_ELEGIDA) & (calibracion["prompt"] == PROMPT_ELEGIDO)
guardar("calibracion.json", [{**r, "consulta_id": int(r["consulta_id"]),
                              "parametros": {**CONFIGS[r["config"]], "prompt": r["prompt"]}}
                             for r in calibracion.to_dict("records")])
guardar("explicaciones.json", [{**r, "consulta_id": int(r["consulta_id"]),
                                "parametros": {**CONFIGS[CONFIG_ELEGIDA], "prompt": PROMPT_ELEGIDO}}
                               for r in exp_df.to_dict("records")])
print("Corrida", CORRIDA_ID, "→", sorted(p.name for p in ART.iterdir()))
""")

nb = nbf.v4.new_notebook(cells=celdas)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nb.metadata["language_info"] = {"name": "python"}
SALIDA.parent.mkdir(exist_ok=True)
nbf.write(nb, SALIDA)
print(f"{SALIDA.relative_to(RAIZ)}: {len(celdas)} celdas")
