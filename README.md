# DistilRoBERTa · Banking77

Clasificación de **77 intenciones bancarias** (`PolyAI/banking77`) con **DistilRoBERTa** afinado y
explicación de sus errores con **`tiiuae/falcon-7b-instruct`** mediante un prompt calibrado.

UNIR · Maestría en Inteligencia Artificial · Sistemas Cognitivos Artificiales · Actividad 2
(individual) — *Transformers y Modelos de Lenguaje Grande*.

**En vivo:** <https://distilroberta.iagentek.com.mx>

| Resultado (prueba, 3 080 consultas) | Valor |
|---|---|
| Accuracy | **93.25 %** |
| F1 macro | **0.932** |
| Errores | 208 |

## Contenido

```text
notebooks/   Notebook entregable (.ipynb) y su PDF
ml/          construir_notebook.py · ejecutar_notebook.py · cargar_resultados.py · exportar_pdf.py
artefactos/  Resultados de la ejecución (JSON/CSV) → base de datos
data/        CSV oficiales de banking77 y orden de etiquetas
db/          Migraciones SQL (tablas "DistilRoBERTa_*", rol propio, RLS)
backend/     FastAPI + Celery (Redis) · clasificación en vivo en CPU
frontend/    Angular 22 + Tailwind CSS 4
deploy/      stack.yml (Docker Swarm + Traefik) y deploy.sh
_bmad*/      BMad Method (épicas, arquitectura, sprint)
```

La planificación (constitución, especificación, plan, modelo de datos, contrato REST y tareas)
se hizo con **Spec Kit** y vive en la carpeta hermana `espesificaciones/`.

## Reproducir

```bash
# 1. Entorno
uv venv .venv-ml --python 3.11 && source .venv-ml/bin/activate
uv pip install torch transformers datasets accelerate scikit-learn pandas matplotlib seaborn \
               nltk wordcloud jupyter nbclient nbconvert "psycopg[binary]"

# 2. Notebook (entrena, evalúa, calibra el prompt y genera las 20 explicaciones)
python ml/construir_notebook.py
python ml/ejecutar_notebook.py        # se pausa hasta que exista artefactos/revision_manual.json

# 3. Base de datos y despliegue
psql "$DSN_ADMIN" -f db/migraciones/001_esquema.sql
psql "$DSN_ADMIN" -v app_password="'…'" -f db/migraciones/002_seguridad.sql
python ml/cargar_resultados.py --dsn "$DSN_APP"
deploy/deploy.sh
```

En Google Colab el notebook detecta CUDA y carga Falcon cuantizado a 4 bits; en Apple Silicon lo
carga en `float16` sobre MPS.

## Arquitectura

```text
Angular ──/api──▶ FastAPI ──▶ Postgres (Supabase · tablas DistilRoBERTa_*)
                     │
                     └─encola─▶ Redis ──▶ worker Celery (DistilRoBERTa en CPU) ──▶ Postgres
```

Falcon-7b **no** se sirve en producción (el VPS no tiene GPU): sus explicaciones se generan en el
notebook y se publican como datos.

## Referencias

Casanueva et al. (2020), *Efficient Intent Detection with Dual Sentence Encoders* · Sanh et al.
(2019), *DistilBERT* · Liu et al. (2019), *RoBERTa* · Almazrouei et al. (2023), *The Falcon Series
of Open Language Models* · Vaswani et al. (2017), *Attention Is All You Need*.
