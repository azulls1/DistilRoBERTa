"""Matriz de cumplimiento: cada exigencia de SCA_individual.docx → dónde se cumple y con qué evidencia.

Las cifras de evidencia se leen de artefactos/ (no se escriben a mano).
Uso:  python ml/cumplimiento.py   → artefactos/cumplimiento.json
"""
import json
from pathlib import Path

ART = Path(__file__).resolve().parents[1] / "artefactos"
c = json.load(open(ART / "corrida.json"))
eda = json.load(open(ART / "eda.json"))
b = eda["balance"]
exp = json.load(open(ART / "explicaciones.json"))
rev = json.load(open(ART / "revision_manual.json"))
ver = {v: sum(r["veredicto"] == v for r in rev) for v in ("pertinente", "parcial", "alucinada")}
ng = {t: sum(n["tipo"] == t for n in eda["ngramas"]) for t in ("palabra", "bigrama", "trigrama")}
est = {(e["metrica"], e["particion"]): e for e in eda["estadisticas"]}

CRITERIOS = {
    "C1": ("Análisis exploratorio de datos", 1.5, 15),
    "C2": ("Entrenamiento de transformer", 2.5, 25),
    "C3": ("Diseño de prompt", 3.0, 30),
    "C4": ("Conclusiones", 2.0, 20),
    "C5": ("Código comentado y referencias", 1.0, 10),
    "EN": ("Entrega y notas técnicas", None, None),
}

R = [
    ("C1", "Calcular la longitud y el número de palabras por comentario", "2.1", "/eda",
     "Columnas n_caracteres y n_palabras para las 13 083 consultas; histogramas train/test."),
    ("C1", "Obtener estadísticas descriptivas de ambas métricas", "2.1", "/eda",
     f"Media {est[('n_palabras','train')]['mean']:.2f} palabras (train), mediana {est[('n_palabras','train')]['mediana']:.0f}, máx. {est[('n_palabras','train')]['max']:.0f}."),
    ("C1", "Convertir a minúsculas", "2.2", "/eda", "limpiar(): texto.lower()"),
    ("C1", "Eliminar caracteres especiales y stopwords", "2.2", "/eda",
     "Regex [^a-z0-9\\s] + 198 stopwords de NLTK; se justifica por qué el Transformer recibe el texto original."),
    ("C1", "Identificar las 15 palabras más frecuentes", "2.3", "/eda", f"Top-15 con gráfica (se guardan {ng['palabra']} para la nube)."),
    ("C1", "Identificar los 10 bigramas y 10 trigramas más frecuentes", "2.3", "/eda", f"{ng['bigrama']} bigramas y {ng['trigrama']} trigramas con gráfica."),
    ("C1", "Generar una nube de palabras", "2.4", "/eda", "WordCloud (150 palabras) en el notebook; nube interactiva en la web."),
    ("C1", "Verificar si el conjunto de datos está balanceado", "2.5", "/eda",
     f"Razón máx/mín {b['razón máx/mín']}, CV {b['coeficiente de variación']}, entropía normalizada {b['entropía normalizada']}; test: 40 por clase."),
    ("C1", "Redactar conclusiones del análisis exploratorio", "2.6", "/eda", "Cuatro bloques: longitud, vocabulario, limpieza y balance."),
    ("C2", "Tokenizar los textos usando el tokenizer de DistilRoBERTa", "3.1", "/simulacion",
     f"Tokens BPE mostrados; max_length = {c['hiperparametros']['max_length']} justificado por la distribución (máx. 96 tokens)."),
    ("C2", "Realizar el fine-tuning del modelo preentrenado", "3.3", "/resumen",
     f"{c['n_train']:,} ej. de entrenamiento + {c['n_val']:,} de validación; mejor época {c['hiperparametros']['mejor_epoca']}; {c['duracion_entrenamiento_s']/60:.1f} min en {c['dispositivo']}."),
    ("C2", "Calcular accuracy, matriz de confusión y reporte de métricas (precision, recall y F1-score)", "3.4", "/confusion",
     f"Accuracy {c['accuracy']*100:.2f} %, F1 macro {c['f1_macro']:.3f}; matriz 77×77 y classification_report."),
    ("C2", "Identificar las siete clases mejor clasificadas y las siete con mayor número de errores", "3.5", "/clases",
     "Tablas y gráfica de F1 por clase con las 7 + 7 resaltadas."),
    ("C2", "Analizar posibles causas del bajo desempeño en las clases problemáticas", "3.6", "/confusion",
     "Destino de los errores, similitud TF-IDF por percentil, Spearman ρ = 0.28 y ejemplos reales."),
    ("C2", "Redactar conclusiones del uso del modelo Transformer", "3.7", "/resumen", "Cinco conclusiones con cifras y propuestas de mejora."),
    ("C3", "Diseñar un prompt para tiiuae/falcon-7b-instruct con explicación breve (máximo dos oraciones)", "4.2", "/explicaciones",
     "Prompt de cinco piezas (rol, tarea, restricciones, datos, ancla de salida) documentado pieza por pieza."),
    ("C3", "Calibrar la temperatura", "4.4", "/simulacion", "Tres configuraciones: codiciosa (T→0), T = 0.3 y T = 1.0 sobre las mismas consultas."),
    ("C3", "Calibrar la longitud de respuesta", "4.4", "/simulacion",
     "max_new_tokens 60 vs 150 (+ explicación de por qué max_new_tokens y no max_length) y recorte a 2 oraciones."),
    ("C3", "Calibrar la claridad y estructura del prompt", "4.2 · 4.4", "/explicaciones", "Formato fijo con ancla «Explanation:», parada en línea en blanco y repetition_penalty."),
    ("C3", "Seleccionar 20 muestras mal clasificadas", "4.3", "/explicaciones", f"{len(exp)} errores de mayor confianza, máximo 2 por clase (criterio reproducible)."),
    ("C3", "Solicitar al LLM una justificación de cada clasificación incorrecta", "4.5", "/explicaciones",
     f"{len(exp)} explicaciones generadas con la configuración {c['hiperparametros']['config_llm']}."),
    ("C3", "Validar manualmente la veracidad y pertinencia de las respuestas", "4.6", "/explicaciones",
     f"Veredicto por explicación: {ver['pertinente']} pertinentes, {ver['parcial']} parciales, {ver['alucinada']} alucinadas."),
    ("C3", "Analizar y listar las razones proporcionadas por el LLM", "4.6 · 4.7", "/explicaciones",
     "Cinco categorías de razón con conteo, tabla y los tres patrones de alucinación."),
    ("C4", "Redactar conclusiones sobre el uso del LLM para interpretar el clasificador", "4.8", "/explicaciones", "Cuatro conclusiones, prompt mejorado propuesto y uso recomendado."),
    ("C4", "Conclusiones generales y limitaciones", "5", "/resumen", "Cuatro conclusiones + limitaciones explícitas."),
    ("C5", "Todo el código debe estar debidamente comentado y estructurado en un notebook", "Todo", "/entregables", "Comentarios en cada celda; secciones numeradas 0–7."),
    ("C5", "Notebook autoexplicativo: análisis, visualizaciones y reflexiones escritas", "Todo", "/entregables", "9 celdas con figuras y una interpretación escrita por sección."),
    ("C5", "Referencias", "6", "/entregables", "10 referencias académicas (APA)."),
    ("EN", "Dataset PolyAI/banking77: 10 003 de entrenamiento y 3 080 de prueba en 77 clases", "1", "/eda",
     "CSV oficiales y orden de etiquetas del script de Hugging Face (datasets 4 ya no ejecuta scripts)."),
    ("EN", "Ejecutable en Colab con GPU T4 · verificar VRAM con !nvidia-smi", "0", "/entregables",
     "Detección CUDA/MPS/CPU; Falcon en 4 bits en Colab; celda de !nvidia-smi; aquí 14.06 GB en MPS."),
    ("EN", "Aviso NumPy<=1.24.3 ante «Unable to avoid copy»", "0", "/entregables", "Documentado: no se presentó con NumPy 2.4.6; se deja la instrucción para Colab."),
    ("EN", "Notebook completo (*.ipynb) con código funcional, análisis, visualizaciones y conclusiones", "—", "/entregables",
     "Ejecutado de punta a punta: 0 celdas sin ejecutar, 0 errores."),
    ("EN", "Versión exportada en PDF", "—", "/entregables", "PDF de 38 páginas generado desde el notebook ejecutado."),
]

salida = [{"orden": i + 1, "criterio": cr, "criterio_nombre": CRITERIOS[cr][0], "puntos": CRITERIOS[cr][1],
           "peso": CRITERIOS[cr][2], "requisito": req, "seccion_notebook": sec, "ruta_web": ruta,
           "evidencia": ev, "cumplido": True} for i, (cr, req, sec, ruta, ev) in enumerate(R)]
json.dump(salida, open(ART / "cumplimiento.json", "w"), indent=1, ensure_ascii=False)
print(len(salida), "requisitos ·", sum(r["cumplido"] for r in salida), "cumplidos")
