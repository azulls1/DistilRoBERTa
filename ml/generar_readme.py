"""Genera README.md y sus recursos gráficos a partir de los resultados REALES de la corrida.

Nada de este README se escribe a mano: las cifras salen de artefactos/, las figuras se extraen del
notebook ejecutado, las gráficas nuevas se dibujan con esos mismos datos y el banner animado muestra
la predicción real del modelo afinado sobre la consulta de ejemplo.

Uso:  python ml/generar_readme.py      → README.md + docs/readme/*
"""
import base64
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib
import nbformat
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
ART = RAIZ / "artefactos"
DOC = RAIZ / "docs" / "readme"
NB = RAIZ / "notebooks" / "SCA_Actividad2_DistilRoBERTa_Banking77.ipynb"
PORTAL = "https://distilroberta.iagentek.com.mx"
REPO = "https://github.com/azulls1/DistilRoBERTa"
CONSULTA_BANNER = "I still have not received my card"

# Paleta Forest (la de las apps de iAgentek)
FOREST, EVERGREEN, PINE, MOSS, FOG, BIEN, MAL, AVISO = "#04202C", "#304040", "#5B7065", "#9EADA3", "#C9D1C8", "#059669", "#DC2626", "#D97706"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": FOG, "axes.labelcolor": EVERGREEN, "xtick.color": PINE, "ytick.color": PINE,
                     "axes.titleweight": "bold", "axes.titlecolor": FOREST, "figure.dpi": 150})


def leer(nombre):
    return json.load(open(ART / nombre, encoding="utf-8"))


def pct(x, d=2):
    return f"{x * 100:.{d}f} %"


# ── 1. Banner con la predicción real ────────────────────────────────────────
def banner():
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(RAIZ / "modelo")
    mod = AutoModelForSequenceClassification.from_pretrained(RAIZ / "modelo").eval()
    with torch.no_grad():
        p = torch.softmax(mod(**tok(CONSULTA_BANNER, return_tensors="pt")).logits[0], -1)
    top = torch.topk(p, 3)
    filas, y = [], 264
    for k, (prob, i) in enumerate(zip(top.values.tolist(), top.indices.tolist())):
        nombre = mod.config.id2label[i]
        ancho = max(3, round(218 * prob))
        estilo = 'font-size="15" fill="#ffffff"' if k == 0 else 'font-size="13" fill="#ffffff" fill-opacity="0.6"'
        filas.append(f'    <text x="714" y="{y}" {estilo}>{nombre} · {prob * 100:.1f} %</text>\n'
                     f'    <rect x="930" y="{y - 11}" width="180" height="{12 if k == 0 else 10}" rx="5" fill="#ffffff" fill-opacity="0.12"/>\n'
                     f'    <rect class="barra" x="930" y="{y - 11}" width="{max(3, round(180 * prob))}" height="{12 if k == 0 else 10}" rx="5" fill="{"#34d399" if k == 0 else MOSS}"/>')
        y += 26 if k == 0 else 24
    svg = (DOC / "banner.plantilla.svg").read_text().replace("{{BARRAS}}", "\n".join(filas))
    (DOC / "banner.svg").write_text(svg)
    return [(mod.config.id2label[i], prob) for prob, i in zip(top.values.tolist(), top.indices.tolist())]


# ── 2. Figuras del notebook ejecutado ───────────────────────────────────────
TITULOS = {"2.1": "longitudes", "2.3": "ngramas", "2.4": "nube", "2.5": "balance", "3.3": "curvas",
           "3.4": "confusion", "3.5": "f1_por_clase", "4.7": "veredictos"}


def figuras():
    nb = nbformat.read(NB, as_version=4)
    seccion, guardadas = "0", {}
    for c in nb.cells:
        if c.cell_type == "markdown":
            m = re.findall(r"^###? (\d+(?:\.\d+)?)", c.source, re.M)
            seccion = m[-1] if m else seccion
            continue
        for o in c.outputs:
            if "image/png" in o.get("data", {}) and seccion in TITULOS and TITULOS[seccion] not in guardadas:
                ruta = DOC / f"fig_{TITULOS[seccion]}.png"
                ruta.write_bytes(base64.b64decode(o["data"]["image/png"]))
                guardadas[TITULOS[seccion]] = ruta.name
    return guardadas


# ── 3. Gráficas nuevas con los mismos datos ─────────────────────────────────
def graficas(cal, exp):
    # Calibración de la decodificación (P1) y de la estructura (configuración elegida)
    dec = cal[cal["prompt"] == "P1"].groupby("config").agg(formato=("formato_ok", "sum"), falsas=("cita_falsa", "sum"),
                                                          n=("formato_ok", "size"))
    elegida = cal.loc[cal["elegida"], "config"].iloc[0]
    est = cal[cal["config"] == elegida].groupby("prompt").agg(verif=("cita_verificable", "sum"), falsas=("cita_falsa", "sum"),
                                                              formato=("formato_ok", "sum"), n=("formato_ok", "size"))
    fig, ejes = plt.subplots(1, 2, figsize=(12, 3.8))
    x = np.arange(len(dec))
    ejes[0].bar(x - 0.2, dec["formato"], 0.4, color=FOREST, label="formato correcto")
    ejes[0].bar(x + 0.2, dec["falsas"], 0.4, color=MAL, label="cita falsa")
    ejes[0].set_xticks(x, [f"{c}\n{p}" for c, p in zip(dec.index, DESCR_CONFIG(cal, dec.index))], fontsize=8)
    ejes[0].set_title("Decodificación (estructura P1)"); ejes[0].legend(frameon=False, fontsize=8)
    ejes[0].set_ylim(0, dec["n"].max() + 0.5); ejes[0].set_ylabel(f"respuestas (de {int(dec['n'].max())})")
    x = np.arange(len(est))
    ejes[1].bar(x - 0.27, est["verif"], 0.27, color=BIEN, label="cita verificable")
    ejes[1].bar(x, est["falsas"], 0.27, color=MAL, label="cita falsa")
    ejes[1].bar(x + 0.27, est["formato"], 0.27, color=FOREST, label="formato correcto")
    ejes[1].set_xticks(x, est.index); ejes[1].set_title(f"Estructura del prompt (configuración {elegida})")
    ejes[1].legend(frameon=False, fontsize=8); ejes[1].set_ylim(0, est["n"].max() + 0.5)
    fig.tight_layout(); fig.savefig(DOC / "grafica_calibracion.png", facecolor="white"); plt.close(fig)

    # Veredictos de la revisión manual
    orden = ["pertinente", "parcial", "alucinada"]
    v = exp["veredicto"].value_counts().reindex(orden).fillna(0)
    fig, eje = plt.subplots(figsize=(6, 3.4))
    eje.pie(v, labels=[f"{k}\n{int(n)}" for k, n in v.items()], colors=[BIEN, AVISO, MAL], startangle=90,
            wedgeprops={"width": 0.38, "edgecolor": "white"}, textprops={"color": EVERGREEN, "fontsize": 10})
    eje.set_title(f"Revisión manual de las {len(exp)} explicaciones")
    fig.tight_layout(); fig.savefig(DOC / "grafica_veredictos.png", facecolor="white"); plt.close(fig)


def DESCR_CONFIG(cal, claves):
    out = []
    for k in claves:
        p = cal.loc[cal["config"] == k, "parametros"].iloc[0]
        t = "T→0" if p.get("temperature") is None else f"T={p['temperature']}"
        out.append(f"{t} · {p['max_new_tokens']}t")
    return out


# ── 4. README ───────────────────────────────────────────────────────────────
def tabla(filas, cab):
    return "\n".join(["| " + " | ".join(cab) + " |", "|" + "|".join("---" for _ in cab) + "|"] +
                     ["| " + " | ".join(str(x) for x in f) + " |" for f in filas])


def main():
    DOC.mkdir(parents=True, exist_ok=True)
    corr, clases, metr = leer("corrida.json"), leer("clases.json"), leer("metricas_clase.json")
    ent, eda, cumpl = leer("entrenamiento.json"), leer("eda.json"), leer("cumplimiento.json")
    cal = pd.DataFrame(leer("calibracion.json"))
    exp = pd.DataFrame(leer("explicaciones.json")).merge(pd.DataFrame(leer("revision_manual.json"))[
        ["consulta_id", "veredicto", "razon_categoria", "nota_revision"]], on="consulta_id", suffixes=("", "_r"))
    if "veredicto_r" in exp:
        exp["veredicto"] = exp["veredicto_r"]
    h = corr["hiperparametros"]
    nombre = {c["id"]: c["nombre"] for c in clases}
    m = pd.DataFrame(metr).assign(nombre=lambda d: d["clase_id"].map(nombre))
    n_err = int(m["errores"].sum())

    top3 = banner()
    figs = figuras()
    graficas(cal, exp)

    mejores = m[m["categoria"] == "mejor"].sort_values(["f1", "errores"], ascending=[False, True])
    peores = m[m["categoria"] == "peor"].sort_values(["errores", "f1"], ascending=[False, True])
    ng = pd.DataFrame(eda["ngramas"])
    est = {(e["metrica"], e["particion"]): e for e in eda["estadisticas"]}
    ver = exp["veredicto"].value_counts()
    razones = exp["razon_categoria"].value_counts()
    por_crit = Counter(q["criterio"] for q in cumpl)
    ok_crit = Counter(q["criterio"] for q in cumpl if q["cumplido"])
    fuentes = Counter(q.get("fuente", "Enunciado") for q in cumpl)
    elegida_cfg, elegido_p = h["config_llm"], h["prompt_llm"]
    cfg_txt = "codiciosa (T→0)" if h.get("llm_temperature") is None else f"temperature={h['llm_temperature']}, top_p={h['llm_top_p']}"
    ejemplo = exp.sort_values("orden").iloc[0]

    badges = " ".join([
        "![Python](https://img.shields.io/badge/Python-3.11-04202C?logo=python&logoColor=white)",
        "![PyTorch](https://img.shields.io/badge/PyTorch-2.x-304040?logo=pytorch&logoColor=white)",
        "![Transformers](https://img.shields.io/badge/🤗_Transformers-5.x-5B7065)",
        "![Angular](https://img.shields.io/badge/Angular-22-04202C?logo=angular&logoColor=white)",
        "![Tailwind](https://img.shields.io/badge/Tailwind_CSS-4-304040?logo=tailwindcss&logoColor=white)",
        "![FastAPI](https://img.shields.io/badge/FastAPI-0.141-5B7065?logo=fastapi&logoColor=white)",
        "![Celery](https://img.shields.io/badge/Celery-5.6-04202C?logo=celery&logoColor=white)",
        "![Redis](https://img.shields.io/badge/Redis-7-304040?logo=redis&logoColor=white)",
        "![Supabase](https://img.shields.io/badge/Supabase-Postgres_15-5B7065?logo=supabase&logoColor=white)",
        "![Docker Swarm](https://img.shields.io/badge/Docker_Swarm-Traefik-04202C?logo=docker&logoColor=white)",
    ])
    kpis = tabla([[f"🎯 **{pct(corr['accuracy'])}**", f"📊 **{corr['f1_macro']:.3f}**", f"❌ **{n_err}**",
                   f"⏱️ **{corr['duracion_entrenamiento_s'] / 60:.1f} min**", f"🧪 **{ver.get('pertinente', 0)}/{len(exp)}**",
                   f"✅ **{sum(ok_crit.values())}/{len(cumpl)}**"]],
                 ["Accuracy (prueba)", "F1 macro", "Errores", "Fine-tuning", "Explicaciones pertinentes", "Exigencias cumplidas"])

    curva = tabla([[e["epoch"], f"{e['train_loss']:.4f}", f"{e['eval_loss']:.4f}", f"{e['eval_accuracy']:.4f}",
                    f"{e['eval_f1_macro']:.4f}" + (" ⭐" if e["epoch"] == h["mejor_epoca"] else "")] for e in ent],
                  ["Época", "Pérdida train", "Pérdida validación", "Accuracy val.", "F1 macro val."])
    t_mejores = tabla([[f"`{r.nombre}`", f"{r.f1:.3f}", r.errores] for r in mejores.itertuples()], ["Clase", "F1", "Errores"])
    t_peores = tabla([[f"`{r.nombre}`", f"{r.f1:.3f}", f"{r.precision:.3f}", f"{r.recall:.3f}", r.errores]
                      for r in peores.itertuples()], ["Clase", "F1", "Precision", "Recall", "Errores"])
    t_ngr = tabla([[f"{i + 1}", *(f"{fila.ngrama} ({fila.frecuencia:,})" if fila is not None else "" for fila in fs)]
                   for i, fs in enumerate(zip(
                       [r for r in ng[ng.tipo == "palabra"].sort_values("rango").head(10).itertuples()],
                       [r for r in ng[ng.tipo == "bigrama"].sort_values("rango").itertuples()],
                       [r for r in ng[ng.tipo == "trigrama"].sort_values("rango").itertuples()]))],
                  ["#", "Palabra", "Bigrama", "Trigrama"])
    t_est = tabla([[met.replace("n_", "").replace("caracteres", "caracteres"), part, f"{e['mean']:.2f}", f"{e['mediana']:.0f}",
                    f"{e['max']:.0f}"] for (met, part), e in sorted(est.items())], ["Métrica", "Partición", "Media", "Mediana", "Máx."])
    t_cal = tabla([[k, *DESCR_CONFIG(cal, [k]), int(g["formato_ok"].sum()), int(g["cita_falsa"].sum()), len(g)]
                   for k, g in cal[cal["prompt"] == "P1"].groupby("config")],
                  ["Config.", "Parámetros", "Formato correcto", "Citas falsas", "Respuestas"])
    t_est_p = tabla([[p, int(g["cita_verificable"].sum()), int(g["cita_falsa"].sum()), int(g["formato_ok"].sum()), len(g)]
                     for p, g in cal[cal["config"] == elegida_cfg].groupby("prompt")],
                    ["Estructura", "Citas verificables", "Citas falsas", "Formato correcto", "Respuestas"])
    t_exp = tabla([[r.orden, f"“{r.texto}”", f"`{r.real}`", f"`{r.predicha}`",
                    {"pertinente": "🟢", "parcial": "🟡", "alucinada": "🔴"}.get(r.veredicto, "") + " " + r.veredicto]
                   for r in exp.sort_values("orden").itertuples()], ["#", "Consulta", "Real", "Predicha", "Veredicto"])
    t_rub = tabla([[c, q[0]["criterio_nombre"], q[0]["puntos"] if q[0]["puntos"] is not None else "—",
                    f"{q[0]['peso']} %" if q[0]["peso"] else "—", f"{ok_crit[c]}/{por_crit[c]} ✅"]
                   for c, q in ((c, [x for x in cumpl if x["criterio"] == c]) for c in dict.fromkeys(x["criterio"] for x in cumpl))],
                  ["Criterio", "Descripción", "Puntos", "Peso", "Exigencias"])
    paginas = tabla([
        ["🏠", "[Resumen](%s/resumen)" % PORTAL, "KPI animados, curvas de entrenamiento y narración guiada"],
        ["📈", "[Análisis exploratorio](%s/eda)" % PORTAL, "Estadísticas, n-gramas, nube de palabras y balance"],
        ["🗂️", "[Desempeño por clase](%s/clases)" % PORTAL, "Las 77 clases, ordenables, con las 7 mejores y 7 peores"],
        ["🔥", "[Matriz de confusión](%s/confusion)" % PORTAL, "Mapa de calor animado con foco por fila/columna"],
        ["💬", "[Explicaciones del LLM](%s/explicaciones)" % PORTAL, "Las 20 explicaciones, su veredicto y el prompt EN/ES"],
        ["🧪", "[Simulación](%s/simulacion)" % PORTAL, "Una consulta real paso a paso: tokens, 6 capas, top-5 y Falcon"],
        ["⚡", "[Clasificar](%s/clasificar)" % PORTAL, "Inferencia en vivo encolada en Celery + Redis"],
        ["🛡️", "[Cumplimiento](%s/cumplimiento)" % PORTAL, "Cada exigencia del enunciado y de la rúbrica con su evidencia"],
        ["📦", "[Entregables](%s/entregables)" % PORTAL, "ZIP para el profesor, notebook, PDF, figuras y datos con SHA-256"],
    ], ["", "Página", "Qué muestra"])

    def fig(nombre, alt):
        return f'<img src="docs/readme/{nombre}" alt="{alt}" width="100%">' if (DOC / nombre).exists() else ""

    demo = '<img src="docs/readme/demo.gif" alt="Demostración del portal" width="100%">' if (DOC / "demo.gif").exists() else ""

    nb_md = nbformat.read(NB, as_version=4)
    txt = "\n".join(o.get("text", "") for c in nb_md.cells if c.cell_type == "code" for o in c.outputs)
    g = lambda patron: re.search(patron, txt).groups()
    h1_rho, h1_p = g(r"H1 · Spearman\(ejemplos de entrenamiento, F1\) = (-?[\d.]+) \(p = ([\d.]+)\)")
    h2_conf, h2_no, h2_x = g(r"se confunden ([\d.]+) · pares que nunca se confunden ([\d.]+) \((\d+)× más\)")
    (h2_rho,) = g(r"Spearman\(bigramas compartidos, errores entre el par\) = ([\d.]+)")
    (tfidf_rho,) = g(r"Spearman\(similitud léxica, errores entre el par\) = ([\d.]+)")
    fam_pct, fam_n = g(r"Errores dentro de la misma familia: ([\d.]+%) \((\d+ de \d+)\)")
    conf_ok, conf_err = g(r"Confianza media: aciertos ([\d.]+) · errores ([\d.]+)")
    conf_pct, conf_n = g(r"confianza > 0.90: ([\d.]+%) \((\d+ de \d+)\)")
    seccion6 = next(c.source for c in nb_md.cells if c.cell_type == "markdown" and c.source.startswith("## 6. Referencias"))
    referencias = [l for l in seccion6.splitlines() if l.startswith("- ")]
    audios = sorted((RAIZ / "frontend" / "public" / "audio").glob("*.mp3"))
    n_narraciones = len(re.findall(r"^  \w+: \{", (RAIZ / "frontend" / "src" / "app" / "core" / "narraciones.ts").read_text(), re.M))
    readme = f"""<div align="center">

<img src="docs/readme/banner.svg" alt="DistilRoBERTa · Banking77 — banner animado" width="100%">

{badges}

**[🌐 Portal en vivo]({PORTAL})** · **[📦 Entregables]({PORTAL}/entregables)** · **[🧪 Simulación]({PORTAL}/simulacion)** · **[🛡️ Cumplimiento]({PORTAL}/cumplimiento)**

*UNIR · Maestría en Inteligencia Artificial · Sistemas Cognitivos Artificiales · Actividad 2 (individual)*
*«Transformers y Modelos de Lenguaje Grande (LLM)» — Adonai Samael Hernández Mata*

</div>

> [!NOTE]
> **Todo lo que aparece en este README se generó a partir de la ejecución real** (`python ml/generar_readme.py`):
> cifras leídas de `artefactos/`, figuras extraídas del notebook ejecutado y, en el banner, la predicción real del modelo
> afinado sobre la consulta «{CONSULTA_BANNER}» → `{top3[0][0]}` ({top3[0][1] * 100:.1f} %).

## ✨ Resultados en una mirada

{kpis}

## 🧭 Contenido

- [🔄 Cómo funciona](#-cómo-funciona)
- [📈 1 · Análisis exploratorio](#-1--análisis-exploratorio)
- [🤖 2 · Clasificación con DistilRoBERTa](#-2--clasificación-con-distilroberta)
- [💬 3 · Explicación de errores con Falcon-7b-instruct](#-3--explicación-de-errores-con-falcon-7b-instruct)
- [🛡️ Cumplimiento de la rúbrica](#️-cumplimiento-de-la-rúbrica)
- [🌐 El portal](#-el-portal)
- [🏗️ Arquitectura y despliegue](#️-arquitectura-y-despliegue)
- [🛠️ Reproducir](#️-reproducir)
- [📚 Referencias](#-referencias)

## 🔄 Cómo funciona

```mermaid
flowchart LR
    A["📥 PolyAI/banking77<br/>{corr['n_train'] + corr['n_val']:,} train · {corr['n_test']:,} test · 77 clases"] --> B["📈 EDA<br/>longitudes · n-gramas<br/>nube · balance"]
    B --> C["🔤 Tokenizer BPE<br/>max_length {h['max_length']}"]
    C --> D["🤖 DistilRoBERTa<br/>fine-tuning {h['mejor_epoca']}/{h['epocas_max']} épocas"]
    D --> E["🎯 Evaluación<br/>acc {pct(corr['accuracy'])} · F1 {corr['f1_macro']:.3f}"]
    E --> F["❌ {n_err} errores"]
    F --> G["💬 Falcon-7b-instruct<br/>config {elegida_cfg} · prompt {elegido_p}"]
    G --> H["🧑‍🏫 Revisión manual<br/>{ver.get('pertinente', 0)} 🟢 · {ver.get('parcial', 0)} 🟡 · {ver.get('alucinada', 0)} 🔴"]
    style D fill:#04202C,color:#fff,stroke:#04202C
    style G fill:#304040,color:#fff,stroke:#304040
```

## 📈 1 · Análisis exploratorio

{fig(figs.get("longitudes", ""), "Distribución de longitudes")}

<details open>
<summary><b>📐 Estadísticas descriptivas</b></summary>

{t_est}

</details>

<details>
<summary><b>🔠 Palabras, bigramas y trigramas más frecuentes (texto limpio)</b></summary>

{t_ngr}

{fig(figs.get("ngramas", ""), "N-gramas más frecuentes")}

</details>

<table><tr>
<td width="50%">{fig(figs.get("nube", ""), "Nube de palabras")}</td>
<td width="50%">{fig(figs.get("balance", ""), "Balance de clases")}</td>
</tr></table>

**Balance:** de {eda['balance']['mínimo (train)']} (`{eda['balance']['clase mínima']}`) a {eda['balance']['máximo (train)']} (`{eda['balance']['clase máxima']}`)
ejemplos por clase en train — razón {eda['balance']['razón máx/mín']}×, entropía normalizada {eda['balance']['entropía normalizada']}. La prueba tiene 40 por clase.

## 🤖 2 · Clasificación con DistilRoBERTa

<table><tr>
<td width="50%">{fig(figs.get("curvas", ""), "Curvas de entrenamiento")}</td>
<td width="50%">{fig(figs.get("f1_por_clase", ""), "F1 por clase")}</td>
</tr></table>

<details open>
<summary><b>📉 Curva por época (⭐ = época elegida por F1 macro en validación)</b></summary>

{curva}

</details>

<table><tr>
<td valign="top" width="45%">

**🟢 Las 7 mejor clasificadas**

{t_mejores}

</td>
<td valign="top" width="55%">

**🔴 Las 7 con más errores**

{t_peores}

</td>
</tr></table>

{fig(figs.get("confusion", ""), "Matriz de confusión")}

### 🔎 ¿Por qué se equivoca? Dos hipótesis del EDA, medidas

| Hipótesis | Medida en el notebook (§ 3.6) | Lectura |
|---|---|---|
| **H1** · el desbalance causa los errores | ρ(ejemplos de train, F1) = **{h1_rho}** (p = {h1_p}) | ⚪ sin evidencia a favor |
| **H2** · el vocabulario compartido se asocia a los errores | bigramas compartidos: **{h2_conf}** vs **{h2_no}** (**{h2_x}×**), ρ = {h2_rho} | 🟢 asociación confirmada |
| Similitud TF-IDF entre clases | ρ(similitud, errores del par) = **{tfidf_rho}** | 🟢 las clases fallan hacia sus vecinas |
| Errores dentro de la misma familia | **{fam_pct}** ({fam_n}) | 🟢 transferencias, recargas, tarjetas… |
| Errores con confianza > 0.90 | **{conf_pct}** ({conf_n}) · confianza media {conf_ok} aciertos vs {conf_err} errores | 🟡 la confianza no basta para abstenerse |

## 💬 3 · Explicación de errores con Falcon-7b-instruct

Se calibraron los **tres ejes** que pide el enunciado: temperatura, longitud de respuesta y estructura del prompt,
con reglas de elección fijadas antes de ver los resultados. Ganó la **configuración {elegida_cfg}** ({cfg_txt},
`max_new_tokens={h['llm_max_new_tokens']}`) con la **estructura {elegido_p}**.

{fig("grafica_calibracion.png", "Calibración del prompt")}

<details>
<summary><b>🎛️ Calibración de la decodificación (estructura P1)</b></summary>

{t_cal}

</details>

<details>
<summary><b>🧱 Calibración de la estructura (configuración {elegida_cfg})</b></summary>

{t_est_p}

</details>

<details>
<summary><b>🇬🇧 / 🇪🇸 El prompt elegido, tal como se envió (ejemplo: error n.º {ejemplo.orden})</b></summary>

```text
{ejemplo.prompt}
```

</details>

<table><tr>
<td width="40%">{fig("grafica_veredictos.png", "Veredictos")}</td>
<td width="60%">

**Razones que da el LLM**

{tabla([[k.replace('_', ' '), int(n)] for k, n in razones.items()], ["Razón", "Explicaciones"])}

</td>
</tr></table>

<details>
<summary><b>📝 Las {len(exp)} explicaciones y su veredicto manual</b></summary>

{t_exp}

</details>

## 🛡️ Cumplimiento de la rúbrica

{t_rub}

{fuentes.get('Enunciado', 0)} exigencias del enunciado · {fuentes.get('Rúbrica detallada', 0)} del nivel 4 de la rúbrica detallada · {fuentes.get('Solicitud', 0)} solicitud adicional.
Detalle con evidencia en **[/cumplimiento]({PORTAL}/cumplimiento)**.

## 🌐 El portal

{demo}

{paginas}

## 🏗️ Arquitectura y despliegue

```mermaid
flowchart TB
    U["👤 Navegador"] -->|HTTPS| T["🔀 Traefik<br/>Let's Encrypt"]
    T -->|"/"| W["🅰️ web · Angular 22 + Tailwind 4<br/>nginx"]
    T -->|"/api"| A["⚙️ api · FastAPI"]
    A -->|encola| R[("🟥 Redis 7")]
    R --> K["👷 worker · Celery<br/>DistilRoBERTa en CPU"]
    A --> P[("🐘 Supabase · Postgres<br/>17 tablas DistilRoBERTa_*")]
    K --> P
    K --> Z["📦 ZIP de entregables"]
    subgraph Swarm["🐳 Docker Swarm · VPS iAgentek"]
      W
      A
      R
      K
    end
```

```mermaid
sequenceDiagram
    autonumber
    participant N as 🌐 Portal
    participant A as ⚙️ API
    participant R as 🟥 Redis
    participant K as 👷 Worker
    participant P as 🐘 Postgres
    N->>A: POST /api/clasificar {{texto}}
    A->>P: inferencia «pendiente»
    A->>R: encola tarea
    A-->>N: 202 · task_id
    R->>K: tarea
    K->>K: tokenizar · 6 capas · softmax
    K->>P: clase, confianza, top-5, tokens
    loop cada 400 ms
      N->>A: GET /api/tareas/{{id}}
      A->>P: estado
    end
    A-->>N: resultado
```

| Pieza | Detalle |
|---|---|
| 🧠 Modelo servido | DistilRoBERTa afinado, CPU, cargado una vez por proceso del worker |
| 💬 LLM | Falcon-7b (≈ 14 GB) **no** se sirve: sus salidas se generaron offline y se publican como datos |
| 🐘 Datos | Tablas `DistilRoBERTa_*` con RLS forzado; solo el rol de la app las lee; `anon` recibe *permission denied* |
| 🔐 Secretos | DSN en un secreto de Docker Swarm, nunca en el repositorio |
| 🛡️ Seguridad web | HSTS, CSP estricta sin scripts inline, anti-*clickjacking*, límite de peticiones por IP, cuerpo máx. 8 KB, escrituras con cabecera anti-CSRF, sin `/docs` públicos, nginx sin privilegios |
| 🎨 Diseño | Forest Design System de iAgentek (el mismo de las demás apps de la maestría) |
| 🔊 Narraciones | {len(audios)} de {n_narraciones} con audio generado con ElevenLabs; todas con transcripción en el portal |

## 🛠️ Reproducir

```bash
# 1 · Entorno
uv venv .venv-ml --python 3.11 && source .venv-ml/bin/activate
uv pip install torch transformers datasets accelerate scikit-learn pandas matplotlib seaborn nltk wordcloud jupyter nbclient nbconvert "psycopg[binary]"

# 2 · Notebook de punta a punta (se pausa para la revisión manual de las 20 explicaciones)
python ml/construir_notebook.py && python ml/ejecutar_notebook.py
python ml/interpretar.py && python ml/exportar_pdf.py

# 3 · Resultados, simulación, entregables y README
python ml/simulacion_llm.py && python ml/cumplimiento.py && python ml/generar_readme.py

# 4 · Base de datos y despliegue
psql "$DSN_ADMIN" -f db/migraciones/001_esquema.sql   # … hasta 005
./deploy/deploy.sh
```

<details>
<summary><b>📁 Estructura del repositorio</b></summary>

```text
notebooks/   Notebook entregable (.ipynb) y su PDF
ml/          construir · ejecutar · interpretar · exportar_pdf · simulacion_llm · cumplimiento · generar_readme
artefactos/  Resultados de la corrida (JSON/CSV) → base de datos y README
data/        CSV oficiales de banking77
db/          Migraciones SQL (tablas DistilRoBERTa_*, rol propio, RLS)
backend/     FastAPI + Celery (Redis)
frontend/    Angular 22 + Tailwind 4 (Forest Design System)
deploy/      stack.yml (Docker Swarm + Traefik) y deploy.sh
docs/readme/ Banner animado, figuras y gráficas de este README
_bmad*/      BMad Method (arquitectura, épicas, sprint)
```

La planificación (constitución, especificación, plan, modelo de datos, contrato REST y tareas) se hizo con
**Spec Kit** en la carpeta hermana `espesificaciones/`; el desarrollo, con **BMad Method**.

</details>

## 📚 Referencias

<details>
<summary><b>{len(referencias)} referencias en formato APA</b> (copiadas de la sección 6 del notebook)</summary>

{chr(10).join(referencias)}

</details>

<div align="center">
<sub>Hecho con 🌲 Forest Design System · <b>powered by iAgentek</b> · Maestría en IA · UNIR 2026</sub>
</div>
"""
    (RAIZ / "README.md").write_text(readme, encoding="utf-8")
    print(f"README.md ({len(readme):,} caracteres) · figuras: {sorted(figs)} · banner: {top3}")


if __name__ == "__main__":
    main()
