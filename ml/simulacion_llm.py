"""Genera, fuera del notebook, las salidas reales de Falcon-7b-instruct que alimentan la simulación web:

1. La explicación (configuración y estructura elegidas por la calibración del notebook) de TODOS los
   errores del conjunto de prueba, no solo de los 20 revisados a mano.
2. Las cinco configuraciones de decodificación (A–E) sobre las 20 consultas revisadas, con la estructura
   elegida, para el simulador de calibración.

Usa las mismas plantillas, semilla y parámetros que el notebook, y lo COMPRUEBA: el prompt que construye
para cada una de las 20 consultas debe ser idéntico al guardado en artefactos/explicaciones.json.
Las 20 explicaciones con la configuración elegida se reutilizan tal cual (no se regeneran).

Uso:  python ml/simulacion_llm.py        → artefactos/simulacion_llm.json
"""
import csv
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

RAIZ = Path(__file__).resolve().parents[1]
ART = RAIZ / "artefactos"
SALIDA = ART / "simulacion_llm.json"
SEED = 42
LLM_ID = "tiiuae/falcon-7b-instruct"

CONFIGS = {
    "A": dict(max_new_tokens=60, temperature=None, top_p=None),
    "B": dict(max_new_tokens=60, temperature=0.3, top_p=0.9),
    "C": dict(max_new_tokens=60, temperature=1.0, top_p=0.95),
    "D": dict(max_new_tokens=150, temperature=1.0, top_p=0.95),
    "E": dict(max_new_tokens=150, temperature=0.3, top_p=0.9),
}
CABECERA = "You are a banking customer-support analyst who audits an automatic intent classifier.\n"
P1 = (CABECERA +
      "The classifier read the customer query below and predicted an intent that is WRONG.\n"
      "In at most two short sentences, explain why the classifier probably chose the predicted "
      "intent instead of the correct one. Refer only to words that appear in the query. "
      "Do not invent facts, do not give advice to the customer.\n\n"
      'Customer query: "{texto}"\nPredicted intent: {pred}\nCorrect intent: {real}\n\nExplanation:')
INSTRUCCION_P2 = (
    "The classifier read the customer query below and predicted an intent that differs from the label in the dataset.\n"
    "In at most two short sentences, explain which words of the query led the classifier to the predicted intent, "
    "quoting those words between single quotes exactly as they appear in the query. "
    "Do not invent facts, do not give advice to the customer.\n\n")
BLOQUE = 'Customer query: "{texto}"\nPredicted intent: {pred}\nDataset label: {real}\n\nExplanation:'
EJEMPLOS = (
    'Customer query: "I think my top up has been reverted"\nPredicted intent: top up failed\n'
    "Dataset label: top up reverted\n\nExplanation: The query mentions 'top up' and a reverted top-up looks "
    "like a failed one, which pushes toward top up failed. The word 'reverted' is what points to the label.\n\n"
    'Customer query: "How long is the wait for my card?"\nPredicted intent: card arrival\n'
    "Dataset label: card delivery estimate\n\nExplanation: The words 'wait' and 'my card' are typical of "
    "messages about cards that have not arrived. 'How long' asks for a delivery time, which fits the label better.\n\n")
PROMPTS = {"P1": P1, "P2": CABECERA + INSTRUCCION_P2 + BLOQUE, "P3": CABECERA + INSTRUCCION_P2 + EJEMPLOS + BLOQUE}


def legible(e):
    return e.replace("_", " ").replace("?", "").lower()


def construir(texto, pred, real, variante):
    return PROMPTS[variante].format(texto=texto.strip(), pred=legible(pred), real=legible(real))


def contar_oraciones(t):
    import re
    return len([s for s in re.split(r"(?<=[.!?])\s+", t.strip()) if s])


def main():
    corrida = json.load(open(ART / "corrida.json"))
    elegida, variante = corrida["hiperparametros"]["config_llm"], corrida["hiperparametros"]["prompt_llm"]
    etiquetas = [c["nombre"] for c in json.load(open(ART / "clases.json"))]
    consultas = {int(r["id"]): r for r in csv.DictReader(open(ART / "consultas.csv"))}
    errores = []
    for r in csv.DictReader(open(ART / "predicciones.csv")):
        q = consultas[int(r["consulta_id"])]
        if int(r["clase_pred_id"]) != int(q["label"]):
            errores.append({"consulta_id": int(r["consulta_id"]), "texto": q["text"],
                            "real": etiquetas[int(q["label"])], "predicha": etiquetas[int(r["clase_pred_id"])],
                            "confianza": float(r["confianza"])})
    revisadas = {e["consulta_id"]: e for e in json.load(open(ART / "explicaciones.json"))}

    # Garantía: las plantillas de este script son las del notebook
    for c, e in revisadas.items():
        assert construir(e["texto"], e["predicha"], e["real"], variante) == e["prompt"], f"prompt distinto en {c}"
    print(f"Plantillas verificadas contra el notebook · configuración {elegida} · estructura {variante}", flush=True)

    resultados = {}
    for c, e in revisadas.items():   # las 20 con la configuración elegida: se reutilizan tal cual
        resultados[(c, elegida)] = {"consulta_id": c, "config": elegida, "prompt": variante, "texto": e["texto"],
                                    "real": e["real"], "predicha": e["predicha"], "confianza": e["confianza"],
                                    "salida_cruda": e["salida_cruda"], "explicacion": e["explicacion"],
                                    "n_oraciones": contar_oraciones(e["salida_cruda"]), "segundos": e["segundos"],
                                    "revisada": True}
    pendientes = [(e, elegida) for e in errores if e["consulta_id"] not in revisadas]
    pendientes += [(e, k) for e in revisadas.values() for k in CONFIGS if k != elegida]
    print(f"{len(errores)} errores · {len(revisadas)} revisados · {len(pendientes)} generaciones pendientes", flush=True)

    tok = AutoTokenizer.from_pretrained(LLM_ID)
    llm = AutoModelForCausalLM.from_pretrained(LLM_ID, dtype=torch.float16).to("mps").eval()
    for i, (e, k) in enumerate(pendientes, 1):
        cfg = CONFIGS[k]
        entrada = tok(construir(e["texto"], e["predicha"], e["real"], variante), return_tensors="pt").to("mps")
        muestreo = cfg["temperature"] is not None
        torch.manual_seed(SEED)
        t0 = time.time()
        with torch.no_grad():
            out = llm.generate(**entrada, max_new_tokens=cfg["max_new_tokens"], do_sample=muestreo,
                               temperature=cfg["temperature"] if muestreo else None,
                               top_p=cfg["top_p"] if muestreo else None, repetition_penalty=1.15,
                               pad_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id,
                               stop_strings=["\n\n", "Customer query:"], tokenizer=tok)
        cruda = tok.decode(out[0, entrada["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        resultados[(e["consulta_id"], k)] = {
            "consulta_id": e["consulta_id"], "config": k, "prompt": variante, "texto": e["texto"], "real": e["real"],
            "predicha": e["predicha"], "confianza": e["confianza"], "salida_cruda": cruda, "explicacion": cruda,
            "n_oraciones": contar_oraciones(cruda), "segundos": round(time.time() - t0, 1),
            "revisada": e["consulta_id"] in revisadas}
        if i % 20 == 0 or i == len(pendientes):
            json.dump(list(resultados.values()), open(SALIDA, "w"), indent=1, ensure_ascii=False)
            print(f"  {i}/{len(pendientes)}", flush=True)
    json.dump(list(resultados.values()), open(SALIDA, "w"), indent=1, ensure_ascii=False)
    print("FIN", len(resultados), "salidas", flush=True)


if __name__ == "__main__":
    main()
