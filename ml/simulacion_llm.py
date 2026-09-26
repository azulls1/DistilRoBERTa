"""Genera, fuera del notebook, las salidas reales de Falcon-7b-instruct que alimentan la simulación web:

1. La explicación (configuración elegida, B) de TODOS los errores del conjunto de prueba, no solo de
   los 20 revisados a mano — así la simulación muestra lo que el LLM respondió de verdad para
   cualquier error que el visitante elija.
2. Las tres configuraciones (A, B, C) sobre las 20 consultas revisadas, para el simulador de
   temperatura.

Usa exactamente el mismo prompt, semilla y parámetros que el notebook. Las 20 explicaciones con
configuración B se reutilizan de artefactos/explicaciones.json (no se regeneran).

Uso:  python ml/simulacion_llm.py        → artefactos/simulacion_llm.json
"""
import csv
import json
import re
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
    "C": dict(max_new_tokens=150, temperature=1.0, top_p=0.95),
}
PLANTILLA = (
    "You are a banking customer-support analyst who audits an automatic intent classifier.\n"
    "The classifier read the customer query below and predicted an intent that is WRONG.\n"
    "In at most two short sentences, explain why the classifier probably chose the predicted "
    "intent instead of the correct one. Refer only to words that appear in the query. "
    "Do not invent facts, do not give advice to the customer.\n\n"
    'Customer query: "{texto}"\n'
    "Predicted intent: {pred}\n"
    "Correct intent: {real}\n\n"
    "Explanation:"
)


def legible(e):
    return e.replace("_", " ").replace("?", "").lower()


def contar_oraciones(t):
    return len([s for s in re.split(r"(?<=[.!?])\s+", t.strip()) if s])


def recortar(t, n=2):
    t = t.strip().split("\n\n")[0].strip()
    return " ".join([s for s in re.split(r"(?<=[.!?])\s+", t) if s][:n])


def main():
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
    previo = {(x["consulta_id"], x["config"]): x for x in json.load(open(SALIDA))} if SALIDA.exists() else {}

    pendientes = [(e, "B") for e in errores if e["consulta_id"] not in revisadas]
    pendientes += [(revisadas[c] | {"predicha": revisadas[c]["predicha"]}, k) for c in revisadas for k in ("A", "C")]
    pendientes = [(e, k) for e, k in pendientes if (e["consulta_id"], k) not in previo]
    print(f"{len(errores)} errores · {len(revisadas)} revisados · {len(pendientes)} generaciones pendientes", flush=True)

    resultados = dict(previo)
    for c, e in revisadas.items():  # las 20 con B ya existen: se reutilizan tal cual
        resultados[(c, "B")] = {"consulta_id": c, "config": "B", "texto": e["texto"], "real": e["real"],
                                "predicha": e["predicha"], "confianza": e["confianza"], "salida_cruda": e["salida_cruda"],
                                "explicacion": e["explicacion"], "n_oraciones": contar_oraciones(e["salida_cruda"]),
                                "segundos": e["segundos"], "revisada": True}
    if pendientes:
        tok = AutoTokenizer.from_pretrained(LLM_ID)
        llm = AutoModelForCausalLM.from_pretrained(LLM_ID, dtype=torch.float16).to("mps").eval()
        for i, (e, k) in enumerate(pendientes, 1):
            cfg = CONFIGS[k]
            prompt = PLANTILLA.format(texto=e["texto"].strip(), pred=legible(e["predicha"]), real=legible(e["real"]))
            torch.manual_seed(SEED)
            entrada = tok(prompt, return_tensors="pt").to("mps")
            muestreo = cfg["temperature"] is not None
            t0 = time.time()
            with torch.no_grad():
                out = llm.generate(**entrada, max_new_tokens=cfg["max_new_tokens"], do_sample=muestreo,
                                   temperature=cfg["temperature"] if muestreo else None,
                                   top_p=cfg["top_p"] if muestreo else None, repetition_penalty=1.15,
                                   pad_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id,
                                   stop_strings=["\n\n", "Customer query:"], tokenizer=tok)
            cruda = tok.decode(out[0, entrada["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            resultados[(e["consulta_id"], k)] = {
                "consulta_id": e["consulta_id"], "config": k, "texto": e["texto"], "real": e["real"],
                "predicha": e["predicha"], "confianza": e["confianza"], "salida_cruda": cruda,
                "explicacion": recortar(cruda), "n_oraciones": contar_oraciones(cruda),
                "segundos": round(time.time() - t0, 1), "revisada": e["consulta_id"] in revisadas}
            if i % 10 == 0 or i == len(pendientes):
                json.dump(list(resultados.values()), open(SALIDA, "w"), indent=1, ensure_ascii=False)
                print(f"  {i}/{len(pendientes)}", flush=True)
    json.dump(list(resultados.values()), open(SALIDA, "w"), indent=1, ensure_ascii=False)
    print("FIN", len(resultados), "salidas", flush=True)


if __name__ == "__main__":
    main()
