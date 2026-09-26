"""Clasificador DistilRoBERTa afinado, cargado una sola vez por proceso del worker (CPU)."""
import re
import time
from functools import lru_cache
from pathlib import Path

from .config import config

# Las mismas 198 stopwords de NLTK que usa el notebook (copiadas para no descargarlas en producción)
STOPWORDS = frozenset(Path(__file__).with_name("stopwords_en.txt").read_text().split())


def limpiar(texto: str) -> str:
    """Idéntica a la del notebook: minúsculas → sin caracteres especiales → sin stopwords."""
    texto = re.sub(r"[^a-z0-9\s]", " ", texto.lower())
    return " ".join(t for t in texto.split() if t not in STOPWORDS and len(t) > 1)


@lru_cache
def _cargar():
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.set_num_threads(2)
    ruta = config().ruta_modelo
    tokenizer = AutoTokenizer.from_pretrained(ruta)
    modelo = AutoModelForSequenceClassification.from_pretrained(ruta).eval()
    return torch, tokenizer, modelo


def clasificar(texto: str, k: int = 5) -> dict:
    """Clase predicha, su probabilidad, las k más probables y los tokens BPE que vio el modelo."""
    torch, tokenizer, modelo = _cargar()
    t0 = time.perf_counter()
    entrada = tokenizer(texto, truncation=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        probs = torch.softmax(modelo(**entrada).logits[0], dim=-1)
    top = torch.topk(probs, k)
    etiquetas = modelo.config.id2label
    top5 = [{"clase_id": int(i), "clase": etiquetas[int(i)], "prob": round(float(p), 5)}
            for p, i in zip(top.values, top.indices)]
    ids = entrada["input_ids"][0].tolist()
    tokens = [{"id": i, "token": t.replace("Ġ", "▁")} for i, t in zip(ids, tokenizer.convert_ids_to_tokens(ids))]
    return {"clase_id": top5[0]["clase_id"], "clase": top5[0]["clase"], "confianza": top5[0]["prob"],
            "top5": top5, "tokens": tokens, "texto_limpio": limpiar(texto),
            "duracion_ms": int((time.perf_counter() - t0) * 1000)}


def precargar() -> None:
    _cargar()
