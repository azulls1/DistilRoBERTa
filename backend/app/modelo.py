"""Clasificador DistilRoBERTa afinado, cargado una sola vez por proceso del worker (CPU)."""
import time
from functools import lru_cache

from .config import config


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
    """Devuelve la clase predicha, su probabilidad y las k más probables."""
    torch, tokenizer, modelo = _cargar()
    t0 = time.perf_counter()
    entrada = tokenizer(texto, truncation=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        probs = torch.softmax(modelo(**entrada).logits[0], dim=-1)
    top = torch.topk(probs, k)
    etiquetas = modelo.config.id2label
    top5 = [{"clase_id": int(i), "clase": etiquetas[int(i)], "prob": round(float(p), 5)}
            for p, i in zip(top.values, top.indices)]
    return {"clase_id": top5[0]["clase_id"], "clase": top5[0]["clase"], "confianza": top5[0]["prob"],
            "top5": top5, "duracion_ms": int((time.perf_counter() - t0) * 1000)}


def precargar() -> None:
    _cargar()
