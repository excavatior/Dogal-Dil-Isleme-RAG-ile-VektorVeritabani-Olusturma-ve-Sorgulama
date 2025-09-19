#!/usr/bin/env python3
"""
Day 1 — Hızlı Embedding Doğrulaması (TR/EN)
Amaç:
  - 2–3 Türkçe/İngilizce cümleyi embedding'e çevir
  - Cosine benzerlik hesapla (soru↔pozitif, soru↔negatif)
  - Sonuçların mantıklı olduğunu gözlemle (pozitif > negatif, margin > 0)

Kurulum:
  pip install -U sentence-transformers torch numpy pandas

Çalıştırma:
  python day1_test.py --model paraphrase-multilingual-MiniLM-L12-v2
"""

import argparse
import time
from typing import List
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


DEFAULT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def encode(model: SentenceTransformer, texts: List[str]) -> np.ndarray:
    return model.encode(texts, normalize_embeddings=True)


def run(model_name: str):
    model_load_t0 = time.time()
    model = SentenceTransformer(model_name)
    model_load_ms = (time.time() - model_load_t0) * 1000.0

    examples = [
        ("TR", "Türkiye'nin başkenti neresidir?", "Ankara, Türkiye'nin başkentidir.", "Kediler köpeklerden farklı hayvanlardır."),
        ("EN", "What is the capital of France?", "Paris is the capital of France.", "Cats are small domesticated animals."),
    ]

    rows = []
    for lang, query, positive, negative in examples:
        embeddings = encode(model, [query, positive, negative])
        q, p, n = embeddings[0], embeddings[1], embeddings[2]

        cos_pos = cosine(q, p)
        cos_neg = cosine(q, n)
        margin = cos_pos - cos_neg

        rows.append({
            "lang": lang,
            "cos_pos": round(cos_pos, 4),
            "cos_neg": round(cos_neg, 4),
            "margin": round(margin, 4),
            "query": query,
            "positive": positive,
            "negative": negative,
            "dim": len(q),
            "load_ms": round(model_load_ms, 2),
        })

    df = pd.DataFrame(rows)
    print(df)

    df.to_csv("day1_results.csv", index=False, encoding="utf-8")
    print("\nSonuçlar 'day1_results.csv' dosyasına kaydedildi.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default=DEFAULT_MODEL)
    args = ap.parse_args()
    run(args.model)
