#!/usr/bin/env python3
"""
Day 4 — Reranker + LLM ile Kaynaklı Cevap Üretimi

Kullanım örnekleri:
  # OpenAI ile
  python day4_rag_answer.py --query "embedding nedir?" --provider openai --top_k 5
  # HuggingFace yerel model ile (hafif, CPU/GPU çalışır)
  python day4_rag_answer.py --query "embedding nedir?" --provider hf --hf_model google/flan-t5-base --top_k 5

Notlar:
- Chroma koleksiyon adı ve yolun Day 2 ile aynı olması gerekir (varsayılan: ./chroma_db, rag_docs).
- Reranker: cross-encoder/ms-marco-MiniLM-L-6-v2 (hızlı, kaliteli).
- Çıktılar:
  - day4_rerank_scores.csv  (her aday için skorlar)
  - day4_answer.txt          (LLM'e gönderilen prompt + nihai cevap)
"""

import os, argparse, textwrap, csv
from typing import List, Dict, Any, Tuple

import numpy as np
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import CrossEncoder, SentenceTransformer


DEFAULT_EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_RERANKER     = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ---------- yardımcılar ----------

def format_hits(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    hits = []
    ids = results.get("ids", [[]])[0]
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0] if results.get("distances") else [None] * len(ids)
    for i in range(len(ids)):
        hits.append({
            "rank": i + 1,
            "id": ids[i],
            "distance": dists[i],
            "document": docs[i],
            "metadata": metas[i],
        })
    return hits

def rerank(query: str, hits: List[Dict[str, Any]], reranker_name: str, top_k: int) -> List[Dict[str, Any]]:
    """Cross-encoder ile yeniden sıralama."""
    ce = CrossEncoder(reranker_name)  # GPU varsa kullanır
    pairs = [(query, h["document"]) for h in hits]
    scores = ce.predict(pairs)  # daha yüksek = daha ilgili
    # Skorları ekle ve sırala
    for h, s in zip(hits, scores):
        h["rerank_score"] = float(s)
    hits.sort(key=lambda x: x["rerank_score"], reverse=True)
    return hits[:top_k]

def build_prompt(query: str, hits: List[Dict[str, Any]], max_chars: int = 2500) -> Tuple[str, str]:
    """En iyi parçaları birleştirip RAG prompt oluştur. Ayrıca kaynak listesini döndürür."""
    blocks, used = [], 0
    citations = []
    for h in hits:
        src   = h["metadata"].get("source")
        chunk = h["metadata"].get("chunk")
        cite  = f"[Kaynak: {src}, Parça: {chunk}]"
        block = f"{cite}\n{h['document']}"
        if used + len(block) > max_chars:
            continue
        blocks.append(block)
        citations.append(cite)
        used += len(block)
    context = "\n\n---\n\n".join(blocks)

    prompt = f"""Aşağıdaki kullanıcı sorusunu, verilen bağlamdan yararlanarak ve bağlam dışına çıkmadan yanıtla.
Yanıtın sonunda kullandığın kaynakları [Kaynak: path, Parça: i] biçiminde listele. Türkçe yanıt ver.

Soru:
{query}

Bağlam:
{context}

Yanıt talimatları:
- Sadece bağlamdan yararlan, uydurma yapma.
- Kısa, net, madde madde veya küçük paragraflarla yanıtla.
- Sonunda "Kaynaklar:" başlığıyla referansları yaz.
"""
    return prompt, "\n".join(citations)

def generate_openai(prompt: str, model: str = "gpt-4o-mini") -> str:
    from openai import OpenAI
    import os
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY bulunamadı. Ortam değişkeni olarak ayarlayın.")
    client = OpenAI(api_key=key)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role":"system","content":"Sen kaynaklı yanıt veren yardımcı bir asistansın."},
            {"role":"user","content":prompt},
        ],
        temperature=0.2,
        max_tokens=600,
    )
    return resp.choices[0].message.content

def generate_hf(prompt: str, model_name: str = "google/flan-t5-base") -> str:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    import torch
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mdl.to(device)
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
    out = mdl.generate(**inputs, max_new_tokens=500)
    return tok.decode(out[0], skip_special_tokens=True)

# ---------- ana akış ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db_dir", default="./chroma_db")
    ap.add_argument("--collection", default="rag_docs")
    ap.add_argument("--embed_model", default=DEFAULT_EMBED_MODEL)
    ap.add_argument("--reranker", default=DEFAULT_RERANKER)
    ap.add_argument("--query", required=True)
    ap.add_argument("--provider", choices=["openai","hf"], default="hf", help="LLM sağlayıcısı")
    ap.add_argument("--openai_model", default="gpt-4o-mini")
    ap.add_argument("--hf_model", default="google/flan-t5-base")
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--candidates", type=int, default=10, help="Reranker'a verilecek ilk aday sayısı")
    ap.add_argument("--max_prompt_chars", type=int, default=2500)
    args = ap.parse_args()

    # 1) Chroma bağlantısı + embed fn
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=args.embed_model, normalize_embeddings=True
    )
    client = chromadb.PersistentClient(path=args.db_dir)
    col = client.get_or_create_collection(name=args.collection, embedding_function=embed_fn)

    # 2) İlk adayları getir (retrieval)
    res = col.query(
        query_texts=[args.query],
        n_results=max(args.candidates, args.top_k),
        include=["documents", "metadatas", "distances"],
    )
    hits = format_hits(res)
    if not hits:
        raise SystemExit("Aday bulunamadı. Day 2 ingestion çalışmış mı?")

    # 3) Rerank (cross-encoder)
    hits = rerank(args.query, hits, args.reranker, args.top_k)

    # 4) Prompt hazırla
    prompt, cite_list = build_prompt(args.query, hits, max_chars=args.max_prompt_chars)

    # 5) LLM'den yanıt al
    if args.provider == "openai":
        answer = generate_openai(prompt, model=args.openai_model)
    else:
        answer = generate_hf(prompt, model_name=args.hf_model)

    # 6) Kayıtlar
    # 6a) Rerank skorlarını CSV'ye yaz
    with open("day4_rerank_scores.csv", "w", newline="", encoding="utf-8") as fw:
        writer = csv.DictWriter(fw, fieldnames=["rank","source","chunk","rerank_score","distance"])
        writer.writeheader()
        for idx, h in enumerate(hits, start=1):
            writer.writerow({
                "rank": idx,
                "source": h["metadata"].get("source"),
                "chunk":  h["metadata"].get("chunk"),
                "rerank_score": round(h.get("rerank_score", 0.0), 4),
                "distance": round(h.get("distance") or 0.0, 4) if h.get("distance") is not None else "",
            })

    # 6b) Prompt + cevap
    with open("day4_answer.txt", "w", encoding="utf-8") as f:
        f.write("=== PROMPT ===\n")
        f.write(prompt)
        f.write("\n\n=== CEVAP ===\n")
        f.write(answer)
        f.write("\n\n=== Kaynaklar ===\n")
        f.write(cite_list)

    # 7) Konsol özeti
    print("\n=== Özet ===")
    for i, h in enumerate(hits, start=1):
        print(f"[{i}] score={h['rerank_score']:.4f} | {h['metadata'].get('source')} (chunk {h['metadata'].get('chunk')})")
    print("\n[OK] Rerank skorları: day4_rerank_scores.csv")
    print("[OK] Cevap + prompt:  day4_answer.txt\n")

    # Cevabın kısa ön izlemesi
    print("=== Cevap (kısaltılmış) ===\n")
    print(textwrap.shorten(answer, width=1000, placeholder=" ... [kısaltıldı] ..."))

if __name__ == "__main__":
    main()
