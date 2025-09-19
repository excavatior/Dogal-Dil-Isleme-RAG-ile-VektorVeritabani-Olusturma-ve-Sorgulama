#!/usr/bin/env python3
"""
Day 3 — Chroma'dan Sorgulama ve RAG Prompt Oluşturma

Kullanım:
  python day3_query_rag.py --query "embedding nedir?" --top_k 5
  python day3_query_rag.py --query "vector database nedir?" --top_k 5 --mmr
  # Filtre (yol içinde geçen bir kelimeye göre) - opsiyonel:
  python day3_query_rag.py --query "embedding nedir?" --filter_source_contains "ornek_tr.txt"

Notlar:
- Model ve DB ayarları Day 2 ile aynı olmalı.
- Çıktılar: terminalde en yakın parçalar + 'day3_rag_prompt.txt' dosyası.
"""

import argparse, os, textwrap
from typing import List, Dict, Any

import chromadb
from chromadb.utils import embedding_functions

DEFAULT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

def format_hits(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Chroma query sonucunu okunur listeye çevirir."""
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

def build_prompt(query: str, hits: List[Dict[str, Any]], max_chars: int = 2500) -> str:
    """Top-k parçaları birleştirip RAG prompt oluşturur."""
    blocks, used = [], 0
    for h in hits:
        src   = h["metadata"].get("source")
        chunk = h["metadata"].get("chunk")
        block = f"[Kaynak: {src}, Parça: {chunk}]\n{h['document']}"
        if used + len(block) > max_chars:
            break
        blocks.append(block)
        used += len(block)
    context = "\n\n---\n\n".join(blocks)
    prompt = f"""Aşağıdaki kullanıcı sorusunu, verilen bağlamdan yararlanarak ve bağlam dışına çıkmadan yanıtla.
Gerekirse maddeler kullan. Yanıtın sonunda kullandığın kaynakları [Kaynak: path, Parça: i] biçiminde referansla.

Soru:
{query}

Bağlam:
{context}

Yanıt talimatları:
- Bağlam dışına çıkma, uydurma yapma.
- Kısa ve net ol.
- Cevabın sonunda referansları listele.
"""
    return prompt

def mmr_select(query_embed, doc_embeds, k: int, lam: float = 0.5):
    """Basit MMR (çeşitlilik) seçimi. Vektörler normalize varsayılır."""
    import numpy as np
    d = np.array(doc_embeds)
    q = np.array(query_embed)
    sims = d @ q  # cosine ~ dot (normalize ise)

    selected, candidates = [], list(range(len(d)))
    while len(selected) < min(k, len(candidates)):
        if not selected:
            j = int(np.argmax(sims[candidates]))
            selected.append(candidates[j]); candidates.pop(j)
        else:
            scores = []
            for idx, c in enumerate(candidates):
                max_sim_to_sel = max(d[c] @ d[s] for s in selected)
                score = lam * sims[c] - (1 - lam) * max_sim_to_sel
                scores.append((score, idx))
            _, best_idx = max(scores, key=lambda x: x[0])
            selected.append(candidates[best_idx]); candidates.pop(best_idx)
    return selected

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db_dir", default="./chroma_db", help="Chroma veritabanı dizini")
    ap.add_argument("--collection", default="rag_docs", help="Koleksiyon adı (Day 2 ile aynı)")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Embedding modeli")
    ap.add_argument("--query", required=True, help="Sorgu metni")
    ap.add_argument("--top_k", type=int, default=5, help="Kaç sonuç getirilsin")
    ap.add_argument("--mmr", action="store_true", help="Çeşitlilik için MMR yeniden sıralama")
    ap.add_argument("--filter_source_contains", type=str, default=None,
                    help="Kaynak yolunda geçmesi gereken kelime (opsiyonel filtre)")
    ap.add_argument("--max_prompt_chars", type=int, default=2500, help="Prompta koyulacak bağlam toplam karakter sınırı")
    args = ap.parse_args()

    # 1) Chroma + embed function
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=args.model, normalize_embeddings=True
    )
    client = chromadb.PersistentClient(path=args.db_dir)
    col = client.get_or_create_collection(name=args.collection, embedding_function=embed_fn)

    # 2) (Opsiyonel) metadata filtre
    where = None
    if args.filter_source_contains:
        # 'source' alanı string olduğundan, contains filtresi için Chroma'nın 'where' söz dizimi:
        where = {"source": {"$contains": args.filter_source_contains}}

    # 3) Sorgu
    res = col.query(
        query_texts=[args.query],
        n_results=args.top_k * (2 if args.mmr else 1),
        include=["documents", "metadatas", "distances"],
        where=where,
    )
    hits = format_hits(res)

    # 4) MMR (opsiyonel)
    if args.mmr and hits:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer(args.model)
        q = m.encode([args.query], normalize_embeddings=True)[0].tolist()
        d = m.encode([h["document"] for h in hits], normalize_embeddings=True).tolist()
        selected_idx = mmr_select(q, d, k=args.top_k, lam=0.5)
        hits = [hits[i] for i in selected_idx]

    # 5) Konsol özeti
    print("\n=== En Yakın Sonuçlar ===")
    for h in hits[:args.top_k]:
        src = h["metadata"].get("source")
        chk = h["metadata"].get("chunk")
        dist = h["distance"]
        prev = h["document"][:220].replace("\n", " ")
        print(f"[{h['rank']}] dist={dist:.4f} | {src} (chunk {chk})")
        print(prev + ("..." if len(h["document"]) > 220 else ""))
        print("-" * 80)

    # 6) RAG promptu
    prompt = build_prompt(args.query, hits[:args.top_k], max_chars=args.max_prompt_chars)
    out_path = "day3_rag_prompt.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(prompt)
    print("\n[OK] RAG prompt kaydedildi ->", out_path)

    print("\n=== RAG Prompt (kısaltılmış görünüm) ===\n")
    print(textwrap.shorten(prompt, width=1200, placeholder=" ... [kısaltıldı] ..."))

if __name__ == "__main__":
    main()
