#!/usr/bin/env python3
# day5_evaluate.py — Batch evaluation: soruları çalıştır, CSV üret
import os, time, argparse, csv
from typing import List, Dict, Any, Tuple
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import CrossEncoder

DEFAULT_EMBED = "paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_RERANK = "cross-encoder/ms-marco-MiniLM-L-6-v2"

def load_questions(path: str) -> List[str]:
    if not os.path.exists(path):
        return [
            "embedding nedir?",
            "RAG yaklaşımı neyi amaçlar?",
            "Chroma nedir ve ne işe yarar?",
            "What is an embedding?",
            "How does RAG reduce hallucinations?",
            "What is a vector database?",
        ]
    with open(path,"r",encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]

def format_hits(results: Dict[str, Any]):
    hits = []
    ids = results.get("ids", [[]])[0]
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0] if results.get("distances") else [None] * len(ids)
    for i in range(len(ids)):
        hits.append({"rank":i+1,"id":ids[i],"distance":dists[i],"document":docs[i],"metadata":metas[i]})
    return hits

def rerank(query: str, hits, reranker_name: str, top_k: int):
    ce = CrossEncoder(reranker_name)
    pairs = [(query, h["document"]) for h in hits]
    scores = ce.predict(pairs)
    for h,s in zip(hits, scores):
        h["rerank_score"] = float(s)
    hits.sort(key=lambda x: x["rerank_score"], reverse=True)
    return hits[:top_k]

def build_prompt(query: str, hits, max_chars: int = 2500):
    blocks, used, cites = [], 0, []
    for h in hits:
        src = h["metadata"].get("source"); chk = h["metadata"].get("chunk")
        cite = f"[Kaynak: {src}, Parça: {chk}]"
        block = f"{cite}\n{h['document']}"
        if used + len(block) > max_chars: continue
        blocks.append(block); cites.append(cite); used += len(block)
    context = "\n\n---\n\n".join(blocks)
    prompt = f"""Aşağıdaki soruyu, verilen bağlamdan yararlanarak ve bağlam dışına çıkmadan yanıtla.
Yanıtın sonunda kullandığın kaynakları [Kaynak: path, Parça: i] biçiminde listele. Türkçe yanıt ver.

Soru:
{query}

Bağlam:
{context}

Yanıt talimatları:
- Sadece bağlamı kullan, uydurma yapma.
- Kısa ve net yaz; gerekiyorsa maddeler kullan.
- Sonunda "Kaynaklar:" başlığıyla referansları yaz.
"""
    return prompt, "; ".join(cites)

def answer_openai(prompt: str, model: str = "gpt-4o-mini") -> str:
    from openai import OpenAI
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY yok. PowerShell: setx OPENAI_API_KEY \"...\" | Git Bash: export OPENAI_API_KEY=\"...\"")
    client = OpenAI(api_key=key)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role":"system","content":"Kaynaklı yanıt üret."},
                  {"role":"user","content":prompt}],
        temperature=0.2, max_tokens=600,
    )
    return resp.choices[0].message.content

def answer_hf(prompt: str, model_name: str = "google/flan-t5-base") -> str:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    import torch
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(
        model_name, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mdl.to(device)
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
    out = mdl.generate(**inputs, max_new_tokens=500)
    return tok.decode(out[0], skip_special_tokens=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db_dir", default="./chroma_db")
    ap.add_argument("--collection", default="rag_docs")
    ap.add_argument("--embed_model", default=DEFAULT_EMBED)
    ap.add_argument("--reranker", default=DEFAULT_RERANK)
    ap.add_argument("--provider", choices=["openai","hf"], default="hf")
    ap.add_argument("--openai_model", default="gpt-4o-mini")
    ap.add_argument("--hf_model", default="google/flan-t5-base")
    ap.add_argument("--questions", default="eval_questions.txt")
    ap.add_argument("--candidates", type=int, default=12)
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--max_prompt_chars", type=int, default=2500)
    args = ap.parse_args()

    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=args.embed_model, normalize_embeddings=True
    )
    client = chromadb.PersistentClient(path=args.db_dir)
    col = client.get_or_create_collection(name=args.collection, embedding_function=embed_fn)

    questions = load_questions(args.questions)
    rows = []

    for q in questions:
        t0 = time.time()
        res = col.query(
            query_texts=[q],
            n_results=max(args.candidates, args.top_k),
            include=["documents","metadatas","distances"],
        )
        hits = format_hits(res)
        if not hits:
            rows.append({"question": q, "answer": "(no hits)", "latency_ms": 0, "sources": ""})
            continue

        hits = rerank(q, hits, args.reranker, args.top_k)
        prompt, cites = build_prompt(q, hits, args.max_prompt_chars)

        if args.provider == "openai":
            ans = answer_openai(prompt, args.openai_model)
        else:
            ans = answer_hf(prompt, args.hf_model)

        latency_ms = int((time.time() - t0) * 1000)
        rows.append({"question": q, "answer": ans, "latency_ms": latency_ms, "sources": cites})
        print(f"[OK] {q} ({latency_ms} ms)")

    out_csv = "evaluation_report.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as fw:
        writer = csv.DictWriter(fw, fieldnames=["question","answer","latency_ms","sources"])
        writer.writeheader()
        writer.writerows(rows)

    print("\n[OK] Değerlendirme kaydedildi ->", out_csv)

if __name__ == "__main__":
    main()
