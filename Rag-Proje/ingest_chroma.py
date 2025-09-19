#!/usr/bin/env python3
"""
Day 2 — Veri Hazırlama ve Chroma'ya Aktarım (Ingestion)

Ne yapar?
- data/ içindeki .txt ve .md dosyalarını okur
- metinleri temizler, parçalara böler (chunk)
- her parçayı embedding'e çevirir ve Chroma'ya ekler
- mapping CSV dosyasına (id, kaynak, chunk, karakter aralığı) kaydeder
- istersen hızlı bir doğrulama sorgusu çalıştırır (--verify)

Kurulum (venv içindeyken):
  pip install -U sentence-transformers chromadb numpy pandas

Kullanım (önerilen):
  python day2_ingest_chroma.py --data_dir ./data --db_dir ./chroma_db --collection rag_docs \
    --model paraphrase-multilingual-MiniLM-L12-v2 --chunk_size 800 --chunk_overlap 100 --reset --verify "embedding nedir?"
"""
import os, re, glob, csv, argparse, hashlib
from typing import List, Tuple

import chromadb
from chromadb.utils import embedding_functions

DEFAULT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
VALID_EXTS = (".txt", ".md")


def clean_text(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s).strip()
    return s


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[Tuple[str, int, int]]:
    """(chunk_text, start_idx, end_idx) listesi döndürür."""
    if chunk_size <= 0:
        return [(text, 0, len(text))]
    chunks = []
    n = len(text)
    start = 0
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append((text[start:end], start, end))
        if end == n:
            break
        start = max(0, end - chunk_overlap)
    return chunks


def list_files(data_dir: str) -> List[str]:
    files = []
    for ext in VALID_EXTS:
        files.extend(glob.glob(os.path.join(data_dir, f"**/*{ext}"), recursive=True))
    return sorted(files)


def stable_id(path: str, chunk_idx: int) -> str:
    key = f"{os.path.abspath(path)}::{chunk_idx}"
    return hashlib.md5(key.encode("utf-8")).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="Metin dosyalarının olduğu klasör (örn. ./data)")
    ap.add_argument("--db_dir", default="./chroma_db", help="Chroma veritabanı dizini")
    ap.add_argument("--collection", default="rag_docs", help="Koleksiyon adı")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="SentenceTransformer model adı")
    ap.add_argument("--chunk_size", type=int, default=800, help="Karakter bazlı parça boyutu")
    ap.add_argument("--chunk_overlap", type=int, default=100, help="Parçalar arası örtüşme")
    ap.add_argument("--min_chars", type=int, default=50, help="Çok kısa parçaları at (karakter)")
    ap.add_argument("--reset", action="store_true", help="Koleksiyonu sıfırla (varsa silip yeniden oluştur)")
    ap.add_argument("--verify", type=str, default=None, help="Ingestion sonrası hızlı doğrulama sorgusu")
    args = ap.parse_args()

    print(f"[INFO] Model           : {args.model}")
    print(f"[INFO] Data dir        : {args.data_dir}")
    print(f"[INFO] DB dir          : {args.db_dir}")
    print(f"[INFO] Koleksiyon      : {args.collection}")
    print(f"[INFO] Chunk/Overlap   : {args.chunk_size}/{args.chunk_overlap}  (min_chars={args.min_chars})")

    files = list_files(args.data_dir)
    if not files:
        print("[WARN] Veri bulunamadı. data/ içine .txt veya .md ekleyin.")
        return

    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=args.model, normalize_embeddings=True
    )
    client = chromadb.PersistentClient(path=args.db_dir)

    if args.reset:
        try:
            client.delete_collection(args.collection)
            print(f"[INFO] Koleksiyon silindi: {args.collection}")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=args.collection,
        embedding_function=embed_fn,
        metadata={"description": "RAG belgeleri (Day 2 ingestion)"},
    )

    # ingestion
    map_rows = []
    batch_ids, batch_docs, batch_metas = [], [], []
    BATCH = 128
    total = 0

    for path in files:
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                raw = f.read()
        except Exception as e:
            print(f"[WARN] Okunamadı: {path} ({e})")
            continue

        text = clean_text(raw)
        chs = chunk_text(text, args.chunk_size, args.chunk_overlap)

        for i, (ch, s0, s1) in enumerate(chs):
            if len(ch.strip()) < args.min_chars:
                continue
            _id = stable_id(path, i)
            meta = {"source": os.path.abspath(path), "chunk": i, "start": s0, "end": s1}
            batch_ids.append(_id)
            batch_docs.append(ch)
            batch_metas.append(meta)

            map_rows.append({
                "id": _id, "source": os.path.abspath(path),
                "chunk": i, "start": s0, "end": s1, "chars": len(ch),
            })

            if len(batch_ids) >= BATCH:
                collection.add(ids=batch_ids, documents=batch_docs, metadatas=batch_metas)
                total += len(batch_ids)
                print(f"[INFO] Eklendi (toplam): {total}")
                batch_ids, batch_docs, batch_metas = [], [], []

    if batch_ids:
        collection.add(ids=batch_ids, documents=batch_docs, metadatas=batch_metas)
        total += len(batch_ids)

    # sayım
    try:
        count = collection.count()
        print(f"[DONE] Toplam parça eklendi: {total} | Koleksiyon sayısı: {count}")
    except Exception:
        print(f"[DONE] Toplam parça eklendi: {total}")

    # mapping CSV
    os.makedirs(args.db_dir, exist_ok=True)
    out_map = os.path.join(args.db_dir, f"{args.collection}__ingestion_map.csv")
    with open(out_map, "w", newline="", encoding="utf-8") as fw:
        writer = csv.DictWriter(fw, fieldnames=["id", "source", "chunk", "start", "end", "chars"])
        writer.writeheader()
        writer.writerows(map_rows)
    print(f"[DONE] Ingestion haritası yazıldı: {out_map}")

    # hızlı doğrulama
    if args.verify:
        res = collection.query(
            query_texts=[args.verify],
            n_results=5,
            include=["documents", "metadatas", "distances"],
        )
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        print("\n=== Doğrulama Sonuçları ===")
        for idx, (doc, meta, dist) in enumerate(zip(docs, metas, dists), start=1):
            src = meta.get("source"); chk = meta.get("chunk")
            preview = doc[:240].replace("\n", " ")
            print(f"[{idx}] dist={dist:.4f} | {src} (chunk {chk})")
            print(preview + ("..." if len(doc) > 240 else ""))
            print("-" * 80)


if __name__ == "__main__":
    main()
