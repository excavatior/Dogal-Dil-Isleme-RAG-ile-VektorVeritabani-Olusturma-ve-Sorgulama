#!/usr/bin/env python3
# delete_from_chroma.py
# Delete all chunks ingested from a specific source file path in a Chroma collection.
# Usage (Windows Git Bash / PowerShell):
#   python delete_from_chroma.py --source_path "C:\path\to\wrong.txt"
# Optional:
#   --db_dir ./chroma_db  --collection rag_docs
#
# Notes:
# - 'source' metadata was stored as absolute path during ingestion.
# - Use the exact absolute path (Right click → Copy as path on Windows).

import os, argparse, sys
import chromadb

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db_dir", default="./chroma_db")
    ap.add_argument("--collection", default="rag_docs")
    ap.add_argument("--source_path", required=True, help="Absolute file path to delete (as stored in metadata 'source')")
    args = ap.parse_args()

    # Normalize path
    abs_src = os.path.abspath(args.source_path)

    client = chromadb.PersistentClient(path=args.db_dir)
    try:
        col = client.get_collection(name=args.collection)
    except Exception as e:
        print(f"[ERR] Collection '{args.collection}' not found in {args.db_dir}: {e}")
        sys.exit(1)

    try:
        before = col.count()
    except Exception:
        before = None

    # Delete by metadata filter
    try:
        col.delete(where={"source": abs_src})
        after = col.count()
        print(f"[OK] Deleted documents with source = {abs_src}")
        print(f"Count before: {before}, after: {after}")
    except Exception as e:
        print(f"[ERR] Delete failed: {e}")
        sys.exit(2)

if __name__ == "__main__":
    main()
