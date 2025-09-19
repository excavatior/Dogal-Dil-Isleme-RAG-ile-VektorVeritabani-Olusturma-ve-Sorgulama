#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rag_desktop_app.py
Tam özellikli bir RAG masaüstü uygulaması (Tkinter).
- Ingest: TXT/MD dosyalarını parçalayıp (chunk) Chroma DB'ye ekler
- Query/RAG: Sorgu çalıştırır, OpenAI (veya yerel HF) ile kaynaklı cevap üretir
- Settings: API key ve provider yönetimi (kalıcı olarak ~/.rag_desktop_app/config.json'a kaydedilir)

Gereksinimler (örnek):
pip install -U sentence-transformers chromadb openai transformers accelerate torch torchvision
"""
import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from datetime import datetime

# Yerel modüller
try:
    from config_manager import load_config, get_provider
    from key_widget_tk import KeySettings
except Exception as e:
    print("[WARN] config_manager/key_widget_tk bulunamadı:", e, file=sys.stderr)

# NLP/DB
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer

# OpenAI
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# HuggingFace
try:
    from transformers import pipeline
except Exception:
    pipeline = None


def safe_int(s, default=5):
    try:
        return int(s)
    except Exception:
        return default


def now_str():
    return datetime.now().strftime("%H:%M:%S")


class RagDesktopApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("RAG Desktop App")
        self.geometry("1000x720")

        # Config yükle
        try:
            self.cfg = load_config()
        except Exception:
            self.cfg = {"provider": "openai", "openai_api_key": ""}

        # Oturuma API KEY yaz (varsa)
        if self.cfg.get("openai_api_key"):
            os.environ["OPENAI_API_KEY"] = self.cfg["openai_api_key"]

        self.provider = self.cfg.get("provider", "openai")

        # Varsayılan yollar/parametreler
        self.db_dir = tk.StringVar(value=os.path.abspath("./chroma_db"))
        self.collection_name = tk.StringVar(value="rag_docs")
        self.embed_model_name = tk.StringVar(value="paraphrase-multilingual-MiniLM-L12-v2")
        self.chunk_size = tk.StringVar(value="800")
        self.chunk_overlap = tk.StringVar(value="100")
        self.top_k = tk.StringVar(value="5")

        # UI
        self._build_ui()

        # Embedding modeli (lazy init)
        self._embedder = None

    # --------------------------- UI kur ---------------------------
    def _build_ui(self):
        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True)

        # Tabs
        self.tab_ingest = ttk.Frame(notebook)
        self.tab_query = ttk.Frame(notebook)
        self.tab_settings = ttk.Frame(notebook)

        notebook.add(self.tab_ingest, text="Ingest")
        notebook.add(self.tab_query, text="Query / RAG")
        notebook.add(self.tab_settings, text="Settings")

        # Ingest tab
        self._build_ingest_tab()
        # Query tab
        self._build_query_tab()
        # Settings tab
        self._build_settings_tab()

    # --------------------------- Ingest Tab ---------------------------
    def _build_ingest_tab(self):
        left = ttk.Frame(self.tab_ingest)
        left.pack(side="left", fill="both", expand=True, padx=8, pady=8)

        right = ttk.Frame(self.tab_ingest)
        right.pack(side="right", fill="both", expand=True, padx=8, pady=8)

        # Dosya listesi
        ttk.Label(left, text="Dosyalar (.txt / .md):", font=("", 10, "bold")).pack(anchor="w")
        self.list_files = tk.Listbox(left, height=14, selectmode="extended")
        self.list_files.pack(fill="both", expand=True)
        btns = ttk.Frame(left)
        btns.pack(fill="x", pady=6)
        ttk.Button(btns, text="Dosya Ekle", command=self._pick_files).pack(side="left", padx=3)
        ttk.Button(btns, text="Seçiliyi Kaldır", command=self._remove_selected).pack(side="left", padx=3)
        ttk.Button(btns, text="Listeyi Temizle", command=self._clear_files).pack(side="left", padx=3)

        # Parametreler
        params = ttk.LabelFrame(right, text="Parametreler")
        params.pack(fill="x")
        row = 0
        ttk.Label(params, text="DB Klasörü:").grid(row=row, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(params, textvariable=self.db_dir, width=50).grid(row=row, column=1, sticky="we", padx=6, pady=4)
        ttk.Button(params, text="Seç", command=self._choose_db_dir).grid(row=row, column=2, padx=6)
        row += 1
        ttk.Label(params, text="Koleksiyon:").grid(row=row, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(params, textvariable=self.collection_name).grid(row=row, column=1, sticky="we", padx=6, pady=4)
        row += 1
        ttk.Label(params, text="Embedding Modeli:").grid(row=row, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(params, textvariable=self.embed_model_name).grid(row=row, column=1, sticky="we", padx=6, pady=4)
        row += 1
        ttk.Label(params, text="Chunk Size:").grid(row=row, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(params, textvariable=self.chunk_size, width=10).grid(row=row, column=1, sticky="w", padx=6, pady=4)
        row += 1
        ttk.Label(params, text="Overlap:").grid(row=row, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(params, textvariable=self.chunk_overlap, width=10).grid(row=row, column=1, sticky="w", padx=6, pady=4)
        row += 1

        ttk.Button(right, text="Ingest ➜ Chroma", command=self._ingest).pack(fill="x", pady=8)

        # Log
        ttk.Label(right, text="Log:", font=("", 10, "bold")).pack(anchor="w")
        self.log_ingest = tk.Text(right, height=22, wrap="word")
        self.log_ingest.pack(fill="both", expand=True)

    def _pick_files(self):
        paths = filedialog.askopenfilenames(filetypes=[("Text/Markdown", "*.txt *.md")])
        for p in paths:
            if p and p not in self.list_files.get(0, "end"):
                self.list_files.insert("end", p)

    def _remove_selected(self):
        sel = list(self.list_files.curselection())
        sel.reverse()
        for i in sel:
            self.list_files.delete(i)

    def _clear_files(self):
        self.list_files.delete(0, "end")

    def _choose_db_dir(self):
        d = filedialog.askdirectory()
        if d:
            self.db_dir.set(d)

    def _log_i(self, msg):
        self.log_ingest.insert("end", f"[{now_str()}] {msg}\n")
        self.log_ingest.see("end")

    def _ensure_embedder(self):
        if self._embedder is None:
            model_name = self.embed_model_name.get().strip()
            self._log_i(f"Embedding modeli yükleniyor: {model_name}")
            self._embedder = SentenceTransformer(model_name)
        return self._embedder

    def _split_text(self, text, chunk_size=800, overlap=100):
        cs = max(1, chunk_size)
        ov = max(0, min(overlap, cs - 1))
        chunks = []
        i = 0
        n = len(text)
        while i < n:
            end = min(i + cs, n)
            chunk = text[i:end]
            if len(chunk.strip()) >= 50:  # çok kısa parçaları atla
                chunks.append(chunk)
            i = i + cs - ov
        return chunks

    def _ingest(self):
        files = list(self.list_files.get(0, "end"))
        if not files:
            messagebox.showwarning("Uyarı", "Lütfen en az bir .txt/.md dosyası ekleyin.")
            return

        db_dir = self.db_dir.get().strip()
        collection = self.collection_name.get().strip()
        cs = safe_int(self.chunk_size.get(), 800)
        ov = safe_int(self.chunk_overlap.get(), 100)

        os.makedirs(db_dir, exist_ok=True)
        client = chromadb.PersistentClient(path=db_dir)

        # Chroma embedding function
        _ = self._ensure_embedder()
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=self.embed_model_name.get().strip())

        # Koleksiyon
        try:
            col = client.get_or_create_collection(name=collection, embedding_function=ef)
        except Exception:
            try:
                col = client.get_collection(name=collection)
            except Exception:
                col = client.create_collection(name=collection, embedding_function=ef)

        total = 0
        for path in files:
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
            except Exception as e:
                self._log_i(f"[ERR] Dosya okunamadı: {path} → {e}")
                continue

            chunks = self._split_text(text, chunk_size=cs, overlap=ov)
            self._log_i(f"[DOSYA] {os.path.basename(path)} için {len(chunks)} parça bulundu.")
            if not chunks:
                continue

            ids = []
            docs = []
            metas = []
            for idx, ch in enumerate(chunks):
                ids.append(f"{os.path.abspath(path)}::{idx}")
                docs.append(ch)
                metas.append({"source": os.path.abspath(path), "chunk": idx})

            try:
                col.add(documents=docs, metadatas=metas, ids=ids)
                total += len(docs)
                self._log_i(f"[OK] Eklendi: {len(docs)}  (Toplam: {total})")
            except Exception as e:
                self._log_i(f"[ERR] Ekleme hatası: {e}")

        self._log_i("[DONE] Ingestion tamamlandı.")

    # --------------------------- Query Tab ---------------------------
    def _build_query_tab(self):
        top = ttk.Frame(self.tab_query)
        top.pack(fill="x", padx=8, pady=8)

        ttk.Label(top, text="Sorgu:", font=("", 10, "bold")).pack(side="left")
        self.var_query = tk.StringVar()
        ttk.Entry(top, textvariable=self.var_query, width=70).pack(side="left", padx=6)
        ttk.Label(top, text="top_k:").pack(side="left", padx=(10, 2))
        ttk.Entry(top, textvariable=self.top_k, width=5).pack(side="left")

        ttk.Button(top, text="Ara (Retrieval)", command=self._do_retrieval).pack(side="left", padx=6)
        ttk.Button(top, text="Cevap Üret (RAG)", command=self._do_rag).pack(side="left", padx=6)

        # Sonuç alanları
        mid = ttk.Panedwindow(self.tab_query, orient="horizontal")
        mid.pack(fill="both", expand=True, padx=8, pady=8)

        # Retrieval sonuçları
        lf_left = ttk.Labelframe(mid, text="Retrieval Sonuçları")
        self.txt_retr = tk.Text(lf_left, wrap="word")
        self.txt_retr.pack(fill="both", expand=True)
        mid.add(lf_left, weight=1)

        # Nihai cevap
        lf_right = ttk.Labelframe(mid, text="RAG Cevabı")
        self.txt_ans = tk.Text(lf_right, wrap="word")
        self.txt_ans.pack(fill="both", expand=True)
        mid.add(lf_right, weight=1)

    def _get_collection(self):
        db_dir = self.db_dir.get().strip()
        collection = self.collection_name.get().strip()
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=self.embed_model_name.get().strip())
        client = chromadb.PersistentClient(path=db_dir)
        try:
            col = client.get_collection(name=collection, embedding_function=ef)
        except Exception:
            col = client.create_collection(name=collection, embedding_function=ef)
        return col

    def _do_retrieval(self):
        q = self.var_query.get().strip()
        if not q:
            messagebox.showwarning("Uyarı", "Lütfen bir sorgu yazın.")
            return
        n = safe_int(self.top_k.get(), 5)

        col = self._get_collection()
        try:
            res = col.query(query_texts=[q], n_results=n)
        except Exception as e:
            messagebox.showerror("Hata", f"Sorgu başarısız: {e}")
            return

        self.txt_retr.delete("1.0", "end")
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        for i, d in enumerate(docs):
            meta = metas[i] if i < len(metas) else {}
            src = meta.get("source", "?")
            ch = meta.get("chunk", "?")
            self.txt_retr.insert("end", f"[{i+1}] Kaynak: {src}  (Parça: {ch})\n{d}\n\n")

    def _do_rag(self):
        # Retrieval sonuçlarını çek
        q = self.var_query.get().strip()
        if not q:
            messagebox.showwarning("Uyarı", "Lütfen bir sorgu yazın.")
            return
        n = safe_int(self.top_k.get(), 5)
        col = self._get_collection()
        res = col.query(query_texts=[q], n_results=n)
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]

        context_blocks = []
        sources = []
        for i, d in enumerate(docs):
            meta = metas[i] if i < len(metas) else {}
            src = meta.get("source", "?")
            ch = meta.get("chunk", "?")
            context_blocks.append(d)
            sources.append((src, ch))

        if not context_blocks:
            messagebox.showwarning("Uyarı", "Retrieval sonucu yok.")
            return

        context = "\n\n---\n\n".join(context_blocks)
        answer = ""

        provider = self.cfg.get("provider", "openai")
        if provider == "openai":
            key = os.getenv("OPENAI_API_KEY") or self.cfg.get("openai_api_key", "")
            if not key or OpenAI is None:
                messagebox.showerror("Hata", "OpenAI için API key gerekli veya openai paketi yüklü değil.")
                return
            try:
                client = OpenAI(api_key=key)
                sys_prompt = (
                    "Aşağıdaki bağlamdan faydalanarak Türkçe ve kaynak belirterek kısa, doğru bir cevap üret. "
                    "Bağlam dışında uydurma yapma. En sonda 'Kaynaklar' başlığı ile dosya ve parça numaralarını listele."
                )
                user_prompt = f"Soru: {q}\n\nBağlam:\n{context}"
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.2,
                )
                answer = resp.choices[0].message.content
            except Exception as e:
                messagebox.showerror("Hata", f"OpenAI hatası: {e}")
                return
        else:
            # HF yerel (ör: flan-t5-base)
            if pipeline is None:
                messagebox.showerror("Hata", "transformers paketi yüklü değil (HF).")
                return
            try:
                prompt = (
                    "Aşağıdaki bağlamdan kısa ve doğru bir yanıt üret. Uydurma yapma.\n\n"
                    f"Soru: {q}\n\nBağlam:\n{context}\n\nCevap:"
                )
                gen = pipeline("text2text-generation", model="google/flan-t5-base", max_new_tokens=256)
                out = gen(prompt)[0]["generated_text"]
                answer = out
            except Exception as e:
                messagebox.showerror("Hata", f"HuggingFace hatası: {e}")
                return

        # Kaynakları ekle
        answer = (answer or "").strip() + "\n\nKaynaklar:\n"
        for i, (src, ch) in enumerate(sources, 1):
            answer += f"[{i}] {src} (Parça: {ch})\n"

        self.txt_ans.delete("1.0", "end")
        self.txt_ans.insert("end", answer)

    # --------------------------- Settings Tab ---------------------------
    def _build_settings_tab(self):
        panel = KeySettings(self.tab_settings, on_change=self._on_settings_changed)
        panel.pack(fill="both", expand=True, padx=10, pady=10)

    def _on_settings_changed(self, provider=None, key=None):
        # config_manager zaten kalıcı yazıyor; burada sadece runtime değişiklikleri yakalayalım
        self.cfg["provider"] = provider or get_provider()
        if key:
            self.cfg["openai_api_key"] = key
        try:
            self.focus_force()
        except Exception:
            pass


if __name__ == "__main__":
    app = RagDesktopApp()
    app.mainloop()
