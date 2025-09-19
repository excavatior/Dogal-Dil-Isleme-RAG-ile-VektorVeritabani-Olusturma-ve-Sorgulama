# Day 4 — Chroma vektör veritabanı (yerel) hızlı başlangıç

Bu paket; ham metinleri (PDF/HTML/TXT/MD) okuyup, chunk'lara bölüp,
embedding üretip Chroma'ya **kalıcı** olarak kaydeder ve sorgulamanı sağlar.

## Hızlı Kurulum
```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows Git Bash
python -m pip install -U pip
python -m pip install -r requirements_day4.txt
cp .env.example .env
```

## Veri ekle
Ham dosyalarını `data/raw/` altına kopyala (PDF/HTML/TXT/MD).

## İndeks oluştur
```bash
python day4_chroma_pipeline.py build --model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 --chunk-size 500 --overlap 80
```

## Sorgu yap
```bash
python day4_chroma_pipeline.py query --q "vector database nedir?" --k 5
```

## Notlar
- Kalıcılık klasörü: `.chroma/` (env ile değiştirilebilir)
- Koleksiyon adı: `staj_docs`
- Metadata tutulan alanlar: `source`, `title`, `mime`, `chunk_id`, `doc_id`
