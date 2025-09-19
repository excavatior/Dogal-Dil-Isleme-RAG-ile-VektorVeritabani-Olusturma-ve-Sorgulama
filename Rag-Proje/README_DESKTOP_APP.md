# Chroma Tabanlı RAG — Masaüstü Uygulama

Bu uygulama, `.txt` ve `.md` dosyalarını **Chroma** vektör veritabanına ingest edip, **RAG** (retrieval-augmented generation) ile **kaynaklı yanıt** üretmenizi sağlar.

## Kurulum
```bash
python -m venv .venv
# Windows (Git Bash):
source .venv/Scripts/activate
pip install -r requirements_desktop.txt
```

> CUDA'lı PyTorch gerekiyorsa, PyTorch'un resmi komutuna göre kurup sonra `pip install -r requirements_desktop.txt` çalıştırın.

## Çalıştırma
```bash
python rag_desktop_app.py
```

## Özellikler
- Sürükle-bırak veya "Dosya Ekle" ile `.txt/.md` seç
- `chunk_size` ve `chunk_overlap` ayarı
- Retrieval (top_k) ve **Reranker**
- **OpenAI** (gpt-4o-mini) veya **HuggingFace** (flan-t5-base) ile yanıt
- Kaynakları yanıtın sonunda listeler

## OpenAI Anahtarı
- PowerShell kalıcı: `setx OPENAI_API_KEY "..."` (yeni terminal aç)
- Git Bash: `export OPENAI_API_KEY="..."` (oturuma özel)

## Notlar
- Prototiptir; büyük veri setlerinde işlem süresi uzayabilir.
- Veritabanı dizini: `./chroma_db`, koleksiyon: `rag_docs`.
