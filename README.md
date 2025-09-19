# RAG Yöntemi ile Vektör Veritabanı Oluşturma ve Sorgulama

Bu proje, **Retrieval-Augmented Generation (RAG)** mimarisi ile metin verilerini embedding’e dönüştürerek **Chroma vektör veritabanı** üzerinde saklamayı ve doğal dilde yapılan sorgulara en uygun cevapları üretmeyi amaçlar.  
Ayrıca proje, kullanıcı dostu bir **Masaüstü Uygulaması** olarak derlenmiştir.

---

## 🚀 Projenin Amacı
- Metin verilerini embedding’e dönüştürmek.  
- Embedding’leri **Chroma** vektör veritabanına aktarmak.  
- Doğal dilde gelen sorgulara en yakın sonuçları getirmek.  
- RAG yaklaşımı ile **LLM + Vektör DB** entegrasyonunu göstermek.  
- Son olarak, teknik süreci kolaylaştırmak için masaüstü uygulaması geliştirmek.  

---

## 🏗 Kullanılan Teknolojiler
- **Python 3.11+**
- **Sentence-Transformers** → metinleri embedding’e dönüştürmek için  
- **ChromaDB** → vektör veritabanı  
- **OpenAI API** → dil modeli tabanlı yanıtlar  
- **Tkinter (GUI)** → masaüstü uygulama arayüzü  
- **PyInstaller** → uygulamayı `.exe` formatına dönüştürmek için  
- **Git & GitHub** → sürüm kontrol ve proje paylaşımı  

---

## 📂 Proje Yapısı
```
rag-proje/
│── data/                 # Örnek veri dosyaları
│── chroma_db/            # Vektör veritabanı
│── rag_desktop_app.py    # Masaüstü uygulaması
│── config_manager.py     # API Key yönetimi
│── key_widget_tk.py      # Key girişi arayüzü
│── ingest_chroma.py      # Ingestion scripti
│── query_rag.py          # Query scripti
│── requirements.txt      # Bağımlılıklar
│── dist/                 # Derlenmiş .exe dosyaları
│── build/                # PyInstaller build geçici dosyaları
│── .venv/                # Sanal ortam
└── README.md
```

---

## ⚙️ Kurulum ve Çalıştırma

### 1. Depoyu Klonla
```bash
git clone https://github.com/<kullanıcı_adı>/<repo_adı>.git
cd rag-proje
```

### 2. Sanal Ortam Kur
```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows (Git Bash)
# veya
.venv\Scripts\activate          # Windows (CMD / PowerShell)
# veya
source .venv/bin/activate       # Linux/Mac
```

### 3. Bağımlılıkları Yükle
```bash
pip install -r requirements.txt
```

### 4. Uygulamayı Çalıştır
```bash
python rag_desktop_app.py
```

### 5. API Key Ayarı
- **Settings** sekmesinden `OpenAI` seç.  
- API key gir → **Kaydet** ve **Test Et**.  

---

## 🖼 Uygulama İçi Görseller

 

---

## 📜 Lisans
Bu proje MIT Lisansı altında sunulmuştur.  
