# RAG Projesi — Final Çalıştırma Rehberi

## 0) Sanal ortamı aç
Git Bash:
```
source .venv/Scripts/activate
```

## 1) Gerekli paketleri kur
```
pip install -U sentence-transformers chromadb numpy pandas transformers accelerate python-docx openai
```

## 2) OpenAI key (opsiyonel — OpenAI kullanacaksan)
- Dün PowerShell ile `setx OPENAI_API_KEY "..."` yaptıysan **yeni terminal açınca** otomatik gelir.
- Git Bash'te `export OPENAI_API_KEY="..."` **sadece o oturumda** geçerlidir; yeni oturumda tekrar yazmalısın.
- Kontrol:
```
# Git Bash
echo $OPENAI_API_KEY
# CMD
echo %OPENAI_API_KEY%
```

## 3) Değerlendirme (CSV üretir)
OpenAI ile:
```
python day5_evaluate.py --provider openai --questions eval_questions.txt --top_k 5 --candidates 12
```
HuggingFace yerel model ile:
```
python day5_evaluate.py --provider hf --hf_model google/flan-t5-base --questions eval_questions.txt --top_k 5 --candidates 12
```

Çıktı: `evaluation_report.csv`

## 4) Word raporu üret
```
python day5_make_report.py
```
Çıktı: `final_RAG_report.docx`

## 5) (Opsiyonel) Tek soru yanıtı
```
python day4_rag_answer.py --query "embedding nedir?" --provider openai --top_k 5
# veya
python day4_rag_answer.py --query "embedding nedir?" --provider hf --hf_model google/flan-t5-base --top_k 5
```

## 6) GitHub push
`.gitignore` içeriğini repo köküne kopyala; sonra:
```
git add .
git commit -m "Final: RAG (Chroma) + Reranker + LLM + Eval + Report"
git push -u origin main
```
