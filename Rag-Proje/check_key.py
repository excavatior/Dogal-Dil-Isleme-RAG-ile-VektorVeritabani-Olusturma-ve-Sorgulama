import os

api_key = os.getenv("OPENAI_API_KEY")

if api_key:
    print("✅ API Key bulundu:", api_key[:5] + "..." + api_key[-5:])
else:
    print("❌ API Key bulunamadı.")
