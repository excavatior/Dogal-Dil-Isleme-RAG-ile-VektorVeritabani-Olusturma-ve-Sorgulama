#!/usr/bin/env python3
# test_openai_key.py — OPENAI_API_KEY var mı, kısa doğrulama
import os
key = os.environ.get("OPENAI_API_KEY")
if not key:
    print("OPENAI_API_KEY bulunamadı. PowerShell: setx OPENAI_API_KEY \"...\" | Git Bash: export OPENAI_API_KEY=\"...\"")
else:
    print("OPENAI_API_KEY OK. Başlangıç:", key[:7] + "..." if len(key) >= 7 else "(kısa)")