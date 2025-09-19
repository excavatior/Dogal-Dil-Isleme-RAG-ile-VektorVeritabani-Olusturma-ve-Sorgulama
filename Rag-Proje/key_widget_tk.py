import tkinter as tk
from tkinter import ttk, messagebox
import os

from config_manager import get_api_key, set_api_key, get_provider, set_provider

class KeySettings(ttk.Frame):
    """A simple Settings panel for provider & OpenAI API key."""
    def __init__(self, parent, on_change=None):
        super().__init__(parent)
        self.on_change = on_change

        self.var_provider = tk.StringVar(value=get_provider())
        self.var_key = tk.StringVar(value=get_api_key())

        row = 0
        ttk.Label(self, text="Sağlayıcı (Provider):", font=("", 10, "bold")).grid(row=row, column=0, sticky="w", padx=6, pady=6)
        row += 1
        ttk.Radiobutton(self, text="OpenAI", variable=self.var_provider, value="openai").grid(row=row, column=0, sticky="w", padx=6)
        ttk.Radiobutton(self, text="HuggingFace (yerel)", variable=self.var_provider, value="hf").grid(row=row, column=1, sticky="w", padx=6)
        row += 1

        ttk.Label(self, text="OpenAI API Key:", font=("", 10, "bold")).grid(row=row, column=0, sticky="w", padx=6, pady=(12, 4))
        row += 1
        self.entry = ttk.Entry(self, textvariable=self.var_key, width=48, show="*")
        self.entry.grid(row=row, column=0, columnspan=2, sticky="we", padx=6)
        row += 1

        btns = ttk.Frame(self)
        btns.grid(row=row, column=0, columnspan=2, sticky="w", padx=6, pady=10)
        ttk.Button(btns, text="Kaydet", command=self.save).pack(side="left", padx=4)
        ttk.Button(btns, text="Test Et", command=self.test_key).pack(side="left", padx=4)
        ttk.Button(btns, text="Göster/Gizle", command=self._toggle_visible).pack(side="left", padx=4)

        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)

    def _toggle_visible(self):
        # toggle password visibility
        self.entry.configure(show="" if self.entry.cget("show") == "*" else "*")

    def save(self):
        set_provider(self.var_provider.get())
        set_api_key(self.var_key.get().strip())
        # also set process env so current session sees it
        if self.var_key.get().strip():
            os.environ["OPENAI_API_KEY"] = self.var_key.get().strip()
        messagebox.showinfo("Kaydedildi", "Ayarlar kaydedildi.")
        if self.on_change:
            try:
                self.on_change(self.var_provider.get(), self.var_key.get().strip())
            except TypeError:
                self.on_change()

    def test_key(self):
        provider = self.var_provider.get()
        if provider == "openai":
            key = self.var_key.get().strip()
            if not key:
                messagebox.showwarning("Eksik", "Lütfen OpenAI API key giriniz.")
                return
            try:
                from openai import OpenAI
                client = OpenAI(api_key=key)
                # lightweight call to validate credentials
                _ = client.models.list()
                messagebox.showinfo("Başarılı", "OpenAI API anahtarı çalışıyor ✔")
            except Exception as e:
                messagebox.showerror("Hata", f"OpenAI doğrulama başarısız:\n{e}")
        else:
            messagebox.showinfo("Bilgi", "HuggingFace yerelde çalışır, API anahtarı gerekmeyebilir.")
