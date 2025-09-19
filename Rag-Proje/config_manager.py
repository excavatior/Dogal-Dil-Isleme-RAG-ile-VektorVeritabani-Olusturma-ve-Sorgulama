import os, json

CONFIG_DIR = os.path.join(os.path.expanduser("~"), ".rag_desktop_app")
CONFIG_PATH = os.path.join(CONFIG_DIR, "config.json")
DEFAULTS = {"provider": "openai", "openai_api_key": ""}

def _ensure_dir():
    os.makedirs(CONFIG_DIR, exist_ok=True)

def load_config():
    _ensure_dir()
    if not os.path.exists(CONFIG_PATH):
        save_config(DEFAULTS.copy())
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        # fill any missing defaults
        for k, v in DEFAULTS.items():
            data.setdefault(k, v)
        return data
    except Exception:
        return DEFAULTS.copy()

def save_config(data: dict):
    _ensure_dir()
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def get_api_key() -> str:
    return load_config().get("openai_api_key", "")

def set_api_key(key: str):
    data = load_config()
    data["openai_api_key"] = (key or "").strip()
    save_config(data)

def get_provider() -> str:
    return load_config().get("provider", "openai")

def set_provider(provider: str):
    data = load_config()
    data["provider"] = provider
    save_config(data)
