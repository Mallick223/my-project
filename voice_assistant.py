# ---------------- assistant_server.py ----------------
import os
import re
import subprocess
import time
import logging
import socket
import shutil
import warnings
from datetime import datetime
from pathlib import Path
import hashlib
import tempfile
import warnings
warnings.filterwarnings("ignore")

# Suppress PostHog and urllib3 warnings
os.environ["POSTHOG_DISABLED"] = "true"
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("backoff").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")
try:
    import eventlet
    eventlet.monkey_patch()
except ImportError:
    eventlet = None

from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify, send_file, send_from_directory, Response
from flask_cors import CORS
from flask_socketio import SocketIO, emit

import requests
import chromadb
from sentence_transformers import SentenceTransformer
import ollama
import pyjokes
import platform
import winreg
import difflib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Optional TTS
try:
    import pyttsx3
except Exception:
    pyttsx3 = None

try:
    from gtts import gTTS
except Exception:
    gTTS = None


# ---------------- Optional imports ----------------
try:
    from dotenv import load_dotenv
except:
    load_dotenv = None
try:
    from gtts import gTTS
except:
    gTTS = None
try:
    import PyPDF2
except:
    PyPDF2 = None
try:
    import docx
except:
    docx = None

# Load .env file if available
if load_dotenv:
    from dotenv import load_dotenv
    load_dotenv(override=False)

# ---------------- Config ----------------
logging.basicConfig(level=logging.INFO)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_BUILD = os.path.join(BASE_DIR, "frontend", "build")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
KB_FILE = os.path.join(BASE_DIR, "kb.csv")

# Load API keys from .env (with defaults for backward compatibility)
MIN_SIMILARITY = float(os.environ.get("MIN_SIMILARITY", "0.35"))
NEWS_API_KEY = os.environ.get("NEWS_API_KEY", "")
WEATHER_API_KEY = os.environ.get("WEATHER_API_KEY", "")
SERPAPI_API_KEY = os.environ.get("SERPAPI_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "tinyllama")
OLLAMA_TEMPERATURE = float(os.environ.get("OLLAMA_TEMPERATURE", "0.0"))
OLLAMA_MAX_TOKENS = int(os.environ.get("OLLAMA_MAX_TOKENS", "512"))

# CPU Optimization: Disable GPU, use thread pooling
OLLAMA_NUM_THREADS = int(os.environ.get("OLLAMA_NUM_THREADS", os.cpu_count() or 4))
OLLAMA_USE_GPU = os.environ.get("OLLAMA_USE_GPU", "false").lower() == "true"

# Model selection for CPU-friendly inference
AVAILABLE_MODELS = {
    "fast": "tinyllama",  # 1.1B, ~600MB, fastest
    "balanced": "neural-chat",  # 7B quantized, ~4GB
    "accurate": "mistral",  # 7B, ~4GB, best quality
}
OLLAMA_SELECTED_MODEL = os.environ.get("OLLAMA_SELECTED_MODEL", "fast")
OLLAMA_MODEL = AVAILABLE_MODELS.get(OLLAMA_SELECTED_MODEL, "tinyllama")

# Hybrid cloud: client-side ASR (Web Speech API), server-side LLM
HYBRID_MODE_ENABLED = os.environ.get("HYBRID_MODE", "true").lower() == "true"

# Lazy loading: only load RAG on first use
LAZY_RAG_ENABLED = os.environ.get("LAZY_RAG", "true").lower() == "true"
RAG_LOADED = False

# Use OS temp directory for TTS cache to avoid permission issues
TEMP_AUDIO_DIR = os.path.join(tempfile.gettempdir(), "voice_cache")
os.makedirs(TEMP_AUDIO_DIR, exist_ok=True)

audio_cache = {}   # text_hash -> filename
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ---------------- Flask ----------------
app = Flask(__name__, static_folder=FRONTEND_BUILD, static_url_path="/")
CORS(app)
# Use threading async mode to avoid requiring eventlet/gevent at runtime
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# ---------------- Helpers ----------------
def normalize(text):
    if not text:
        return ""
    t = text.lower()
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()

def internet_available(host="8.8.8.8", port=53, timeout=2):
    try:
        sock = socket.create_connection((host, port), timeout)
        sock.close()
        return True
    except:
        return False

# ---------------- TTS ----------------
def speak_to_file(text):
    if not text or not isinstance(text, str):
        return None

    # Hash text for caching
    text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
    filename = f"{text_hash}.mp3"
    path = os.path.join(TEMP_AUDIO_DIR, filename)

    # Serve from cache if exists
    if text_hash in audio_cache and os.path.exists(path):
        return path

    # Try online gTTS if available and internet is up
    if gTTS and internet_available():
        try:
            tts = gTTS(text=text, lang="en")
            tts.save(path)
            audio_cache[text_hash] = filename
            return path
        except Exception as e:
            logging.debug("gTTS failed: %s", e)

    # Fallback to pyttsx3 (offline)
    if pyttsx3:
        try:
            engine = pyttsx3.init()
            engine.save_to_file(text, path)
            engine.runAndWait()
            audio_cache[text_hash] = filename
            return path
        except Exception as e:
            logging.debug("pyttsx3 failed: %s", e)

    logging.error("TTS generation unavailable")
    return None


# ---------------- Load KB ----------------
def load_kb(file=KB_FILE):
    if not os.path.exists(file):
        pd.DataFrame(columns=["input","response"]).to_csv(file, index=False)
    try:
        return pd.read_csv(file).fillna("")
    except:
        return pd.DataFrame(columns=["input","response"])

df = load_kb()
kb_inputs = df["input"].tolist()
kb_responses = df["response"].tolist()

# ---------------- Embedding Model ----------------
kb_model = SentenceTransformer('all-MiniLM-L6-v2')  # Small, fast, accurate

# Embed all KB inputs
if kb_inputs:
    kb_embeddings = kb_model.encode(kb_inputs, convert_to_numpy=True, normalize_embeddings=True)
else:
    kb_embeddings = np.zeros((0, kb_model.get_sentence_embedding_dimension()))

# ---------------- KB Matching ----------------
def kb_match(query, threshold=MIN_SIMILARITY):
    if not query or kb_embeddings.size == 0:
        return None
    query_vec = kb_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    sims = cosine_similarity(query_vec, kb_embeddings).flatten()
    best_idx = int(np.argmax(sims))
    if sims[best_idx] >= threshold:
        return kb_responses[best_idx]
    return None

# ---------------- Teach New ----------------
def teach_new(query, answer, df=df):
    new_row = pd.DataFrame([{"input": query, "response": answer}])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(KB_FILE, index=False)
    # Recompute embeddings
    global kb_inputs, kb_responses, kb_embeddings
    kb_inputs = df["input"].tolist()
    kb_responses = df["response"].tolist()
    kb_embeddings = kb_model.encode(kb_inputs, convert_to_numpy=True, normalize_embeddings=True)
    return df

# ---------------- RAG (Retrieval Augmented Generation) ----------------
import faiss

rag_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
rag_index, rag_texts = None, []

def build_rag_index(docs):
    global rag_index, rag_texts
    if not docs:
        return
    embeddings = rag_model.encode(docs, convert_to_numpy=True, normalize_embeddings=True)
    dim = embeddings.shape[1]
    rag_index = faiss.IndexFlatIP(dim)
    rag_index.add(embeddings.astype('float32'))
    rag_texts = docs.copy()

def rag_add_docs(new_docs):
    global rag_index, rag_texts
    if not new_docs:
        return
    embeddings = rag_model.encode(new_docs, convert_to_numpy=True, normalize_embeddings=True)
    if rag_index is None:
        build_rag_index(new_docs)
    else:
        rag_index.add(embeddings.astype('float32'))
        rag_texts.extend(new_docs)

def rag_retrieve(query, top_k=3):
    global lazy_rag
    if LAZY_RAG_ENABLED:
        lazy_rag.ensure_loaded()
    if rag_index is None or not query:
        return []
    q_emb = rag_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    D, I = rag_index.search(q_emb.astype('float32'), top_k)
    results = []
    for idx in I[0]:
        if 0 <= idx < len(rag_texts):
            results.append(rag_texts[idx])
    return results


# Lazy RAG Loader (for low-memory devices)
class LazyRAGLoader:
    """Load RAG index only when needed."""
    def __init__(self):
        self.loaded = False
        self.docs_file = None
    
    def set_docs_file(self, file_path):
        """Set path to documents file for lazy loading."""
        self.docs_file = file_path
    
    def ensure_loaded(self):
        """Load RAG if not already loaded."""
        global RAG_LOADED, rag_index, rag_texts
        if not self.loaded and LAZY_RAG_ENABLED:
            if self.docs_file and os.path.exists(self.docs_file):
                try:
                    with open(self.docs_file, 'r', encoding='utf-8') as f:
                        docs = [line.strip() for line in f if line.strip()]
                    if docs:
                        build_rag_index(docs)
                        self.loaded = True
                        RAG_LOADED = True
                        logging.info("Lazy-loaded RAG with %d documents", len(docs))
                except Exception as e:
                    logging.error("Failed to lazy-load RAG: %s", e)
        elif not LAZY_RAG_ENABLED:
            self.loaded = True

lazy_rag = LazyRAGLoader()

# ---------------- Persistent Memory (ChromaDB) ----------------
MEMORY_DIR = os.path.join(BASE_DIR, "assistant_memory_db")
os.makedirs(MEMORY_DIR, exist_ok=True)

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="assistant_memory")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
# Warmup flag (models loaded during import)
WARMUP_DONE = True

def remember(user_text, assistant_reply):
    try:
        embedding = embedding_model.encode(user_text).tolist()
        collection.add(
            documents=[assistant_reply],
            embeddings=[embedding],
            metadatas=[{"query": user_text}],
            ids=[f"mem_{hash(user_text)}"]
        )
    except Exception as e:
        print(f"Memory error: {e}")

def recall(query, top_k=2):
    try:
        query_embedding = embedding_model.encode(query).tolist()
        results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
        docs = results.get("documents", [None])
        if docs and docs[0]:
            past_memories = [doc for doc in docs[0]]
            return "\n".join(past_memories)
        return ""
    except Exception as e:
        print(f"Recall error: {e}")
        return ""

# ---------------- web search ----------------
def is_web_search_query(text):
    if not text:
        return False
    t = text.lower()
    triggers = [
        "search", "google", "look up", "find online",
        "web search", "on the internet"
    ]
    return any(trigger in t for trigger in triggers)

def web_search(query, num_results=5):
    if not SERPAPI_API_KEY:
        return []

    try:
        r = requests.get(
            "https://serpapi.com/search",
            params={
                "q": query,
                "api_key": SERPAPI_API_KEY,
                "num": num_results
            },
            timeout=8
        )
        data = r.json()
        results = []

        for item in data.get("organic_results", [])[:num_results]:
            title = item.get("title", "")
            snippet = item.get("snippet", "")
            link = item.get("link", "")
            if title and snippet:
                results.append(f"{title}\n{snippet}\nSource: {link}")

        return results
    except Exception as e:
        logging.error(f"Web search error: {e}")
        return []


# ---------------- System Apps (STABLE + FUZZY + SAFE) ----------------

BUILTIN_APPS = {
    "calculator": "calc.exe",
    "notepad": "notepad.exe",
    "cmd": "cmd.exe",
    "explorer": "explorer.exe"
}

INSTALLED_APPS = {}   # classic exe apps
UWP_APPS = {}         # store apps (WhatsApp, Camera, Mail, etc.)


def _clean_exe_path(exe_path):
    if not exe_path:
        return exe_path
    p = str(exe_path).strip().strip('"')
    if "," in p:
        parts = p.split(",")
        if parts[-1].strip().isdigit():
            p = ",".join(parts[:-1]).strip()
    return p


def index_classic_apps():
    paths = [
        (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"),
        (winreg.HKEY_CURRENT_USER, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"),
    ]
    for h, p in paths:
        try:
            k = winreg.OpenKey(h, p)
            for i in range(winreg.QueryInfoKey(k)[0]):
                try:
                    s = winreg.OpenKey(k, winreg.EnumKey(k, i))
                    name, _ = winreg.QueryValueEx(s, "DisplayName")
                    icon, _ = winreg.QueryValueEx(s, "DisplayIcon")
                    if icon:
                        INSTALLED_APPS[normalize(name)] = icon.split(",")[0]
                except:
                    pass
        except:
            pass


def index_uwp_apps():
    try:
        output = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command",
             "Get-StartApps | ForEach-Object { $_.Name + '|' + $_.AppID }"],
            text=True
        )
        for line in output.splitlines():
            if "|" in line:
                name, appid = line.split("|", 1)
                UWP_APPS[normalize(name)] = appid
    except:
        pass


index_classic_apps()
index_uwp_apps()

def index_system_apps():
    """Index .exe files from common system folders"""
    global SYSTEM_APPS
    search_dirs = [
        r"C:\Program Files",
        r"C:\Program Files (x86)",
        r"C:\Windows\System32",
        r"C:\Windows"
    ]

    apps = {}
    for base in search_dirs:
        if not os.path.exists(base):
            continue
        for root, _, files in os.walk(base):
            for f in files:
                if f.lower().endswith(".exe"):
                    name = normalize(os.path.splitext(f)[0])
                    apps.setdefault(name, os.path.join(root, f))

    SYSTEM_APPS = apps
    logging.info("Indexed %d system apps", len(SYSTEM_APPS))

# Initialize app indexing
try:
    index_system_apps()
except Exception as e:
    logging.warning("App indexing failed: %s", e)

def fuzzy_find(app_dict, key):
    matches = difflib.get_close_matches(key, app_dict.keys(), n=1, cutoff=0.6)
    return matches[0] if matches else None

# ---------------- Ollama TinyLLaMA ----------------
def call_ollama_tinyllama(prompt: str, stream_callback=None, force_hallucinate: bool = False) -> str:
    """Call Ollama with optional token streaming.

    If `force_hallucinate` is True and the model returns no useful content,
    the function will ask the model to provide a plausible (speculative)
    final answer and mark it as speculative.
    """
    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            stream=False
        )

        # Extract response
        if isinstance(response, dict) and "message" in response:
            content = response["message"].get("content", "").strip()
            if content:
                if stream_callback:
                    for token in content.split():
                        stream_callback(token + " ")
                return content

        # If model gave no content and hallucination is allowed, ask it to speculate
        if force_hallucinate:
            try:
                halluc_prompt = (
                    "You were not able to provide a confident answer. "
                    "Now provide a plausible, speculative answer based on your knowledge, "
                    "and prefix it with '(Speculative) '. Be concise."
                )
                resp2 = ollama.chat(
                    model=OLLAMA_MODEL,
                    messages=[
                        {"role": "system", "content": "You are a creative assistant that may speculate when asked."},
                        {"role": "user", "content": prompt + "\n\n" + halluc_prompt}
                    ],
                    stream=False
                )
                if isinstance(resp2, dict) and "message" in resp2:
                    content2 = resp2["message"].get("content", "").strip()
                    if content2:
                        if stream_callback:
                            for token in content2.split():
                                stream_callback(token + " ")
                        return "(Speculative) " + content2
            except Exception as e:
                logging.debug("Hallucination attempt failed: %s", e)

        return "I don't have information about that. Please try another query."
    except Exception as e:
        logging.error(f"Ollama error: {e}")
        return f"I encountered an error processing your request."

#---------------- Social Media Search ----------------
SOCIAL_APPS = {
    "instagram": "https://www.instagram.com",
    "facebook": "https://www.facebook.com",
    "twitter": "https://twitter.com",
    "linkedin": "https://www.linkedin.com"
}

def social_web_search(query):
    query_lower = query.lower()
    for name, url in SOCIAL_APPS.items():
        if name in query_lower:
            return f"You can visit {name.capitalize()} here: {url}"
    return None


#---------------- System Command Handling ----------------
def is_system_command(text):
    return text.lower().startswith("open ")

def handle_system_command(text):
    app_name = text[5:].strip()
    logging.info("Attempting to run system command: '%s'", app_name)
    if not app_name:
        return "Please specify an application to open."
    return open_app(app_name)
#---------------- Open Any App ----------------
def open_app(name):
    key = normalize(name)

    # 1️⃣ Built-in apps
    if key in BUILTIN_APPS:
        subprocess.Popen(BUILTIN_APPS[key], shell=True)
        return f"Opened {name}"

    # 2️⃣ WhatsApp HARD FIX (Microsoft Store)
    if "whatsapp" in key:
        subprocess.Popen(
            "explorer.exe shell:AppsFolder\\5319275A.WhatsAppDesktop_cv1g1gvanyjgm!App",
            shell=True
        )
        return "Opened WhatsApp"

    # 3️⃣ UWP apps (Camera, Mail, etc.)
    uwp = fuzzy_find(UWP_APPS, key)
    if uwp:
        subprocess.Popen(
            f'explorer.exe shell:AppsFolder\\{UWP_APPS[uwp]}',
            shell=True
        )
        return f"Opened {uwp}"

    # 4️⃣ VS CODE HARD FIX (NO PATH NEEDED)
    if key in ("vscode", "visual studio code", "code"):
        paths = [
            os.path.expandvars(r"%LOCALAPPDATA%\Programs\Microsoft VS Code\Code.exe"),
            r"C:\Program Files\Microsoft VS Code\Code.exe",
        ]
        for p in paths:
            if os.path.exists(p):
                subprocess.Popen(f'"{p}"', shell=True)
                return "Opened Visual Studio Code"
        return "Visual Studio Code is not installed."

    # 5️⃣ Classic EXE apps
    exe = fuzzy_find(INSTALLED_APPS, key)
    if exe:
        subprocess.Popen(f'"{INSTALLED_APPS[exe]}"', shell=True)
        return f"Opened {exe}"

    return "I couldn't find that application."


#---------------- Joke Handling ----------------    
def get_random_joke():
    try:
        return pyjokes.get_joke()
    except:
        return "I tried to be funny, but failed."

def get_latest_news(query):
    # If NEWS_API_KEY provided, query NewsAPI for headlines optionally filtered by topic
    try:
        if not NEWS_API_KEY:
            return None
        params = {
            "language": "en",
            "pageSize": 3,
            "apiKey": NEWS_API_KEY,
        }
        if query and query.strip():
            params["q"] = query
        url = "https://newsapi.org/v2/top-headlines"
        res = requests.get(url, params=params, timeout=5).json()
        articles = res.get("articles", [])
        if not articles:
            return None
        return "\n".join(f"- {a['title']}" for a in articles if a.get("title"))
    except Exception:
        return None


def get_weather(query):
    """Return short weather for a location extracted from query.
    Uses OpenWeatherMap if WEATHER_API_KEY env var is set, otherwise falls back to wttr.in.
    """
    # extract location after 'in' or the last word
    loc = None
    q = query.lower()
    if " in " in q:
        loc = q.split(" in ")[-1].strip()
    else:
        parts = q.split()
        if parts:
            loc = parts[-1]
    if not loc:
        return "Please specify a location for weather (e.g., 'weather in London')."

    try:
        if WEATHER_API_KEY:
            url = f"https://api.openweathermap.org/data/2.5/weather"
            params = {"q": loc, "appid": WEATHER_API_KEY, "units": "metric"}
            r = requests.get(url, params=params, timeout=5).json()
            if r.get("cod") != 200:
                return None
            desc = r["weather"][0]["description"].capitalize()
            temp = r["main"]["temp"]
            return f"Weather in {loc}: {desc}, {temp}°C"
        else:
            # fallback to wttr.in short format
            url = f"https://wttr.in/{loc}?format=3"
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                return r.text.strip()
            return None
    except Exception:
        return None


def get_time(query):
    """Return current local time or time in a requested city (best-effort).
    Uses worldtimeapi to find a timezone that contains the city name.
    """
    q = query.lower()
    if " in " in q:
        loc = q.split(" in ")[-1].strip()
    else:
        loc = None

    try:
        if not loc:
            now = datetime.now()
            return now.strftime("%Y-%m-%d %H:%M:%S")
        # try worldtimeapi lookup
        zones = requests.get("http://worldtimeapi.org/api/timezone", timeout=5).json()
        # find zone containing city name
        match = None
        for z in zones:
            if loc.replace(" ", "_").lower() in z.lower() or loc.lower() in z.lower():
                match = z
                break
        if match:
            data = requests.get(f"http://worldtimeapi.org/api/timezone/{match}", timeout=5).json()
            dt = data.get("datetime")
            if dt:
                # format: 2025-12-15T12:34:56.123456+00:00
                dt = dt.split(".")[0].replace("T", " ")
                return f"Local time in {loc}: {dt}"
        # fallback to local time
        now = datetime.now()
        return now.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        now = datetime.now()
        return now.strftime("%Y-%m-%d %H:%M:%S")

# ---------------- Real-Time Informational APIs (Free & Reliable) ----------------

import requests

def get_current_weather(location="Delhi"):
    """Fetch current weather using Open-Meteo (free, no key needed)"""
    try:
        # Simple geocode fallback: use a known city or default
        # In production, you'd use a geocoder, but for simplicity:
        city_coords = {
            "delhi": (28.7041, 77.1025),
            "mumbai": (19.0760, 72.8777),
            "london": (51.5074, -0.1278),
            "new york": (40.7128, -74.0060),
            "paris": (48.8566, 2.3522),
            "tokyo": (35.6762, 139.6503),
            "sydney": (-33.8688, 151.2093),
        }
        loc_lower = location.lower().strip()
        coords = city_coords.get(loc_lower)
        if not coords:
            # Default fallback
            coords = (28.7041, 77.1025)
            location = "your area"

        lat, lon = coords
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,weather_code,wind_speed_10m"
        response = requests.get(url, timeout=8)
        response.raise_for_status()
        data = response.json()["current"]

        temp = data["temperature_2m"]
        wind = data["wind_speed_10m"]
        code = data["weather_code"]

        desc_map = {
            0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
            45: "Fog", 48: "Fog", 51: "Light drizzle", 61: "Light rain", 63: "Rain",
            80: "Heavy rain showers", 95: "Thunderstorm"
        }
        description = desc_map.get(code, "Unknown")

        return f"Weather in {location.title()}: {description}, {temp}°C, Wind: {wind} km/h."
    except Exception as e:
        logging.debug(f"Weather API failed: {e}")
        return "Sorry, I couldn't fetch the weather right now."

def get_world_time(location=None):
    """Get current time in any city using WorldTimeAPI (free, no key)"""
    try:
        if not location:
            from datetime import datetime
            return datetime.now().strftime("Current time: %H:%M:%S on %Y-%m-%d")

        # Common timezone mappings
        zone_map = {
            "london": "Europe/London",
            "paris": "Europe/Paris",
            "berlin": "Europe/Berlin",
            "tokyo": "Asia/Tokyo",
            "new york": "America/New_York",
            "los angeles": "America/Los_Angeles",
            "sydney": "Australia/Sydney",
            "mumbai": "Asia/Kolkata",
            "delhi": "Asia/Kolkata",
            "dubai": "Asia/Dubai",
        }
        zone = zone_map.get(location.lower().strip())
        if not zone:
            return "I don't recognize that location for time lookup."

        url = f"http://worldtimeapi.org/api/timezone/{zone}"
        response = requests.get(url, timeout=8)
        response.raise_for_status()
        data = response.json()
        dt = data["datetime"]
        time_str = dt[11:16]  # HH:MM
        date_str = dt[:10]
        return f"Time in {location.title()}: {time_str} on {date_str}"
    except Exception as e:
        logging.debug(f"Time API failed: {e}")
        return "Couldn't fetch time for that location."

def get_crypto_price(coin="bitcoin"):
    """Get live crypto price via CoinGecko (free, no key)"""
    try:
        coin_map = {
            "bitcoin": "bitcoin", "btc": "bitcoin",
            "ethereum": "ethereum", "eth": "ethereum",
            "cardano": "cardano", "ada": "cardano",
            "solana": "solana", "sol": "solana"
        }
        coin_id = coin_map.get(coin.lower(), "bitcoin")
        url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies=usd&include_24hr_change=true"
        response = requests.get(url, timeout=8)
        response.raise_for_status()
        data = response.json()[coin_id]
        price = data["usd"]
        change = data.get("usd_24h_change", 0)
        return f"{coin.title()} is currently ${price:,.2f} USD (24h: {change:+.2f}%)"
    except Exception as e:
        logging.debug(f"Crypto API failed: {e}")
        return "Couldn't fetch crypto price right now."

def get_stock_price(symbol="AAPL"):
    """Simple stock price via Alpha Vantage (requires free key)"""
    if not os.environ.get("ALPHA_VANTAGE_KEY"):
        return "Stock lookup not configured (missing ALPHA_VANTAGE_KEY)."
    try:
        key = os.environ["ALPHA_VANTAGE_KEY"]
        url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol.upper()}&apikey={key}"
        response = requests.get(url, timeout=8)
        response.raise_for_status()
        quote = response.json()["Global Quote"]
        price = quote["05. price"]
        change = quote["10. change percent"]
        return f"{symbol.upper()} stock: ${price} ({change} today)"
    except Exception as e:
        logging.debug(f"Stock API failed: {e}")
        return "Couldn't fetch stock price."

def verify_with_serpapi(claims, top_n=3, api_key=None, timeout=5):
    """Verify one or more claims using SerpAPI.

    Args:
        claims (str | list): A single claim string or a list of claim strings/sentences.
        top_n (int): Number of top organic results to fetch per claim.
        api_key (str): Optional SerpAPI key. If None, uses env var `SERPAPI_API_KEY`.
        timeout (int): HTTP timeout in seconds.

    Returns:
        dict: Mapping from claim -> verification result dict with keys:
              - status: 'verified'|'supported'|'unverified' or error message
              - verified: bool
              - supported: bool
              - top_results: list of {title, link, snippet}
    """
    api_key = api_key or os.environ.get("SERPAPI_API_KEY")
    if not api_key:
        logging.info("SERPAPI_API_KEY not set; verification unavailable.")
        return {"error": "SERPAPI_API_KEY not set"}

    # Normalize claims into a list of non-empty sentences
    if isinstance(claims, str):
        # split on sentence boundaries but keep reasonably sized clauses
        parts = [c.strip() for c in re.split(r'[\.\?!]\s+', claims) if c.strip()]
        claim_list = parts if parts else [claims.strip()]
    elif isinstance(claims, (list, tuple)):
        claim_list = [str(c).strip() for c in claims if str(c).strip()]
    else:
        return {"error": "Invalid claims type; must be string or list"}

    results = {}
    authoritative_domains = ["wikipedia.org", ".gov", ".edu", "nytimes.com", "bbc.co", "theguardian.com", "nature.com", "sciencedirect.com"]

    for claim in claim_list:
        try:
            params = {"q": claim, "api_key": api_key, "num": top_n}
            resp = requests.get("https://serpapi.com/search", params=params, timeout=timeout)
            if resp.status_code != 200:
                results[claim] = {"error": f"SerpAPI returned {resp.status_code}", "raw": resp.text}
                continue

            data = resp.json()
            organic = data.get("organic_results") or data.get("knowledge_graph") or []
            if isinstance(organic, dict):
                organic = [organic]

            top_results = []
            for item in organic[:top_n]:
                title = item.get("title") or item.get("position") or ""
                link = item.get("link") or item.get("url") or item.get("displayed_link") or ""
                snippet = item.get("snippet") or item.get("description") or ""
                top_results.append({"title": title, "link": link, "snippet": snippet})

            # Heuristic: check word-overlap support and authoritative sources
            norm_claim = normalize(claim)
            claim_words = set(re.findall(r"\w+", norm_claim))
            supported = False
            verified = False

            for ritem in top_results:
                text_blob = " ".join([ritem.get("title", ""), ritem.get("snippet", "")]).lower()
                words = set(re.findall(r"\w+", text_blob))
                if not words:
                    continue
                overlap = len(claim_words & words) / max(1, min(len(claim_words), 6))
                if overlap >= 0.5:
                    supported = True
                link = (ritem.get("link") or "").lower()
                for dom in authoritative_domains:
                    if dom in link:
                        verified = True
                        break
                if supported and verified:
                    break

            status = "verified" if verified else ("supported" if supported else "unverified")
            results[claim] = {"status": status, "verified": verified, "supported": supported, "top_results": top_results}

        except Exception as e:
            logging.debug("SerpAPI error for claim '%s': %s", claim, e)
            results[claim] = {"error": str(e)}

    return results


def extract_and_verify_claims(text, verify=True, top_n=3):
    """Extract key factual claims from text and optionally verify them.
    
    Args:
        text (str): Response text to extract claims from.
        verify (bool): Whether to call SerpAPI for verification.
        top_n (int): Number of search results per claim.
    
    Returns:
        dict: {"original": text, "verified_annotations": {...}, "summary": str}
              where verified_annotations maps claims -> {verified: bool, ...}
    """
    if not text or not isinstance(text, str):
        return {"original": text, "verified_annotations": {}, "summary": ""}
    
    # Extract sentences with numbers, named entities, or claims
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
    
    # Simple heuristic: prioritize sentences with numbers or capitalized proper nouns
    claim_candidates = []
    for s in sentences[:5]:  # Check first 5 sentences max
        if any(c.isdigit() for c in s) or any(word[0].isupper() for word in s.split() if len(word) > 2):
            claim_candidates.append(s)
    
    claim_candidates = claim_candidates[:3]  # Verify top 3 claims
    
    verified_annotations = {}
    if verify and claim_candidates and SERPAPI_API_KEY:
        try:
            results = verify_with_serpapi(claim_candidates, top_n=top_n, api_key=SERPAPI_API_KEY, timeout=5)
            for claim, result in results.items():
                if "error" not in result:
                    verified_annotations[claim] = {
                        "status": result.get("status", "unverified"),
                        "verified": result.get("verified", False),
                        "supported": result.get("supported", False)
                    }
        except Exception as e:
            logging.debug("Claim verification failed: %s", e)
    
    # Build summary annotation
    summary_parts = []
    if verified_annotations:
        verified_count = sum(1 for r in verified_annotations.values() if r.get("verified"))
        total_count = len(verified_annotations)
        if verified_count > 0:
            summary_parts.append(f"[{verified_count}/{total_count} claims verified]")
        else:
            summary_parts.append(f"[{total_count} claims checked, some unverified]")
    
    return {
        "original": text,
        "verified_annotations": verified_annotations,
        "summary": " ".join(summary_parts) if summary_parts else ""
    }


# ---------------- Generate Response ----------------
def generate_response(query):
    ql = query.strip()
    ql_lower = ql.lower()

    if not ql:
        return "Please say something."

    # -------------------------------------------------------
    # 0️⃣ Simple greetings (Quick response)
    # -------------------------------------------------------
    greetings = {
        "hello": "Hello there! How can I help you today?",
        "hi": "Hi! What can I do for you?",
        "hey": "Hey! What's on your mind?",
        "good morning": "Good morning! Ready to help you.",
        "good evening": "Good evening! How can I assist?",
        "how are you": "I'm doing great, thanks for asking! How about you?",
        "thanks": "You're welcome! Happy to help.",
        "thank you": "You're welcome! My pleasure.",
        "goodbye": "Goodbye! Have a great day.",
        "bye": "Bye! Take care."

    }
    
    for greeting, response in greetings.items():
        if ql_lower == greeting or ql_lower.startswith(greeting):
            return response

    # -------------------------------------------------------
    # 1️⃣ System commands
    # -------------------------------------------------------
    if is_system_command(ql):
        return handle_system_command(ql)
    #-------------------------------------------------------
    # 5️⃣ Web Search (Explicit request)
    # -------------------------------------------------------
    if is_web_search_query(ql):
        social_result = social_web_search(ql)
        if social_result:
            return social_result

        clean_query = re.sub(
            r"(search|google|look up|find online|web search|on the internet)",
            "",
            ql_lower
        ).strip()

        web_results = web_search(clean_query or ql, num_results=5)

        if web_results:
            # Summarize web results using LLM (NO hallucination)
            summary_prompt = (
                "You are given web search results.\n"
                "Summarize them clearly and accurately.\n"
                "Do NOT add any information that is not present.\n\n"
                "Web Results:\n" + "\n\n".join(web_results) +
                "\n\nAnswer:"
            )

            try:
                llm_resp = ollama.chat(
                    model=OLLAMA_MODEL,
                    messages=[
                        {"role": "system", "content": "You summarize web data factually."},
                        {"role": "user", "content": summary_prompt}
                    ]
                )
                if isinstance(llm_resp, dict) and "message" in llm_resp:
                    return llm_resp["message"].get("content", "")
                return str(llm_resp)
            except Exception:
                return "\n\n".join(web_results)

        return "I couldn't find relevant results on the web."


    # -------------------------------------------------------
    # 2️⃣ Jokes
    # -------------------------------------------------------
    if "joke" in ql_lower:
        return get_random_joke()

        # Enhanced Weather
    if "weather" in ql_lower:
        loc = ql_lower.replace("weather", "").replace("in", "").strip() or "Delhi"
        return get_current_weather(loc)

    # Enhanced Time
    if any(word in ql_lower for word in ["time", "clock", "what time"]):
        loc = ql_lower.split("in")[-1].strip() if " in " in ql_lower else None
        return get_world_time(loc)

    # Crypto Price
    if any(word in ql_lower for word in ["bitcoin", "btc", "ethereum", "eth", "crypto", "coin price"]):
        coin = "bitcoin"
        if "eth" in ql_lower:
            coin = "ethereum"
        return get_crypto_price(coin)

    # Stock Price
    if any(word in ql_lower for word in ["stock", "aapl", "apple stock", "tesla stock", "google stock"]):
        symbol = "AAPL"
        if "tesla" in ql_lower: symbol = "TSLA"
        if "google" in ql_lower: symbol = "GOOGL"
        return get_stock_price(symbol)
    # -------------------------------------------------------
    # 5️⃣ News
    # -------------------------------------------------------
    if "news" in ql_lower:
        news = get_latest_news(ql)
        if news:
            return news

    # -------------------------------------------------------
    # 6️⃣ Knowledge Base (KB) (embedding-based)
    # -------------------------------------------------------
    # -------------------------------------------------------
    kb_answer = None

    if df is not None and not df.empty:
     kb_answer = kb_match(ql)

    if kb_answer:
        remember(ql, kb_answer)
        # Generate TTS (non-blocking, no return dependency)
        audio_file = speak_to_file(kb_answer)
        if audio_file:
            print(f"TTS ready: {audio_file}")
            return kb_answer

    # -------------------------------------------------------
    # 7️⃣ Memory recall for context
    # -------------------------------------------------------
    memory_context = recall(ql)
    # -------------------------------------------------------
    # 8️⃣ RAG retrieval (always attempt)
    # -------------------------------------------------------
    rag_docs = rag_retrieve(ql, top_k=5)

    # -------------------------------------------------------
    # 9️⃣ Wikipedia lookup removed; proceed with LLM / RAG fallback
    # -------------------------------------------------------
    # -------------------------------------------------------
    # 🔟 Build comprehensive prompt for LLM
    # -------------------------------------------------------
    prompt_parts = []
    
    # System instruction for better answers
    system_instruction = (
    "You are a knowledgeable, careful, and truthful AI assistant.\n"
    "- Prefer factual accuracy over creativity.\n"
    "- If information is uncertain, say so clearly.\n"
    "- Use retrieved context if available.\n"
    "- Do NOT hallucinate names, dates, or sources.\n"
    "- Be concise but informative."
    )
    
    if memory_context:
        prompt_parts.append(f"Previous context:\n{memory_context}")

    if rag_docs:
        prompt_parts.append(
            "Reference information:\n" + "\n---\n".join(rag_docs[:3])
        )

    prompt_parts.append(f"Question: {ql}\n\nProvide a helpful and accurate answer:")
    
    final_prompt = "\n\n".join(prompt_parts)

    # -------------------------------------------------------
    # Determine whether user requested speculation
    # -------------------------------------------------------
    speculative_keywords = ["speculate", "speculative", "guess", "hypothesize", "make a guess", "please speculate"]
    speculative_requested = any(k in ql_lower for k in speculative_keywords)

    # -------------------------------------------------------
    # Call LLM with improved prompt
    # -------------------------------------------------------
    try:
        # First attempt: use Ollama with system instruction
        response_text = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": final_prompt}
            ],
            stream=False
        )
        
        if isinstance(response_text, dict) and "message" in response_text:
            response_text = response_text["message"].get("content", "").strip()
        
        # If we got a good response, verify and return
        if response_text and len(response_text.strip()) > 5:
            # Optionally verify claims (auto-verify unless user requests speculation)
            if not speculative_requested and SERPAPI_API_KEY:
                try:
                    verification_result = extract_and_verify_claims(response_text, verify=True, top_n=3)
                    if verification_result.get("summary"):
                        response_text = response_text + "\n\n" + verification_result["summary"]
                except Exception as e:
                    logging.debug("Verification annotation failed: %s", e)
            
            remember(ql, response_text)
            return response_text
    except Exception as e:
        logging.error(f"LLM error: {e}")

    # -------------------------------------------------------
    # Fallback: return apologetic message or suggestion
    # -------------------------------------------------------
    return "I'm having difficulty answering that. Please try rephrasing or ask me something else."

# Health check endpoint
@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok", "service": "assistant"}), 200

# Status endpoint used by frontend to show warmup and basic stats
@app.route("/status", methods=["GET"])
def status_route():
    try:
        kb_count = len(kb_inputs) if 'kb_inputs' in globals() else 0
        rag_count = len(rag_texts) if 'rag_texts' in globals() else 0
        return jsonify({
            "warmup_done": bool(globals().get('WARMUP_DONE', False)),
            "kb_count": kb_count,
            "rag_count": rag_count,
            "model": OLLAMA_MODEL,
            "use_gpu": OLLAMA_USE_GPU,
            "cpu_threads": OLLAMA_NUM_THREADS,
            "hybrid_mode": HYBRID_MODE_ENABLED,
            "lazy_rag": LAZY_RAG_ENABLED,
            "rag_loaded": RAG_LOADED
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------- Document Parsing Routes ----------------
def parse_document(file_path):
    text = ""
    try:
        if file_path.endswith(".txt"):
            with open(file_path,"r",encoding="utf-8",errors="ignore") as f:
                text = f.read()
        elif file_path.endswith(".pdf") and PyPDF2:
            with open(file_path,"rb") as fh:
                reader = PyPDF2.PdfReader(fh)
                text = " ".join(page.extract_text() or "" for page in reader.pages)
        elif file_path.endswith(".docx") and docx:
            text = " ".join(p.text for p in docx.Document(file_path).paragraphs)
        elif file_path.endswith(".csv"):
            text = pd.read_csv(file_path).to_string()
    except Exception:
        text = ""
    return text.strip()

@app.route("/upload_doc", methods=["POST"])
def upload_doc():
    if "file" not in request.files:
        return jsonify({"status":"error","message":"No file uploaded"}), 400
    file = request.files["file"]
    if not file.filename:
        return jsonify({"status":"error","message":"No file selected"}), 400
    filename = secure_filename(file.filename)
    path = os.path.join(UPLOAD_DIR, filename)
    file.save(path)
    text = parse_document(path)
    if text:
        chunks = [text[i:i+500] for i in range(0,len(text),500)]
        rag_add_docs(chunks)
        return jsonify({"status":"success","message":f"{filename} added to RAG"}), 200
    return jsonify({"status":"error","message":"Unsupported or empty file"}), 400

# ---------------- TTS Routes ----------------
@app.route("/tts", methods=["POST"])
def tts_route():
    data = request.get_json() or {}
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "No text provided"}), 400

    audio_path = speak_to_file(text)
    if not audio_path:
        return jsonify({"error": "TTS generation failed"}), 500

    return send_file(audio_path, mimetype="audio/mpeg")
@app.route("/audio/<filename>")
def serve_audio(filename):
    path = os.path.join(TEMP_AUDIO_DIR, filename)
    if os.path.exists(path):
        return send_file(path, mimetype="audio/mpeg")
    return jsonify({"error": "File not found"}), 404


# ---------------- Verification & Web Search Routes ----------------
@app.route("/verify", methods=["POST"])
def verify_route():
    """Verify claims via SerpAPI.
    
    Request body: {"claims": "text to verify" | ["claim1", "claim2"]}
    Response: {"results": {claim: {"status", "verified", "supported", "top_results"}}}
    """
    try:
        data = request.get_json() or {}
        claims = data.get("claims")
        if not claims:
            return jsonify({"error": "No claims provided"}), 400
        
        results = verify_with_serpapi(claims, top_n=5, api_key=SERPAPI_API_KEY, timeout=10)
        return jsonify({"results": results}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/search", methods=["POST"])
def search_route():
    """Web search via SerpAPI.
    
    Request body: {"query": "search query"}
    Response: {"results": [{"title", "link", "snippet"}]}
    """
    try:
        data = request.get_json() or {}
        query = (data.get("query") or "").strip()
        if not query:
            return jsonify({"error": "No query provided"}), 400
        
        if not SERPAPI_API_KEY:
            return jsonify({"error": "SERPAPI_API_KEY not configured"}), 500
        
        params = {"q": query, "api_key": SERPAPI_API_KEY, "num": 5}
        resp = requests.get("https://serpapi.com/search", params=params, timeout=10)
        if resp.status_code != 200:
            return jsonify({"error": f"Search failed: {resp.status_code}"}), 500
        
        data = resp.json()
        organic = data.get("organic_results", [])
        results = [
            {"title": item.get("title", ""), "link": item.get("link", ""), "snippet": item.get("snippet", "")}
            for item in organic[:5]
        ]
        return jsonify({"results": results}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ============= DATASET & RESOURCE ENDPOINTS =============

@app.route("/datasets/arxiv", methods=["POST"])
def load_arxiv_papers():
    """Fetch and index arXiv papers by topic.
    
    Body: {"query": "machine learning", "max_results": 10}
    """
    try:
        data = request.get_json() or {}
        query = (data.get("query") or "machine learning").strip()
        max_results = int(data.get("max_results", 10))
        
        url = f"http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results={max_results}"
        r = requests.get(url, timeout=15)
        
        if r.status_code != 200:
            return jsonify({"error": f"arXiv API error: {r.status_code}"}), 500
        
        # Simple XML parsing (avoid heavy dependencies)
        import xml.etree.ElementTree as ET
        root = ET.fromstring(r.content)
        docs = []
        
        for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
            title_elem = entry.find('{http://www.w3.org/2005/Atom}title')
            summary_elem = entry.find('{http://www.w3.org/2005/Atom}summary')
            if title_elem is not None and summary_elem is not None:
                title = title_elem.text.strip()
                summary = summary_elem.text.strip()
                doc = f"Title: {title}\n{summary}"
                docs.append(doc)
        
        if docs:
            rag_add_docs(docs)
            return jsonify({"loaded": len(docs), "docs": docs[:3]}), 200
        
        return jsonify({"error": "No papers found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/datasets/kb-import", methods=["POST"])
def import_kb_csv():
    """Bulk import Q&A pairs from CSV file.
    
    CSV format: question,answer
    """
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files["file"]
        df = pd.read_csv(file)
        
        if "question" not in df.columns or "answer" not in df.columns:
            return jsonify({"error": "CSV must have 'question' and 'answer' columns"}), 400
        
        count = 0
        for _, row in df.iterrows():
            try:
                teach_new(row["question"], row["answer"])
                count += 1
            except Exception as e:
                logging.debug(f"Failed to import row: {e}")
        
        return jsonify({"imported": count, "total": len(df)}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/datasets/custom-docs", methods=["POST"])
def add_custom_docs():
    """Add custom documents to RAG index.
    
    Body: {"documents": ["doc1 text", "doc2 text", ...]}
    """
    try:
        data = request.get_json() or {}
        docs = data.get("documents", [])
        
        if not docs or not isinstance(docs, list):
            return jsonify({"error": "Provide 'documents' as a list of strings"}), 400
        
        # Clean and chunk docs
        chunks = []
        for doc in docs:
            doc_str = str(doc).strip()
            if doc_str:
                # Chunk into 500-char segments
                doc_chunks = [doc_str[i:i+500] for i in range(0, len(doc_str), 500)]
                chunks.extend(doc_chunks)
        
        if chunks:
            rag_add_docs(chunks)
            return jsonify({"added": len(chunks)}), 200
        
        return jsonify({"error": "No valid documents to add"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============= MODEL & OPTIMIZATION ENDPOINTS =============

@app.route("/model/select", methods=["POST"])
def select_model():
    """Switch between CPU-optimized models.
    
    Body: {"model": "fast|balanced|accurate"}
    - fast: TinyLLaMA (1.1B), ~600MB, fastest
    - balanced: Neural-Chat (7B), ~4GB, good quality
    - accurate: Mistral (7B), ~4GB, best quality
    """
    try:
        data = request.get_json() or {}
        model_key = data.get("model", "fast")
        
        if model_key not in AVAILABLE_MODELS:
            return jsonify({"error": f"Available models: {list(AVAILABLE_MODELS.keys())}"}), 400
        
        new_model = AVAILABLE_MODELS[model_key]
        global OLLAMA_MODEL
        OLLAMA_MODEL = new_model
        
        return jsonify({"model": OLLAMA_MODEL, "key": model_key}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/model/list", methods=["GET"])
def list_models():
    """List available models and their specs."""
    return jsonify({
        "available": AVAILABLE_MODELS,
        "current": OLLAMA_MODEL,
        "specs": {
            "tinyllama": {"params": "1.1B", "size": "~600MB", "vram": "minimal", "cpu": "yes"},
            "neural-chat": {"params": "7B (quantized)", "size": "~4GB", "vram": "1GB", "cpu": "yes"},
            "mistral": {"params": "7B", "size": "~4GB", "vram": "2GB", "cpu": "yes"},
        }
    }), 200


@app.route("/query-stream", methods=["POST"])
def query_stream():
    """Stream response tokens for low-latency perception.
    
    Body: {"query": "your question"}
    Returns: Server-Sent Events (SSE) with tokens
    """
    try:
        data = request.get_json() or {}
        query = (data.get("query") or "").strip()
        
        if not query:
            return jsonify({"error": "No query provided"}), 400
        
        def stream_tokens():
            try:
                # Use ollama streaming mode
                prompt = f"Q: {query}\nA:"
                response = ollama.generate(
                    model=OLLAMA_MODEL,
                    prompt=prompt,
                    stream=True,
                    options={"num_threads": OLLAMA_NUM_THREADS}
                )
                
                for chunk in response:
                    token = chunk.get("response", "")
                    if token:
                        yield f"data: {token}\n\n"
                
                yield "data: [DONE]\n\n"
            except Exception as e:
                yield f"data: [ERROR] {str(e)}\n\n"
        
        return Response(stream_tokens(), mimetype="text/event-stream")
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/", defaults={"path":""})
@app.route("/<path:path>")
def serve_frontend(path):
    index_path = os.path.join(FRONTEND_BUILD, "index.html")
    if os.path.exists(index_path):
        requested = os.path.join(FRONTEND_BUILD, path)
        if path and os.path.exists(requested) and os.path.isfile(requested):
            return send_from_directory(FRONTEND_BUILD, path)
        return send_from_directory(FRONTEND_BUILD, "index.html")
    return jsonify({"error":"Frontend build not found"}), 404

# ---------------- Socket.IO ----------------
@socketio.on("user_message")
def handle_user_message(data):
    query = (data.get("query") or "").strip()

    if not query:
        emit("bot_response", {"audio": None, "text": "No query provided."})
        return

    text = ""
    try:
        # Generate assistant response
        text = generate_response(query)
    except Exception as e:
        logging.exception("generate_response failed: %s", e)
        emit("bot_response", {"audio": None, "text": f"Error: {e}"})
        return

    try:
        # Generate TTS audio (if available)
        audio_path = speak_to_file(text)
        audio_url = None
        if audio_path:
            try:
                base = request.host_url.rstrip('/') if request and getattr(request, 'host_url', None) else 'http://127.0.0.1:5000'
            except Exception:
                base = 'http://127.0.0.1:5000'
            audio_url = f"{base}/audio/{os.path.basename(audio_path)}"

        emit("bot_response", {
            "text": text,
            "audio": audio_url,
            "verified": True
        })
    except Exception as e:
        logging.exception("Failed to send response: %s", e)
        emit("bot_response", {"audio": None, "text": "Internal error generating response."})

if __name__ == "__main__":
    logging.info("Assistant server starting on http://0.0.0.0:5000")

    # Best effort to use proper async server
    try:
        import eventlet
        eventlet.monkey_patch()
        async_mode = "eventlet"
        logging.info("Using eventlet (recommended)")
    except ImportError:
        try:
            from gevent import monkey
            monkey.patch_all()
            async_mode = "gevent"
            logging.info("Using gevent")
        except ImportError:
            async_mode = "threading"
            logging.warning("Using threading mode - install 'eventlet' for stability")

    # Re-initialize SocketIO with correct async mode
    socketio = SocketIO(app, cors_allowed_origins="*", async_mode=async_mode)

    @socketio.on("disconnect")
    def on_disconnect():
        logging.debug("Client disconnected")

    socketio.run(app, host="0.0.0.0", port=5000, debug=False) 
    # Replace your current socketio line with this:
async_mode = "eventlet" if eventlet else "threading"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode=async_mode)
