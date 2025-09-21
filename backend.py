# assistant_server.py
from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import os
import re
import subprocess
import pandas as pd
import webbrowser
import random
import socket
import threading
import time
import logging
import ctypes
from ctransformers import AutoModelForCausalLM
import pyttsx3
from gtts import gTTS
import winreg
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz
import speech_recognition as sr
from pydub import AudioSegment
from pydub.utils import which
from io import BytesIO

logging.basicConfig(level=logging.INFO)

# ---------------- Config ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_BUILD = os.path.join(BASE_DIR, "frontend", "build")
KB_FILE = os.path.join(BASE_DIR, "kb.csv")
MIN_SIMILARITY = 0.35
TEMP_AUDIO_DIR = os.path.join(BASE_DIR, "temp_audio")
os.makedirs(TEMP_AUDIO_DIR, exist_ok=True)

# Make sure pydub finds ffmpeg if available
AudioSegment.converter = which("ffmpeg")
AudioSegment.ffmpeg = which("ffmpeg")
AudioSegment.ffprobe = which("ffprobe")

# ---------------- Flask ----------------
app = Flask(__name__, static_folder=FRONTEND_BUILD, static_url_path="/")
CORS(app)

# ---------------- Helpers ----------------
def internet_available(host="8.8.8.8", port=53, timeout=2):
    try:
        sock = socket.create_connection((host, port), timeout)
        sock.close()
        return True
    except:
        return False

# ---------------- Mini Orca (local LLM) ----------------
MINI_ORCA_MODEL = os.path.join(BASE_DIR, "models", "orca-mini-3b.gguf")
mini_orca_model = None
if os.path.exists(MINI_ORCA_MODEL):
    try:
        # Try two common constructors for ctransformers
        try:
            mini_orca_model = AutoModelForCausalLM(MINI_ORCA_MODEL)
        except TypeError:
            # fallback name
            mini_orca_model = AutoModelForCausalLM.from_pretrained(MINI_ORCA_MODEL)
        logging.info("✅ Mini Orca loaded successfully.")
    except Exception as e:
        mini_orca_model = None
        logging.warning("⚠️ Mini Orca failed to load: %s", e)
else:
    logging.info("❌ Mini Orca model not found at %s", MINI_ORCA_MODEL)

# ---------------- NLP & KB ----------------
def normalize(text):
    if not text:
        return ""
    t = text.lower()
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()

def load_kb(file=KB_FILE):
    if not os.path.exists(file):
        pd.DataFrame(columns=["input", "response"]).to_csv(file, index=False)
    return pd.read_csv(file).fillna("")

def build_tfidf(sentences):
    sentences = [s if isinstance(s, str) else "" for s in sentences]
    if not any(s.strip() for s in sentences):
        return None, None
    vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    X = vec.fit_transform(sentences)
    return vec, X

def best_match(query, vec, X, df):
    if df.empty or vec is None or X is None or not query:
        return None, 0.0
    qv = vec.transform([query])
    sims = cosine_similarity(qv, X).flatten()
    idx = int(sims.argmax())
    score = float(sims.max())
    if score >= MIN_SIMILARITY:
        return df.iloc[idx]["response"], score
    return None, score

def teach_new(query, answer, df):
    new_row = pd.DataFrame([{"input": query, "response": answer}])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(KB_FILE, index=False)
    return df

df = load_kb(KB_FILE)
vec, X = build_tfidf(df["input"].tolist()) if not df.empty else (None, None)

# ---------------- TTS ----------------
engine = pyttsx3.init()
engine.setProperty("rate", 170)

def speak_to_file(text):
    temp_file = os.path.join(TEMP_AUDIO_DIR, f"tts_{int(time.time())}.mp3")
    # prefer online gTTS if internet
    if internet_available():
        try:
            tts = gTTS(text=text, lang="en")
            tts.save(temp_file)
            return temp_file
        except Exception as e:
            logging.warning("gTTS failed, falling back to pyttsx3: %s", e)
    # fallback to pyttsx3
    engine.save_to_file(text, temp_file)
    engine.runAndWait()
    return temp_file

# ---------------- Apps (Windows) ----------------
BUILTIN_APPS = {
    "calculator": "calc.exe",
    "notepad": "notepad.exe",
    "paint": "mspaint.exe",
    "wordpad": "write.exe",
    "task manager": "taskmgr.exe",
    "control panel": "control.exe",
    "file explorer": "explorer.exe",
    "command prompt": "cmd.exe",
    "powershell": "powershell.exe",
    "settings": "ms-settings:"
}

SYSTEM_APPS = {}
INSTALLED_APPS = {}

def _clean_exe_path(exe_path):
    if not exe_path:
        return exe_path
    p = str(exe_path).strip().strip('"')
    # Remove trailing comma,index patterns like path,0
    if "," in p:
        parts = p.split(",")
        if len(parts[-1].strip()) <= 3 and parts[-1].strip().isdigit():
            p = ",".join(parts[:-1]).strip()
    return p

def index_installed_apps():
    global INSTALLED_APPS
    apps = {}
    reg_paths = [
        (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"),
        (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Wow6432Node\Microsoft\Windows\CurrentVersion\Uninstall"),
        (winreg.HKEY_CURRENT_USER, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall")
    ]
    for hive, reg_path in reg_paths:
        try:
            key = winreg.OpenKey(hive, reg_path, 0, winreg.KEY_READ | winreg.KEY_WOW64_64KEY)
        except Exception:
            try:
                key = winreg.OpenKey(hive, reg_path)
            except Exception:
                continue
        try:
            for i in range(winreg.QueryInfoKey(key)[0]):
                try:
                    sub_key_name = winreg.EnumKey(key, i)
                    sub_key = winreg.OpenKey(key, sub_key_name)
                    try:
                        name, _ = winreg.QueryValueEx(sub_key, "DisplayName")
                        exe_path = ""
                        try:
                            exe_path, _ = winreg.QueryValueEx(sub_key, "DisplayIcon")
                        except Exception:
                            pass
                        if not exe_path:
                            try:
                                exe_path, _ = winreg.QueryValueEx(sub_key, "UninstallString")
                            except Exception:
                                pass
                        if name and exe_path:
                            apps[normalize(name)] = _clean_exe_path(exe_path)
                    except Exception:
                        pass
                    finally:
                        try:
                            sub_key.Close()
                        except:
                            pass
                except Exception:
                    continue
        finally:
            try:
                key.Close()
            except:
                pass
    INSTALLED_APPS = apps
    logging.info("Indexed %d installed apps", len(INSTALLED_APPS))

def index_system_apps():
    global SYSTEM_APPS
    search_dirs = [r"C:\Program Files", r"C:\Program Files (x86)", r"C:\Windows\System32", r"C:\Windows"]
    apps = {}
    for base_dir in search_dirs:
        if not os.path.exists(base_dir):
            continue
        for root, dirs, files in os.walk(base_dir):
            for file in files:
                if file.lower().endswith(".exe"):
                    name = normalize(os.path.splitext(file)[0])
                    path = os.path.join(root, file)
                    if name not in apps:
                        apps[name] = path
    SYSTEM_APPS = apps
    logging.info("Indexed %d system apps", len(SYSTEM_APPS))

def _refresh_apps():
    index_installed_apps()
    index_system_apps()
    return len(INSTALLED_APPS), len(SYSTEM_APPS)

def auto_refresh_apps(interval=600):
    while True:
        _refresh_apps()
        logging.info("Auto-refreshed apps.")
        time.sleep(interval)

threading.Thread(target=auto_refresh_apps, args=(600,), daemon=True).start()

# ---------------- Open Apps & WhatsApp ----------------
def open_uwp_app(app_id):
    """Open UWP / Microsoft Store apps via PowerShell shell:AppsFolder\<AppID>"""
    try:
        cmd = ['powershell', '-NoProfile', '-Command', f'Start-Process "shell:AppsFolder\\{app_id}"']
        subprocess.run(cmd, check=True)
        return True, f"Opened UWP app {app_id}"
    except Exception as e:
        logging.warning("open_uwp_app failed for %s: %s", app_id, e)
        return False, f"Failed to open UWP app {app_id}: {e}"

def _try_open(exe_path):
    if not exe_path:
        return False, "No executable path provided."
    exe_path = os.path.expandvars(str(exe_path))
    try:
        # URIs & shell handlers
        if exe_path.lower().startswith(("http:", "https:", "ms-settings:", "shell:")):
            webbrowser.open(exe_path, new=2)
            return True, f"Opening {exe_path}..."
        # Existing file
        if os.path.exists(exe_path):
            os.startfile(exe_path)
            return True, f"Opening {exe_path}..."
        # fallback attempts
        try:
            subprocess.run(f'start "" "{exe_path}"', shell=True, check=False)
        except Exception:
            pass
        try:
            ctypes.windll.shell32.ShellExecuteW(None, "open", exe_path, None, None, 1)
        except Exception:
            pass
        return True, f"Attempted to open {exe_path}"
    except Exception as e:
        logging.exception("Failed to open app %s", exe_path)
        return False, f"Failed to open {exe_path}: {e}"

def open_whatsapp():
    """
    Try multiple methods to open WhatsApp:
      1) UWP AppID via PowerShell (shell:AppsFolder\<AppID>)
      2) Common EXE locations (%LocalAppData%\WhatsApp\WhatsApp.exe)
      3) Search WindowsApps folder for WhatsApp
      4) Fallback: return not found
    """
    # NOTE: AppID must match the installed package. If your AppID differs, update it here.
    common_appids = [
        "5319275A.WhatsAppDesktop_cv1g1gvanyjgm!App",  # common official ID
    ]
    for app_id in common_appids:
        ok, msg = open_uwp_app(app_id)
        if ok:
            return True, "Opening WhatsApp (UWP)..."

    # Check common EXE paths
    possible_paths = [
        os.path.expanduser(r"~\AppData\Local\WhatsApp\WhatsApp.exe"),
        os.path.join(os.environ.get("ProgramFiles", ""), "WhatsApp", "WhatsApp.exe"),
        os.path.join(os.environ.get("ProgramFiles(x86)", ""), "WhatsApp", "WhatsApp.exe")
    ]
    for p in possible_paths:
        if p and os.path.exists(p):
            return _try_open(p)

    # Search WindowsApps folder (may be protected; still attempt)
    store_path = os.path.join(os.environ.get("LOCALAPPDATA", ""), "Microsoft", "WindowsApps")
    if os.path.exists(store_path):
        for file in os.listdir(store_path):
            if "whatsapp" in normalize(file) and file.lower().endswith(".exe"):
                return _try_open(os.path.join(store_path, file))

    return False, "WhatsApp not found. If you have WhatsApp Desktop installed from the Microsoft Store, run `Get-StartApps | Select Name, AppID` in PowerShell to get the AppID and add it to the known AppIDs list."

def open_app(app_name):
    name_norm = normalize(app_name)

    # special-case whatsapp to try UWP/EXE flows
    if "whatsapp" in name_norm:
        return open_whatsapp()

    # 1. Builtin
    for n, exe in BUILTIN_APPS.items():
        if fuzz.partial_ratio(name_norm, normalize(n)) >= 70:
            return _try_open(exe)

    # 2. Installed apps from registry
    best_name, best_score = None, 0
    for n, exe in INSTALLED_APPS.items():
        score = fuzz.partial_ratio(name_norm, n)
        if score > best_score:
            best_name, best_score = n, score
    if best_name and best_score >= 70:
        return _try_open(INSTALLED_APPS[best_name])

    # 3. System apps (scanned .exe)
    best_name, best_score = None, 0
    for n, exe in SYSTEM_APPS.items():
        score = fuzz.partial_ratio(name_norm, n)
        if score > best_score:
            best_name, best_score = n, score
    if best_name and best_score >= 85:
        return _try_open(SYSTEM_APPS[best_name])
    if best_name and best_score >= 75:
        return False, f"Best match is '{best_name}' (score {best_score}). Be more specific."

    # 4. WindowsApps folder (store exes)
    store_path = os.path.join(os.environ.get("LOCALAPPDATA", ""), "Microsoft", "WindowsApps")
    if os.path.exists(store_path):
        for file in os.listdir(store_path):
            if name_norm in normalize(file) and file.lower().endswith(".exe"):
                return _try_open(os.path.join(store_path, file))

    # 5. Try raw command/path
    return _try_open(app_name)

# ---------------- Web Search ----------------
def web_search(query, open_in_browser=True):
    if not query:
        return False, "Empty query"
    url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
    if open_in_browser:
        try:
            webbrowser.open(url, new=2)
            return True, f"Opened browser search for '{query}'."
        except Exception as e:
            logging.warning("webbrowser.open failed: %s", e)
            return False, f"Failed to open browser: {e}"
    return True, url

# ---------------- Jokes ----------------
JOKES = [
    "Why don’t scientists trust atoms? Because they make up everything!",
    "Why did the math book look sad? Because it had too many problems.",
    "I told my computer I needed a break, and it said: 'No problem, I’ll go to sleep.'",
    "Why don’t programmers like nature? It has too many bugs.",
    "What’s a computer’s favorite beat? An algo-rhythm!"
]
def get_random_joke():
    return random.choice(JOKES)

# ---------------- Generate Response ----------------
def generate_response(query):
    q_norm = normalize(query)

    # KB
    if vec is not None and X is not None:
        kb_resp, score = best_match(q_norm, vec, X, df)
        if kb_resp and score >= MIN_SIMILARITY:
            return kb_resp

    # detect search
    if any(token in q_norm for token in ("search ", "look up", "google ", "find ")):
        ok, msg = web_search(query)
        return msg if isinstance(msg, str) else str(msg)

    # local LLM
    if mini_orca_model:
        try:
            prompt = f"You are a helpful assistant. Answer concisely:\n{query}"
            resp = mini_orca_model(prompt, max_new_tokens=200)
            if isinstance(resp, (list, tuple)):
                resp = "".join(map(str, resp))
            return str(resp).strip()
        except Exception as e:
            logging.warning("Mini Orca inference error: %s", e)

    return "Sorry, I don't know the answer. Try rephrasing your question or teach me via /teach."

# ---------------- Routes ----------------
@app.route("/message", methods=["POST"])
def message_route():
    data = request.get_json() or {}
    user_input = (data.get("message", "") or "").strip()
    if not user_input:
        return jsonify({"response": "Please say something!", "opened": False})
    lowered = user_input.lower()
    if lowered == "ping":
        return jsonify({"response": "pong", "opened": True})
    if lowered.startswith("open "):
        ok, msg = open_app(user_input[5:].strip())
        return jsonify({"response": msg, "opened": ok})
    if lowered.startswith("search "):
        ok, msg = web_search(user_input[7:].strip())
        return jsonify({"response": msg, "opened": ok})
    if "joke" in lowered:
        return jsonify({"response": get_random_joke(), "opened": True, "source": "joke"})
    resp, score = best_match(normalize(user_input), vec, X, df) if vec else (None, 0)
    if resp and score >= MIN_SIMILARITY:
        return jsonify({"response": resp, "opened": True, "source": "kb", "score": score})
    final = generate_response(user_input)
    return jsonify({"response": final, "opened": True})

@app.route("/voice", methods=["POST"])
def voice_route():
    if "file" not in request.files:
        return jsonify({"error": "No audio file uploaded", "opened": False}), 400

    file = request.files["file"]
    temp_wav = os.path.join(TEMP_AUDIO_DIR, f"voice_{int(time.time())}.wav")
    try:
        # read bytes -> BytesIO for pydub
        audio_data = BytesIO(file.read())
        audio = AudioSegment.from_file(audio_data).set_channels(1).set_frame_rate(16000)
        audio.export(temp_wav, format="wav")

        recognizer = sr.Recognizer()
        with sr.AudioFile(temp_wav) as source:
            recorded_audio = recognizer.record(source)

        try:
            text = recognizer.recognize_google(recorded_audio)
        except sr.UnknownValueError:
            text = ""
        except sr.RequestError as e:
            return jsonify({"error": f"Google API error: {e}", "opened": False}), 500

        lowered = text.lower()
        if lowered.startswith("open "):
            ok, msg = open_app(text[5:].strip())
            return jsonify({"transcribed": text, "response": msg, "opened": ok})
        if lowered.startswith("search "):
            ok, msg = web_search(text[7:].strip())
            return jsonify({"transcribed": text, "response": msg, "opened": ok})

        resp, score = best_match(normalize(text), vec, X, df) if vec else (None, 0)
        if resp and score >= MIN_SIMILARITY:
            return jsonify({"transcribed": text, "response": resp, "opened": True, "source": "kb", "score": score})

        if mini_orca_model:
            try:
                response = mini_orca_model(text, max_new_tokens=200).strip()
                return jsonify({"transcribed": text, "response": response, "opened": True, "source": "mini_orca"})
            except Exception as e:
                logging.warning("Mini Orca inference failed: %s", e)
                return jsonify({"transcribed": text, "response": "Sorry, I cannot answer that right now.", "opened": False, "source": "error"})

        return jsonify({"transcribed": text, "response": "Sorry, I cannot answer that right now.", "opened": False, "source": "none"})

    except Exception as e:
        logging.exception("Voice processing error")
        return jsonify({"error": f"Voice processing failed: {e}", "opened": False}), 500

@app.route("/teach", methods=["POST"])
def teach_route():
    global df, vec, X
    data = request.get_json() or {}
    query = (data.get("query", "") or "").strip()
    answer = (data.get("answer", "") or "").strip()
    if not query or not answer:
        return jsonify({"error": "Both 'query' and 'answer' required"}), 400
    df = teach_new(query, answer, df)
    vec, X = build_tfidf(df["input"].tolist())
    return jsonify({"response": f"Learned: '{query}' → '{answer}'", "opened": True})

@app.route("/tts", methods=["POST"])
def tts_route():
    text = (request.get_json() or {}).get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400
    try:
        path = speak_to_file(text)
        return send_file(path, mimetype="audio/mpeg")
    except Exception as e:
        logging.exception("TTS failed")
        return jsonify({"error": str(e)}), 500

@app.route("/refresh_apps", methods=["POST"])
def refresh_apps_route():
    reg_count, sys_count = _refresh_apps()
    return jsonify({"response": f"Apps refreshed. Registry: {reg_count}, System: {sys_count}", "opened": True})

@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_frontend(path):
    # Serve React build if present, else helpful JSON
    index_path = os.path.join(FRONTEND_BUILD, "index.html")
    if os.path.exists(index_path):
        # If a specific file requested and exists, serve that; otherwise serve index.html
        requested = os.path.join(FRONTEND_BUILD, path)
        if path and os.path.exists(requested) and os.path.isfile(requested):
            return send_from_directory(FRONTEND_BUILD, path)
        return send_from_directory(FRONTEND_BUILD, "index.html")
    return jsonify({"error": "Frontend build not found"}), 404

# ---------------- Run ----------------
if __name__ == "__main__":
    # initial app indexing
    _refresh_apps()
    logging.info("Assistant server running at http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
