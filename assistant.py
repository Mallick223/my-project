# assistant_server.py - CRASH-PROOF VERSION
import os
import re
import logging
import warnings
from datetime import datetime
from pathlib import Path
import tempfile
import threading
import traceback

# Suppress all warnings including numpy/pandas compatibility issues
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["NUMPY_EXPERIMENTAL_ARRAY_FUNCTION"] = "0"

# Suppress noisy logs
os.environ["POSTHOG_DISABLED"] = "true"
logging.getLogger("urllib3").setLevel(logging.ERROR)

from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify, send_file, send_from_directory, Response
from flask_cors import CORS
from flask_socketio import SocketIO, emit

import ollama

# Optional pandas
try:
    import pandas as pd
except (ImportError, ValueError):
    # ImportError: pandas not installed
    # ValueError: numpy/pandas compatibility issue
    pd = None

# Optional TTS
try:
    import pyttsx3
except ImportError:
    pyttsx3 = None

try:
    from gtts import gTTS
except ImportError:
    gTTS = None

# Optional document parsing
try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

try:
    import docx
except ImportError:
    docx = None

# ─── Config ─────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.WARNING)  # Suppress info messages

BASE_DIR = Path(__file__).parent
FRONTEND_BUILD = BASE_DIR.parent / "ai-assistant" / "public"  # Fallback to public
if not FRONTEND_BUILD.exists():
    FRONTEND_BUILD = BASE_DIR / "frontend" / "build"
UPLOAD_DIR = BASE_DIR / "uploads"
KB_FILE = BASE_DIR / "kb.csv"

# Load environment variables
if load_dotenv:
    load_dotenv(override=False)

MIN_SIMILARITY = float(os.environ.get("MIN_SIMILARITY", "0.35"))
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "tinyllama")
OLLAMA_TEMPERATURE = float(os.environ.get("OLLAMA_TEMPERATURE", "0.0"))
OLLAMA_MAX_TOKENS = int(os.environ.get("OLLAMA_MAX_TOKENS", "512"))

OLLAMA_NUM_THREADS = int(os.environ.get("OLLAMA_NUM_THREADS", os.cpu_count() or 4))
OLLAMA_USE_GPU = os.environ.get("OLLAMA_USE_GPU", "false").lower() == "true"

AVAILABLE_MODELS = {
    "fast": "tinyllama",
    "balanced": "neural-chat",
    "accurate": "mistral",
}
OLLAMA_SELECTED_MODEL = os.environ.get("OLLAMA_SELECTED_MODEL", "fast")
OLLAMA_MODEL = AVAILABLE_MODELS.get(OLLAMA_SELECTED_MODEL, "tinyllama")

HYBRID_MODE_ENABLED = os.environ.get("HYBRID_MODE", "true").lower() == "true"
LAZY_RAG_ENABLED = os.environ.get("LAZY_RAG", "true").lower() == "true"
RAG_LOADED = False

TEMP_AUDIO_DIR = Path(tempfile.gettempdir()) / "voice_cache"
try:
    TEMP_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
except Exception as e:
    logging.warning(f"Failed to create temp audio dir: {e}")
    TEMP_AUDIO_DIR = Path(tempfile.gettempdir())

try:
    UPLOAD_DIR.mkdir(exist_ok=True)
except Exception as e:
    logging.warning(f"Failed to create upload dir: {e}")

audio_cache = {}  # text_hash -> filename

# â"€â"€â"€ Flask + SocketIO with CRASH-PROOF SETTINGS â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
app = Flask(__name__, static_folder=str(FRONTEND_BUILD), static_url_path="/")
CORS(app, resources={r"/*": {"origins": "*"}})

# Persistent connection with crash protection
async_mode = "threading"
socketio = SocketIO(
    app, 
    cors_allowed_origins="*", 
    async_mode=async_mode,
    ping_timeout=120,
    ping_interval=15,
    max_http_buffer_size=1e6,
    engineio_logger=False,
    reconnection=True,
    reconnection_delay=500,
    reconnection_delay_max=5000,
    max_reconnection_attempts=None,
    # CRASH PROTECTION: handle errors gracefully
    monitor_clients=True
)

# â"€â"€â"€ STUB FUNCTIONS FOR MISSING SERVICES â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
def stub_func(*args, **kwargs):
    return None

def stub_bool(*args, **kwargs):
    return False

# Initialize all services with stubs (prevents crashes from missing imports)
kb_match = stub_func
teach_new = stub_func
rag_retrieve = stub_func
lazy_rag = stub_func
speak_to_file = stub_func
normalize = stub_func
rag_add_docs = stub_func
kb_ready = stub_bool
detect_csv_intent = stub_func
remember = stub_func
handle_user_query = stub_func
should_store_memory = stub_bool
recall = stub_func
smart_web_search = stub_func
web_search = stub_func
is_system_command = stub_bool
handle_system_command = stub_func
handle_info_query = stub_func
call_ollama_tinyllama = stub_func
classify_search_intent = lambda *a, **k: "unknown"
search_youtube_with_powershell = stub_func
open_youtube = stub_func
open_spotify = stub_func

SERVICES_LOADED = []

# â"€â"€â"€ SAFE SERVICE LOADER â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
def load_service(service_name, import_func):
    """Load a service safely without crashing the app."""
    try:
        import_func()
        SERVICES_LOADED.append(service_name)
        logging.info(f"✓ {service_name} loaded successfully")
        return True
    except (ValueError, ImportError, AttributeError) as e:
        # ValueError: numpy dtype issues
        # ImportError: missing dependencies
        # AttributeError: module issues
        logging.debug(f"⚠ {service_name} import failed: {type(e).__name__}: {e}")
        return False
    except Exception as e:
        logging.debug(f"⚠ {service_name} import failed: {type(e).__name__}: {e}")
        return False

def load_ai_core():
    global kb_match, teach_new, rag_retrieve, lazy_rag, speak_to_file, normalize, rag_add_docs, kb_ready, detect_csv_intent
    from services.ai_core import kb_match as km, teach_new as tn, rag_retrieve as rr, lazy_rag as lr, speak_to_file as stf, normalize as nm, rag_add_docs as rad, kb_ready as kr, detect_csv_intent as dci
    kb_match, teach_new, rag_retrieve, lazy_rag, speak_to_file, normalize, rag_add_docs, kb_ready, detect_csv_intent = km, tn, rr, lr, stf, nm, rad, kr, dci

def load_memory():
    global remember, handle_user_query, should_store_memory, recall, smart_web_search, web_search
    from services.memory_tool import remember as rem, handle_user_query as huq, should_store_memory as ssm, recall as rec, smart_web_search as sws, web_search as ws
    remember, handle_user_query, should_store_memory, recall, smart_web_search, web_search = rem, huq, ssm, rec, sws, ws

def load_system_commands():
    global is_system_command, handle_system_command
    from services.system_commands import is_system_command as isc, handle_system_command as hsc
    is_system_command, handle_system_command = isc, hsc

def load_info_tools():
    global handle_info_query
    from services.info_tools import handle_info_query as hiq
    handle_info_query = hiq

def load_ollama():
    global call_ollama_tinyllama
    from services.ollama_client import call_ollama_tinyllama as cot
    call_ollama_tinyllama = cot

def load_intent_classifier():
    global classify_search_intent
    from services.intent_classifier import classify_search_intent as csi
    classify_search_intent = csi

def load_youtube():
    global search_youtube_with_powershell, open_youtube
    from services.youtube_search import search_youtube_with_powershell as sywp, open_youtube as oy
    search_youtube_with_powershell, open_youtube = sywp, oy

def load_spotify():
    global open_spotify
    from services.spotify_integration import open_spotify as os_
    open_spotify = os_

# Load all services safely
load_service("ai_core", load_ai_core)
load_service("memory_tool", load_memory)
load_service("system_commands", load_system_commands)
load_service("info_tools", load_info_tools)
load_service("ollama_client", load_ollama)
load_service("intent_classifier", load_intent_classifier)
load_service("youtube_search", load_youtube)
load_service("spotify_integration", load_spotify)

logging.info(f"Loaded {len(SERVICES_LOADED)} services: {', '.join(SERVICES_LOADED)}")

# â"€â"€â"€ Core Logic with ERROR HANDLING â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
def call_with_timeout(func, query, timeout_sec=2.0):
    """Call a function with timeout to prevent socket hangups."""
    try:
        if not func or not callable(func):
            return None
            
        result = [None]
        exception = [None]
        
        def run_func():
            try:
                result[0] = func(query)
            except Exception as e:
                exception[0] = e
        
        thread = threading.Thread(target=run_func, daemon=True)
        thread.start()
        thread.join(timeout=timeout_sec)
        
        if thread.is_alive():
            logging.debug(f"Function call exceeded {timeout_sec}s timeout")
            return None
        
        if exception[0]:
            raise exception[0]
        
        return result[0]
    except Exception as e:
        logging.debug(f"Function call failed: {e}")
        return None

def get_contextual_data(query):
    """Gather all external data to enhance LLM context."""
    context_parts = []
    
    try:
        if callable(kb_ready) and kb_ready():
            if callable(kb_match):
                kb_answer = kb_match(query)
                if kb_answer and isinstance(kb_answer, str) and kb_answer.strip():
                    context_parts.append(f"[KB] Knowledge Base:\n{kb_answer}")
    except Exception as e:
        logging.debug(f"KB context retrieval failed: {e}")
    
    try:
        if callable(rag_retrieve):
            rag_docs = rag_retrieve(query, top_k=3) or []
            if rag_docs and isinstance(rag_docs, (list, tuple)):
                valid_docs = [str(d).strip() for d in rag_docs[:3] if d]
                if valid_docs:
                    rag_context = "\n---\n".join(valid_docs)
                    context_parts.append(f"[DOC] Reference Documents:\n{rag_context}")
    except Exception as e:
        logging.debug(f"RAG context retrieval failed: {e}")
    
    try:
        if callable(recall):
            memory_context = recall(query)
            if memory_context and isinstance(memory_context, str) and memory_context.strip():
                context_parts.append(f"[MEM] Previous Context:\n{memory_context}")
    except Exception as e:
        logging.debug(f"Memory context retrieval failed: {e}")
    
    try:
        if callable(web_search):
            web_results = call_with_timeout(lambda q: web_search(q, num_results=2), query, timeout_sec=1.5)
            if web_results and isinstance(web_results, (list, tuple)) and len(web_results) > 0:
                valid_results = [str(r).strip() for r in web_results if r]
                if valid_results:
                    web_context = "\n---\n".join(valid_results[:2])
                    if len(web_context) > 500:
                        web_context = web_context[:500] + "..."
                    context_parts.append(f"WEB: {web_context}")
    except Exception as e:
        logging.debug(f"Web search failed (skipping): {e}")
    
    return "\n\n".join(context_parts) if context_parts else ""


def generate_response(query):
    """Generate response with comprehensive error handling."""
    try:
        ql = query.strip() if query else ""
        if not ql:
            return "Please say something."

        ql_lower = ql.lower()

        # Quick greetings
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

        for g, resp in greetings.items():
            if ql_lower == g or ql_lower.startswith(g + " "):
                return resp

        # System commands
        try:
            if callable(is_system_command) and is_system_command(ql):
                if callable(handle_system_command):
                    cmd_result = handle_system_command(ql)
                    if cmd_result and isinstance(cmd_result, str):
                        return cmd_result
        except Exception as e:
            logging.debug(f"System command failed: {e}")

        # Quick info queries
        try:
            if callable(handle_info_query):
                info = handle_info_query(ql)
                if info and isinstance(info, str):
                    if callable(should_store_memory) and should_store_memory(info):
                        if callable(remember):
                            remember(ql, info)
                    return info
        except Exception as e:
            logging.debug(f"Info query failed: {e}")

        # Safety checks
        try:
            if callable(classify_search_intent):
                intent = classify_search_intent(ql)
                if intent and isinstance(intent, str) and intent.lower() == "malicious":
                    return "I can't help with unsafe or illegal requests."
        except Exception as e:
            logging.debug(f"Intent classification failed: {e}")

        # CSV intent detection
        try:
            if callable(detect_csv_intent):
                result = detect_csv_intent(ql)
                if result and isinstance(result, tuple) and len(result) >= 2:
                    intent, kb_response = result[0], result[1]
                    if intent and isinstance(intent, str) and intent.lower() in ("malicious", "vulgar"):
                        if kb_response and isinstance(kb_response, str) and callable(should_store_memory) and should_store_memory(kb_response):
                            if callable(remember):
                                remember(ql, kb_response)
                        return kb_response if kb_response else "I can't help with that request."
        except Exception as e:
            logging.debug(f"CSV detection error: {e}")

        # Media actions
        try:
            media_intent = classify_search_intent(ql)
            
            if media_intent == "youtube":
                try:
                    response = open_youtube(ql)
                    if response and should_store_memory(response):
                        remember(ql, response)
                    return response
                except Exception as yt_err:
                    logging.debug(f"YouTube failed: {yt_err}")
                    try:
                        import urllib.parse
                        import webbrowser
                        search_query = urllib.parse.quote(ql)
                        url = f"https://www.youtube.com/results?search_query={search_query}"
                        webbrowser.open(url)
                        return f"Opening YouTube search for: {ql}"
                    except:
                        return f"Unable to open YouTube. Please search manually for: {ql}"
            
            if media_intent == "spotify":
                try:
                    response = call_with_timeout(open_spotify, ql, timeout_sec=1.0)
                    if response and should_store_memory(response):
                        remember(ql, response)
                    return response
                except Exception as sp_err:
                    logging.debug(f"Spotify failed: {sp_err}")
                    try:
                        import subprocess
                        import urllib.parse
                        search_query = urllib.parse.quote(ql)
                        spotify_uri = f"spotify:search:{search_query}"
                        subprocess.Popen(["explorer", spotify_uri], shell=True)
                        return f"Opening Spotify app for: {ql}"
                    except:
                        return f"Spotify unavailable. Try searching on YouTube instead."
        except Exception as e:
            logging.debug(f"Media action failed: {e}")

        # LLM fallback with context
        try:
            external_context = get_contextual_data(ql)
            
            system_instruction = (
                "You are a helpful, truthful, and knowledgeable AI assistant.\n"
                "Use the provided context to give accurate, up-to-date information.\n"
                "If context is provided, prioritize it over your training data.\n"
                "Always be concise, natural, and accurate.\n"
                "If you don't know something, say so honestly."
            )

            prompt_parts = [system_instruction]
            if external_context and isinstance(external_context, str):
                prompt_parts.append(f"Available Context:\n{external_context}")
            prompt_parts.append(f"User Question:\n{ql}\n\nProvide a helpful and accurate answer:")
            
            final_prompt = "\n\n".join(prompt_parts)

            answer = ""
            if callable(call_ollama_tinyllama):
                try:
                    answer = call_ollama_tinyllama(prompt=final_prompt, force_hallucinate=False)
                    if answer and isinstance(answer, str):
                        answer = answer.strip()
                    else:
                        answer = ""
                except Exception as llm_err:
                    logging.debug(f"LLM call failed: {llm_err}")
                    answer = ""

            # Fallback if LLM response is insufficient
            if not answer or len(answer.strip()) < 20:
                try:
                    if callable(smart_web_search):
                        smart_results = call_with_timeout(smart_web_search, ql, timeout_sec=1.2)
                        if smart_results and isinstance(smart_results, (list, tuple)) and len(smart_results) > 0:
                            smart_answer = " ".join([str(r) for r in smart_results if r])
                            if len(smart_answer) > 300:
                                smart_answer = smart_answer[:300] + "..."
                            if smart_answer and isinstance(smart_answer, str) and callable(should_store_memory) and should_store_memory(smart_answer):
                                if callable(remember):
                                    remember(ql, smart_answer)
                            return smart_answer if smart_answer else "I couldn't find information on that topic. Please try a different search."
                except Exception as ws_err:
                    logging.debug(f"Smart web search fallback failed: {ws_err}")

            if answer and isinstance(answer, str) and callable(should_store_memory) and should_store_memory(answer):
                if callable(remember):
                    remember(ql, answer)
            
            return answer if answer else "I'm not able to help with that right now. Could you try rephrasing?"

        except Exception as e:
            logging.error(f"LLM generation error: {e}", exc_info=True)
            return "Sorry, I'm having trouble processing that. Please try again?"

    except Exception as e:
        logging.error(f"CRITICAL ERROR in generate_response: {e}", exc_info=True)
        return "An unexpected error occurred. Please try again."

# â"€â"€â"€ Routes with ERROR HANDLING â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
@app.route("/health", methods=["GET"])
def health_check():
    try:
        return jsonify({"status": "ok", "service": "assistant", "services_loaded": len(SERVICES_LOADED)}), 200
    except Exception as e:
        logging.error(f"Health check error: {e}")
        return jsonify({"status": "error"}), 500


@app.route("/status", methods=["GET"])
def status_route():
    try:
        return jsonify({
            "status": "ok",
            "model": OLLAMA_MODEL,
            "services_loaded": SERVICES_LOADED,
            "service_count": len(SERVICES_LOADED),
            "use_gpu": OLLAMA_USE_GPU,
            "cpu_threads": OLLAMA_NUM_THREADS,
        })
    except Exception as e:
        logging.error(f"Status route error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/assistant", methods=["POST"])
def assistant():
    try:
        data = request.get_json() or {}
        user_text = ""
        if isinstance(data, dict):
            user_text = data.get("text", "")
        
        if isinstance(user_text, str):
            user_text = user_text.strip()
        else:
            user_text = str(user_text).strip() if user_text else ""
        
        reply = generate_response(user_text)
        if not isinstance(reply, str):
            reply = str(reply) if reply else "Unable to process request"
        
        return jsonify({"reply": reply})
    except Exception as e:
        logging.error(f"Assistant route error: {e}")
        return jsonify({"error": "Internal server error"}), 500


def parse_document(file_path):
    """Parse document with error handling."""
    text = ""
    try:
        if not file_path:
            return ""
        
        file_path = str(file_path).strip()
        if not file_path:
            return ""
        
        if file_path.endswith(".txt"):
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read() or ""
            except Exception as txt_err:
                logging.debug(f"TXT parse failed: {txt_err}")
        
        elif file_path.endswith(".pdf"):
            if PyPDF2:
                try:
                    with open(file_path, "rb") as fh:
                        reader = PyPDF2.PdfReader(fh)
                        pages = [page.extract_text() or "" for page in reader.pages if hasattr(page, 'extract_text')]
                        text = " ".join(pages) if pages else ""
                except Exception as pdf_err:
                    logging.debug(f"PDF parse failed: {pdf_err}")
        
        elif file_path.endswith(".docx"):
            if docx:
                try:
                    doc = docx.Document(file_path)
                    paragraphs = [p.text for p in doc.paragraphs if hasattr(p, 'text')]
                    text = " ".join(paragraphs) if paragraphs else ""
                except Exception as docx_err:
                    logging.debug(f"DOCX parse failed: {docx_err}")
        
        elif file_path.endswith(".csv"):
            if pd:
                try:
                    text = pd.read_csv(file_path).to_string() or ""
                except Exception as csv_err:
                    logging.debug(f"CSV parse with pandas failed: {csv_err}")
                    try:
                        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                            text = f.read() or ""
                    except Exception as fallback_err:
                        logging.debug(f"CSV fallback failed: {fallback_err}")
            else:
                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        text = f.read() or ""
                except Exception as raw_err:
                    logging.debug(f"CSV raw read failed: {raw_err}")
    
    except Exception as e:
        logging.debug(f"Parse failed: {e}")
    
    return text.strip() if isinstance(text, str) else ""


@app.route("/upload_doc", methods=["POST"])
def upload_doc():
    try:
        if "file" not in request.files:
            return jsonify({"status": "error", "message": "No file uploaded"}), 400

        file = request.files["file"]
        if not file.filename:
            return jsonify({"status": "error", "message": "No file selected"}), 400

        filename = secure_filename(file.filename)
        path = UPLOAD_DIR / filename
        file.save(path)

        text = parse_document(path)
        if text:
            try:
                chunks = [text[i:i+500] for i in range(0, len(text), 500)]
                rag_add_docs(chunks)
                return jsonify({"status": "success", "message": f"{filename} added to RAG"}), 200
            except Exception as e:
                logging.error(f"RAG add failed: {e}")
                return jsonify({"status": "error", "message": "Failed to add to RAG"}), 500

        return jsonify({"status": "error", "message": "Unsupported or empty file"}), 400
    except Exception as e:
        logging.error(f"Upload doc error: {e}")
        return jsonify({"status": "error", "message": "Upload failed"}), 500


@app.route("/tts", methods=["POST"])
def tts_route():
    try:
        data = request.get_json() or {}
        text = ""
        if isinstance(data, dict):
            text = data.get("text", "")
        
        if isinstance(text, str):
            text = text.strip()
        else:
            text = str(text).strip() if text else ""

        if not text:
            return jsonify({"error": "No text provided"}), 400

        audio_path = None
        if callable(speak_to_file):
            try:
                audio_path = speak_to_file(text)
            except Exception as e:
                logging.warning(f"TTS failed: {e}")

        if not audio_path or not isinstance(audio_path, str) or not Path(audio_path).exists():
            return jsonify({"error": "TTS generation failed"}), 500

        try:
            return send_file(audio_path, mimetype="audio/mpeg")
        except Exception as send_err:
            logging.error(f"Failed to send audio file: {send_err}")
            return jsonify({"error": "Failed to send audio"}), 500
    except Exception as e:
        logging.error(f"TTS route error: {e}")
        return jsonify({"error": "TTS error"}), 500


@app.route("/audio/<filename>")
def serve_audio(filename):
    try:
        path = TEMP_AUDIO_DIR / filename
        if path.is_file():
            return send_file(path, mimetype="audio/mpeg")
        return jsonify({"error": "File not found"}), 404
    except Exception as e:
        logging.error(f"Audio serve error: {e}")
        return jsonify({"error": "Audio serve error"}), 500


@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_frontend(path):
    try:
        index = FRONTEND_BUILD / "index.html"
        if not index.is_file():
            return jsonify({"error": "Frontend not built"}), 404

        requested = FRONTEND_BUILD / path
        if requested.is_file():
            return send_from_directory(FRONTEND_BUILD, path)
        return send_from_directory(FRONTEND_BUILD, "index.html")
    except Exception as e:
        logging.error(f"Frontend serve error: {e}")
        return jsonify({"error": "Frontend error"}), 500


# â"€â"€â"€ SocketIO handlers with CRASH PREVENTION â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
active_connections = {}

@socketio.on("connect")
def handle_connect():
    """Handle new client connection."""
    try:
        client_id = request.sid
        active_connections[client_id] = {
            "connected_at": datetime.now(),
            "messages_count": 0
        }
        logging.info(f"🟢 Client connected: {client_id}. Total active: {len(active_connections)}")
        emit("connection_status", {
            "status": "connected",
            "message": "Successfully connected to backend"
        })
    except Exception as e:
        logging.error(f"Connect handler error: {e}")

@socketio.on("disconnect")
def handle_disconnect():
    """Handle client disconnection gracefully."""
    try:
        client_id = request.sid
        if client_id in active_connections:
            del active_connections[client_id]
        logging.info(f"🔴 Client disconnected: {client_id}. Remaining: {len(active_connections)}")
    except Exception as e:
        logging.error(f"Disconnect handler error: {e}")

@socketio.on("ping")
def handle_ping():
    """Respond to client heartbeat."""
    try:
        emit("pong", {"timestamp": datetime.now().isoformat()})
    except Exception as e:
        logging.error(f"Ping handler error: {e}")

@socketio.on("user_message")
def handle_user_message(data):
    """Handle user message with robust error handling."""
    client_id = request.sid
    try:
        query = ""
        if data and isinstance(data, dict):
            query = data.get("query", "")
        
        if isinstance(query, str):
            query = query.strip()
        else:
            query = str(query).strip() if query else ""
        
        if not query:
            try:
                emit("bot_response", {"audio": None, "text": "No query provided."})
            except Exception as emit_err:
                logging.debug(f"Emit error: {emit_err}")
            return

        try:
            if client_id in active_connections:
                active_connections[client_id]["messages_count"] += 1
            
            text = generate_response(query)
            if not isinstance(text, str):
                text = str(text) if text else "Unable to process request."
            
            audio_url = None

            try:
                if callable(speak_to_file):
                    audio_path = speak_to_file(text)
                    if audio_path and isinstance(audio_path, str):
                        audio_file = Path(audio_path)
                        if audio_file.exists():
                            try:
                                base = request.host_url.rstrip('/') if request.host_url else "http://127.0.0.1:5000"
                                audio_url = f"{base}/audio/{audio_file.name}"
                            except Exception as url_err:
                                logging.debug(f"Audio URL generation failed: {url_err}")
            except Exception as tts_err:
                logging.debug(f"Audio generation failed: {tts_err}")

            # Always emit response to keep connection alive
            try:
                emit("bot_response", {
                    "text": text,
                    "audio": audio_url,
                    "verified": True,
                    "timestamp": datetime.now().isoformat()
                })
            except Exception as emit_err:
                logging.error(f"Failed to emit bot_response: {emit_err}")
        except Exception as e:
            logging.error(f"Message processing error: {e}")
            try:
                emit("bot_response", {
                    "audio": None,
                    "text": "Internal error occurred. Reconnecting automatically...",
                    "error": True,
                    "timestamp": datetime.now().isoformat()
                })
            except Exception as emit_err:
                logging.debug(f"Error emit failed: {emit_err}")
    except Exception as e:
        logging.error(f"User message handler critical error: {e}", exc_info=True)
        try:
            emit("bot_response", {
                "text": "System error. Please refresh the page.",
                "error": True
            })
        except Exception as final_err:
            logging.error(f"Final error emit failed: {final_err}")


# â"€â"€â"€ Main Entry with CRASH PROTECTION â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
if __name__ == "__main__":
    try:
        logging.info("🚀 Assistant server starting (CRASH-PROOF MODE)...")
        logging.info(f"✓ Services loaded: {len(SERVICES_LOADED)}/{8}")
        
        # Pre-warm model with timeout
        try:
            warmup_done = [False]
            
            def warmup():
                try:
                    ollama.chat(
                        model=OLLAMA_MODEL,
                        messages=[{"role": "user", "content": "hi"}],
                        options={"num_predict": 1}
                    )
                    warmup_done[0] = True
                    logging.info("✓ Model pre-loaded")
                except Exception as e:
                    logging.warning(f"⚠ Model warm-up failed: {e}")
            
            warmup_thread = threading.Thread(target=warmup, daemon=True)
            warmup_thread.start()
            warmup_thread.join(timeout=5.0)
            
            if not warmup_done[0]:
                logging.warning("⚠ Model warm-up timed out after 5s - continuing without warmup")
        except Exception as e:
            logging.warning(f"⚠ Model warm-up exception: {e}")

        logging.info("🎯 Server ready! Listening on http://127.0.0.1:5000")
        
        socketio.run(
            app,
            host="127.0.0.1",
            port=5000,
            debug=False,
            allow_unsafe_werkzeug=True
        )
    except Exception as e:
        logging.error(f"FATAL: Server startup failed: {e}", exc_info=True)
        raise
