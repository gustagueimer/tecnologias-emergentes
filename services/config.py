import os
from dotenv import load_dotenv

load_dotenv()

class settings():
    app_name = "pindamonhangaba"
    model_path = os.getenv("MODEL_PATH")
    confidence_threshold = float(os.getenv("CONFIDENCE_THRESHOLD"))
    save_dir = os.getenv("SAVE_DIR")
    captures_dir = os.getenv("SAVE_DIR")
    db_path = os.getenv("DB_PATH")

    target_classes = set(os.getenv("TARGET_CLASSES").split())

    min_consecutive_frames = int(os.getenv("MIN_CONSECUTIVE_FRAMES"))
    alert_cooldown_seconds = int(os.getenv("ALERT_COOLDOWN_SECONDS"))   

    ollama_url = os.getenv("OLLAMA_URL")
    ollama_model = os.getenv("OLLAMA_MODEL")
    ollama_timeout = int(os.getenv("OLLAMA_TIMEOUT"))
    ollama_keep_alive = os.getenv("OLLAMA_KEEP_ALIVE")
    agent_event_limit = int(os.getenv("AGENT_EVENT_LIMIT"))

    camera_source = os.getenv("CAMERA_SOURCE")
    camera_reconnect_seconds = int(os.getenv("CAMERA_RECONNECT_SECONDS"))