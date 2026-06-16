import os
import cv2
import threading
import shutil

from services.capture_store import CaptureStore
from services.config import settings
from services import event_repository
from services.monitoring_agent import MonitoringAgent
from services.ollama_client import OllamaClient
from services.schemas import ChatRequest, ChatResponse
from services.video_monitor import VideoMonitorService

from dotenv import load_dotenv

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# =========================
# APP
# =========================

load_dotenv()
app = FastAPI(title="AgroVision AI")

os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)
os.makedirs(os.getenv("SAVE_DIR"), exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
capture_store = CaptureStore(settings.captures_dir)
ollama_client = OllamaClient(base_url=settings.ollama_url, model_name=settings.ollama_model)
monitoring_agent = MonitoringAgent(ollama_client=ollama_client)
video_monitor = VideoMonitorService(
    camera_source=settings.camera_source,
    model_path=settings.model_path,
    confidence_threshold=settings.confidence_threshold,
    captures_dir=settings.captures_dir,
    event_repository=event_repository,
    target_classes=settings.target_classes,
    min_consecutive_frames=settings.min_consecutive_frames,
    alert_cooldown_seconds=settings.alert_cooldown_seconds,
)

# =========================
# EVENTO DE INICIALIZAÇÃO
# =========================
@app.on_event("startup")
def startup_event():
    event_repository.init_db()
    thread = threading.Thread(target=video_monitor._process_stream, daemon=True)
    thread.start()

# =========================
# ROTAS
# =========================
@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    events = event_repository.list_events(20)
    return templates.TemplateResponse("index.html", {"request": request, "events": events})


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "app": settings.app_name,
        "detector_backend": video_monitor.get_detector_backend(),
        "camera_source": settings.camera_source_raw,
        "ollama_url": settings.ollama_url,
        "events": event_repository.count_events(),
    }


@app.get("/events")
def get_events():
    return JSONResponse(content=event_repository.list_events(50))


@app.get("/captures")
def get_captures():
    return JSONResponse(content=capture_store.list_captures(30))


@app.get("/stream")
def stream_frame():
    return StreamingResponse(
        video_monitor.mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/frame")
def get_frame():
    frame_bytes = video_monitor.encode_current_frame()
    if frame_bytes is None:
        return JSONResponse(content={"message": "Ainda sem frame disponivel."}, status_code=503)
    return Response(content=frame_bytes, media_type="image/jpeg")


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    events = event_repository.list_events(payload.limit)
    summary = monitoring_agent.analyze(events=events, user_message=payload.message)
    return ChatResponse(
        agent_name=monitoring_agent.name,
        agent_role=monitoring_agent.role,
        events_analyzed=len(events),
        summary=summary,
    )


@app.post("/chat/stream")
def chat_stream(payload: ChatRequest):
    events = event_repository.list_events(payload.limit)

    def generator():
        for chunk in monitoring_agent.analyze_stream(events=events, user_message=payload.message):
            yield f"data: {chunk}\n\n"

    return StreamingResponse(generator(), media_type="text/event-stream")