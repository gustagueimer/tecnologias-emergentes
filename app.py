import os
import cv2
import threading
import shutil
from services.video_monitor import process_stream, last_frame_lock, returnLastFrame
from services.event_repository import init_db, list_events
from dotenv import load_dotenv

from fastapi import FastAPI, Request, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response
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

# =========================
# EVENTO DE INICIALIZAÇÃO
# =========================
@app.on_event("startup")
def startup_event():
    init_db()
    thread = threading.Thread(target=process_stream, daemon=True)
    thread.start()

# =========================
# ROTAS
# =========================
@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    events = list_events(20)
    return templates.TemplateResponse("index.html", {"request": request, "events": events})


@app.get("/health")
def health():
    return {"status": "ok", "service": "AgroVision AI"}


@app.get("/events")
def get_events():
    return JSONResponse(content=list_events(50))


@app.get("/frame")
def get_frame():
    lost_frame = returnLastFrame()

    with last_frame_lock:
        if lost_frame is None:
            return JSONResponse(
                content={"message": "Ainda sem frame disponível."},
                status_code=503
            )

        success, buffer = cv2.imencode(".jpg", lost_frame)
        if not success:
            return JSONResponse(
                content={"message": "Erro ao converter frame."},
                status_code=500
            )

        return Response(content=buffer.tobytes(), media_type="image/jpeg")
    
@app.post("/upload", response_class=HTMLResponse)
async def upload_imagem(request: Request, file: UploadFile = File(...)):
    caminho_arquivo = os.path.join(os.getenv("SAVE_DIR"), file.filename)
    with open(caminho_arquivo, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return templates.TemplateResponse("index.html", {"request": request, "mensagem": f"Imagem enviada com sucesso: {file.filename}"})