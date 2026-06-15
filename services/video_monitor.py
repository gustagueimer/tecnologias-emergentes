import os
import cv2
import time
import uuid
import threading
import json
from numpy import ndarray
from dotenv import load_dotenv
from datetime import datetime
from collections import defaultdict
from .event_repository import save_event
from ultralytics import YOLO

# =========================
# FUNÇÕES DE DETECÇÃO
# =========================

load_dotenv()

model = YOLO(os.getenv("MODEL_PATH"))

last_frame = None
last_frame_lock = threading.Lock()

detection_state = defaultdict(int)
last_alert_time = defaultdict(lambda: 0.0)

def draw_box(frame, x1, y1, x2, y2, label, conf):
    text = f"{label} {conf:.2f}"
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(
        frame,
        text,
        (x1, max(20, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )


def should_alert(label: str):
    now = time.time()
    return (now - last_alert_time[label]) > int(os.getenv("ALERT_COOLDOWN_SECONDS"))


def process_stream():
    global last_frame

    cap = cv2.VideoCapture(int(os.getenv("CAMERA_SOURCE")))

    if not cap.isOpened():
        print("Erro ao abrir câmera.")
        return

    print("Câmera iniciada com sucesso.")

    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(1)
            continue

        results = model(frame, conf=float(os.getenv("CONFIDENCE_THRESHOLD")), verbose=False)

        found_labels_in_frame = set()
        best_conf_by_label = {}

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                label = model.names[cls_id]

                if label not in set(os.getenv("TARGET_CLASSES").split()):
                    continue

                found_labels_in_frame.add(label)

                if label not in best_conf_by_label or conf > best_conf_by_label[label]:
                    best_conf_by_label[label] = conf

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                draw_box(frame, x1, y1, x2, y2, label, conf)

        for label in set(os.getenv("TARGET_CLASSES").split()):
            if label in found_labels_in_frame:
                detection_state[label] += 1
            else:
                detection_state[label] = 0

        for label in found_labels_in_frame:
            if detection_state[label] >= int(os.getenv("MIN_CONSECUTIVE_FRAMES")) and should_alert(label):
                event_id = str(uuid.uuid4())[:8]
                filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{label}_{event_id}.jpg"
                filepath = os.path.join(os.getenv("SAVE_DIR"), filename)

                cv2.imwrite(filepath, frame)
                image_path = f"/static/captures/{filename}"

                confidence = best_conf_by_label.get(label, 0.0)
                save_event(event_id, label, confidence, image_path)

                last_alert_time[label] = time.time()
                print(f"[ALERTA] {label} detectado. Evidência salva em {filepath}")

        with last_frame_lock:
            last_frame = frame.copy()

        time.sleep(0.05)

def returnLastFrame() -> (ndarray | None):
    global last_frame
    return last_frame