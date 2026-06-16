import threading
import time
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any
from numpy import ndarray

import cv2

from services import event_repository


class VideoMonitorService:
    def __init__(
        self,
        camera_source: Any,
        model_path: str,
        confidence_threshold: float,
        captures_dir: Path,
        event_repository: any,
        target_classes: set[str],
        min_consecutive_frames: int,
        alert_cooldown_seconds: int,
    ):
        self.camera_source = camera_source
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.captures_dir = captures_dir
        self.event_repository = event_repository
        self.target_classes = target_classes
        self.min_consecutive_frames = min_consecutive_frames
        self.alert_cooldown_seconds = alert_cooldown_seconds

        self.model = None
        self.detector_backend = "iniciando"
        self.last_frame = None
        self.last_frame_lock = threading.Lock()
        self.detection_state = defaultdict(int)
        self.last_alert_time = defaultdict(lambda: 0.0)

    def start(self) -> None:
        thread = threading.Thread(target=self._process_stream, daemon=True)
        thread.start()

    def get_detector_backend(self) -> str:
        return self.detector_backend

    def encode_current_frame(self) -> bytes | None:
        with self.last_frame_lock:
            if self.last_frame is None:
                return None
            ok, buffer = cv2.imencode(".jpg", self.last_frame)
            if not ok:
                return None
            return buffer.tobytes()

    def mjpeg_generator(self):
        while True:
            frame_bytes = self.encode_current_frame()
            if frame_bytes is None:
                time.sleep(0.1)
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )
            time.sleep(0.05)

    def _load_model(self) -> None:
        try:
            from ultralytics import YOLO

            self.model = YOLO(self.model_path)
            self.detector_backend = "yolo"
        except Exception as exc:
            self.detector_backend = "indisponivel"
            self.model = None
            print(f"Falha ao carregar YOLO: {exc}")

    def _build_status_frame(self, message: str, width: int = 640, height: int = 360):
        frame = cv2.cvtColor(
            cv2.merge([
                cv2.UMat(height, width, cv2.CV_8UC1, 24),
                cv2.UMat(height, width, cv2.CV_8UC1, 24),
                cv2.UMat(height, width, cv2.CV_8UC1, 24),
            ]).get(),
            cv2.COLOR_BGR2RGB,
        )
        cv2.putText(frame, "AgroVision AI", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
        cv2.putText(frame, message, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 180, 255), 2)
        return frame

    def _open_camera(self):
        if isinstance(self.camera_source, str):
            for backend in (cv2.CAP_FFMPEG, cv2.CAP_ANY):
                cap = cv2.VideoCapture(self.camera_source, backend)
                if cap.isOpened():
                    print(f"Stream aberto: source={self.camera_source}, backend={backend}")
                    return cap
            return None

        candidates = [
            (self.camera_source, cv2.CAP_DSHOW),
            (self.camera_source, cv2.CAP_MSMF),
            (self.camera_source, cv2.CAP_ANY),
            (1, cv2.CAP_DSHOW),
            (1, cv2.CAP_MSMF),
        ]
        for source, backend in candidates:
            cap = cv2.VideoCapture(source, backend)
            if cap.isOpened():
                print(f"Camera aberta: source={source}, backend={backend}")
                return cap
        return None

    def _draw_box(self, frame, x1, y1, x2, y2, label, conf):
        text = f"{label} {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, text, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


    def should_alert(self, label: str):
        now = time.time()
        return (now - self.last_alert_time[label]) > self.alert_cooldown_seconds


    def _process_stream(self) -> None:
        self._load_model()
        if self.model is None:
            with self.last_frame_lock:
                self.last_frame = self._build_status_frame("YOLO indisponivel")
            return

        cap = self._open_camera()
        if cap is None:
            with self.last_frame_lock:
                self.last_frame = self._build_status_frame("Camera indisponivel")
            return

        while True:
            ok, frame = cap.read()
            if not ok:
                cap.release()
                with self.last_frame_lock:
                    self.last_frame = self._build_status_frame("Reconectando stream...")
                time.sleep(1)
                cap = self._open_camera()
                if cap is None:
                    time.sleep(1)
                continue

            found_labels: set[str] = set()
            best_conf: dict[str, float] = {}

            results = self.model(frame, conf=self.confidence_threshold, verbose=False)
            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue

                for box in boxes:
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    label = self.model.names[cls_id]

                    if label not in self.target_classes:
                        continue

                    found_labels.add(label)
                    best_conf[label] = max(conf, best_conf.get(label, 0.0))
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    self._draw_box(frame, x1, y1, x2, y2, label, conf)

            tracked = set(self.target_classes) | set(self.detection_state.keys())
            for label in tracked:
                self.detection_state[label] = self.detection_state[label] + 1 if label in found_labels else 0

            for label in found_labels:
                if self.detection_state[label] >= self.min_consecutive_frames and self.should_alert(label):
                    event_id = str(uuid.uuid4())[:8]
                    filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{label}_{event_id}.jpg"
                    filepath = f"{self.captures_dir} / {filename}"
                    cv2.imwrite(str(filepath), frame)
                    image_path = f"/static/captures/{filename}"
                    self.event_repository.save_event(event_id, label, best_conf.get(label, 0.0), image_path)
                    self.last_alert_time[label] = time.time()

            with self.last_frame_lock:
                self.last_frame = frame.copy()

            time.sleep(0.05)

    def returnLastFrame() -> (ndarray | None):
        global last_frame
        return last_frame