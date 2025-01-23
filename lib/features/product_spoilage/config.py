# config.py (обновленный)
import torch
from pathlib import Path


class Config:
    # Пути
    BASE_DIR = Path(__file__).parent.parent.parent.parent

    MODELS_DIR = BASE_DIR / "models"
    DATA_DIR = BASE_DIR / "datasets"

    # Устройство
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Модели
    MODELS = {
        "pose": {
            "path": "yolo11s-pose.pt",
            "conf": 0.25,
            "iou": 0.4,
            "max_det": 10
        },
        "box": {
            "path": MODELS_DIR / "yolo11_product_spoilage_detection_yolo11s" / "weights/best.pt",
            "conf": 0.3,
            "tracker": "botsort.yaml"
        }
    }

    # Параметры событий
    EVENTS = {
        "kick": {
            "distance_threshold": 100,
            "foot_speed_threshold": 5,
            "box_accel_threshold": 5
        },
        "fall": {
            "speed_threshold": 5,
            "history_length": 5
        }
    }

    # Визуализация
    COLORS = {
        "left_foot": (0, 0, 255),
        "right_foot": (255, 0, 0),
        "box": (0, 255, 0),
        "fallen_box": (0, 0, 255),
        "speed_text": (255, 255, 0)
    }

    WINDOW_SIZE = (480, 640)  # (width, height)

    # Логирование
    LOG_FILE = BASE_DIR / "events.log"


config = Config()
