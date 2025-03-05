import sys
import os
import cv2
import time
from collections import defaultdict
import numpy as np

# Если у вас своя структура каталогов:
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from ultralytics import YOLO
from lib.core.config import MODEL_NAME, DEVICE

# Задаём устройство
DEVICE = "cuda"

# Конфигурация путей
VIDEO_PATH = "datasets/long_delay_shelf/video/train.mp4"
SEGMENTATION_MODEL_PATH = "путь/к/модели_сегментации_рук.pt"  # Укажите путь к модели сегментации
LOG_FILE = "hand_in_shelf_events.log"

# Параметры
CONFIRMATION_THRESHOLD = 10  # Число кадров для подтверждения
FPS = 30

class HandInShelfDetector:
    def __init__(self):
        # Загружаем модель сегментации
        self.segmentation_model = YOLO(SEGMENTATION_MODEL_PATH)
        # Состояния для фиксации события для каждой руки
        self.last_hand_event = {}  # Ключ: (hand_idx, side) -> shelf_side или None
        self.confirmation_counter = {}  # Ключ: (hand_idx, side) -> число кадров подтверждения

    def process_frame(self, frame):
        hands = self._detect_hands(frame)
        self._analyze_interactions(frame, hands)
        return frame

    def _detect_hands(self, frame):
        """Детектирует руки с помощью модели сегментации."""
        hands = []
        results = self.segmentation_model.predict(frame, conf=0.3, iou=0.4, verbose=False, device=DEVICE)
        if len(results) > 0 and results[0].masks is not None:
            masks = results[0].masks.data.cpu().numpy()  # Маски сегментации
            for i, mask in enumerate(masks):
                # Предполагаем, что класс 0 соответствует руке
                if results[0].boxes.cls[i] == 0:  # Убедитесь, что класс соответствует руке
                    hands.append(mask)
        return hands

    def _analyze_interactions(self, frame, hands):
        """Анализирует, находится ли рука в зоне полки."""
        frame_width = frame.shape[1]
        mid_x = frame_width // 2  # Разделяем кадр на левую и правую половины

        for hand_idx, hand_mask in enumerate(hands):
            # Определяем положение руки
            hand_pixels = np.where(hand_mask > 0)
            if len(hand_pixels[0]) == 0:
                continue
            mean_x = np.mean(hand_pixels[1])  # Средняя координата X
            shelf_side = "left" if mean_x < mid_x else "right"

            key = (hand_idx, shelf_side)
            self.confirmation_counter[key] = self.confirmation_counter.get(key, 0) + 1

            if self.confirmation_counter[key] >= CONFIRMATION_THRESHOLD:
                prev_side = self.last_hand_event.get(key, None)
                if prev_side != shelf_side:
                    self._draw_hand_in_shelf(frame, hand_mask, shelf_side)
                    self._log_event(f"HAND IN SHELF - Hand {hand_idx} on {shelf_side} side")
                    self.last_hand_event[key] = shelf_side
            else:
                self.confirmation_counter[key] = 0
                self.last_hand_event[key] = None

    def _draw_hand_in_shelf(self, frame, hand_mask, shelf_side):
        """Отображает сообщение о нахождении руки в зоне полки."""
        hand_pixels = np.where(hand_mask > 0)
        if len(hand_pixels[0]) > 0:
            mean_x = int(np.mean(hand_pixels[1]))
            mean_y = int(np.mean(hand_pixels[0]))
            cv2.putText(frame, f"Hand in box ({shelf_side} side)",
                        (mean_x, mean_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    def _log_event(self, message):
        """Записывает событие в лог-файл."""
        with open(LOG_FILE, "a") as f:
            f.write(f"{time.strftime('%H:%M:%S')} | {message}\n")

def main():
    detector = HandInShelfDetector()
    cap = cv2.VideoCapture(VIDEO_PATH)
    print(f"Используется модель: {MODEL_NAME}, устройство: {DEVICE}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (480, 640))
        processed_frame = detector.process_frame(frame)

        # Линия разделения кадра
        cv2.line(frame, (frame.shape[1] // 2, 0), (frame.shape[1] // 2, frame.shape[0]), (255, 255, 255), 1)

        cv2.imshow("Hand In Shelf Detection", processed_frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()