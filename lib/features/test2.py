import cv2
import numpy as np
from ultralytics import YOLO
import time

# Пути и параметры
VIDEO_PATH = "datasets/long_delay_shelf/video/train.mp4"  # Укажите путь к вашему видео
SEGMENTATION_MODEL_PATH = "yolo11s-seg.pt"  # Путь к модели YOLOv11 для сегментации
LOG_FILE = "hand_in_shelf_events.log"
CONFIRMATION_THRESHOLD = 10  # Число кадров для подтверждения
HAND_CLASS_ID = 0  # ID класса "рука" (проверьте в вашей модели)

class HandInShelfDetector:
    def __init__(self):
        # Загружаем модель сегментации
        self.model = YOLO(SEGMENTATION_MODEL_PATH)
        self.confirmation_counter = {}  # Счётчик кадров для подтверждения
        self.last_hand_event = {}  # Последнее событие для каждой руки

    def process_frame(self, frame):
        hands = self._detect_hands(frame)
        self._analyze_position(frame, hands)
        return frame

    def _detect_hands(self, frame):
        """Детектируем руки с помощью YOLOv11."""
        hands = []
        results = self.model.predict(frame, conf=0.3, iou=0.4, verbose=False)
        if len(results) > 0 and results[0].masks is not None:
            masks = results[0].masks.data.cpu().numpy()  # Маски объектов
            classes = results[0].boxes.cls.cpu().numpy()  # Классы объектов
            for i, mask in enumerate(masks):
                if classes[i] == HAND_CLASS_ID:  # Проверяем, что это рука
                    hands.append(mask)
        return hands

    def _analyze_position(self, frame, hands):
        """Определяем положение руки относительно зон полок."""
        frame_width = frame.shape[1]
        mid_x = frame_width // 2  # Середина кадра

        for hand_idx, hand_mask in enumerate(hands):
            # Находим координаты пикселей маски руки
            hand_pixels = np.where(hand_mask > 0)
            if len(hand_pixels[0]) == 0:
                continue
            mean_x = np.mean(hand_pixels[1])  # Средняя X-координата
            shelf_side = "left" if mean_x < mid_x else "right"

            key = (hand_idx, shelf_side)
            self.confirmation_counter[key] = self.confirmation_counter.get(key, 0) + 1

            # Проверяем, достаточно ли кадров для подтверждения
            if self.confirmation_counter[key] >= CONFIRMATION_THRESHOLD:
                prev_side = self.last_hand_event.get(key, None)
                if prev_side != shelf_side:
                    self._draw_text(frame, hand_mask, shelf_side)
                    self._log_event(f"Рука {hand_idx} в зоне {shelf_side}")
                    self.last_hand_event[key] = shelf_side
            else:
                self.confirmation_counter[key] = 0
                self.last_hand_event[key] = None

    def _draw_text(self, frame, hand_mask, shelf_side):
        """Отображаем текст на кадре."""
        hand_pixels = np.where(hand_mask > 0)
        if len(hand_pixels[0]) > 0:
            mean_x = int(np.mean(hand_pixels[1]))
            mean_y = int(np.mean(hand_pixels[0]))
            cv2.putText(frame, f"Рука в коробке ({shelf_side})",
                        (mean_x, mean_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    def _log_event(self, message):
        """Записываем событие в лог-файл."""
        with open(LOG_FILE, "a") as f:
            f.write(f"{time.strftime('%H:%M:%S')} | {message}\n")

def main():
    detector = HandInShelfDetector()
    cap = cv2.VideoCapture(VIDEO_PATH)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (480, 640))
        processed_frame = detector.process_frame(frame)

        # Линия разделения зон
        cv2.line(frame, (frame.shape[1] // 2, 0), (frame.shape[1] // 2, frame.shape[0]), (255, 255, 255), 1)

        cv2.imshow("Hand Detection", processed_frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()