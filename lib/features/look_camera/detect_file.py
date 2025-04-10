import sys
import os
import cv2
import time
from ultralytics import YOLO
from lib.core.config import MODEL_NAME, DEVICE

# Формируем путь к обученной модели
MODEL_PATH = f'models/yolo11_fire_smoke_detection_{MODEL_NAME.split(".")[0]}/weights/best.pt'
print(f"Using model: {MODEL_PATH} on device: {DEVICE}")

# Загружаем модель
model = YOLO(MODEL_PATH)
model.to(DEVICE)
model.overrides["conf"] = 0.3
model.overrides["show"] = False

# Пути к видео
VIDEO_PATH = "datasets/fire_smoke/video/input.mp4"
OUTPUT_PATH = "datasets/fire_smoke/video/output.mp4"

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Failed to open video: {VIDEO_PATH}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 25

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

    print("Starting fire and smoke detection...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Выполняем предсказание
        results = model.predict(frame, conf=0.3, verbose=False)
        fire_detected = False

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                cls = int(box.cls[0])
                label = model.names.get(cls, str(cls))
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Рисуем прямоугольник и метку
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                # Проверка на огонь
                if label.lower() == "fire":
                    fire_detected = True
                    print(f"{time.strftime('%H:%M:%S')} - Detected {label} with confidence {conf:.2f} at {(x1, y1, x2, y2)}")

        # 🔥 Уведомление на экране
        if fire_detected:
            cv2.rectangle(frame, (10, 10), (300, 60), (0, 0, 255), -1)  # красный фон
            cv2.putText(frame, "FIRE DETECTED!", (20, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        writer.write(frame)
        cv2.imshow("Fire and Smoke Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
