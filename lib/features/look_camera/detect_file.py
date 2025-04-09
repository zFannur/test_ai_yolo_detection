import sys
import os
import cv2
import time
from ultralytics import YOLO
from lib.core.config import MODEL_NAME, DEVICE

# Формируем путь к обученной модели по шаблону
MODEL_PATH = f'models/yolo11_fire_smoke_detection_{MODEL_NAME.split(".")[0]}/weights/best.pt'
print(f"Используем модель: {MODEL_PATH} на устройстве: {DEVICE}")

# Загружаем обученную модель
model = YOLO(MODEL_PATH)
model.to(DEVICE)
model.overrides["conf"] = 0.5
model.overrides["show"] = False

# Пути к видео
VIDEO_PATH = "datasets/fire_smoke/video/input2.mp4"   # Входное видео
OUTPUT_PATH = "datasets/fire_smoke/video/output2.mp4"   # Выходной файл

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Не удалось открыть видео: {VIDEO_PATH}")
        return

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 25

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

    print("Начало детекции огня и дыма...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Выполнение детекции объектов на кадре
        results = model.predict(frame, conf=0.5, verbose=False)

        # Для каждого обнаруженного результата (обычно один результат на кадр)
        for result in results:
            if result.boxes is None:
                continue
            # Проходим по всем обнаруженным bounding box
            for box in result.boxes:
                cls = int(box.cls[0])
                label = model.names.get(cls, str(cls))
                conf = float(box.conf[0])
                # Получаем координаты bounding box: (x1, y1, x2, y2)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # Рисуем прямоугольник
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Рисуем подпись
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                # Логирование обнаружения
                print(f"{time.strftime('%H:%M:%S')} - Detected {label} with confidence {conf:.2f} at {(x1, y1, x2, y2)}")

        writer.write(frame)
        cv2.imshow("Fire and Smoke Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
