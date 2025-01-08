# python scripts/open_product/detect.py

import sys
import os

# Добавляем путь к корневой директории
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from ultralytics import YOLO
import cv2
from scripts.config import MODEL_NAME, EPOCHS, IMG_SIZE, DEVICE, DETECTION_DISTANCE

# Путь к обученной модели
MODEL_PATH = f'models/yolo11_fall_detection_{MODEL_NAME.split(".")[0]}/weights/best.pt'

# Загрузка обученной модели
model = YOLO(MODEL_PATH)
model.to(DEVICE)

print(f"Используется модель: {MODEL_NAME}, устройство: {DEVICE}")

# Инициализация видеопотока из файла
VIDEO_PATH = 'datasets/fall/video/train.mp4'

# Проверка существования файла
if os.path.exists(VIDEO_PATH):
    print("Путь корректный, файл найден.")
else:
    print("Файл не найден. Проверьте путь.")

cap = cv2.VideoCapture(VIDEO_PATH)

# Задаем разрешение окна
WINDOW_WIDTH, WINDOW_HEIGHT = 480, 640

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Изменение размера кадра до 640x480
    frame = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))

    # Обнаружение объектов с помощью модели YOLO
    results = model.track(frame, conf=0.5)

    bag_boxes = []
    product_boxes = []

    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls)
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            if cls == 0:  # Класс 'bag'
                bag_boxes.append((x1, y1, x2, y2))
                # Рисуем прямоугольник вокруг пакета
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, 'Look_camera', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (255, 0, 0), 2)
            elif cls == 1:  # Класс 'count_product'
                product_boxes.append((x1, y1, x2, y2))
                # Рисуем прямоугольник вокруг продукта
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, 'Normal', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (0, 255, 0), 2)

    # Отображение кадра
    cv2.imshow('Look camera Product Detection', frame)

    # Выход по нажатию клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
