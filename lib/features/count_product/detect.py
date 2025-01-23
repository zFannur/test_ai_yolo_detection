from ultralytics import YOLO
import cv2
import os
import numpy as np
from config import MODEL_NAME, DEVICE, DETECTION_DISTANCE

# Путь к обученной модели
MODEL_PATH = f'models/yolo11_count_product_detection_{MODEL_NAME.split(".")[0]}/weights/best.pt'

# Загрузка обученной модели
model = YOLO(MODEL_PATH)
model.to(DEVICE)

print(f"Используется модель: {MODEL_NAME}, устройство: {DEVICE}")

# Инициализация видеопотока из файла
VIDEO_PATH = 'datasets/count_product/video/train.mp4'

# Проверка существования файла
if os.path.exists(VIDEO_PATH):
    print("Путь корректный, файл найден.")
else:
    print("Файл не найден. Проверьте путь.")

cap = cv2.VideoCapture(VIDEO_PATH)

# Задаем разрешение окна
#WINDOW_WIDTH, WINDOW_HEIGHT = 640, 480
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
                cv2.putText(frame, 'Bag', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (255, 0, 0), 2)
            elif cls == 1:  # Класс 'count_product'
                product_boxes.append((x1, y1, x2, y2))
                # Рисуем прямоугольник вокруг продукта
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, 'Product', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (0, 255, 0), 2)

    # Проверка расстояния между пакетом и продуктом
    for (x1_prod, y1_prod, x2_prod, y2_prod) in product_boxes:
        product_center = ((x1_prod + x2_prod) // 2, (y1_prod + y2_prod) // 2)
        for (x1_bag, y1_bag, x2_bag, y2_bag) in bag_boxes:
            bag_center = ((x1_bag + x2_bag) // 2, (y1_bag + y2_bag) // 2)
            distance = np.linalg.norm(np.array(product_center) - np.array(bag_center))

            # Если расстояние меньше порога, считаем, что продукт в пакете
            if distance < DETECTION_DISTANCE:  # Настройте порог
                cv2.putText(frame, 'Product in Bag', (x1_bag, y1_bag - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (0, 0, 255), 2)
                # Обновляем цвет прямоугольника вокруг пакета
                cv2.rectangle(frame, (x1_bag, y1_bag), (x2_bag, y2_bag), (0, 0, 255), 2)

    # Отображение кадра
    cv2.imshow('Bag and Product Detection', frame)

    # Выход по нажатию клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
