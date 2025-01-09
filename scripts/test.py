import cv2
import torch
from ultralytics import YOLO

print("CUDA available:", torch.cuda.is_available())

# -----------------------------------------------------
# 1) ЗАГРУЗКА МОДЕЛИ
# -----------------------------------------------------
# Важно: должна быть модель *-pose, например "yolo11s-pose.pt"
POSE_MODEL_PATH = "yolo11s-pose.pt"
model = YOLO(POSE_MODEL_PATH)

# -----------------------------------------------------
# 2) ОТКРЫВАЕМ ВИДЕО
# -----------------------------------------------------
VIDEO_PATH = "datasets/product_spoilage/video/train.mp4"  # Или индекс камеры (например, 0)
WINDOW_WIDTH  = 480
WINDOW_HEIGHT = 640

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("[ERROR] Не удалось открыть видео/камеру:", VIDEO_PATH)
    exit()

# -----------------------------------------------------
# 3) СПИСОК ПАР КЛЮЧЕВЫХ ТОЧЕК (COCO) ДЛЯ РИСОВАНИЯ ЛИНИЙ
# -----------------------------------------------------
# Каждая пара (idx1, idx2) будет связана линией
# Ниже описаны руки и ноги:
#   - Левая рука (5 -> 7 -> 9)
#   - Правая рука (6 -> 8 -> 10)
#   - Левая нога (11 -> 13 -> 15)
#   - Правая нога (12 -> 14 -> 16)
LIMB_PAIRS = [
    (5, 7), (7, 9),     # Левая рука
    (6, 8), (8, 10),    # Правая рука
    (11, 13), (13, 15), # Левая нога
    (12, 14), (14, 16), # Правая нога
]

def draw_limb(image, x1, y1, x2, y2, color=(0, 255, 0), thickness=2):
    """Рисует линию между (x1,y1) и (x2,y2), если координаты валидны."""
    if x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0:
        cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)

# -----------------------------------------------------
# 4) ЦИКЛ ПО КАДРАМ
# -----------------------------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Изменяем размер
    frame = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))

    # -----------------------------------------------------
    # 4.1) МОДЕЛЬ (без трекинга!)
    # -----------------------------------------------------
    # Прогоняем через model.predict, чтобы гарантированно получить keypoints
    results = model.predict(frame, conf=0.3)

    # Если модель что-то нашла:
    if len(results) > 0:
        # results[0] - это инфо по первому (и единственному) кадру
        # keypoints - объект. В новых версиях ultralytics это: results[0].keypoints
        kpts_all = results[0].keypoints  # может быть None, если нет позы
        if kpts_all is not None:
            # kpts_all.xy: список [N] (по количеству людей), внутри shape [17, 2]
            # где 17 - число ключевых точек COCO
            # Пример: kpts_all.xy[i][j] - j-я точка у i-го человека
            persons_kpts = kpts_all.xy  # список numpy-массивов

            # -----------------------------------------------------
            # 4.2) РИСОВАНИЕ ДЛЯ КАЖДОГО ЧЕЛОВЕКА
            # -----------------------------------------------------
            for person_id, kpts in enumerate(persons_kpts):
                # kpts.shape -> (17, 2), kpts[j] = (x_j, y_j) для j в [0..16]

                # Рисуем линии
                for (idx1, idx2) in LIMB_PAIRS:
                    x1, y1 = kpts[idx1]
                    x2, y2 = kpts[idx2]
                    draw_limb(frame, x1, y1, x2, y2, color=(0, 255, 0), thickness=2)

                # Рисуем точки
                # - запястья (9, 10) и лодыжки (15, 16) делаем красными кружками
                for j, (kx, ky) in enumerate(kpts):
                    if kx > 0 and ky > 0:  # точка валидна
                        if j in [9, 10, 15, 16]:
                            cv2.circle(frame, (int(kx), int(ky)), 5, (0, 0, 255), -1)
                        else:
                            cv2.circle(frame, (int(kx), int(ky)), 5, (255, 0, 0), -1)
        else:
            # Нет keypoints
            pass

    # -----------------------------------------------------
    # 4.3) ПОКАЗЫВАЕМ РЕЗУЛЬТАТ
    # -----------------------------------------------------
    cv2.imshow("Arms & Legs Pose", frame)

    # Выход по 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
