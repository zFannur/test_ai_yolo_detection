import sys
import os
import math
import cv2

# Если у вас есть специфические пути к проекту, раскомментируйте при необходимости
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from ultralytics import YOLO
from scripts.config import MODEL_NAME, DEVICE

# -----------------------------------------------------------------------------
# ПУТИ К МОДЕЛЯМ
# -----------------------------------------------------------------------------
POSE_MODEL_PATH = 'yolo11s-pose.pt'  # модель позы (обязательно *-pose)
BOX_MODEL_PATH  = f'models/yolo11_product_spoilage_detection_{MODEL_NAME.split(".")[0]}/weights/best.pt'

# -----------------------------------------------------------------------------
# ПАРАМЕТРЫ
# -----------------------------------------------------------------------------
VIDEO_PATH    = "datasets/product_spoilage/video/train.mp4"
OUTPUT_PATH   = "datasets/product_spoilage/video/train_detect_kick.mp4"

WINDOW_WIDTH  = 480
WINDOW_HEIGHT = 640

DISTANCE_THRESHOLD   = 5.0   # px
FOOT_SPEED_THRESHOLD = 2.0   # px/frame
BOX_SPEED_THRESHOLD  = 1.0   # px/frame

# DISTANCE_THRESHOLD   = 50.0   # px
# FOOT_SPEED_THRESHOLD = 15.0   # px/frame
# BOX_SPEED_THRESHOLD  = 10.0   # px/frame

# -----------------------------------------------------------------------------
# ИНИЦИАЛИЗАЦИЯ МОДЕЛЕЙ
# -----------------------------------------------------------------------------
pose_model = YOLO(POSE_MODEL_PATH)  # модель позы
box_model  = YOLO(BOX_MODEL_PATH)   # модель для коробки

pose_model.to(DEVICE)
box_model.to(DEVICE)

print(f"[INFO] Используются модели:\n"
      f"      Pose  : {POSE_MODEL_PATH}\n"
      f"      Box   : {BOX_MODEL_PATH}\n"
      f"      Устройство : {DEVICE}")

# -----------------------------------------------------------------------------
# ОТКРЫВАЕМ ВИДЕО
# -----------------------------------------------------------------------------
if not os.path.exists(VIDEO_PATH):
    print(f"[ERROR] Файл не найден: {VIDEO_PATH}")
    sys.exit(1)

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"[ERROR] Невозможно открыть видео: {VIDEO_PATH}")
    sys.exit(1)

fps = cap.get(cv2.CAP_PROP_FPS)
print(f"[INFO] FPS входного видео: {fps:.2f}")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_video = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (WINDOW_WIDTH, WINDOW_HEIGHT))

# -----------------------------------------------------------------------------
# ХРАНИЛИЩЕ ПРЕДЫДУЩИХ КООРДИНАТ
# -----------------------------------------------------------------------------
foot_positions_prev = {}  # (person_id, foot_label) -> (x, y, frame_idx)
box_positions_prev  = {}  # box_id -> (cx, cy, frame_idx)

frame_idx = 0

# -----------------------------------------------------------------------------
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# -----------------------------------------------------------------------------
def euclidean_distance(p1, p2):
    return math.dist(p1, p2)

def compute_speed(old_pos, new_pos, old_frame, new_frame):
    """Вычисляем скорость (пиксели/кадр) между двумя точками (old_pos, new_pos)."""
    dt = new_frame - old_frame
    if dt <= 0:
        return 0.0
    dist = euclidean_distance(old_pos, new_pos)
    return dist / dt

def draw_limb(image, kpts, idx1, idx2, color=(0, 255, 0), thickness=2):
    """
    Рисует линию между kpts[idx1] и kpts[idx2], если координаты валидны.
    kpts — массив shape [17, 2] (или [17, 3]) с координатами COCO-ключевых точек.
    """
    if idx1 < 0 or idx2 < 0 or idx1 >= len(kpts) or idx2 >= len(kpts):
        return

    x1, y1 = kpts[idx1]
    x2, y2 = kpts[idx2]

    # Проверяем, что координаты не (0,0)
    if x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0:
        cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)

def is_valid_point(pt):
    """Простая проверка, что точка (x, y) имеет положительные координаты."""
    return pt[0] > 0 and pt[1] > 0

# -----------------------------------------------------------------------------
# ОСНОВНОЙ ЦИКЛ
# -----------------------------------------------------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1

    # Меняем размер кадра
    frame = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))

    # -----------------------------------------------------------------------------
    # 1) ПОЛУЧАЕМ KEYPOINTS ЧЕЛОВЕКА - через .predict(...) !!!
    # -----------------------------------------------------------------------------
    pose_results = pose_model.predict(
        frame,
        conf=0.3,
        max_det=10,
        device=DEVICE
    )

    # -----------------------------------------------------------------------------
    # 2) КОРОБКА (tracking)
    # -----------------------------------------------------------------------------
    box_results = box_model.track(
        frame,
        conf=0.5,
        persist=True,    # нужно для трекинга
        max_det=10,
        device=DEVICE
    )

    # -----------------------------------------------------------------------------
    # ОБРАБОТКА KEYPOINTS (рисуем руки, ноги, лодыжки)
    # -----------------------------------------------------------------------------
    foot_positions_current = {}  # (person_id, 'L'/'R') -> (x, y, frame_idx)

    # pose_results[0] содержит объекты детекции (boxes, keypoints, и т.д.)
    # Для позы: results[0].keypoints -> Keypoints
    if len(pose_results) > 0 and pose_results[0].keypoints is not None:
        # keypoints.xy -> список [N], где N - число людей; каждый элемент shape [17,2]
        all_person_kpts = pose_results[0].keypoints.xy
        for person_idx, kpts in enumerate(all_person_kpts):
            # Рисуем руки:
            # Левая рука: (5->7->9)
            draw_limb(frame, kpts, 5, 7, (0, 255, 0), 2)
            draw_limb(frame, kpts, 7, 9, (0, 255, 0), 2)
            # Правая рука: (6->8->10)
            draw_limb(frame, kpts, 6, 8, (0, 255, 0), 2)
            draw_limb(frame, kpts, 8, 10, (0, 255, 0), 2)

            # Рисуем ноги:
            # Левая нога: (11->13->15)
            draw_limb(frame, kpts, 11, 13, (0, 255, 0), 2)
            draw_limb(frame, kpts, 13, 15, (0, 255, 0), 2)
            # Правая нога: (12->14->16)
            draw_limb(frame, kpts, 12, 14, (0, 255, 0), 2)
            draw_limb(frame, kpts, 14, 16, (0, 255, 0), 2)

            # Лодыжки (15,16)
            left_ankle  = kpts[15]
            right_ankle = kpts[16]

            if is_valid_point(left_ankle):
                cv2.circle(frame, (int(left_ankle[0]), int(left_ankle[1])), 5, (0, 0, 255), -1)
                foot_positions_current[(person_idx, 'L')] = (left_ankle[0], left_ankle[1], frame_idx)

            if is_valid_point(right_ankle):
                cv2.circle(frame, (int(right_ankle[0]), int(right_ankle[1])), 5, (0, 0, 255), -1)
                foot_positions_current[(person_idx, 'R')] = (right_ankle[0], right_ankle[1], frame_idx)

    # -----------------------------------------------------------------------------
    # ОБРАБОТКА КОРОБКИ (tracking)
    # -----------------------------------------------------------------------------
    box_positions_current = {}
    if len(box_results) > 0 and len(box_results[0].boxes) > 0:
        for b in box_results[0].boxes:
            box_id = b.id
            if box_id is None:
                continue
            box_id = int(box_id.item())

            x1, y1, x2, y2 = b.xyxy[0].tolist()
            cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            cv2.putText(frame, f"BoxID: {box_id}", (int(x1), int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            box_positions_current[box_id] = (cx, cy, frame_idx)

    # -----------------------------------------------------------------------------
    # ВЫЧИСЛЯЕМ СКОРОСТИ (НОГ И КОРОБКИ)
    # -----------------------------------------------------------------------------
    speeds_feet = {}
    for key_foot, (fx, fy, f_frame) in foot_positions_current.items():
        if key_foot in foot_positions_prev:
            old_x, old_y, old_f_frame = foot_positions_prev[key_foot]
            speed = compute_speed((old_x, old_y), (fx, fy), old_f_frame, f_frame)
        else:
            speed = 0.0
        speeds_feet[key_foot] = speed

    speeds_box = {}
    for box_id, (cx, cy, c_frame) in box_positions_current.items():
        if box_id in box_positions_prev:
            old_cx, old_cy, old_c_frame = box_positions_prev[box_id]
            speed_box = compute_speed((old_cx, old_cy), (cx, cy), old_c_frame, c_frame)
        else:
            speed_box = 0.0
        speeds_box[box_id] = speed_box

    # -----------------------------------------------------------------------------
    # ПРОВЕРЯЕМ "ПИНОК"
    # -----------------------------------------------------------------------------
    # Для каждого (person_idx, foot_label) находим ближайшую коробку
    for (pid, flabel), (fx, fy, f_frame) in foot_positions_current.items():
        best_box_id = None
        best_dist = float('inf')

        for b_id, (bcx, bcy, b_frame) in box_positions_current.items():
            dist = euclidean_distance((fx, fy), (bcx, bcy))
            if dist < best_dist:
                best_dist = dist
                best_box_id = b_id

        if best_box_id is not None and best_dist < DISTANCE_THRESHOLD:
            foot_speed = speeds_feet.get((pid, flabel), 0.0)
            box_speed  = speeds_box.get(best_box_id, 0.0)

            if foot_speed > FOOT_SPEED_THRESHOLD and box_speed > BOX_SPEED_THRESHOLD:
                cv2.putText(frame, "KICK!!!",
                            (int(fx), int(fy)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

    # -----------------------------------------------------------------------------
    # СОХРАНЯЕМ ТЕКУЩИЕ ПОЗИЦИИ
    # -----------------------------------------------------------------------------
    foot_positions_prev = foot_positions_current
    box_positions_prev  = box_positions_current

    # -----------------------------------------------------------------------------
    # ВЫВОД
    # -----------------------------------------------------------------------------
    out_video.write(frame)
    cv2.imshow("Kick Detection (Predict Pose + Track Box)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out_video.release()
cv2.destroyAllWindows()

print(f"[INFO] Готово! Видео сохранено в: {OUTPUT_PATH}")
