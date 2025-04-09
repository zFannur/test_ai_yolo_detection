import cv2
import time
import math
import numpy as np
from ultralytics import YOLO

# ---------- ПАРАМЕТРЫ -------------
VIDEO_PATH = "datasets/hide_product/video/train.mp4"
OUTPUT_PATH = "datasets/hide_product/video/output.mp4"

FINAL_WIDTH = 480
FINAL_HEIGHT = 640
FPS_FALLBACK = 25

# Пороги
KNEE_ANGLE_THRESHOLD = 70      # угол в колене < 70° => сильно согнуты
GROIN_ANGLE_THRESHOLD = 100    # groin-угол < 100° => человек присел
TIME_THRESHOLD = 2.0           # если поза держится дольше 2 секунд
HIGHLIGHT_DURATION = 3.0       # подсветка на 3 секунды
MATCH_THRESHOLD = 50.0         # порог сопоставления детекций (по центроидам)

# Загружаем модель YOLO Pose
model = YOLO("yolo11m-pose.pt")  # или "yolo11n-pose.pt"
model.overrides["conf"] = 0.5
model.overrides["show"] = False
model.overrides["line_width"] = 0


class SuspiciousPoseDetector:
    """
    Для каждого обнаруженного человека вычисляются:
      - Углы локтей, коленей.
      - Groin-углы: вычисляются как угол между (колено, средняя точка бедер, плечо) для левой и правой сторон.
    Если оба колена < KNEE_ANGLE_THRESHOLD и хотя бы один groin-угол < GROIN_ANGLE_THRESHOLD,
    а такая поза держится > TIME_THRESHOLD секунд,
    для данного человека отображается событие (текст и красный контур по выпуклой оболочке) на HIGHLIGHT_DURATION секунд.
    Состояния для каждого человека сохраняются отдельно.
    """
    def __init__(self):
        # person_states: ключ – уникальный идентификатор (ID) человека, значение – словарь состояния
        self.person_states = {}

    def process_all(self, frame, kpts_all):
        """
        kpts_all: NumPy-массив формы (N, 17, 2), где N – количество обнаруженных людей.
        Сопоставляем текущие детекции с сохранёнными состояниями по центроидам.
        """
        h, w, _ = frame.shape
        current_ids = {}
        for i in range(kpts_all.shape[0]):
            person = kpts_all[i]
            if person.shape[0] < 17:
                continue
            centroid = np.mean(person, axis=0)  # (x,y)
            assigned_id = None
            for pid, state in self.person_states.items():
                prev_centroid = state.get("centroid", None)
                if prev_centroid is not None:
                    dist = np.linalg.norm(np.array(centroid) - np.array(prev_centroid))
                    if dist < MATCH_THRESHOLD:
                        assigned_id = pid
                        state["centroid"] = tuple(centroid)
                        break
            if assigned_id is None:
                assigned_id = max(self.person_states.keys(), default=-1) + 1
                self.person_states[assigned_id] = {"centroid": tuple(centroid),
                                                    "start_time": None,
                                                    "event_logged": False,
                                                    "highlight_end_time": None}
            current_ids[i] = assigned_id
            state = self.person_states[assigned_id]
            frame = self._process_person_for_state(frame, person, state)
        # Удаляем состояния для людей, которых нет на кадре
        keys_to_remove = [pid for pid in self.person_states if pid not in current_ids.values()]
        for pid in keys_to_remove:
            self.person_states.pop(pid)
        return frame

    def _process_person_for_state(self, frame, kpts_array, state):
        """
        Обрабатывает одного человека с ключевыми точками kpts_array (17,2) и состоянием state.
        Вычисляет углы, отрисовывает подписи и, если поза подозрительная, обновляет состояние для подсветки.
        """
        h, w, _ = frame.shape

        # Извлекаем ключевые точки согласно стандартной COCO:
        # Локти: левый (5,7,9), правый (6,8,10)
        lx_sh, ly_sh = kpts_array[5]
        lx_elb, ly_elb = kpts_array[7]
        lx_wri, ly_wri = kpts_array[9]

        rx_sh, ry_sh = kpts_array[6]
        rx_elb, ry_elb = kpts_array[8]
        rx_wri, ry_wri = kpts_array[10]

        # Колени: левое (11,13,15), правое (12,14,16)
        lx_hip, ly_hip = kpts_array[11]
        lx_kn, ly_kn = kpts_array[13]
        lx_ank, ly_ank = kpts_array[15]

        rx_hip, ry_hip = kpts_array[12]
        rx_kn, ry_kn = kpts_array[14]
        rx_ank, ry_ank = kpts_array[16]

        # Для groin-угла: средняя точка между бедрами
        groin = ((lx_hip + rx_hip) / 2, (ly_hip + ry_hip) / 2)

        # Вычисляем groin-углы:
        left_groin_angle = self._calculate_angle((lx_kn, ly_kn), groin, (lx_sh, ly_sh))
        right_groin_angle = self._calculate_angle((rx_kn, ry_kn), groin, (rx_sh, ry_sh))

        # Вычисляем углы локтей и коленей (для справки)
        left_elbow_angle = self._calculate_angle((lx_sh, ly_sh), (lx_elb, ly_elb), (lx_wri, ly_wri))
        right_elbow_angle = self._calculate_angle((rx_sh, ry_sh), (rx_elb, ry_elb), (rx_wri, ry_wri))
        left_knee_angle = self._calculate_angle((lx_hip, ly_hip), (lx_kn, ly_kn), (lx_ank, ly_ank))
        right_knee_angle = self._calculate_angle((rx_hip, ry_hip), (rx_kn, ry_kn), (rx_ank, ry_ank))

        # Отрисовка контрольных точек
        cv2.circle(frame, (int(lx_elb), int(ly_elb)), 3, (0,255,0), -1)
        cv2.circle(frame, (int(rx_elb), int(ry_elb)), 3, (0,255,0), -1)
        cv2.circle(frame, (int(lx_kn), int(ly_kn)), 3, (255,0,0), -1)
        cv2.circle(frame, (int(rx_kn), int(ry_kn)), 3, (255,0,0), -1)
        cv2.circle(frame, (int(groin[0]), int(groin[1])), 3, (0,255,255), -1)

        # Отрисовка текстовых подписей (шрифт 0.5)
        font_scale = 0.5
        color_blue = (255, 0, 0)
        color_red = (0, 0, 255)

        if left_elbow_angle is not None:
            cv2.putText(frame, f"LElb:{int(left_elbow_angle)}", (int(lx_elb), int(ly_elb)-8),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color_blue, 1)
        if right_elbow_angle is not None:
            cv2.putText(frame, f"RElb:{int(right_elbow_angle)}", (int(rx_elb), int(ry_elb)-8),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color_blue, 1)
        if left_knee_angle is not None:
            cv2.putText(frame, f"LKnee:{int(left_knee_angle)}", (int(lx_kn), int(ly_kn)-8),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color_blue, 1)
        if right_knee_angle is not None:
            cv2.putText(frame, f"RKnee:{int(right_knee_angle)}", (int(rx_kn), int(ry_kn)-8),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color_blue, 1)
        # Сдвигаем подписи для groin-углов, чтобы они не перекрывались
        if left_groin_angle is not None:
            cv2.putText(frame, f"LGroin:{int(left_groin_angle)}", (int(groin[0])-60, int(groin[1])-10),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color_red, 1)
        if right_groin_angle is not None:
            cv2.putText(frame, f"RGroin:{int(right_groin_angle)}", (int(groin[0])+20, int(groin[1])+10),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color_red, 1)

        # Логика подозрительной позы: если оба колена < порога и хотя бы один groin-угол < порога
        suspicious = False
        if (left_knee_angle is not None and right_knee_angle is not None and
            left_groin_angle is not None and right_groin_angle is not None):
            knees_ok = (left_knee_angle < KNEE_ANGLE_THRESHOLD and right_knee_angle < KNEE_ANGLE_THRESHOLD)
            groin_ok = (left_groin_angle < GROIN_ANGLE_THRESHOLD or right_groin_angle < GROIN_ANGLE_THRESHOLD)
            if knees_ok and groin_ok:
                suspicious = True

        now = time.time()
        if suspicious:
            if state["start_time"] is None:
                state["start_time"] = now
            elapsed = now - state["start_time"]
            cv2.putText(frame, f"Suspicious {elapsed:.1f}s", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,255,255), 2)
            if elapsed > TIME_THRESHOLD and not state["event_logged"]:
                state["event_logged"] = True
                cv2.putText(frame, "EVENT: squat pose!", (30, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale+0.2, (0,0,255), 2)
                state["highlight_end_time"] = time.time() + HIGHLIGHT_DURATION
        else:
            state["start_time"] = None
            state["event_logged"] = False

        # Отрисовка подсветки (контур по выпуклой оболочке) только для данного человека
        if state.get("highlight_end_time") is not None:
            if now < state["highlight_end_time"]:
                hull = self._get_hull(kpts_array)
                if hull is not None:
                    cv2.polylines(frame, [hull], isClosed=True, color=(0, 0, 255), thickness=2)
            else:
                state["highlight_end_time"] = None

        return frame

    def reset(self):
        self.person_states = {}

    @staticmethod
    def _calculate_angle(a, b, c):
        """
        Вычисляет угол ABC (в точке b) в градусах.
        Ограничивает значение косинуса в диапазоне [-1,1] для предотвращения math domain error.
        a, b, c – (x,y)
        """
        if not a or not b or not c:
            return None
        ba = (a[0] - b[0], a[1] - b[1])
        bc = (c[0] - b[0], c[1] - b[1])
        mag_ba = math.hypot(*ba)
        mag_bc = math.hypot(*bc)
        if mag_ba == 0 or mag_bc == 0:
            return None
        dot = ba[0]*bc[0] + ba[1]*bc[1]
        cos_angle = dot / (mag_ba * mag_bc)
        cos_angle = max(-1.0, min(1.0, cos_angle))
        angle = math.degrees(math.acos(cos_angle))
        return angle

    @staticmethod
    def _get_hull(kpts_array):
        """
        Принимает (17,2) NumPy-массив (x,y) – ключевые точки человека.
        Фильтрует нулевые точки и строит выпуклую оболочку (convex hull), возвращая массив точек.
        """
        valid_pts = []
        for (x, y) in kpts_array:
            if x > 0 and y > 0:
                valid_pts.append([x, y])
        if len(valid_pts) < 3:
            return None
        pts = np.array(valid_pts, dtype=np.int32).reshape(-1, 1, 2)
        hull = cv2.convexHull(pts)
        return hull


def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Не удалось открыть видео:", VIDEO_PATH)
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps < 1:
        fps = FPS_FALLBACK

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (FINAL_WIDTH, FINAL_HEIGHT))
    detector = SuspiciousPoseDetector()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Поворот и ресайз
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        frame = cv2.resize(frame, (FINAL_WIDTH, FINAL_HEIGHT))

        # Запускаем модель YOLO Pose
        results = model.predict(frame, conf=0.5, verbose=False)
        if len(results) == 0 or results[0].keypoints is None:
            detector.reset()
            writer.write(frame)
            cv2.imshow("YOLO Pose Demo", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        # Получаем keypoints для всех людей (shape: (N, 17, 2))
        kpts_tensor = results[0].keypoints.xy
        if not isinstance(kpts_tensor, np.ndarray):
            kpts_tensor = kpts_tensor.cpu().numpy()

        if kpts_tensor.shape[0] == 0:
            detector.reset()
        else:
            frame = detector.process_all(frame, kpts_tensor)

        writer.write(frame)
        cv2.imshow("YOLO Pose Demo", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
