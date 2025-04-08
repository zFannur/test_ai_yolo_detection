import cv2
import math
import time
import mediapipe as mp

# Пути к файлам
VIDEO_PATH = "datasets/nasvay/video/train.mp4"       # Входное видео
OUTPUT_PATH = "datasets/nasvay/video/train_out.mp4"  # Выходное видео
LOG_FILE = "nasvay_events.log"

# ----- ПАРАМЕТРЫ -----
ELBOW_ANGLE_THRESHOLD = 40        # угол в локте (градусы), когда рука сильно согнута
HAND_MOUTH_THRESHOLD = 60.0       # дистанция (пиксели) между центром рта и кисти
TIME_THRESHOLD = 2.0              # если рука у рта больше этого времени, считаем "приём"
HIGHLIGHT_DURATION = 3.0          # сколько секунд подсвечивать человека красным после детекта

FINAL_WIDTH = 480
FINAL_HEIGHT = 640

# Инициализация Mediapipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5
)

class NasvayDetector:
    def __init__(self):
        # Отдельные таймеры для каждой руки (подсчёт времени, пока рука у рта)
        self.start_time_left = None
        self.start_time_right = None
        # Флаги, чтобы не спамить лог при долгом удержании
        self.logged_left = False
        self.logged_right = False
        # Время, до которого подсвечиваем человека красным
        self.highlight_end_time = None

    def process_frame(self, frame):
        """Обработка кадра: определяем приём насвая, при необходимости подсвечиваем человека."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if not results.pose_landmarks:
            # Если позы нет, сбрасываем
            self.reset()
            return frame

        h, w, _ = frame.shape
        landmarks = results.pose_landmarks.landmark

        # Получаем bounding box рта
        mouth_bbox, mouth_center = self._get_mouth_bbox_and_center(landmarks, w, h)

        # Для отладки нарисуем рот
        if mouth_bbox:
            x_min, y_min, x_max, y_max = mouth_bbox
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)
        if mouth_center:
            cv2.circle(frame, mouth_center, 4, (0, 255, 255), -1)

        # Координаты плеча/локтя/запястья для левой руки (11, 13, 15)
        left_shoulder = self._get_xy(landmarks, 11, w, h)
        left_elbow   = self._get_xy(landmarks, 13, w, h)
        left_wrist   = self._get_xy(landmarks, 15, w, h)

        # Координаты плеча/локтя/запястья для правой руки (12, 14, 16)
        right_shoulder = self._get_xy(landmarks, 12, w, h)
        right_elbow   = self._get_xy(landmarks, 14, w, h)
        right_wrist   = self._get_xy(landmarks, 16, w, h)

        # Углы локтей
        left_angle = self._calculate_angle(left_shoulder, left_elbow, left_wrist)
        right_angle = self._calculate_angle(right_shoulder, right_elbow, right_wrist)

        # Отображаем углы для наглядности
        if left_elbow and left_angle is not None:
            cv2.putText(frame, f"{int(left_angle)}", left_elbow,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
        if right_elbow and right_angle is not None:
            cv2.putText(frame, f"{int(right_angle)}", right_elbow,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # "Имитация" bbox для левой кисти (вокруг zapястья)
        left_wrist_bbox, left_wrist_center = self._make_bbox_around_point(left_wrist, box_size=10)
        if left_wrist_bbox:
            lx_min, ly_min, lx_max, ly_max = left_wrist_bbox
            cv2.rectangle(frame, (lx_min, ly_min), (lx_max, ly_max), (255,0,0), 2)
        if left_wrist_center:
            cv2.circle(frame, left_wrist_center, 3, (255,0,0), -1)

        # "Имитация" bbox для правой кисти
        right_wrist_bbox, right_wrist_center = self._make_bbox_around_point(right_wrist, box_size=10)
        if right_wrist_bbox:
            rx_min, ry_min, rx_max, ry_max = right_wrist_bbox
            cv2.rectangle(frame, (rx_min, ry_min), (rx_max, ry_max), (0,255,0), 2)
        if right_wrist_center:
            cv2.circle(frame, right_wrist_center, 3, (0,255,0), -1)

        # Проверяем каждую руку — угол + дистанция
        self._check_arm(frame, "left", left_angle, left_wrist_center, mouth_center)
        self._check_arm(frame, "right", right_angle, right_wrist_center, mouth_center)

        # Если нужно подсветить человека, проверим, не истёк ли таймер highlight
        self._maybe_highlight_person(frame, landmarks, w, h)

        return frame

    def _check_arm(self, frame, side, angle, wrist_center, mouth_center):
        """
        Условие: угол < ELBOW_ANGLE_THRESHOLD и дистанция < HAND_MOUTH_THRESHOLD.
        Если удерживается > TIME_THRESHOLD секунд, фиксируем событие и подсвечиваем человека.
        """
        if angle is None or wrist_center is None or mouth_center is None:
            self._reset_side(side)
            return

        dist = self._distance(wrist_center, mouth_center)

        # Для наглядности выводим расстояние
        if side == "left":
            cv2.putText(frame, f"Dist(L)={int(dist)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
        else:
            cv2.putText(frame, f"Dist(R)={int(dist)}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

        if angle < ELBOW_ANGLE_THRESHOLD and dist < HAND_MOUTH_THRESHOLD:
            current = time.time()
            if side == "left":
                if self.start_time_left is None:
                    self.start_time_left = current
                elapsed = current - self.start_time_left
                cv2.putText(frame, f"L-Elapsed {elapsed:.1f}s", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
                if elapsed > TIME_THRESHOLD and not self.logged_left:
                    self.logged_left = True
                    self._log_event("Left arm: nasvay intake detected!")
                    cv2.putText(frame, "NASVAY (L)!", (10, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                    # Запускаем подсветку человека
                    self._start_highlight()

            else:  # "right"
                if self.start_time_right is None:
                    self.start_time_right = current
                elapsed = current - self.start_time_right
                cv2.putText(frame, f"R-Elapsed {elapsed:.1f}s", (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
                if elapsed > TIME_THRESHOLD and not self.logged_right:
                    self.logged_right = True
                    self._log_event("Right arm: nasvay intake detected!")
                    cv2.putText(frame, "NASVAY (R)!", (10, 180),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                    self._start_highlight()
        else:
            self._reset_side(side)

    def _start_highlight(self):
        """Запускаем таймер, в течение которого человек будет подсвечен красным."""
        self.highlight_end_time = time.time() + HIGHLIGHT_DURATION

    def _maybe_highlight_person(self, frame, landmarks, w, h):
        """
        Если сейчас мы находимся в режиме подсветки (highlight_end_time не истек),
        рисуем красный bounding box вокруг всего человека.
        """
        if self.highlight_end_time is None:
            return

        current = time.time()
        if current < self.highlight_end_time:
            # Получаем bbox всех точек pose (чтобы обвести всего человека)
            bbox = self._get_person_bbox(landmarks, w, h)
            if bbox is not None:
                x_min, y_min, x_max, y_max = bbox
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0,0,255), 3)
        else:
            # Время истекло, сбрасываем
            self.highlight_end_time = None

    def _reset_side(self, side):
        """Сброс для отдельной руки."""
        if side == "left":
            self.start_time_left = None
            self.logged_left = False
        else:
            self.start_time_right = None
            self.logged_right = False

    def reset(self):
        """Сброс для обеих рук, если поза не найдена."""
        self.start_time_left = None
        self.logged_left = False
        self.start_time_right = None
        self.logged_right = False
        self.highlight_end_time = None

    def _log_event(self, message):
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {message}\n")

    @staticmethod
    def _get_xy(landmarks, idx, width, height):
        """Возвращает (x, y) в пикселях для заданного индекса landmark."""
        if idx < 0 or idx >= len(landmarks):
            return None
        lm = landmarks[idx]
        x = int(lm.x * width)
        y = int(lm.y * height)
        return (x, y)

    @staticmethod
    def _calculate_angle(a, b, c):
        """Вычисление угла ABC (угол в точке b)."""
        if not a or not b or not c:
            return None
        ba = (a[0] - b[0], a[1] - b[1])
        bc = (c[0] - b[0], c[1] - b[1])
        dot_prod = ba[0]*bc[0] + ba[1]*bc[1]
        mag_ba = math.hypot(*ba)
        mag_bc = math.hypot(*bc)
        if mag_ba == 0 or mag_bc == 0:
            return None
        angle = math.degrees(math.acos(dot_prod / (mag_ba * mag_bc)))
        return angle

    @staticmethod
    def _distance(p1, p2):
        """Евклидово расстояние между p1 и p2."""
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def _get_mouth_bbox_and_center(self, landmarks, width, height):
        """
        Создаём bounding box рта по точкам 9 и 10 (левый и правый угол рта).
        Возвращаем (bbox, center).
        """
        pt9 = self._get_xy(landmarks, 9, width, height)
        pt10 = self._get_xy(landmarks, 10, width, height)
        if not pt9 or not pt10:
            return None, None
        x_min = min(pt9[0], pt10[0])
        x_max = max(pt9[0], pt10[0])
        y_min = min(pt9[1], pt10[1])
        y_max = max(pt9[1], pt10[1])
        bbox = (x_min, y_min, x_max, y_max)
        center_x = (x_min + x_max) // 2
        center_y = (y_min + y_max) // 2
        return bbox, (center_x, center_y)

    @staticmethod
    def _make_bbox_around_point(p, box_size=10):
        """Пример мини-bbox вокруг одной точки (кисти)."""
        if not p:
            return None, None
        x, y = p
        half = box_size // 2
        x_min = x - half
        x_max = x + half
        y_min = y - half
        y_max = y + half
        bbox = (x_min, y_min, x_max, y_max)
        center = (x, y)
        return bbox, center

    @staticmethod
    def _get_person_bbox(landmarks, width, height):
        """
        Возвращаем (x_min, y_min, x_max, y_max) по всем точкам pose,
        чтобы можно было «подсветить» всего человека.
        """
        if not landmarks:
            return None
        x_vals = []
        y_vals = []
        for lm in landmarks:
            x_vals.append(lm.x * width)
            y_vals.append(lm.y * height)
        if not x_vals or not y_vals:
            return None
        x_min = int(min(x_vals))
        x_max = int(max(x_vals))
        y_min = int(min(y_vals))
        y_max = int(max(y_vals))
        return (x_min, y_min, x_max, y_max)


def main():
    detector = NasvayDetector()
    cap = cv2.VideoCapture(VIDEO_PATH)

    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (FINAL_WIDTH, FINAL_HEIGHT))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Повернём кадр на 90° вправо
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        # Ресайз до нужного размера
        frame = cv2.resize(frame, (FINAL_WIDTH, FINAL_HEIGHT))

        processed = detector.process_frame(frame)

        cv2.imshow("Nasvay Detection", processed)
        writer.write(processed)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
