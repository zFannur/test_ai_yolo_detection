# detectors/kick_detector.py
from collections import defaultdict, deque

from ultralytics import YOLO

from lib.features.product_spoilage.config import config
from lib.features.product_spoilage.detectors.base_detector import BaseDetector
from lib.features.product_spoilage.utils import geometry, visualization


class KickDetector(BaseDetector):
    def __init__(self):
        self._load_models()
        self._init_tracking()

    def _load_models(self):
        """Загрузка моделей"""
        self.pose_model = YOLO(config.MODELS["pose"]["path"]).to(config.DEVICE)
        self.box_model = YOLO(config.MODELS["box"]["path"]).to(config.DEVICE)

    def _init_tracking(self):
        """Инициализация систем отслеживания"""
        self.foot_history = deque(maxlen=config.EVENTS["fall"]["history_length"])
        self.box_history = defaultdict(lambda: deque(
            maxlen=config.EVENTS["fall"]["history_length"]
        ))
        self.frame_idx = 0

    def process_frame(self, frame):
        self.frame_idx += 1
        results = {
            "kicks": [],
            "falls": [],
            "boxes": {},
            "limbs": []
        }

        try:
            # Детекция объектов
            pose_results = self.pose_model.predict(
                frame,
                conf=config.MODELS["pose"]["conf"],
                iou=config.MODELS["pose"]["iou"],
                device=config.DEVICE,
                verbose=False
            )

            box_results = self.box_model.track(
                frame,
                conf=config.MODELS["box"]["conf"],
                persist=True,
                tracker=config.MODELS["box"]["tracker"],
                device=config.DEVICE,
                verbose=False
            )

            # Обработка результатов
            limbs = self._process_pose(pose_results)
            boxes = self._process_boxes(box_results)

            # Анализ событий
            self._update_histories(limbs, boxes)
            results.update({
                "limbs": limbs,
                "boxes": boxes,
                "falls": self._detect_falls(boxes),
                "kicks": self._detect_kicks(limbs, boxes)
            })

        except Exception as e:
            print(f"Ошибка обработки кадра: {str(e)}")

        return results

    def _process_pose(self, results):
        limbs = []
        if results[0].keypoints is None:
            return limbs

        for kpts in results[0].keypoints.xy.cpu().numpy():
            if len(kpts) < 17:
                continue

            limbs.append({
                "left_foot": geometry.get_valid_point(kpts[15]),
                "right_foot": geometry.get_valid_point(kpts[16])
            })
        return limbs

    def _process_boxes(self, results):
        boxes = {}
        if results[0].boxes.id is None:
            return boxes

        for box, box_id in zip(results[0].boxes.xyxy, results[0].boxes.id):
            box_id = int(box_id.item())
            x1, y1, x2, y2 = map(int, box[:4])
            boxes[box_id] = {
                'coords': (x1, y1, x2, y2),
                'center': ((x1 + x2) // 2, (y1 + y2) // 2)
            }
        return boxes

    def _update_histories(self, limbs, boxes):
        # Обновление истории ног
        current_feet = {}
        for limb in limbs:
            for ft in ["left_foot", "right_foot"]:
                if limb[ft] != (0, 0):
                    current_feet[ft] = limb[ft]
        self.foot_history.append(current_feet)

        # Обновление истории коробок
        for box_id, box_data in boxes.items():
            self.box_history[box_id].append(box_data['center'])

    def _detect_falls(self, boxes):
        falls = []
        for box_id, box_data in boxes.items():
            pos = box_data['center']
            if len(self.box_history[box_id]) < 2:
                continue

            speed = geometry.calculate_speed(
                self.box_history[box_id][-2],
                self.box_history[box_id][-1]
            )

            if speed > config.EVENTS["fall"]["speed_threshold"]:
                falls.append({
                    "box_id": box_id,
                    "position": pos,
                    "speed": speed
                })
        return falls

    def _detect_kicks(self, limbs, boxes):
        kicks = []
        for limb in limbs:
            for foot_type in ["left_foot", "right_foot"]:
                foot_pos = limb[foot_type]
                if foot_pos == (0, 0):
                    continue

                closest = geometry.find_closest_box(foot_pos, boxes)
                if closest is None:
                    continue

                box_id, box_data, distance = closest
                box_pos = box_data['center']

                if distance > config.EVENTS["kick"]["distance_threshold"]:
                    continue

                foot_speed = self._calculate_limb_speed(foot_type)
                box_speed = self._calculate_box_speed(box_id)

                if (foot_speed > config.EVENTS["kick"]["foot_speed_threshold"] and
                        box_speed > config.EVENTS["kick"]["box_accel_threshold"]):
                    kicks.append({
                        "foot_type": foot_type,
                        "foot_pos": foot_pos,
                        "box_id": box_id,
                        "box_pos": box_pos,
                        "speed": foot_speed
                    })
        return kicks

    def _calculate_limb_speed(self, limb_type):
        return geometry.calculate_limb_speed(
            self.foot_history,
            limb_type,
            min_frames=2
        )

    def _calculate_box_speed(self, box_id):
        return geometry.calculate_box_speed(
            self.box_history[box_id],
            min_frames=2
        )

    def visualize(self, frame, results):
        # Отрисовка коробок
        for box_id, box_data in results["boxes"].items():
            visualization.draw_box(frame, box_id, box_data)

        # Отрисовка конечностей
        for limb in results["limbs"]:
            for ft in ["left_foot", "right_foot"]:
                if limb[ft] != (0, 0):
                    visualization.draw_limb(frame, limb[ft], ft)

        # Отрисовка событий
        for kick in results["kicks"]:
            visualization.draw_kick_event(
                frame,
                kick["foot_pos"],
                kick["box_pos"],
                kick["speed"],
                kick["box_id"]
            )

        for fall in results["falls"]:
            visualization.draw_fall_event(
                frame,
                fall["position"],
                fall["box_id"],
                fall["speed"]
            )

        return frame