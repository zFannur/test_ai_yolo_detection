# detectors/base_detector.py
from abc import ABC, abstractmethod
import cv2


class BaseDetector(ABC):
    @abstractmethod
    def process_frame(self, frame):
        """Обработка кадра и обнаружение событий"""
        pass

    @abstractmethod
    def visualize(self, frame):
        """Визуализация результатов на кадре"""
        return frame
