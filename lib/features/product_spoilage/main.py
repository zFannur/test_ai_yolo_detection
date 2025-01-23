# main.py
import cv2
from detectors.kick_detector import KickDetector
from lib.features.product_spoilage.config import config
from utils.logger import setup_logger


class VideoProcessor:
    def __init__(self, source, output=None):
        self.cap = self._init_capture(source)
        self.detector = KickDetector()
        self.writer = None
        self.logger = setup_logger()

        if output:
            self._init_writer(output)

    def _init_capture(self, source):
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise ValueError(f"Error opening video source: {source}")
        return cap

    def _init_writer(self, output_path):
        # Используем размер из конфига (width, height)
        frame_size = config.WINDOW_SIZE
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(
            output_path,
            fourcc,
            30,
            frame_size  # (width, height)
        )

    def _resize_frame(self, frame):
        return cv2.resize(frame, config.WINDOW_SIZE)

    def process(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Изменение размера кадра
            resized_frame = self._resize_frame(frame)

            # Обработка кадра
            results = self.detector.process_frame(resized_frame)
            processed_frame = self.detector.visualize(resized_frame, results)

            # Запись и отображение
            if self.writer:
                self.writer.write(processed_frame)

            cv2.imshow("Frame", processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        if self.writer:
            self.writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    source = "datasets/product_spoilage/video/train2.mp4"
    output = "datasets/product_spoilage/video/train_detect_kick.mp4"

    processor = VideoProcessor(source, output)
    processor.process()