# lib/config.py

"""
Общий конфиг для параметров обучения и детекции.
"""

# Зерно (seed) для воспроизводимости
SEED = 42

# Параметры обучения
NUM_EPOCHS = 6         # Увеличили с 5 до 10 для более основательной тренировки
BATCH_SIZE = 32         # Размер батча; может увеличить требование к памяти
LEARNING_RATE = 1e-4    # Скорость обучения
WEIGHT_DECAY = 1e-5     # Уменьшили с 1e-4 -> 1e-5 для более «мягкой» регуляризации
DROPOUT_PROB = 0.6      # Вероятность dropout

# Параметры данных
NUM_FRAMES = 16         # Число кадров на один клип
IMG_SIZE = 224          # Размер кадра (высота и ширина)
NUM_CLASSES = 2         # fight / noFight

# Прочее
STABLE_THRESHOLD = 3    # Для стабилизации предсказания в detect.py

# Пути
MODEL_SAVE_DIR = r"C:\Users\zfann\PycharmProjects\test_ai_yolo_detect\models\fight"
TRAIN_DATA_ROOT = r"C:\Users\zfann\PycharmProjects\test_ai_yolo_detect\datasets\fight\train"
TEST_VIDEO_PATH = r"C:\Users\zfann\PycharmProjects\test_ai_yolo_detect\datasets\fight\test3.mp4"
