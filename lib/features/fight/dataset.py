# lib/features/fight/dataset.py

import os

import av
import torch
from torch.utils.data import Dataset
from torchvision.io import read_video, VideoReader # Используем VideoReader для большей информации
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, ToPILImage, RandomHorizontalFlip # Добавляем нужные
import random

# --- Улучшенный monkey-patch для PyAV (попробуем несколько вариантов) ---
try:
    # Современный PyAV >= 10
    from av import AVError as BaseAVError
    av.AVError = BaseAVError
    print("Используется av.AVError из 'av'")
except ImportError:
    try:
        # Старый PyAV
        import av.error
        if not hasattr(av, 'AVError'):
            av.AVError = av.error.OSError # Или другое подходящее, например av.error.PermissionError?
            print("Установлен av.AVError = av.error.OSError")
    except (ImportError, AttributeError):
        if not hasattr(av, 'AVError'):
             av.AVError = OSError # Общий фолбэк
             print("Предупреждение: Не удалось найти AVError в PyAV. Установлен фолбэк av.AVError = OSError")
# -----------------------------------------------------------------------


def subsample_frames(total_frames, target_num_frames=16):
    """
    Генерирует индексы для выбора ровно target_num_frames кадров.
    :param total_frames: Общее количество кадров в видео.
    :param target_num_frames: Сколько кадров нужно выбрать.
    :return: Тензор с индексами кадров.
    """
    if total_frames >= target_num_frames:
        # Выбираем равномерно
        indices = torch.linspace(0, total_frames - 1, target_num_frames).long()
    else:
        # Повторяем последний кадр, если не хватает
        indices = torch.cat([
            torch.arange(0, total_frames),
            torch.full((target_num_frames - total_frames,), total_frames - 1)
        ]).long()
    return indices


class VideoDataset(Dataset):
    def __init__(self, file_paths, labels, num_frames=16, transform=None):
        """
        :param file_paths: Список путей к видеофайлам.
        :param labels: Список соответствующих меток (0 или 1).
        :param num_frames: Целевое количество кадров для каждого видео.
        :param transform: Трансформации torchvision, применяемые к каждому кадру.
        """
        super().__init__()
        self.video_paths = file_paths
        self.labels = labels
        self.num_frames = num_frames
        self.transform = transform

        if len(self.video_paths) != len(self.labels):
             raise ValueError("Количество путей к видео и меток должно совпадать!")

        print(f"Dataset инициализирован. Найдено файлов: {len(self.video_paths)}")

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        label_tensor = torch.tensor(label, dtype=torch.long)

        try:
            # Используем VideoReader для получения метаданных (fps, duration) если нужно
            # reader = VideoReader(video_path, "video")
            # reader_md = reader.get_metadata()
            # total_frames = int(reader_md["video"]['duration'][0] * reader_md["video"]['fps'][0]) # Примерно

            # Простой способ - читать все кадры и потом сэмплировать
            # pts_unit='sec' может быть не надежным для точного числа кадров
            video_data, _, info = read_video(video_path, output_format="TCHW", pts_unit="sec")
            # video_data форма: [T, C, H, W]

            if video_data is None or video_data.shape[0] == 0:
                 raise ValueError(f"Не удалось прочитать кадры или видео пустое: {video_path}")

            total_frames = video_data.shape[0]
            indices = subsample_frames(total_frames, self.num_frames)
            video = video_data[indices] # Выбираем кадры -> [num_frames, C, H, W]

            # --- Применение трансформаций ---
            # Трансформации применяются к каждому кадру
            if self.transform:
                 # Важно: Некоторые трансформации (например, Normalize) ожидают [C, H, W]
                 # и могут работать некорректно с [T, C, H, W] напрямую.
                 # Применяем последовательно к каждому кадру.
                 transformed_video = torch.stack([self.transform(frame) for frame in video])
                 # -> Результат будет [num_frames, C, H_new, W_new]
            else:
                 transformed_video = video # Если трансформаций нет

            # Меняем порядок на [C, T, H, W] как ожидает 3D CNN
            final_video = transformed_video.permute(1, 0, 2, 3)

            # Проверка финальной формы (опционально)
            # expected_shape = (video.shape[1], self.num_frames, H_new, W_new) # H_new, W_new после Resize
            # if final_video.shape != expected_shape:
            #    print(f"Предупреждение: Неожиданная форма видео {final_video.shape} для {video_path}")


        except av.AVError as e:
            print(f"!!! Ошибка PyAV при загрузке {video_path}: {e}. Пропускаем...")
            # Возвращаем None, чтобы DataLoader мог пропустить (требует кастомный collate_fn)
            # return None, None
            # Или возвращаем плейсхолдеры (если модель сможет обработать)
            # return torch.zeros((3, self.num_frames, 112, 112)), torch.tensor(-1, dtype=torch.long) # Пример плейсхолдера
            # Самый простой вариант - перевыбросить ошибку, если это редкость
            raise RuntimeError(f"Ошибка PyAV при загрузке {video_path}") from e
        except Exception as e:
            print(f"!!! Общая ошибка при обработке {video_path}: {type(e).__name__} - {e}. Пропускаем...")
            # return None, None
            # return torch.zeros((3, self.num_frames, 112, 112)), torch.tensor(-1, dtype=torch.long)
            raise RuntimeError(f"Общая ошибка при обработке {video_path}") from e


        return final_video, label_tensor