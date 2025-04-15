import os
import torch
from torch.utils.data import Dataset
from torchvision.io import read_video

def subsample_frames(total_frames, target_num_frames=16):
    if total_frames >= target_num_frames:
        indices = torch.linspace(0, total_frames - 1, target_num_frames).long()
    else:
        indices = torch.cat([
            torch.arange(0, total_frames),
            torch.full((target_num_frames - total_frames,), total_frames - 1)
        ]).long()
    return indices

class VideoDataset(Dataset):
    def __init__(self, file_paths, labels, num_frames=16, transform=None):
        self.video_paths = file_paths
        self.labels = labels
        self.num_frames = num_frames
        self.transform = transform
        if len(self.video_paths) != len(self.labels):
            raise ValueError("Количество видеофайлов и меток должно совпадать!")
        print(f"Dataset инициализирован. Файлов: {len(self.video_paths)}")

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)
        video_data, _, _ = read_video(video_path, output_format="TCHW", pts_unit="sec")
        if video_data is None or video_data.shape[0] == 0:
            raise ValueError(f"Видео пустое или не прочиталось: {video_path}")
        total_frames = video_data.shape[0]
        indices = subsample_frames(total_frames, self.num_frames)
        video = video_data[indices]
        if self.transform:
            video = torch.stack([self.transform(frame) for frame in video])
        # Меняем размерность: [num_frames, C, H, W] -> [C, num_frames, H, W]
        video = video.permute(1, 0, 2, 3)
        return video, label_tensor
