import os
import torch
import torchaudio
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pickle
import numpy as np

class AVSBenchDataset(Dataset):
    def __init__(self, root_dir, use_mel=True, transform=None):
        """
        Data loader: Visual (Image) -> Audio (Mel Spectrogram or WAV)
        :param root_dir: Root directory of dataset (AVSBench/Single/s4_data)
        :param use_mel: Whether to use Mel spectrogram (True: Mel, False: WAV)
        :param transform: Visual data preprocessing
        """
        self.root_dir = os.path.abspath(root_dir)
        self.use_mel = use_mel
        self.transform = transform or transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

        self.visual_path = os.path.join(self.root_dir, "visual_frames")
        self.audio_path = os.path.join(self.root_dir, "audio_log_mel" if use_mel else "audio_wav")

        if not os.path.exists(self.visual_path):
            raise FileNotFoundError(f"Visual data path does not exist: {self.visual_path}")
        if not os.path.exists(self.audio_path):
            raise FileNotFoundError(f"Audio data path does not exist: {self.audio_path}")

        self.data_pairs = self._load_data()
        print(f"Loaded {len(self.data_pairs)} (image, audio) pairs")
        if len(self.data_pairs) == 0:
            raise ValueError("No (image, audio) pairs found, check if paths are correct!")

    def _load_data(self):
        """ Get (image, audio) pairs """
        data_pairs = []
        
        for split in ["train"]:
            split_visual_path = os.path.join(self.visual_path, split)
            split_audio_path = os.path.join(self.audio_path, split)

            if not os.path.exists(split_visual_path) or not os.path.exists(split_audio_path):
                print(f"Skipping: {split_visual_path} or {split_audio_path} does not exist")
                continue

            for category in os.listdir(split_visual_path):
                category_path = os.path.join(split_visual_path, category)
                audio_category_path = os.path.join(split_audio_path, category)

                if not os.path.isdir(category_path) or not os.path.exists(audio_category_path):
                    continue

                for video_id in os.listdir(category_path):
                    video_frame_base_path = os.path.join(category_path, video_id)

                    if not os.path.isdir(video_frame_base_path):
                        print(f"Visual data folder does not exist: {video_frame_base_path}")
                        continue

                    image_files = sorted([
                        os.path.join(video_frame_base_path, f) 
                        for f in os.listdir(video_frame_base_path) 
                        if f.endswith(".png")
                    ])

                    audio_file = os.path.join(audio_category_path, f"{video_id}.pkl")

                    if not os.path.exists(audio_file):
                        print(f"Audio not found: {audio_file}")
                        continue

                    for img in image_files:
                        data_pairs.append((img, audio_file))

        print(f"Loaded {len(data_pairs)} data pairs")
        return data_pairs

    def __len__(self):
        """ Return the size of the dataset """
        return len(self.data_pairs)

    def __getitem__(self, idx):
        """ Get single image and corresponding audio """
        image_path, audio_path = self.data_pairs[idx]

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        try:
            with open(audio_path, "rb") as f:
                audio_data = pickle.load(f)

            if isinstance(audio_data, np.ndarray):
                audio_data = torch.tensor(audio_data)

            if not isinstance(audio_data, torch.Tensor):
                raise ValueError(f"Audio data type error: {type(audio_data)}, should be `torch.Tensor`")

        except Exception as e:
            print(f"Error reading {audio_path}: {e}")
            audio_data = torch.zeros(1, 96, 64)

        return image, audio_data



