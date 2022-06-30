import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
import cv2 
import numpy as np
import os
import magic
from albumentations import Compose, RandomBrightnessContrast, HorizontalFlip, FancyPCA, HueSaturationValue, OneOf, ToGray, ShiftScaleRotate, ImageCompression, PadIfNeeded, GaussNoise, GaussianBlur, Rotate, Normalize
from PIL import Image
from transforms.albu import IsotropicResize
from concurrent.futures import ThreadPoolExecutor
from os import cpu_count
import re
import cv2
from itertools import compress


DATA_DIR = "../datasets/ForgeryNet/faces"
ORIGINAL_VIDEOS_PATH = {"train": "../datasets/ForgeryNet/Training/video/train_video_release", "val": "../datasets/ForgeryNet/Training/video/train_video_release", "test": "../datasets/ForgeryNet/Validation/video/val_video_release"}
RANGE_SIZE = 5
SIZE_EMB_DICT = [(1+i*RANGE_SIZE, (i+1)*RANGE_SIZE) if i != 0 else (0, RANGE_SIZE) for i in range(20)]


class DeepFakesDataset(Dataset):
    def __init__(self, videos_paths, labels, image_size, mode = 'train', model = 0, sequence_length = 8):
        self.x = videos_paths
        self.y = labels
        self.image_size = image_size
        self.mode = mode
        self.n_samples = len(videos_paths)
        self.sequence_length = sequence_length
    
    def create_train_transforms(self, size):
        return Compose([
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ImageCompression(quality_lower=60, quality_upper=100, p=0.2),
            GaussNoise(p=0.3),
            GaussianBlur(blur_limit=3, p=0.05),
            HorizontalFlip(),
            OneOf([
                IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
                IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_LINEAR),
                IsotropicResize(max_side=size, interpolation_down=cv2.INTER_LINEAR, interpolation_up=cv2.INTER_LINEAR),
            ], p=1),
            PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
            OneOf([RandomBrightnessContrast(), FancyPCA(), HueSaturationValue()], p=0.4),
            ToGray(p=0.2),
            ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=5, border_mode=cv2.BORDER_CONSTANT, p=0.5),
        ]
        )
        
    def create_val_transform(self, size):
        return Compose([
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
            PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
        ])
        
    def get_largest_identity(self, identities):
        max_side = 0
        max_identity = None
        for identity in identities:
            if os.path.basename(identity) == "discarded":
                continue
            faces = os.listdir(identity)
            t = magic.from_file(os.path.join(identity, faces[0]))
            shape = re.search('(\d+) x (\d+)', t).groups()
            if int(shape[0]) > max_side:
                max_side = int(shape[0])
                max_identity = identity
        return max_identity
    
        
    def __getitem__(self, index):
        video_path = self.x[index]
        video_path = os.path.join(DATA_DIR, video_path)
        if self.mode == 'train':
            transform = self.create_train_transforms(self.image_size)
        else:
            transform = self.create_val_transform(self.image_size)   
        
        original_video_path = os.path.join(ORIGINAL_VIDEOS_PATH[self.mode], video_path.split(self.mode + os.path.sep)[1])
        identities = os.listdir(video_path)
        consider_discarded = False
        if len(identities) == 1 and identities[0] == "discarded":
            consider_discarded = True
            video_path = os.path.join(video_path, "discarded")
            identities = os.listdir(video_path)


        identity_path = self.get_largest_identity([os.path.join(video_path, identity) for identity in identities])
        faces = np.asarray(sorted(os.listdir(identity_path), key=lambda x:int(x.split("_")[0])))

        if len(faces) > self.sequence_length:
            idx = np.round(np.linspace(0, len(faces) - 1, self.sequence_length)).astype(int)
            faces = faces[idx]

        
        identity_faces = []
        for face in faces:
            face_path = os.path.join(identity_path, face)
            identity_faces.append(face_path)

        identity_images = []
        
        capture = cv2.VideoCapture(original_video_path)
        width  = capture.get(3)  
        height = capture.get(4) 
        video_area = width*height/2
        size_embeddings = []
        for image_path in identity_faces:
            image = np.asarray(Image.open(image_path))
            face_area = image.shape[0] * image.shape[1] / 2
            ratio = int(face_area * 100 / video_area)
             
            side_ranges = list(map(lambda a_: ratio in range(a_[0], a_[1] + 1), SIZE_EMB_DICT))
            size_embeddings.append(np.where(side_ranges)[0][0]+1)
            image = transform(image=image)['image']
            identity_images.append(image)
            
        size_embeddings = torch.tensor(size_embeddings)
        if len(size_embeddings) < self.sequence_length:
            diff = self.sequence_length - len(size_embeddings)
            size_embeddings = torch.cat((size_embeddings, torch.zeros(diff)))

            
        identity_images = torch.tensor(identity_images)

        if len(identity_images) < self.sequence_length:
            diff = self.sequence_length - len(identity_images)
            identity_images = torch.cat((identity_images, torch.zeros(diff, self.image_size, self.image_size, 3, dtype=torch.double)))
            mask = [1 if i < self.sequence_length - diff else 0 for i in range(self.sequence_length)]
        else:
            mask = [1 for i in range(self.sequence_length)]

        return identity_images.float(), torch.tensor(size_embeddings).int(), torch.tensor(mask).bool(), self.y[index]

    def __len__(self):
        return self.n_samples