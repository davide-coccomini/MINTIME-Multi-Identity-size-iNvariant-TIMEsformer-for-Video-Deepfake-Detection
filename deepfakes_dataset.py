import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
import cv2 
import numpy as np
from datetime import datetime
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


DATA_DIR = "../datasets/ForgeryNet/faces_test"
ORIGINAL_VIDEOS_PATH = {"train": "../datasets/ForgeryNet/Training/video/train_video_release", "val": "../datasets/ForgeryNet/Training/video/train_video_release", "test": "../datasets/ForgeryNet/Validation/video/val_video_release"}
RANGE_SIZE = 5
SIZE_EMB_DICT = [(1+i*RANGE_SIZE, (i+1)*RANGE_SIZE) if i != 0 else (0, RANGE_SIZE) for i in range(20)]

class DeepFakesDataset(Dataset):
    def __init__(self, videos_paths, labels, image_size, mode = 'train', model = 0, num_frames = 8, max_identities = 3, num_patches=49):
        self.x = videos_paths
        self.y = labels
        self.image_size = image_size
        self.mode = mode
        self.n_samples = len(videos_paths)
        self.num_frames = num_frames
        self.num_patches = num_patches
        self.max_identities = max_identities
        self.max_faces_per_identity = {1: [num_frames], 
                  2:  [int(num_frames/2), int(num_frames/2)],
                  3:  [int(num_frames/3), int(num_frames/3), int(num_frames/4)],
                  4:  [int(num_frames/3), int(num_frames/3), int(num_frames/8), int(num_frames/8)]}

    
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


    
    def get_sorted_identities(self, identities):
        sorted_identities = []
    
        for identity in identities:
            if not os.path.isdir(identity):
                continue
            
            faces = sorted(os.listdir(identity), key=lambda x:int(x.split("_")[0]))
            t = magic.from_file(os.path.join(identity, faces[0]))
            shape = re.search('(\d+) x (\d+)', t).groups()
            max_side = int(shape[0])
            number_of_faces = len(faces)
            sorted_identities.append([identity, max_side, number_of_faces])


        sorted_identities = sorted(sorted_identities, key=lambda x:x[1], reverse=True)
        if len(sorted_identities) > self.max_identities:
            sorted_identities = sorted_identities[:self.max_identities]

        identities_number = len(sorted_identities)
        if identities_number > 1:
            max_faces_per_identity = self.max_faces_per_identity[identities_number]
            # Adjust the list
            for i in range(identities_number):
                if sorted_identities[i][2] < max_faces_per_identity[i] and i < identities_number - 1:
                    sorted_identities[i+1][2] += max_faces_per_identity[i] - sorted_identities[i][2] 
                elif sorted_identities[i][2] > max_faces_per_identity[i]:
                    sorted_identities[i][2] = max_faces_per_identity[i]
        else:
            sorted_identities[0][2] = self.num_frames

        return sorted_identities
    

    def __getitem__(self, index):
        video_path = self.x[index]
        video_path = os.path.join(DATA_DIR, video_path)
        if self.mode == 'train':
            transform = self.create_train_transforms(self.image_size)
        else:
            transform = self.create_val_transform(self.image_size)   

        original_video_path = os.path.join(ORIGINAL_VIDEOS_PATH[self.mode], video_path.split(self.mode + os.path.sep)[1])
        identities = os.listdir(video_path)
        identities = self.get_sorted_identities([os.path.join(video_path, identity) for identity in identities])
        mask = []
        last_range_end = 0
        sequence = []
        size_embeddings = []
        
        images_frames = []
        for identity_index, identity in enumerate(identities):
            identity_path = identity[0]
            max_faces = identity[2]
            faces = np.asarray(sorted(os.listdir(identity_path), key=lambda x:int(x.split("_")[0])))
            if len(faces) > max_faces:
                idx = np.round(np.linspace(0, len(faces) - 1, max_faces)).astype(int)
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
            identity_size_embeddings = []
            for image_path in identity_faces:
                image = np.asarray(Image.open(image_path))
                face_area = image.shape[0] * image.shape[1] / 2
                ratio = int(face_area * 100 / video_area)
                
                side_ranges = list(map(lambda a_: ratio in range(a_[0], a_[1] + 1), SIZE_EMB_DICT))
                identity_size_embeddings.append(np.where(side_ranges)[0][0]+1)
                try:
                    image = transform(image=image)['image']
                except:
                    image = np.zeros((self.image_size, self.image_size, 3))
                frame = int(os.path.basename(image_path).split("_")[0])
                images_frames.append(frame)
                identity_images.append(image)
            if len(identity_images) < max_faces: # The faces are less than max_faces so we need to add empty images
                diff = max_faces - len(identity_size_embeddings)
                identity_size_embeddings = np.concatenate((identity_size_embeddings, np.zeros(diff)))
                identity_images = np.concatenate((identity_images, np.zeros((diff, self.image_size, self.image_size, 3), dtype=np.double)))
                mask.extend([1 if i < max_faces - diff else 0 for i in range(max_faces)])
            else: # Otherwise all the faces are valid
                mask.extend([1 for i in range(max_faces)])
            size_embeddings.extend(identity_size_embeddings)
            sequence.extend(identity_images)


        identities_mask = []
        last_range_end = 0
        for identity_index in range(len(identities)):
            identity_mask = [True if i >= last_range_end and i < last_range_end + identities[identity_index][2] else False for i in range(0, self.num_frames)]
            for k in range(identities[identity_index][2]):
                identities_mask.append(identity_mask)
            last_range_end += identities[identity_index][2]
        
        # Generate coherent temporal-positional embedding
        images_frames_positions = {k: v+1 for v, k in enumerate(sorted(set(images_frames)))}
        frame_positions = [images_frames_positions[frame] for frame in images_frames]     
        if self.num_patches != None: 
            positions = [[i+1 for i in range(((frame_position-1)*self.num_patches), self.num_patches*(frame_position))] for frame_position in frame_positions]
            positions = sum(positions, []) # Merge the lists
            positions.insert(0,0) # Add CLS
        else:
            positions = []
        return torch.tensor(sequence).float(), torch.tensor(size_embeddings).int(), torch.tensor(mask).bool(), torch.tensor(identities_mask).bool(), torch.tensor(positions), self.y[index]

    def __len__(self):
        return self.n_samples
