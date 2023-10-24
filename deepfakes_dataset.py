# DeepFakesDataset class used for data loading
# In this step the identities are also refined and organized in order to fit into the available number of frames per video. 
# The data augmentation is also applied to each face extracted from the video and several embeddings and masks are generated:
# 1. The Size Embedding, responsible to induct the information about face-frame area ratio of each face to the model. 
# 2. The Temporal Positional Embedding, responsible to maintain a coherent spatial and temporal positional information of the input tokens
# 3. The Mask, responsible to make the model ignore the "empty faces" added to fill wholes in the input sequence, if occur
# 4. The Identity Mask, used to tell the model each face to which identity it corresponds

import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
import cv2 
import random
import numpy as np
from datetime import datetime
import os
import magic
from albumentations import Cutout, CoarseDropout, RandomGamma, MedianBlur, ToSepia, RandomShadow, MultiplicativeNoise, RandomSunFlare, GlassBlur, RandomBrightness, MotionBlur, RandomRain, RGBShift, RandomFog, RandomContrast, Downscale, InvertImg, RandomContrast, ColorJitter, Compose, RandomBrightnessContrast, CLAHE, ISONoise, JpegCompression, HorizontalFlip, FancyPCA, HueSaturationValue, OneOf, ToGray, ShiftScaleRotate, ImageCompression, PadIfNeeded, GaussNoise, GaussianBlur, Rotate, Normalize, Resize
from PIL import Image
from transforms.albu import IsotropicResize
from concurrent.futures import ThreadPoolExecutor
from os import cpu_count
import re
import cv2
from itertools import compress
from statistics import mean


ORIGINAL_VIDEOS_PATH = {"train": "../datasets/ForgeryNet/Training/video/train_video_release", "val": "../datasets/ForgeryNet/Training/video/train_video_release", "test": "../datasets/ForgeryNet/Validation/video/val_video_release"}
MODES = ["train", "val", "test"]
RANGE_SIZE = 5
SIZE_EMB_DICT = [(1+i*RANGE_SIZE, (i+1)*RANGE_SIZE) if i != 0 else (0, RANGE_SIZE) for i in range(20)]

class DeepFakesDataset(Dataset):
    def __init__(self, videos_paths, labels, data_path, video_path, image_size, augmentation = None, multiclass_labels = None, save_attention_plots = False, mode = 'train', model = 0, num_frames = 8, max_identities = 3, num_patches=49, enable_identity_attention = True, identities_ordering = 0):
        self.x = videos_paths
        self.y = labels
        self.multiclass_labels = multiclass_labels
        self.save_attention_plots = save_attention_plots
        self.data_path = data_path
        self.video_path = video_path
        self.image_size = image_size
        if mode not in MODES:
            raise Exception("Invalid dataloader mode.")
        self.mode = mode
        self.n_samples = len(videos_paths)
        self.num_frames = num_frames
        self.num_patches = num_patches
        self.max_identities = max_identities
        self.augmentation = augmentation
        self.max_faces_per_identity = {1: [num_frames], 
                  2:  [int(num_frames/2), int(num_frames/2)],
                  3:  [int(num_frames/3), int(num_frames/3), int(num_frames/4)],
                  4:  [int(num_frames/3), int(num_frames/3), int(num_frames/8), int(num_frames/8)]}
        self.enable_identity_attention = enable_identity_attention
        self.identities_ordering = identities_ordering
    
    def create_train_transforms(self, size, additional_targets, augmentation):
        if augmentation == "min":
            return Compose([
                OneOf([
                    IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
                    IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_LINEAR),
                    IsotropicResize(max_side=size, interpolation_down=cv2.INTER_LINEAR, interpolation_up=cv2.INTER_LINEAR),
                ], p=1),
                PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
                Resize(height=size, width=size),
                ImageCompression(quality_lower=60, quality_upper=100, p=0.2),
                GaussNoise(p=0.3),
                GaussianBlur(blur_limit=3, p=0.05),
                HorizontalFlip(),
                OneOf([RandomBrightnessContrast(), FancyPCA(), HueSaturationValue()], p=0.4),
                ToGray(p=0.2),
                ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=5, border_mode=cv2.BORDER_CONSTANT, p=0.5),
            ], additional_targets = additional_targets
            )
        else:
            return Compose([
                OneOf([
                    IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
                    IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_LINEAR),
                    IsotropicResize(max_side=size, interpolation_down=cv2.INTER_LINEAR, interpolation_up=cv2.INTER_LINEAR),
                ], p=1),
                PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
                Resize(height=size, width=size),
                ImageCompression(quality_lower=60, quality_upper=100, p=0.2),
                OneOf([GaussianBlur(blur_limit=3), MedianBlur(), GlassBlur(), MotionBlur()], p=0.1),
                OneOf([HorizontalFlip(), InvertImg()], p=0.5),
                OneOf([RandomBrightnessContrast(), RandomContrast(), RandomBrightness(), FancyPCA(), HueSaturationValue()], p=0.5),
                OneOf([RGBShift(), ColorJitter()], p=0.1),
                OneOf([MultiplicativeNoise(), ISONoise(), GaussNoise()], p=0.3),
                OneOf([Cutout(), CoarseDropout()], p=0.1),
                OneOf([RandomFog(), RandomRain(), RandomSunFlare()], p=0.02),
                RandomShadow(p=0.05),
                RandomGamma(p=0.1),
                CLAHE(p=0.05),
                ToGray(p=0.2),
                ToSepia(p=0.05),
                ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=5, border_mode=cv2.BORDER_CONSTANT, p=0.5),
            ], additional_targets = additional_targets
            )
            
    def create_val_transform(self, size, additional_targets):
        return Compose([
            IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
            PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
            Resize(height=size, width=size)
        ],  additional_targets = additional_targets
        )

    # Input the identity path and return a row with path, size and number of faces available
    def get_identity_information(self, identity):
        faces = [os.path.join(identity, face) for face in os.listdir(identity)]
        try:
            mean_side = mean([int(re.search('(\d+) x (\d+)', magic.from_file(face)).groups()[0]) for face in faces])
        except:
            mean_side = 0

        number_of_faces = len(faces)
        return [identity, mean_side, number_of_faces]


    # Returns the identities, size-based sorted, with the number of faces for each identity to be readed
    def get_sorted_identities(self, video_path):
        identities = [os.path.join(video_path, identity) for identity in os.listdir(video_path)]
        sorted_identities = []
        discarded_faces = []
        for identity in identities:
            if not os.path.isdir(identity): # The faces are not inside an identity folder but we save them to fill temporal wholes in identities if occurs
                discarded_faces.append(identity)
                continue
            
            # Sort faces based on temporal order
            sorted_identities.append(self.get_identity_information(identity))
        
        
        # If no faces have been found, use the discarded faces
        if len(sorted_identities) == 0:
            sorted_identities.append(self.get_identity_information(os.path.dirname(discarded_faces[0])))
            discarded_faces = []

        # Sort identities
        if self.identities_ordering == 0: # Based on faces size 
            sorted_identities = sorted(sorted_identities, key=lambda x:x[1], reverse=True)
        elif self.identities_ordering == 1: # Based on identities length
            sorted_identities = sorted(sorted_identities, key=lambda x:x[2], reverse=True)
        else: # Random shuffle
            random.shuffle(sorted_identities)

        if len(sorted_identities) > self.max_identities:
            sorted_identities = sorted_identities[:self.max_identities]
            
        # Adjust the identities list faces number
        identities_number = len(sorted_identities)
        available_additional_faces = []
        if identities_number > 1:
            max_faces_per_identity = self.max_faces_per_identity[identities_number]
            for i in range(identities_number):
                if sorted_identities[i][2] < max_faces_per_identity[i] and i < identities_number - 1:
                    sorted_identities[i+1][2] += max_faces_per_identity[i] - sorted_identities[i][2] 
                    available_additional_faces.append(0)
                elif sorted_identities[i][2] > max_faces_per_identity[i]:
                    available_additional_faces.append(sorted_identities[i][2] - max_faces_per_identity[i])
                    sorted_identities[i][2] = max_faces_per_identity[i]
                else:
                    available_additional_faces.append(0)

        else: # If only one identity is in the video, all the frames are assigned to this identity
            sorted_identities[0][2] = self.num_frames
            available_additional_faces.append(0)

        # Check if we found enough faces to fullfill the input sequence, otherwise go back and add some faces from previous identities
        input_sequence_length = sum(faces for _, _, faces in sorted_identities)
        if input_sequence_length < self.num_frames:
            for i in range(identities_number):
                needed_faces = self.num_frames - input_sequence_length
                if available_additional_faces[i] > 0:
                    added_faces = min(available_additional_faces[i], needed_faces)
                    sorted_identities[i][2] += added_faces
                    input_sequence_length += added_faces
                    if input_sequence_length == self.num_frames:
                        break
            # If not enough faces have been found, add some "dummy" images in the last identity
            if input_sequence_length < self.num_frames:
                needed_faces = self.num_frames - input_sequence_length
                sorted_identities[-1][2] += needed_faces
                input_sequence_length += needed_faces
        
        return sorted_identities, discarded_faces
    

    def __getitem__(self, index):
        video_path = self.x[index]
        video_path = os.path.join(self.data_path, video_path) 
        if self.mode not in video_path:
            for mode in MODES:
                if mode in video_path:
                    self.mode = mode
                    break

        video_id =  video_path.split(self.mode + os.path.sep)[1]

        original_video_path = os.path.join(self.video_path, self.mode, video_id)
        if ".mp4" not in original_video_path:
            original_video_path += ".mp4"
        if not os.path.exists(original_video_path) and self.mode == "val":
            original_video_path = os.path.join(self.video_path, "train", video_id)


            
        if not os.path.exists(original_video_path):
            raise Exception("Invalid video path for video.", original_video_path)


        identities, discarded_faces = self.get_sorted_identities(video_path)
  
        mask = []
        last_range_end = 0
        sequence = []
        size_embeddings = []
        
        images_frames = []
        for identity_index, identity in enumerate(identities):
            identity_path = identity[0]
            max_faces = identity[2]
            identity_faces = [os.path.join(identity_path, face) for face in os.listdir(identity_path)]

            # If no faces were considered for a frame during clustering, probably it is inside the discarded faces
            if identity_index == 0 and len(discarded_faces) > 0:
                frames = [int(os.path.basename(image_path).split("_")[0]) for image_path in identity_faces]
                discarded_frames = [int(os.path.basename(image_path).split("_")[0]) for image_path in discarded_faces]
                missing_frames = list(set(discarded_frames) - set(frames))
                missing_faces = [discarded_faces[discarded_frames.index(missing_frame)] for missing_frame in missing_frames]
                
                if len(missing_faces) > 0:
                    identity_faces = identity_faces + missing_faces # Add the missing faces to the identity
                
            identity_faces = np.asarray(sorted(identity_faces, key=lambda x:int(os.path.basename(x).split("_")[0])))

            # Select uniformly the frames in an alternate way
            if len(identity_faces) > max_faces:
                if index % 2:
                    idx = np.round(np.linspace(0, len(identity_faces) - 2, max_faces)).astype(int)
                else:
                    idx = np.round(np.linspace(1, len(identity_faces) - 1, max_faces)).astype(int)
                    
                identity_faces = identity_faces[idx]

            # Read all images files
            identity_images = []
            capture = cv2.VideoCapture(original_video_path)
            width  = capture.get(3)  
            height = capture.get(4) 
            video_area = width*height/2
            identity_size_embeddings = []
            for image_index, image_path in enumerate(identity_faces):
                # Read face image
                image = cv2.imread(image_path)
                
                # Get face-frame area ratio for size embedding
                face_area = image.shape[0] * image.shape[1] / 2
                ratio = int(face_area * 100 / video_area)
                side_ranges = list(map(lambda a_: ratio in range(a_[0], a_[1] + 1), SIZE_EMB_DICT))
                identity_size_embeddings.append(np.where(side_ranges)[0][0]+1)
               
                # Read the frame number associated with the image in order to generate the correct temporal-positional embedding
                frame = int(os.path.basename(image_path).split("_")[0])
                images_frames.append(frame)

                # Append the image to the list of readed images
                identity_images.append(image)


            # If the readed faces are less than max_faces we need to add empty images and generate the mask
            if len(identity_images) < max_faces:
                diff = max_faces - len(identity_size_embeddings)
                identity_size_embeddings = np.concatenate((identity_size_embeddings, np.zeros(diff)))
                identity_images.extend([np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8) for i in range(diff)])
                try:
                    images_frames.extend([max(images_frames) for i in range(diff)])
                except:
                    print("Error", original_video_path)
                    images_frames.extend([0 for i in range(diff)])
                    
            if self.enable_identity_attention and len(identity_images) < max_faces: # Calculate attention only between faces of the same identity
                mask.extend([1 if i < max_faces - diff else 0 for i in range(max_faces)])
            else: # Otherwise all the faces are valid
                mask.extend([1 for i in range(max_faces)])

            # Compose the size_embedding and sequence list
            size_embeddings.extend(identity_size_embeddings)
            sequence.extend(identity_images)

        # Transform the images for data augmentation, the same transformation is applied to all the faces in the same video
        additional_targets_keys = ["image" + str(i) for i in range(self.num_frames)]
        additional_targets_values = ["image" for i in range(self.num_frames)]
        additional_targets = dict(zip(additional_targets_keys, additional_targets_values))

        if self.mode == 'train':
            transform = self.create_train_transforms(self.image_size, additional_targets, self.augmentation)
        else:
            transform = self.create_val_transform(self.image_size, additional_targets) 

        if len(sequence) == 8:
            transformed_images = transform(image=sequence[0], image1=sequence[1], image2=sequence[2], image3=sequence[3], image4=sequence[4], image5=sequence[5], image6=sequence[6], image7=sequence[7])
        elif len(sequence) == 16:
            transformed_images = transform(image=sequence[0], image1=sequence[1], image2=sequence[2], image3=sequence[3], image4=sequence[4], image5=sequence[5], image6=sequence[6], image7=sequence[7], image8=sequence[8], image9=sequence[9], image10=sequence[10], image11=sequence[11], image12=sequence[12], image13=sequence[13], image14=sequence[14], image15=sequence[15])
        elif len(sequence) == 32:
            transformed_images = transform(image=sequence[0], image1=sequence[1], image2=sequence[2], image3=sequence[3], image4=sequence[4], image5=sequence[5], image6=sequence[6], image7=sequence[7], image8=sequence[8], image9=sequence[9], image10=sequence[10], image11=sequence[11], image12=sequence[12], image13=sequence[13], image14=sequence[14], image15=sequence[15], image16=sequence[16], image17=sequence[17], image18=sequence[18], image19=sequence[19], image20=sequence[20], image21=sequence[21], image22=sequence[22], image23=sequence[23], image24=sequence[24], image25=sequence[25], image26=sequence[26], image27=sequence[27], image28=sequence[28], image29=sequence[29], image30=sequence[30], image31=sequence[31])
        else:
            raise Exception("Invalid number of frames.")

        sequence = [transformed_images[key] for key in transformed_images]
          
        # Generate the identities_mask telling to the model which faces attend to an identity and which to another one
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
        if self.num_patches is not None: 
            positions = [[i+1 for i in range(((frame_position-1)*self.num_patches), self.num_patches*(frame_position))] for frame_position in frame_positions]
            positions = sum(positions, []) # Merge the lists
            positions.insert(0,0) # Add CLS
            tokens_per_identity = [(os.path.basename(identities[i][0]), identities[i][2]*self.num_patches + identities[i-1][2]*self.num_patches) if i > 0 else (os.path.basename(identities[i][0]), identities[i][2]*self.num_patches) for i in range(len(identities))]     
        else:
            positions = []
            tokens_per_identity = []

        if self.save_attention_plots == False:
            tokens_per_identity = []

        if self.multiclass_labels == None:
            return torch.tensor(sequence).float(), torch.tensor(size_embeddings).int(), torch.tensor(mask).bool(), torch.tensor(identities_mask).bool(), torch.tensor(positions), self.y[index]
        else:       
            return torch.tensor(sequence).float(), torch.tensor(size_embeddings).int(), torch.tensor(mask).bool(), torch.tensor(identities_mask).bool(), torch.tensor(positions), tokens_per_identity, self.y[index], self.multiclass_labels[index], video_id.replace("/", "_")


    def __len__(self):
        return self.n_samples
