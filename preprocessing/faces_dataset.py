# Class used in extract_features.py file

from torch.utils.data import Dataset
from PIL import Image
import torch
from torchvision import transforms
import os
import cv2
class FacesDataset(Dataset):

    def __init__(self, faces, output_dir) -> None:
        super().__init__()
        self.faces = faces
        self.output_dir = output_dir

    def __getitem__(self, index: int):
        # Preprocess the image as required by EfficientNet
        face_path = self.faces[index]
        tfms = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
        img = tfms(Image.open(face_path))

        # Compose the output path
        output_path = self.output_dir + face_path.split("faces")[1] + ".pt"
    
        return img, output_path

    def __len__(self) -> int:
        return len(self.faces)
