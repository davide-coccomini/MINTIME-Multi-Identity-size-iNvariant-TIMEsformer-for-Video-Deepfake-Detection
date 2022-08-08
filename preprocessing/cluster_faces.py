# Since several subjects can be found within a video, it is necessary to cluster them into groups based on similarity. 
# This operation is carried out in the following code with additional attention to maintaining the temporal coherence of faces.
# The extracted faces are reorganised into consecutive sequences of similar faces so as to be more suitable for network processing.


import argparse
import os
import glob
import torch
import numpy as np
import pandas as pd
import shutil
from functools import partial
from multiprocessing.pool import Pool
from numpy.linalg import norm
from PIL import Image
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1, fixed_image_standardization
from collections import OrderedDict
from sklearn.cluster import KMeans
from torch.utils.data.dataloader import DataLoader
from progress.bar import ChargingBar
from utils import preprocess_images, _generate_connected_components

seed = 42
def move_files(face_paths):
    src_path, dst_path = face_paths
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    shutil.move(src_path, dst_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--faces_path', default="../../datasets/ForgeryNet/faces", type=str,
                        help='Path of folder containing train/val/test with extracted cropped faces to be clustered.')
    parser.add_argument('--gpu_id', default=0, type=int,
                        help='ID of GPU to be used.')
    parser.add_argument('--similarity_threshold', default=0.45, type=float,
                        help='Threshold to discard faces with high distance.')
    parser.add_argument('--valid_cluster_size_ratio', default=0.20, type=int,
                        help='Valid cluster size ratio.')
    parser.add_argument('--workers', default=40, type=int,
                        help='Number of data loader workers.')

    opt = parser.parse_args()
    print(opt)

    # Get all the paths of the videos to be clustered 
    for dataset in os.listdir(opt.faces_path):
        dataset_path = os.path.join(opt.faces_path, dataset)
        if not os.path.isdir(dataset_path):
            continue

        print()
        print("Clustering videos in ", dataset_path)
        set_paths = glob.glob(f'{dataset_path}/*/**/*.mp4', recursive=True)
    
        excluded_videos = []
        for path in set_paths:
            if os.path.exists(os.path.join(path, "0")):
                excluded_videos.append(path)

        set_paths = [video_path for video_path in set_paths if video_path not in excluded_videos]
        print("Excluded already clustered videos: ", len(excluded_videos))
        
        # For each video in each set, perform faces clustering
        bar = ChargingBar('Clustered videos', max=(len(set_paths)))
        for path in set_paths:
            # Read all faces, load them into a dictionary 
            faces_files = [face_file for face_file in os.listdir(path) if not os.path.isdir(os.path.join(path, face_file))]
            faces_files = sorted(faces_files, key=lambda x:(int(x.split("_")[0]), int(os.path.splitext(x)[0].split("_")[1])))
            mapping = {}
            faces = []
            
            for index, face_file in enumerate(faces_files):
                face_path = os.path.join(path, face_file)
                frame_number = int(os.path.splitext(face_file)[0].split("_")[0])
                face = Image.open(face_path)
                faces.append(face)
                mapping[index] = face_path
               

            
            # Extract the embeddings
            embeddings_extractor = InceptionResnetV1(pretrained='vggface2').eval().to(opt.gpu_id)
            faces = [preprocess_images(face) for face in faces]
            faces = np.stack([np.uint8(face) for face in faces])
            faces = torch.as_tensor(faces)
            faces = faces.permute(0, 3, 1, 2).float()
            faces = fixed_image_standardization(faces)
            face_recognition_input = faces.cuda()
            embeddings = []
            embeddings = embeddings_extractor(face_recognition_input).detach().cpu().numpy()

            # Clustering
            valid_cluster_size = int(len(mapping) * opt.valid_cluster_size_ratio)
            similarities = np.dot(np.array(embeddings), np.array(embeddings).T)

            components = _generate_connected_components(
                similarities, similarity_threshold=opt.similarity_threshold
            )
            components = [sorted(component) for component in components]

            mapped_components = []
            for identity_index, component in enumerate(components):
                for index in component:
                    src_path = mapping[index]
                    folder_path = os.path.dirname(src_path)
                    file_name = os.path.basename(src_path)
                    dst_path = os.path.join(folder_path, str(identity_index), file_name)
                    mapped_components.append((src_path, dst_path))

            
            # Organize the clusters inside the folder
            with Pool(processes=opt.workers) as p:
                for v in p.imap_unordered(move_files, mapped_components):
                    continue

            bar.next()
        
        print()
