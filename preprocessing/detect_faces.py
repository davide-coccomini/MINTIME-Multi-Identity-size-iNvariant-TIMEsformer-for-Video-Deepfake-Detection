# The videos are given as input to the network for training and inference in the form of sequences of faces extracted from the frames. 
# Faces are detected using a MTCNN in order to extract one per second. In the case of multiple faces within the same frame, all faces are extracted.

import argparse
import json
import os
import numpy as np
from typing import Type

from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import pandas as pd
import face_detector
from face_detector import VideoDataset, VideoFaceDetector
import argparse


def process_videos(videos, detector_cls: Type[VideoFaceDetector], opt):
    
    detector = face_detector.__dict__[detector_cls](device=opt.gpu_id)

    dataset = VideoDataset(videos)
    loader = DataLoader(dataset, shuffle=False, num_workers=opt.workers, batch_size=1, collate_fn=lambda x: x)

    missed_videos = [] # Used to print videos with no detected faces
    
    # For each video in the dataset, detect faces
    for item in tqdm(loader): 
        result = {}
        video, indices, fps, frames = item[0]
        id = video.split(opt.data_path)[-1]
        out_dir = opt.output_path + id
        out_dir = out_dir.replace("video.mp4", '')

        # Skip already detected videos to improve speed
        if os.path.exists(out_dir) and "video.json" in os.listdir(out_dir):
            continue
        
        if fps == 0:
            print("Zero fps video", video)
            continue

        
        result.update({i : b for i, b in zip(indices, detector._detect_faces(frames))})

        # Save faces as json dictionary into output folder
        os.makedirs(out_dir, exist_ok=True)
        
        with open(os.path.join(out_dir, "video.json"), "w") as f:
            json.dump(result, f)
        
        # Check if some faces have been detected
        found_faces = False
        for key in result:
            if type(result[key]) == list:
                found_faces = True
                break

        if not found_faces:
            print("Faces not found", video)
            missed_videos.append(video)

    # Display the missed videos
    if len(missed_videos) > 0:
        print("The detector did not find faces inside the following videos:")
        print(missed_videos)
        print(len(missed_videos))
        print("We suggest to re-run the code decreasing the detector threshold.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--list_file', default="../../datasets/ForgeryNet/Validation/video_list.txt", type=str,
                        help='Video List txt file path)')
    parser.add_argument('--data_path', type=str,
                        help='Data directory', default='../../datasets/ForgeryNet/Validation/video')
    parser.add_argument('--output_path', type=str,
                        help='Output directory', default='../../datasets/ForgeryNet/Validation/boxes')
    parser.add_argument("--detector_type", help="type of the detector", default="FacenetDetector",
                        choices=["FacenetDetector"])
    parser.add_argument('--gpu_id', default=0, type=int,
                        help='ID of GPU to be used')
    parser.add_argument('--workers', default=40, type=int,
                        help='Number of data loader workers.')

    opt = parser.parse_args()
    print(opt)


    # Read videos paths from which the user wants to detect faces
    with open(opt.list_file, 'r') as temp_f:
        col_count = [ len(l.split(" ")) for l in temp_f.readlines() ]

    column_names = [i for i in range(0, max(col_count))]
    df = pd.read_csv(opt.list_file, sep=' ', names=column_names)   
    videos_paths = df.values.tolist()
    videos_paths = list(dict.fromkeys([os.path.join(opt.data_path, os.path.dirname(row[1].split(" ")[0]), "video.mp4") for row in videos_paths]))
    
    # Ignore already extracted videos to improve speed
    excluded_videos = []
    for path in videos_paths:
        id = path.split(opt.data_path)[-1]
        out_dir = opt.output_path + id
        out_dir = out_dir.replace("video.mp4", '')
        if os.path.exists(out_dir) and "video.json" in os.listdir(out_dir):
            excluded_videos.append(path)
    
    videos_paths = [video_path for video_path in videos_paths if video_path not in excluded_videos]
    print("Excluded videos:", len(excluded_videos))

    # Start face detection
    process_videos(videos_paths, opt.detector_type, opt)
    
if __name__ == "__main__":
    main()
