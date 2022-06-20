import argparse
import json
import os
import numpy as np
from typing import Type

from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import pandas as pd
import face_detector
from face_detector import VideoDataset
from face_detector import VideoFaceDetector
import argparse


def process_videos(videos, detector_cls: Type[VideoFaceDetector], opt):
    detector = face_detector.__dict__[detector_cls](device="cuda:0")
    dataset = VideoDataset(videos)

    loader = DataLoader(dataset, shuffle=False, num_workers=40, batch_size=1, collate_fn=lambda x: x)
    missed_videos = []
    for item in tqdm(loader): 
        result = {}
        video, indices, fps, frames = item[0]
        id = os.path.splitext(os.path.basename(video))[0]
        tmp = video.split("Training/video")[-1]
        out_dir = opt.output_path + tmp
        out_dir = out_dir.replace("video.mp4", '')

        if os.path.exists(out_dir) and "{}.json".format(id) in os.listdir(out_dir):
            continue

        if fps == 0:
            print("Zero fps video", video)
            continue

        frames = [frames[i] for i in range(0, len(frames), fps)]
        result.update({i : b for i, b in zip(indices, detector._detect_faces(frames))})
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "{}.json".format(id)), "w") as f:
            json.dump(result, f)
        
        found_faces = False
        for key in result:
            if type(result[key]) == list:
                found_faces = True
                break
        if not found_faces:
            print("Faces not found", video)
            missed_videos.append(video)

    if len(missed_videos) > 0:
        print("The detector did not find faces inside the following videos:")
        print(missed_videos)
        print(len(missed_videos))
        print("We suggest to re-run the code decreasing the detector threshold.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--list_file', default="outputs/2.txt", type=str,
                        help='Video List txt file path)')
    parser.add_argument('--data_path', type=str,
                        help='Data directory', default='../../datasets/ForgeryNet/Training/video')
    parser.add_argument('--output_path', type=str,
                        help='Output directory', default='../../datasets/ForgeryNet/Training/boxes')
    parser.add_argument("--detector_type", help="type of the detector", default="FacenetDetector",
                        choices=["FacenetDetector"])
                        
    opt = parser.parse_args()
    print(opt)
    with open(opt.list_file, 'r') as temp_f:
        col_count = [ len(l.split(" ")) for l in temp_f.readlines() ]

    column_names = [i for i in range(0, max(col_count))]
    df = pd.read_csv(opt.list_file, sep=' ', names=column_names)   

    videos_paths = df.values.tolist()

    videos_paths = list(dict.fromkeys([os.path.join(opt.data_path, os.path.dirname(row[1].split(" ")[0]), "video.mp4") for row in videos_paths]))
    excluded_videos = []
    for path in videos_paths:
        id = os.path.splitext(os.path.basename(path))[0]
        tmp = path.split("Training/video")[-1]
        out_dir = opt.output_path + tmp
        out_dir = out_dir.replace("video.mp4", '')
        if os.path.exists(out_dir) and "{}.json".format(id) in os.listdir(out_dir):
            excluded_videos.append(path)

    paths = [video_path for video_path in videos_paths if video_path not in excluded_videos]
    print("Excluded videos:", len(excluded_videos))
    process_videos(paths, opt.detector_type, opt)
    
if __name__ == "__main__":
    main()