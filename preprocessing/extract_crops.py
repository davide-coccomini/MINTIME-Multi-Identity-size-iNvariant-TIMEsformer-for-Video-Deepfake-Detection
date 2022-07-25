# Following face detection, the json files containing the coordinates framing the faces identified by the MTCNN must be converted into images. 

import argparse
import json
import os
from os import cpu_count
from pathlib import Path
from collections import OrderedDict

import pandas as pd
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
from functools import partial
from glob import glob
from multiprocessing.pool import Pool

import cv2

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)
from tqdm import tqdm

def extract_video(video, data_path):
    # Composes the path where the coordinates of the detected faces were saved
    bboxes_path = data_path +  "/boxes_better/" + video.split("video/")[-1].split(".")[0] + ".json"
    if not os.path.exists(bboxes_path) or not os.path.exists(video):
        print(bboxes_path, "not found\n")
        return

    # Load the json dictionary and the corresponding video
    with open(bboxes_path, "r") as bbox_f:
        bboxes_dict = json.load(bbox_f)
    capture = cv2.VideoCapture(video)
    frames_num = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(capture.get(5))
   

    # For each frame, save the detected faces into files
    frames = []
    for i in range(frames_num):
        capture.grab()
        success, frame = capture.retrieve()
        if not success:
            continue
        frames.append(frame)

    explored_indexes = []

    for i in range(0, len(frames), fps):
        while str(i) not in bboxes_dict:
            if i == frames_num - 1:
                i -= 1
            if i in explored_indexes:
                break
            else:
                explored_indexes.append(i)

        frame = frames[i]
        id = os.path.splitext(os.path.basename(video))[0]
        crops = []
        index = i
        limit = i + fps - 1
        keys = [int(x) for x in list(bboxes_dict.keys())]

        while index < limit:
            index += 1
            if index in keys and bboxes_dict[str(index)] is not None:
                break
        if index == limit:
            continue

        bboxes = bboxes_dict[str(index)]

        for bbox in bboxes:
            xmin, ymin, xmax, ymax = [int(b * 2) for b in bbox]
            w = xmax - xmin
            h = ymax - ymin

            # Add some padding to catch background too
            p_h = h // 3
            p_w = w // 3
           
            crop_h = (ymax + p_h) - max(ymin - p_h, 0)
            crop_w = (xmax + p_w) - max(xmin - p_w, 0)

            # Make the image square
            if crop_h > crop_w:
                p_h -= int(((crop_h - crop_w)/2))
            else:
                p_w -= int(((crop_w - crop_h)/2))

            # Extract the face from the frame
            crop = frame[max(ymin - p_h, 0):ymax + p_h, max(xmin - p_w, 0):xmax + p_w]
            
            # Check if out of bound and correct
            h, w = crop.shape[:2]
            if h > w:
                diff = int((h - w)/2)
                if diff > 0:         
                    crop = crop[diff:-diff,:]
                else:
                    crop = crop[1:,:]
            elif h < w:
                diff = int((w - h)/2)
                if diff > 0:
                    crop = crop[:,diff:-diff]
                else:
                    crop = crop[:,:-1]

            
            # Add the extracted face to the list
            crops.append(crop)

        # Save the extracted faces into files
        tmp = video.split("release")[1]
        out_dir = opt.output_path + tmp
        os.makedirs(out_dir, exist_ok=True)
        for j, crop in enumerate(crops):
            try:
                cv2.imwrite(os.path.join(out_dir, "{}_{}.png".format(i, j)), crop)
            except:
                print("Error writing image")
    
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--list_file', default="../../datasets/ForgeryNet/Training/video_list.txt", type=str,
                        help='Images List txt file path)')
    parser.add_argument('--data_path', default='../../datasets/ForgeryNet/Training', type=str,
                        help='Videos directory')
    parser.add_argument('--output_path', default='../../datasets/ForgeryNet/Training/faces_fix/crops_fix', type=str,
                        help='Output directory')
    parser.add_argument('--gpu_id', default=0, type=int,
                        help='ID of GPU to be used')
    parser.add_argument('--workers', default=40, type=int,
                        help='Number of data loader workers.')

    opt = parser.parse_args()
    print(opt)

    # Read the dataset
    with open(opt.list_file, 'r') as temp_f:
        col_count = [ len(l.split(" ")) for l in temp_f.readlines() ]
    column_names = [i for i in range(0, max(col_count))]
    os.makedirs(opt.output_path, exist_ok=True)
    df = pd.read_csv(opt.list_file, sep=' ', names=column_names)
    videos_paths = df.values.tolist()
    videos_paths = list(dict.fromkeys([os.path.join(opt.data_path, "video", os.path.dirname(row[1].split(" ")[0]), "video.mp4") for row in videos_paths]))
  
    # Start face extraction
    with Pool(processes=opt.workers) as p:
        with tqdm(total=len(videos_paths)) as pbar:
            for v in p.imap_unordered(partial(extract_video, data_path=opt.data_path), videos_paths):
                pbar.update()

    