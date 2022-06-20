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

    bboxes_path = data_path +  "/boxes" + video.split("Training/video")[-1].split(".")[0] + ".json"
    if not os.path.exists(bboxes_path) or not os.path.exists(video):
        print(bboxes_path, "not found")
        return
    with open(bboxes_path, "r") as bbox_f:
        bboxes_dict = json.load(bbox_f)
    capture = cv2.VideoCapture(video)
    frames_num = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
   
    counter = 0
    for i in range(frames_num):
        capture.grab()
        success, frame = capture.retrieve()
        if not success or str(i) not in bboxes_dict:
            continue
        id = os.path.splitext(os.path.basename(video))[0]
        crops = []
        bboxes = bboxes_dict[str(i)]
        if bboxes is None:
            continue
        else:
            counter += 1

        for bbox in bboxes:
            xmin, ymin, xmax, ymax = [int(b * 2) for b in bbox]
            w = xmax - xmin
            h = ymax - ymin
            p_h = h // 3
            p_w = w // 3
           
            '''
            if h > w:
                p_w += int((h - w)/2)
            else:
                p_h += int((w - h)/2)
            '''

            crop_h = (ymax + p_h) - max(ymin - p_h, 0)
            crop_w = (xmax + p_w) - max(xmin - p_w, 0)

            if crop_h > crop_w:
                p_h -= int(((crop_h - crop_w)/2))
            else:
                p_w -= int(((crop_w - crop_h)/2))

            

            crop = frame[max(ymin - p_h, 0):ymax + p_h, max(xmin - p_w, 0):xmax + p_w]
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


            crops.append(crop)

        tmp = video.split("release")[1]
        out_dir = opt.output_path + tmp
        os.makedirs(out_dir, exist_ok=True)
        for j, crop in enumerate(crops):
            try:
                cv2.imwrite(os.path.join(out_dir, "{}_{}.png".format(i, j)), crop)
            except:
                print("Error writing image")
    #if counter == 0:
        #print(video, counter)
    
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--list_file', default="outputs/8.txt", type=str,
                        help='Images List txt file path)')
    parser.add_argument('--data_path', default='../../datasets/ForgeryNet/Training', type=str,
                        help='Videos directory')
    parser.add_argument('--output_path', default='../../datasets/ForgeryNet/Training/crops', type=str,
                        help='Output directory')

    opt = parser.parse_args()
    print(opt)
    with open(opt.list_file, 'r') as temp_f:
        col_count = [ len(l.split(" ")) for l in temp_f.readlines() ]

    column_names = [i for i in range(0, max(col_count))]
    

    
    os.makedirs(opt.output_path, exist_ok=True)
    #excluded_videos = os.listdir(os.path.join(opt.output_dir)) # Useful to avoid to extract from already extracted videos
    #excluded_videos = os.listdir(opt.output_path)
    df = pd.read_csv(opt.list_file, sep=' ', names=column_names)
   

    videos_paths = df.values.tolist()

    videos_paths = list(dict.fromkeys([os.path.join(opt.data_path, "video", os.path.dirname(row[1].split(" ")[0]), "video.mp4") for row in videos_paths]))

    with Pool(processes=cpu_count()-2) as p:
        with tqdm(total=len(videos_paths)) as pbar:
            for v in p.imap_unordered(partial(extract_video, data_path=opt.data_path), videos_paths):
                pbar.update()






def scale_bounding_box(coords, height, width, scale=1.2, minsize=None):
    """
    Transforms bounding boxes to square and muptilpies it with a scale
    """
    if len(coords) == 3:
        x1, y1, size_bb = coords
        x1, y1, x2, y2 = x1, y1, x1 + size_bb, y1 + size_bb
    else:
        x1, y1, x2, y2 = coords
    size_bb = int(max(x2 - x1, y2 - y1) * scale)
    if minsize:
        if size_bb < minsize:
            size_bb = minsize
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
    # Check for out of bounds, x-y top left corner
    x1 = max(int(center_x - size_bb // 2), 0)
    y1 = max(int(center_y - size_bb // 2), 0)
    # Check for too big bb size for given x, y
    size_bb = min(width - x1, size_bb)
    size_bb = min(height - y1, size_bb)
    return x1, y1, x1 + size_bb, y1 + size_bb


def isotropically_resize_image(
    img, size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC
):
    h, w = img.shape[:2]
    if max(w, h) == size:
        return img
    if w > h:
        scale = size / w
        h = h * scale
        w = size
    else:
        scale = size / h
        w = w * scale
        h = size
    interpolation = interpolation_up if scale > 1 else interpolation_down
    resized = cv2.resize(img, (int(w), int(h)), interpolation=interpolation)
    return resized
