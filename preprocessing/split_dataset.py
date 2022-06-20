import os
import argparse
import pandas as pd
import math
import matplotlib.pyplot as plt
import collections
import random
import shutil
import glob
import csv

seed = 42

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_list_file', default="../../datasets/ForgeryNet/Training/video_list_complete.txt", type=str,
                        help='Videos List txt file path for training set (to be splitted in train and validation)')
    parser.add_argument('--validation_list_file', default="../../datasets/ForgeryNet/Validation/video_list.txt", type=str,
                        help='Videos List txt file path for validation set (our test set)')
    parser.add_argument('--plots_output_path', default="../outputs", type=str,
                        help='Plots output path')
    parser.add_argument('--faces_path', default="../../datasets/ForgeryNet/faces", type=str,
                        help='Images path')
    parser.add_argument('--validation_set_output_path', default="../../datasets/ForgeryNet/faces/val", type=str,
                        help='Test set output path')
    parser.add_argument('--train_faces_path', default="../../datasets/ForgeryNet/faces/train", type=str,
                        help='Train images path')
    parser.add_argument('--test_faces_path', default="../../datasets/ForgeryNet/faces/test", type=str,
                        help='Test images path')

    opt = parser.parse_args()
    print(opt)
    datasets = {"train": {}, "val": {}, "test": {}}
    paths = glob.glob(f'{opt.train_faces_path}/*/**/*.mp4', recursive=True)

    with open(opt.training_list_file, 'r') as temp_f:
        col_count = [ len(l.split(" ")) for l in temp_f.readlines() ]

        column_names = [i for i in range(0, max(col_count))]
        df = pd.read_csv(opt.training_list_file, sep=' ', names=column_names)
      
        training_counter = {}        
        column_names.reverse()
        skipped = 0
        for index, row in df.iterrows():
            video_name = os.path.join(opt.train_faces_path, row[1].split("train_video_release/")[-1])
            if video_name not in paths:
                skipped += 1
                continue

            for column_name in column_names:
                if not math.isnan(row[column_name]):
                    deepfake_class = row[column_name]
                    break
            
            if deepfake_class in training_counter:
                training_counter[deepfake_class] += 1
            else:
                training_counter[deepfake_class] = 1

            if deepfake_class in datasets["train"]:
                datasets["train"][deepfake_class].append(video_name.replace("train_video_release", "train").replace(opt.train_faces_path, "train"))
            else:
                datasets["train"][deepfake_class] = [video_name.replace("train_video_release", "train").replace(opt.train_faces_path, "train")]

        print(skipped, "videos in training set without detected faces skipped.")
        training_counter = collections.OrderedDict(sorted(training_counter.items()))
        total_training_samples = len(df)
        validation_size = total_training_samples/10
        total = 0
        validation_counter = {}
        for key in training_counter:
            percentage = training_counter[key]/total_training_samples
            elements = validation_size*percentage
            validation_counter[key] = int(elements)
            training_counter[key] -= elements

        validation_counter = collections.OrderedDict(sorted(validation_counter.items()))


        names = list(training_counter.keys())
        values = list(training_counter.values())
        x = [i-0.3 for i in range(len(training_counter))]
        plt.bar(x, values, 0.3, tick_label=names, label = "Training Set")

        
        names = list(validation_counter.keys())
        values = list(validation_counter.values())
        x = [i for i in range(len(training_counter))]
        plt.bar(x, values, 0.3, tick_label=names, label = "Validation Set")

    
    skipped = 0

    with open(opt.validation_list_file, 'r') as temp_f:
        col_count = [ len(l.split(" ")) for l in temp_f.readlines() ]

        column_names = [i for i in range(0, max(col_count))]
        df = pd.read_csv(opt.validation_list_file, sep=' ', names=column_names)
      
        test_counter = {}        
        column_names.reverse()

        paths = glob.glob(f'{opt.test_faces_path}/*/**/*.mp4', recursive=True)
        for index, row in df.iterrows():
            video_name = os.path.join(opt.test_faces_path, row[1].split("val_video_release/")[-1])
            if video_name not in paths:
                skipped += 1
                continue

            for column_name in column_names:
                if not math.isnan(row[column_name]):
                    deepfake_class = row[column_name]
                    break
            
            if deepfake_class in test_counter:
                test_counter[deepfake_class] += 1
            else:
                test_counter[deepfake_class] = 1

            if deepfake_class in datasets["test"]:
                datasets["test"][deepfake_class].append(video_name.replace("val_video_release", "test").replace(opt.test_faces_path, "test"))
            else:
                datasets["test"][deepfake_class] = [video_name.replace("val_video_release", "test").replace(opt.test_faces_path, "test")]
    
        print(skipped, "videos in test set without detected faces skipped.")
    test_counter = collections.OrderedDict(sorted(test_counter.items()))

    names = list(test_counter.keys())
    values = list(test_counter.values())
    
    x = [i+0.3 for i in range(len(test_counter))]
    plt.bar(x, values, 0.3, tick_label=names, label = "Test Set")

    plt.legend()
    plt.savefig(os.path.join(opt.plots_output_path, "distribution"))


# Move files
for deepfake_class in datasets["train"]:
    number_of_elements = validation_counter[deepfake_class]
    extracted_elements = random.Random(seed).sample(datasets["train"][deepfake_class],number_of_elements)
    for index, video_name in enumerate(extracted_elements):
        out_path = os.path.join(opt.validation_set_output_path, video_name.split("Training/video")[-1])
        src_path = os.path.join(opt.faces_path, video_name).replace("train_video_release", "train")
        datasets["train"][deepfake_class].remove(video_name)
        if deepfake_class in datasets["val"]:
            datasets["val"][deepfake_class].append(video_name.replace("train", "val"))
        else:
            datasets["val"][deepfake_class] = [video_name.replace("train", "val")]
        if index % 500 == 0:
            print("Moved", index, "videos into validation set.")
        shutil.move(src_path, out_path)
    
# Save labels
for key in datasets:
    f = open(os.path.join(opt.faces_path, key+".csv"), 'w+')
    dataset = datasets[key]
    for deepfake_class in dataset:
        if deepfake_class == 0:
            binary_class = "0"
        else:
            binary_class = "1"
        for video in dataset[deepfake_class]:
            row = video + " " + binary_class + " " + str(int(deepfake_class)) + "\n"
            f.write(row)
            
    f.close()

    