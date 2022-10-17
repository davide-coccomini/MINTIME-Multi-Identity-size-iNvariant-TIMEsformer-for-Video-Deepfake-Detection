# Utility functions for training process

import numpy as np
import torch
from matplotlib import pyplot as plt
from random import random
from scipy.special import softmax
from einops import rearrange
from statistics import mean
import cv2
import math
from typing import Dict
import json
import urllib
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
    UniformCropVideo
) 

PLOTS_NAMES = ["space", "time", "combined"]


# Convert the preds into final video-level prediction
def check_correct(preds, labels, multiclass_labels = None, multiclass_errors = None, videos_ids = None):
    preds = [np.asarray(torch.sigmoid(pred).detach().numpy()).round() for pred in preds]

    correct = 0
    positive_class = 0
    negative_class = 0
    videos_errors = []
    for i in range(len(labels)):
        pred = int(preds[i])
        if labels[i] == pred:
            correct += 1
        if labels[i] != pred:
            if multiclass_labels is not None and not math.isnan(multiclass_labels[i]):
                multiclass_errors[multiclass_labels[i].item()][0] += 1
            if videos_ids != None:
                videos_errors.append(videos_ids[i])
            
        if pred == 1:
            positive_class += 1
        else:
            negative_class += 1

    if multiclass_errors != None:
        return correct, positive_class, negative_class, multiclass_errors, videos_errors
    else:
        return correct, positive_class, negative_class


def unix_time_millis(dt):
    return dt.total_seconds() * 1000.0


def multiple_lists_mean(a):
    return sum(a) / len(a)

# Aggregate space and time attention 
def aggregate_attentions(attentions, heads, num_frames, frames_per_identity, scale_factor = 50000):

    # Collapse attentions heads for each attention separated
    aggregated_attentions = []
    for attention in attentions:
        attention = attention.squeeze(1)
        attention = rearrange(attention, '(b h) t -> b h t', h = heads)
        tokens_means = [torch.max(attention[:, :, i]).item() for i in range(attention.shape[2])]        
        aggregated_attentions.append(tokens_means)

    # Combined space and time attention
    tokens_means_combined = list(np.sum(np.asarray(aggregated_attentions), axis=0))
    aggregated_attentions.append(tokens_means_combined)

    # Softmax all the attentions
    for i in range(len(aggregated_attentions)):
        aggregated_attentions[i] = np.array_split(np.asarray(aggregated_attentions[i]), num_frames)
        aggregated_attentions[i] = softmax([mean(values)*scale_factor for values in aggregated_attentions[i]])

    identity_attentions = []
    for index, identity_frames in enumerate(frames_per_identity):
        if index == 0:
            identity_attention = sum(aggregated_attentions[-1][:identity_frames-1])
        else:
            previous_identity_frames = frames_per_identity[index-1]
            identity_attention = sum(aggregated_attentions[-1][previous_identity_frames-1:identity_frames-1])
        identity_attentions.append(identity_attention)

    return aggregated_attentions, identity_attentions


# Visualize the attention
def save_attention_plots(aggregated_attentions, identity_names, frames_per_identity, num_frames, video_id):
    colors = np.random.rand(len(frames_per_identity), 4)
    for index, tokens_means in enumerate(aggregated_attentions):
        plt.bar([i+1 for i in range(num_frames)], tokens_means)
        for i in range(len(frames_per_identity)):
            plt.vlines(frames_per_identity[i], ymin=min(tokens_means), ymax=max(tokens_means), colors=colors[i], label = str(identity_names[i]))
        plt.legend()
        plt.savefig("outputs/tokens/" + video_id + "_" + PLOTS_NAMES[index] + ".jpg")
        plt.clf()


def draw_border(img, pt1, pt2, color, thickness, r, d):
    x1,y1 = pt1
    x2,y2 = pt2

    # Top left
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)

    # Top right
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)

    # Bottom left
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)

    # Bottom right
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)
    return img



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)




SLOWFAST_ALPHA = 4

class PackPathway(torch.nn.Module):
    """
    Transform for converting video frames as a list of tensors. 
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, frames: torch.Tensor):
        fast_pathway = frames
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(
                0, frames.shape[1] - 1, frames.shape[1] // SLOWFAST_ALPHA
            ).long(),
        )
        frame_list = [slow_pathway, fast_pathway]
        return frame_list

def slowfast_input_transform(videos, crop_size = 256, side_size = 256, num_frames = 32, sampling_rate = 2, frames_per_second = 30, mean = [0.45, 0.45, 0.45], std = [0.225, 0.225, 0.225]):
    transform=Compose(
        [
            UniformTemporalSubsample(num_frames),
            Lambda(lambda x: x/255.0),
            NormalizeVideo(mean, std),
            ShortSideScale(
                size=side_size
            ),
            CenterCropVideo(crop_size),
            PackPathway()
        ]
    )
    transformed_videos = [[],[]]
    for video in videos:
        output = transform(video)
        transformed_videos[0].append(output[0])
        transformed_videos[1].append(output[1])
    
        
    return transformed_videos