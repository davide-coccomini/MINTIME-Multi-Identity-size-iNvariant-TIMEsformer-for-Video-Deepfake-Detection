# Utility functions for training process

import numpy as np
import torch
from matplotlib import pyplot as plt
from random import random
from scipy.special import softmax
from einops import rearrange
from statistics import mean

PLOTS_NAMES = ["space", "time", "combined"]


# Convert the preds into final video-level prediction
def check_correct(preds, labels, multiclass_labels = None, multiclass_errors = None):
    preds = [np.asarray(torch.sigmoid(pred).detach().numpy()).round() for pred in preds]

    correct = 0
    positive_class = 0
    negative_class = 0
    for i in range(len(labels)):
        pred = int(preds[i])
        if labels[i] == pred:
            correct += 1
        elif multiclass_labels != None:
            multiclass_errors[multiclass_labels[i].item()][0] += 1
        if pred == 1:
            positive_class += 1
        else:
            negative_class += 1

    if multiclass_errors != None:
        return correct, positive_class, negative_class, multiclass_errors
    else:
        return correct, positive_class, negative_class

def multiple_lists_mean(a):
    return sum(a) / len(a)

# Aggregate space and time attention 
def aggregate_attentions(attentions, heads, num_frames, frames_per_identity, scale_factor = 100000):

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
            plt.vlines(frames_per_identity[i], ymin=min(tokens_means), ymax=max(tokens_means), colors=colors[i], label = str(identity_names[i][0]))
        plt.legend()
        if len(frames_per_identity) > 1:
            plt.savefig("outputs/tokens/" + video_id + "_" + PLOTS_NAMES[index] + ".jpg")
        plt.clf()