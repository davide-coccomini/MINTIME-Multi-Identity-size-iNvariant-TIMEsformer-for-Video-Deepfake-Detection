# Utility functions for training process

import numpy as np
import torch

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