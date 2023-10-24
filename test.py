
import torch
import numpy as np
import argparse
from tqdm import tqdm
import math
import yaml
from utils import check_correct, aggregate_attentions, save_attention_plots, count_parameters,  slowfast_input_transform
from torch.optim.lr_scheduler import LambdaLR
from datetime import datetime, timedelta
from statistics import mean
import tensorflow as tf
import collections
import os
import json
from sklearn import metrics
from sklearn.metrics import f1_score
from itertools import chain
import random
from einops import rearrange, reduce
import pandas as pd
from os import cpu_count
from multiprocessing.pool import Pool
from functools import partial
from multiprocessing import Manager
from progress.bar import ChargingBar
from torch.optim import lr_scheduler
from deepfakes_dataset import DeepFakesDataset
from models.size_invariant_timesformer import SizeInvariantTimeSformer
from models.efficientnet.efficientnet_pytorch import EfficientNet
from torch.utils.tensorboard import SummaryWriter
import torch_optimizer as optim
from timm.scheduler.cosine_lr import CosineLRScheduler
from models.baseline import Baseline
from models.xception import xception




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--test_list_file', default="../../datasets/ForgeryNet/faces/test.csv", type=str,
                        help='Test List txt file path)')  
    parser.add_argument('--data_path', default="../../datasets/ForgeryNet/faces", type=str,
                        help='Path to the dataset converted into identities.')
    parser.add_argument('--video_path', default="../../datasets/ForgeryNet/videos", type=str,
                        help='Path to the dataset original videos (.mp4 files).')
    parser.add_argument('--deepfake_methods', nargs='*', required=False,
                        help="For ForgeryNet dataset, filter some deepfake methods for partial training.")
    parser.add_argument('--workers', default=8, type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--random_state', default=42, type=int,
                        help='Random state value')
    parser.add_argument('--model_weights', type=str,
                        help='Model weights.')
    parser.add_argument('--extractor_model', type=int, default=0, 
                        help="Which model use for features extraction (0: EfficientNet; 1: XceptionNet).")
    parser.add_argument('--extractor_weights', default='ImageNet', type=str,
                        help='Path to extractor weights or "imagenet".')
    parser.add_argument('--gpu_id', default=0, type=int,
                        help='ID of GPU to be used.')
    parser.add_argument('--max_videos', type=int, default=-1, 
                        help="Maximum number of videos to use for training (default: all).")                  
    parser.add_argument('--only_multiidentity', default=False, action="store_true",
                        help='Use only multiidentity videos.')
    parser.add_argument('--config', type=str, 
                        help="Which configuration to use. See into 'config' folder.")
    parser.add_argument('--model', type=int, 
                        help="Which model to use. (0: Baseline | 1: Size Invariant TimeSformer | 2: SlowFast).")
    parser.add_argument('--identities_ordering', type=int,  default = 0,
                        help="Which ordering rule to use. (0: Size-based | 1: Frequency-based | 2: Random).")
    parser.add_argument('--save_attentions', default=False, action="store_true",
                        help='Save attentions plots.')
    opt = parser.parse_args()
    
    print(opt)
    with open(opt.config, 'r') as ymlfile:
        config = yaml.safe_load(ymlfile)

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Check for integrity
    if config['model']['num-frames'] != 8 and config['model']['num-frames'] != 16:
        raise Exception("Invalid number of frames.")
        
        
    # Setup CUDA settings 
    torch.backends.cudnn.deterministic = True
    random.seed(opt.random_state)
    torch.manual_seed(opt.random_state)
    torch.cuda.manual_seed(opt.random_state)
    np.random.seed(opt.random_state)
   
    # Load required weights for feature extractor
    if opt.model != 2:
        if opt.extractor_model == 0: # EfficientNet-B0
            if opt.extractor_weights.lower() == 'imagenet':
                features_extractor = EfficientNet.from_pretrained('efficientnet-b0')
            else:
                features_extractor = EfficientNet.from_name('efficientnet-b0')
                features_extractor.load_matching_state_dict(torch.load(opt.extractor_weights, map_location=torch.device('cpu')))
                print("Custom features extractor weights loaded.")
        else: # XceptionNet
            if opt.extractor_weights.lower() == 'pretrained':
                features_extractor = xception(num_classes=1, pretrain_path="weights/ckpt_iter.pth.tar")
            else:
                features_extractor = xception(num_classes=1, pretrain_path=opt.extractor_weights)
    else:
        features_extractor = None

    # Init the required model
    if opt.model == 0:
        model = Baseline(config=config)
        num_patches = None
    elif opt.model == 1:
        model = SizeInvariantTimeSformer(config=config, require_attention=True)
        num_patches = config['model']['num-patches']
    elif opt.model == 2:
        torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
        model = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=True)   
        output_layer = torch.nn.Linear(2304 , 1)
        model.blocks[6].proj = output_layer
        num_patches = None


    if features_extractor != None and opt.gpu_id == -1:
        features_extractor = torch.nn.DataParallel(features_extractor)

    if opt.gpu_id == -1:
        model = torch.nn.DataParallel(model)


    if os.path.exists(opt.model_weights):
        model.load_state_dict(torch.load(opt.model_weights))
    else:
        raise Exception("No checkpoint loaded for the model.")    

    loss_fn = torch.nn.BCEWithLogitsLoss()

    # Move into GPU
    if features_extractor != None:
        features_extractor = features_extractor.to(device)   
        features_extractor.eval()
        print("Extractor Parameters: ", count_parameters(features_extractor))  
    print("Model Parameters: ", count_parameters(model)) 
    model = model.to(device)
    model.eval()
       
    # Read all the paths and initialize data loaders for train and validation
    paths = []
    col_names = ["video", "label", "8_cls"]
    df_test = pd.read_csv(opt.test_list_file, sep=' ', names=col_names)
    df_test = df_test.sample(frac=1, random_state=opt.random_state).reset_index(drop=True)

    
    # Filter out deepfake methods if requested for ForgeryNet
    if opt.deepfake_methods is not None and len(opt.deepfake_methods) > 0:
        opt.deepfake_methods = [int(method) for method in opt.deepfake_methods]
        indexes_to_drop = []
        for index, row in df_test.iterrows():
            if row['8_cls'] not in opt.deepfake_methods:
                indexes_to_drop.append(index)
        df_test.drop(df_test.index[indexes_to_drop], inplace=True)
    
    # Filter out non-multi-identity videos if requested
    if opt.only_multiidentity:
        indexes_to_drop = []
        for index, row in df_test.iterrows():
            video_path = os.path.join(opt.data_path, row['video']) 
            folders = os.listdir(video_path)
            if len(folders) < 2:
                indexes_to_drop.append(index)
            else:
                counter = 0
                for folder in folders:
                    if os.path.isdir(os.path.join(opt.data_path, row['video'], folder)):
                        counter += 1
                if counter < 2:
                    indexes_to_drop.append(index)
                
        df_test.drop(df_test.index[indexes_to_drop], inplace=True)
            
    # Split videos and labels and reduce to the required number of videos
    test_videos = df_test['video'].tolist()
    test_labels = df_test['label'].tolist()
    multiclass_labels = df_test['8_cls'].tolist()
    class_counter = collections.Counter(multiclass_labels)

    if opt.max_videos > -1:
        test_videos = test_videos[:opt.max_videos]
        test_labels = test_labels[:opt.max_videos]
    
    test_samples = len(test_videos)

    # Create the data loaders 
    test_dataset = DeepFakesDataset(test_videos, test_labels, multiclass_labels = multiclass_labels, image_size=config['model']['image-size'], data_path=opt.data_path, video_path=opt.video_path, num_frames=config['model']['num-frames'], num_patches=num_patches, max_identities=config['model']['max-identities'], enable_identity_attention=config['model']['enable-identity-attention'], identities_ordering = opt.identities_ordering, mode='test')
    test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=config['test']['bs'], shuffle=False, sampler=None,
                                    batch_sampler=None, num_workers=opt.workers, collate_fn=None,
                                    pin_memory=False, drop_last=False, timeout=0,
                                    worker_init_fn=None, prefetch_factor=2,
                                    persistent_workers=False)

    # Print some useful statistics
    print("Test videos:", test_samples)
    print("__TEST STATS__")
    test_counters = collections.Counter(test_labels)
    print(test_counters)

    # Init variables
    total_test_loss = 0
    test_correct = 0
    test_positive = 0
    test_negative = 0
    test_counter = 0

    multiclass_errors = dict.fromkeys([i for i in range(9)])
    for key in multiclass_errors:
        multiclass_errors[key] = [0, class_counter[key]]
    
    bar = ChargingBar('PREDICT', max=(len(test_dl)))
    preds = []
    videos_errors = []

    # Test loop
    for index, (videos, size_embeddings, masks, identities_masks, positions, tokens_per_identity, labels, multiclass_labels, video_ids) in enumerate(test_dl):
        b, f, h, w, c = videos.shape
        labels = labels.unsqueeze(1).float()
        identities_masks = identities_masks.to(device)
        masks = masks.to(device)
        positions = positions.to(device)

        with torch.no_grad():
            
            if opt.model != 2: # Use the features extractor
                videos = rearrange(videos, "b f h w c -> (b f) c h w")
                videos = videos.to(device)
                
                features = features_extractor(videos)  
                if opt.model == 0: 
                    test_pred = model(features)
                    test_pred = torch.mean(test_pred.reshape(-1, config["model"]["num-frames"]), axis=1).unsqueeze(1)
                elif opt.model == 1:
                    features = rearrange(features, '(b f) c h w -> b f c h w', b = b, f = f)
                    test_pred, attentions = model(features, mask=masks, size_embedding=size_embeddings, identities_mask=identities_masks, positions=positions)
                    if opt.save_attentions:
                        identity_names = [row[0] for row in tokens_per_identity]
                        frames_per_identity = [int(row[1] / config["model"]["num-patches"]) for row in tokens_per_identity]
                        
                        aggregated_attentions, identity_attentions = aggregate_attentions(attentions, config['model']['heads'], config['model']['num-frames'], frames_per_identity)
                        
                        save_attention_plots(aggregated_attentions, identity_names, frames_per_identity, config['model']['num-frames'], video_ids[0])
            elif opt.model == 2:
                videos = rearrange(videos, 'b f h w c -> b c f h w')
                videos = slowfast_input_transform(videos)
                videos = [torch.cat([v[None, ...].to(device) for v in videos[0]]), torch.cat([v[None, ...].to(device) for v in videos[1]])]
                test_pred = model(videos)
                    
        
        if opt.model != 2:
            videos = videos.cpu()
        else:
            videos = [torch.cat([v[None, ...].cpu() for v in videos[0]]), torch.cat([v[None, ...].cpu() for v in videos[1]])]

        test_pred = test_pred.cpu()
        
        test_loss = loss_fn(test_pred, labels)
        total_test_loss += round(test_loss.item(), 2)
        corrects, positive_class, negative_class, multiclass_errors, batch_errors = check_correct(test_pred, labels, multiclass_labels, multiclass_errors, video_ids)
        videos_errors.extend(batch_errors)
        test_correct += corrects
        test_positive += positive_class
        test_counter += 1
        test_negative += negative_class
        preds.extend(test_pred)
        bar.next()
    
    preds = [torch.sigmoid(torch.tensor(pred)) for pred in preds]
    fpr, tpr, th = metrics.roc_curve(test_labels, preds)
    auc = metrics.auc(fpr, tpr)
    f1 = f1_score(test_labels, [round(pred.item()) for pred in preds])
    bar.finish()
    total_test_loss /= test_counter
    test_correct /= test_samples
    print("Videos errors", videos_errors)
    print("Class errors", multiclass_errors)
    print(str(opt.model_weights) + " test loss:" +
            str(total_test_loss) + " f1 score: " + str(f1) + " test accuracy:" + str(test_correct) + " test_0s:" + str(test_negative) + "/" + str(test_counters[0]) + " test_1s:" + str(test_positive) + "/" + str(test_counters[1]) + " AUC " + str(auc))
    