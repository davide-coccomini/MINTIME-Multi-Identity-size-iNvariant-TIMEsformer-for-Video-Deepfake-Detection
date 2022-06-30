import torch
import numpy as np
import argparse
from tqdm import tqdm
import math
import yaml
from utils import get_n_params, shuffle_dataset, check_correct
from sklearn.utils.class_weight import compute_class_weight 
from torch.optim.lr_scheduler import LambdaLR
import collections
import os
import json
from einops import rearrange
import pandas as pd
from os import cpu_count
from multiprocessing.pool import Pool
from functools import partial
from multiprocessing import Manager
from progress.bar import ChargingBar
from torch.optim import lr_scheduler
from models.baseline import Baseline
from efficientnet_pytorch import EfficientNet
from deepfakes_dataset import DeepFakesDataset
from timesformer_pytorch import TimeSformer
from models.size_invariant_timesformer import SizeInvariantTimeSformer
from models.efficientnet.efficientnet_pytorch import EfficientNet

BASE_DIR = './'
DATA_DIR = os.path.join(BASE_DIR, "datasets/ForgeryNet/faces")
TRAINING_DIR = os.path.join(DATA_DIR, "train")
VALIDATION_DIR = os.path.join(DATA_DIR, "val")
TEST_DIR = os.path.join(DATA_DIR, "test")
MODELS_PATH = "outputs/models"



# Main body
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--train_list_file', default="../datasets/ForgeryNet/faces/train.csv", type=str,
                        help='Training List txt file path)')
    parser.add_argument('--validation_list_file', default="../datasets/ForgeryNet/faces/val.csv", type=str,
                        help='Validation List txt file path)')
    parser.add_argument('--num_epochs', default=300, type=int,
                        help='Number of training epochs.')
    parser.add_argument('--workers', default=1, type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='Path to latest checkpoint (default: none).')
    parser.add_argument('--max_videos', type=int, default=-1, 
                        help="Maximum number of videos to use for training (default: all).")
    parser.add_argument('--config', type=str, 
                        help="Which configuration to use. See into 'config' folder.")
    parser.add_argument('--model', type=int, 
                        help="Which model to use. (0: Baseline | 1: Convolutional TimeSformer).")
    parser.add_argument('--patience', type=int, default=5, 
                        help="How many epochs wait before stopping for validation loss not improving.")
    
    opt = parser.parse_args()
    print(opt)

    with open(opt.config, 'r') as ymlfile:
        config = yaml.safe_load(ymlfile)
 

    if opt.model == 0:
        model = Baseline()
    else:
        model = SizeInvariantTimeSformer(config=config)
      
    model.train()
    
    optimizer = torch.optim.SGD(model.parameters(), lr=config['training']['lr'], weight_decay=config['training']['weight-decay'])
    scheduler = lr_scheduler.StepLR(optimizer, step_size=config['training']['step-size'], gamma=config['training']['gamma'])
    starting_epoch = 0
    if os.path.exists(opt.resume):
        model.load_state_dict(torch.load(opt.resume))
        starting_epoch = int(opt.resume.split("checkpoint")[1].split("_")[0]) + 1 # The checkpoint's file name format should be "checkpoint_EPOCH"
    else:
        print("No checkpoint loaded.")


    print("Model Parameters:", get_n_params(model))
    

    
    # READ ALL PATHS
    paths = []
    # 1. Read csv of training and validation set
    col_names = ["video", "label", "8_cls"]
    df_train = pd.read_csv(opt.train_list_file, sep=' ', names=col_names)
    df_validation = pd.read_csv(opt.validation_list_file, sep=' ', names=col_names)
    
    df_train = df_train.sample(frac=1, random_state=42).reset_index(drop=True)
    df_validation = df_validation.sample(frac=1, random_state=42).reset_index(drop=True)

    train_videos = df_train['video'].tolist()
    train_videos = train_videos
    train_labels = df_train['label'].tolist()
    train_labels = train_labels
    validation_videos = df_validation['video'].tolist()
    validation_videos = validation_videos
    validation_labels = df_validation['label'].tolist()
    validation_labels = validation_labels

    train_samples = len(train_videos)
    validation_samples = len(validation_videos)

    # Print some useful statistics
    print("Train videos:", train_samples, "Validation videos:", validation_samples)
    print("__TRAINING STATS__")
    train_counters = collections.Counter(train_labels)
    print(train_counters)
    
    class_weights = train_counters[0] / train_counters[1]
    print("Weights", class_weights)

    print("__VALIDATION STATS__")
    val_counters = collections.Counter(validation_labels)
    print(val_counters)
    print("___________________")

    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([class_weights]))

    # Create the data loaders
    
    train_dataset = DeepFakesDataset(train_videos, train_labels, config['model']['image-size'], sequence_length=config['model']['num-frames'])
    train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=config['training']['bs'], shuffle=True, sampler=None,
                                 batch_sampler=None, num_workers=opt.workers, collate_fn=None,
                                 pin_memory=False, drop_last=False, timeout=0,
                                 worker_init_fn=None, prefetch_factor=2,
                                 persistent_workers=False)

    validation_dataset = DeepFakesDataset(validation_videos, validation_labels, config['model']['image-size'], sequence_length=config['model']['num-frames'], mode='val')
    val_dl = torch.utils.data.DataLoader(validation_dataset, batch_size=config['training']['val_bs'], shuffle=True, sampler=None,
                                    batch_sampler=None, num_workers=opt.workers, collate_fn=None,
                                    pin_memory=False, drop_last=False, timeout=0,
                                    worker_init_fn=None, prefetch_factor=2,
                                    persistent_workers=False)

    
    features_extractor = EfficientNet.from_pretrained('efficientnet-b0')
 
    counter = 0
    not_improved_loss = 0
    previous_loss = math.inf
    for t in range(starting_epoch, opt.num_epochs + 1):
        if not_improved_loss == opt.patience:
            break
        counter = 0

        total_loss = 0
        total_val_loss = 0
        
        bar = ChargingBar('EPOCH #' + str(t), max=(len(train_dl)*config['training']['bs'])+len(val_dl))
        train_correct = 0
        positive = 0
        negative = 0
        for index, (videos, size_embeddings, masks, labels) in enumerate(train_dl):
            b, f, _, _, _= videos.shape
            labels = labels.unsqueeze(1).float()
            videos = videos.cuda()
            masks = masks.cuda()
            
            with torch.no_grad():
                features_extractor = features_extractor.cuda()
                videos = rearrange(videos, 'b f h w c -> (b f) c h w')                                               # B*8 x 3 x 224 x 224
                features = features_extractor.extract_features(videos)                                                                # B*8 x 1280 x 7 x 7
                features = rearrange(features, '(b f) c h w -> b f c h w', b = b, f = f)                             # B x 8 x 1280 x 7 x 7
                features_extractor = features_extractor.cpu()

            model = model.cuda()
            y_pred = model(features, mask=masks, size_embedding=size_embeddings)
            
            videos = videos.cpu()
            y_pred = y_pred.cpu()
            loss = loss_fn(y_pred, labels)
            corrects, positive_class, negative_class = check_correct(y_pred, labels)  
            train_correct += corrects
            positive += positive_class
            negative += negative_class
            optimizer.zero_grad()
            
            loss.backward()
            
            optimizer.step()
            counter += 1
            total_loss += round(loss.item(), 2)
            
            if index%1200 == 0: # Intermediate metrics print
                print("\nLoss: ", total_loss/counter, "Accuracy: ", train_correct/(counter*config['training']['bs']) ,"Train 0s: ", negative, "Train 1s:", positive)


            for i in range(config['training']['bs']):
                bar.next()
        
        torch.cuda.empty_cache() 
        
        val_correct = 0
        val_positive = 0
        val_negative = 0
        val_counter = 0
        train_correct /= train_samples
        total_loss /= counter
        for index, (videos, size_embeddings, masks, labels) in enumerate(val_dl):
            videos = videos.cuda()
            masks = masks.cuda()
            labels = labels.unsqueeze(1).float()

            with torch.no_grad():
                features_extractor = features_extractor.cuda()
                videos = rearrange(videos, 'b f h w c -> (b f) c h w')                                       # B*8 x 3 x 224 x 224
                features = features_extractor.extract_features(videos)                                           # B*8 x 1280 x 7 x 7
                features = rearrange(features, '(b f) c h w -> b f c h w', b = b, f = f)                             # B x 8 x 1280 x 7 x 7
                features_extractor = features_extractor.cpu()


            val_pred = model(features, mask=masks, size_embedding=size_embeddings)
            videos = videos.cpu()
            val_pred = val_pred.cpu()
            val_loss = loss_fn(val_pred, labels)
            total_val_loss += round(val_loss.item(), 2)
            corrects, positive_class, negative_class = check_correct(val_pred, labels)
            val_correct += corrects
            val_positive += positive_class
            val_counter += 1
            val_negative += negative_class
            bar.next()
        
        torch.cuda.empty_cache() 

    
        scheduler.step()
        bar.finish()
            
        
        total_val_loss /= val_counter
        val_correct /= validation_samples
        if previous_loss <= total_val_loss:
            print("Validation loss did not improved")
            not_improved_loss += 1
        else:
            not_improved_loss = 0
        
        previous_loss = total_val_loss
        print("#" + str(t) + "/" + str(opt.num_epochs) + " loss:" +
            str(total_loss) + " accuracy:" + str(train_correct) +" val_loss:" + str(total_val_loss) + " val_accuracy:" + str(val_correct) + " val_0s:" + str(val_negative) + "/" + str(np.count_nonzero(validation_labels == 0)) + " val_1s:" + str(val_positive) + "/" + str(np.count_nonzero(validation_labels == 1)))
    
        if not os.path.exists(MODELS_PATH):
            os.makedirs(MODELS_PATH)
        torch.save(model.state_dict(), os.path.join(MODELS_PATH,  "ConvolutionalTimeSformer_checkpoint" + str(t)))
        