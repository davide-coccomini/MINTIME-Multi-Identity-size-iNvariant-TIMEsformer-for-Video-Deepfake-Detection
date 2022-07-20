# The training process is conducted using this code and it can be customized on the specific model that you want to train.

import torch
import numpy as np
import argparse
from tqdm import tqdm
import math
import yaml
from utils import check_correct
from torch.optim.lr_scheduler import LambdaLR
from datetime import datetime, timedelta
from statistics import mean
import tensorflow as tf
import collections
import os
import json
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--train_list_file', default="../datasets/ForgeryNet/faces/train.csv", type=str,
                        help='Training List txt file path)')
    parser.add_argument('--validation_list_file', default="../datasets/ForgeryNet/faces/val.csv", type=str,
                        help='Validation List txt file path)')  
    parser.add_argument('--data_path', default="../datasets/ForgeryNet/faces", type=str,
                        help='Path to the dataset converted into identities')
    parser.add_argument('--num_epochs', default=30, type=int,
                        help='Number of training epochs.')
    parser.add_argument('--workers', default=8, type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--random_state', default=42, type=int,
                        help='Random state value')
    parser.add_argument('--freeze_backbone', default=False, action="store_true",
                        help='Maintain the backbone freezed or train it.')
    parser.add_argument('--extractor_unfreeze_blocks', type=int, default=-1, 
                        help="How many layers unfreeze in the extractor.")
    parser.add_argument('--extractor_weights', default='ImageNet', type=str,
                        help='Path to extractor weights or "imagenet".')
    parser.add_argument('--gpu_id', default=0, type=int,
                        help='ID of GPU to be used.')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='Path to latest checkpoint (default: none).')
    parser.add_argument('--max_videos', type=int, default=-1, 
                        help="Maximum number of videos to use for training (default: all).")
    parser.add_argument('--config', type=str, 
                        help="Which configuration to use. See into 'config' folder.")
    parser.add_argument('--model', type=int, 
                        help="Which model to use. (0: Baseline | 1: Size Invariant TimeSformer).")
    parser.add_argument('--patience', type=int, default=5, 
                        help="How many epochs wait before stopping for validation loss not improving.")
    parser.add_argument('--logger_name', default='runs/train',
                        help='Path to save the model and Tensorboard log.')
    parser.add_argument('--models_output_path', default='"outputs/models"',
                        help='Output path for checkpoints.')
    opt = parser.parse_args()
    
    print(opt)
    with open(opt.config, 'r') as ymlfile:
        config = yaml.safe_load(ymlfile)

    # Setup CUDA settings
    torch.cuda.set_device(opt.gpu_id) 
    torch.backends.cudnn.deterministic = True
    random.seed(opt.random_state)
    torch.manual_seed(opt.random_state)
    torch.cuda.manual_seed(opt.random_state)
    np.random.seed(opt.random_state)
   
    # Create useful dirs
    os.makedirs(opt.logger_name, exist_ok=True)
    os.makedirs(opt.models_output_path, exist_ok=True)

    # Load required weights for feature extractor
    if opt.extractor_weights.lower() == 'imagenet':
        features_extractor = EfficientNet.from_pretrained('efficientnet-b0')
    else:
        features_extractor = EfficientNet.from_name('efficientnet-b0')
        features_extractor.load_matching_state_dict(torch.load(opt.extractor_weights, map_location=torch.device('cpu')))
        print("Custom features extractor weights loaded.")
    
    # Init the required model
    if opt.model == 0:
        model = Baseline(config=config)
        num_patches = None
    else:
        model = SizeInvariantTimeSformer(config=config)
        num_patches = config['model']['num-patches']

    # Setup the requiring grad layers for features extractor
    if opt.freeze_backbone:
        features_extractor.eval()
    else:
        features_extractor.train()
        if opt.extractor_unfreeze_blocks > -1:
            for i in range(0, len(features_extractor._blocks)):
                for index, param in enumerate(features_extractor._blocks[i].parameters()):
                    if i >= len(features_extractor._blocks)-opt.extractor_unfreeze_blocks:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False

    # Move models to GPU 
    features_extractor = features_extractor.to(opt.gpu_id)    
    model = model.to(opt.gpu_id)
    model.train()
    

    # Init optimizers
    if opt.freeze_backbone:
        parameters = model.parameters()
    else:
        parameters = chain(features_extractor.parameters(), model.parameters())

    if config['training']['optimizer'].lower() == 'sgd':
        optimizer = torch.optim.SGD(parameters, lr=config['training']['lr'], weight_decay=config['training']['weight-decay'])
    elif config['training']['optimizer'].lower() == 'adamw':
        optimizer = torch.optim.AdamW(parameters, lr=config['training']['lr'], weight_decay=config['training']['weight-decay'])
    elif config['training']['optimizer'].lower() == 'adam':
        optimizer = torch.optim.Adam(parameters, lr=config['training']['lr'], weight_decay=config['training']['weight-decay'])
    else:
        print("Error: Invalid optimizer specified in the config file.")
        exit()

    # Init LR schedulers
    if config['training']['scheduler'].lower() == 'steplr':   
        scheduler = lr_scheduler.StepLR(optimizer, step_size=config['training']['step-size'], gamma=config['training']['gamma'])
    elif config['training']['scheduler'].lower() == 'cosinelr':
        lr_scheduler = CosineLRScheduler(
                optimizer,
                t_initial=opt.num_epochs,
                lr_min=config['training']['lr'] * 1e-2,
                cycle_limit=1,
                t_in_epochs=False,
        )
    else:
        print("Warning: Invalid scheduler specified in the config file.")


    starting_epoch = 0
    if os.path.exists(opt.resume):
        model.load_state_dict(torch.load(opt.resume))
        starting_epoch = int(opt.resume.split("checkpoint")[1].split("_")[0]) + 1 # The checkpoint's file name format should be "checkpoint_EPOCH"
    else:
        print("No checkpoint loaded for TimeSformer.")

    
    # Read all the paths and initialize data loaders for train and validation
    paths = []
    col_names = ["video", "label", "8_cls"]
    
    df_train = pd.read_csv(opt.train_list_file, sep=' ', names=col_names)
    df_validation = pd.read_csv(opt.validation_list_file, sep=' ', names=col_names)
    
    df_train = df_train.sample(frac=1, random_state=opt.random_state).reset_index(drop=True)
    df_validation = df_validation.sample(frac=1, random_state=opt.random_state).reset_index(drop=True)

    train_videos = df_train['video'].tolist()
    train_labels = df_train['label'].tolist()
    validation_videos = df_validation['video'].tolist()
    validation_labels = df_validation['label'].tolist()

    if opt.max_videos > -1:
        train_videos = train_videos[:opt.max_videos]
        train_labels = train_labels[:opt.max_videos]
        validation_videos = validation_videos[:opt.max_videos]
        validation_labels = validation_labels[:opt.max_videos]
    
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


    # Init logger
    tb_logger = SummaryWriter(log_dir=opt.logger_name, comment='')
    experiment_path = tb_logger.get_logdir()
    
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([class_weights]))

    # Create the data loaders 
    train_dataset = DeepFakesDataset(train_videos, train_labels, image_size=config['model']['image-size'], data_path=opt.data_path, num_frames=config['model']['num-frames'], num_patches=num_patches, max_identities=config['model']['max-identities'])
    train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=config['training']['bs'], shuffle=True, sampler=None,
                                 batch_sampler=None, num_workers=opt.workers, collate_fn=None,
                                 pin_memory=False, drop_last=False, timeout=0,
                                 worker_init_fn=None, prefetch_factor=2,
                                 persistent_workers=False)

    validation_dataset = DeepFakesDataset(validation_videos, validation_labels, image_size=config['model']['image-size'], data_path=opt.data_path, num_frames=config['model']['num-frames'], num_patches=num_patches, max_identities=config['model']['max-identities'], mode='val')
    val_dl = torch.utils.data.DataLoader(validation_dataset, batch_size=config['training']['val_bs'], shuffle=True, sampler=None,
                                    batch_sampler=None, num_workers=opt.workers, collate_fn=None,
                                    pin_memory=False, drop_last=False, timeout=0,
                                    worker_init_fn=None, prefetch_factor=2,
                                    persistent_workers=False)
   
    # Init variables for training
    not_improved_loss = 0
    previous_loss = math.inf
       
    # Training loop
    for t in range(starting_epoch, opt.num_epochs + 1):
        model.train()
        if not_improved_loss == opt.patience:
            break

        # Init epoch variables
        counter = 0
        total_loss = 0
        total_val_loss = 0
        train_correct = 0
        positive = 0
        negative = 0
        times_per_batch = 0
        train_batches = len(train_dl)
        val_batches = len(val_dl)
        total_batches = train_batches + val_batches

        # Epoch loop
        bar = ChargingBar('EPOCH #' + str(t), max=(len(train_dl)+len(val_dl)))
        for index, (videos, size_embeddings, masks, identities_masks, positions, labels) in enumerate(train_dl):
            start_time = datetime.now()
            b, f, h, w, c = videos.shape
            labels = labels.unsqueeze(1).float()
            videos = videos.to(opt.gpu_id)
            identities_masks = identities_masks.to(opt.gpu_id)
            masks = masks.to(opt.gpu_id)
            positions = positions.to(opt.gpu_id)

            if opt.model == 0: # Baseline
                videos = rearrange(videos, "b f h w c -> (b f) c h w")
                features = features_extractor.extract_features(videos)  
                y_pred = model(features)
                y_pred = torch.mean(y_pred.reshape(-1, self.num_frames), axis=1)
            elif opt.model == 1: # Size-Invariant TimeSformer
                videos = rearrange(videos, 'b f h w c -> (b f) c h w')                                               # B*8 x 3 x 224 x 224
                if opt.freeze_backbone:
                    with torch.no_grad():
                        features = features_extractor.extract_features(videos)                                               # B*8 x 1280 x 7 x 7
                else:
                    features = features_extractor.extract_features(videos)                                               # B*8 x 1280 x 7 x 7
    
                features = rearrange(features, '(b f) c h w -> b f c h w', b = b, f = f)
                y_pred = model(features, mask=masks, size_embedding=size_embeddings, identities_mask=identities_masks, positions=positions)
        
            # Calculate loss
            videos = videos.cpu()
            y_pred = y_pred.cpu()
            loss = loss_fn(y_pred, labels)
            corrects, positive_class, negative_class = check_correct(y_pred, labels)  
            train_correct += corrects
            positive += positive_class
            negative += negative_class
            counter += 1
            total_loss += round(loss.item(), 2)
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if config['training']['scheduler'].lower() == 'cosinelr':
                lr_scheduler.step_update((t * (train_batches) + index))

            # Update time per epoch
            time_diff = datetime.now()-start_time
            print(time_diff)
            duration = float(str(time_diff.seconds) + "." +str(time_diff.microseconds))
            times_per_batch += duration
            
            # Print intermediate metrics
            if index%1 == 0:
                expected_time = str(timedelta(seconds=(times_per_batch / (index+1))*total_batches-index))
                print("\nLoss: ", total_loss/counter, "Accuracy: ", train_correct/(counter*config['training']['bs']) ,"Train 0s: ", negative, "Train 1s:", positive, "Expected Time:", expected_time)


            bar.next()
        
        # Clean variables before moving into validation
        torch.cuda.empty_cache() 
        val_correct = 0
        val_positive = 0
        val_negative = 0
        val_counter = 0
        train_correct /= train_samples
        total_loss /= counter
        model.eval()

        # Epoch validation loop
        for index, (videos, size_embeddings, masks, identities_masks, positions, labels) in enumerate(val_dl):
            b, f, _, _, _= videos.shape
            videos = videos.to(opt.gpu_id)
            masks = masks.to(opt.gpu_id)
            positions = positions.to(opt.gpu_id)
            identities_masks = identities_masks.to(opt.gpu_id)
            labels = labels.unsqueeze(1).float()

            # Do not update the gradient during validation
            with torch.no_grad():
                if opt.model == 0: 
                    videos = reduce(videos, "b f h w c -> b h w c", 'mean')
                    videos = rearrange(videos, "b h w c -> b c h w")
                    features = features_extractor.extract_features(videos)  
                    val_pred = model(features)
                elif opt.model == 1:
                    videos = rearrange(videos, 'b f h w c -> (b f) c h w')                                       # B*8 x 3 x 224 x 224
                    features = features_extractor.extract_features(videos)                                           # B*8 x 1280 x 7 x 7
                    features = rearrange(features, '(b f) c h w -> b f c h w', b = b, f = f)   
                    val_pred = model(features, mask=masks, size_embedding=size_embeddings, identities_masks=identities_masks, positions=positions)

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
        
        if config['training']['scheduler'].lower() == 'steplr':
            scheduler.step()


        torch.cuda.empty_cache() 
        bar.finish()
            
        total_val_loss /= val_counter
        val_correct /= validation_samples

        if previous_loss <= total_val_loss:
            print("Validation loss did not improved")
            not_improved_loss += 1
        else:
            not_improved_loss = 0

        previous_loss = total_val_loss

        # Save checkpoint if the model's validation loss is improving
        if previous_loss > total_val_loss:
            torch.save(features_extractor.state_dict(), os.path.join(opt.models_output_path,  "EfficientNetExtractor_checkpoint" + str(t)))
            torch.save(model.state_dict(), os.path.join(opt.models_output_path,  "SizeInvariantTimeSformer_checkpoint" + str(t)))

        # Log some metrics into Tensorboard
        tb_logger.add_scalar("Training/Accuracy", train_correct, t)
        tb_logger.add_scalar("Training/Loss", total_loss, t)
        tb_logger.add_scalar("Training/Learning_Rate", optimizer.param_groups[0]['lr'], t)
        tb_logger.add_scalar("Validation/Loss", total_val_loss, t)
        tb_logger.add_scalar("Validation/Accuracy", val_correct, t)

        # Print epoch metrics            
        print("#" + str(t) + "/" + str(opt.num_epochs) + " loss:" +
            str(total_loss) + " accuracy:" + str(train_correct) +" val_loss:" + str(total_val_loss) + " val_accuracy:" + str(val_correct) + " val_0s:" + str(val_negative) + "/" + str(val_counters[0]) + " val_1s:" + str(val_positive) + "/" + str(val_counters[1]))
    
        
        
        