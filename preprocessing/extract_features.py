# It is possible to decide to extract features from previously detected face images in advance. This is done via an EfficientNet B0 and is useful if you are using the convolutional
# backbone freezed architecture. 
# ATTENTION: The features take up a lot of disk space and it may therefore be unavoidable to have to extract them in the training phase as the images are loaded from the data loader.

from utils import get_paths
from tqdm import tqdm
from efficientnet_pytorch import EfficientNet
from faces_dataset import FacesDataset
from torch.utils.data.dataloader import DataLoader
import argparse
import os
import pickle
import torch
from progress.bar import ChargingBar


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='', type=str,
                        help='Faces images directory')
    parser.add_argument('--support_files_path', default='support_files', type=str,
                        help='Path to save support files')
    parser.add_argument('--gpu_id', default=0, type=int,
                        help='ID of GPU to be used')
    parser.add_argument('--workers', default=40, type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--batch_size', default=48, type=int,
                        help='Batch size.')
    parser.add_argument('--output_path', default='', type=str,
                        help='Features output directory')


    opt = parser.parse_args()
    print(opt)
    
    # Reading or saving the file containing previously saved paths to improve speed in the case of multiple executions.
    print("Searching for faces...")
    list_file_path = os.path.join(opt.support_files_path, "faces.txt")
    if os.path.exists(list_file_path):
        with open(list_file_path, 'rb') as fp:
            paths = pickle.load(fp)
        print("Backup file found, loaded", len(paths), "faces.")
    else:
        paths = get_paths(opt.data_path)
        with open(list_file_path, 'wb') as fp:
            pickle.dump(paths, fp)
        print(len(paths), "faces found.")

    # Read faces and prepare them for extraction
    dataset = FacesDataset(paths, output_dir = opt.output_path)
    dl = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, sampler=None,
                                 batch_sampler=None, num_workers=opt.workers, collate_fn=None,
                                 pin_memory=False, drop_last=False, timeout=0,
                                 worker_init_fn=None, prefetch_factor=2,
                                 persistent_workers=False)

    # Load the pretrained convolutional backbone
    model = EfficientNet.from_pretrained('efficientnet-b0')
    model = model.cuda(device=opt.gpu_id)


    # Extract the features and save them into disk
    bar = ChargingBar('Extracted: ', max=(len(dl)))
    os.makedirs(opt.output_path, exist_ok=True)
    for index, (faces, output_paths) in enumerate(dl):
        faces = faces.cuda(device=opt.gpu_id)
        features = model.extract_features(faces)

        for i in range(len(faces)):
            os.makedirs(os.path.dirname(output_paths[i]), exist_ok = True)
            torch.save(features[i], output_paths[i])
        
        bar.next()


