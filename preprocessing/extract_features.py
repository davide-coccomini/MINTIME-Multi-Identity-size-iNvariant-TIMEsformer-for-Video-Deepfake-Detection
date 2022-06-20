#################################################################################################################
#################### Executing this code you will be able to preliminary extract the features ###################
####################       from the detected faces using a pretrained EfficientNet B0         ###################
#################################################################################################################


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

    os.makedirs(opt.output_path, exist_ok=True)
    
    dataset = FacesDataset(paths, output_dir = opt.output_path)

    dl = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, sampler=None,
                                 batch_sampler=None, num_workers=opt.workers, collate_fn=None,
                                 pin_memory=False, drop_last=False, timeout=0,
                                 worker_init_fn=None, prefetch_factor=2,
                                 persistent_workers=False)
    model = EfficientNet.from_pretrained('efficientnet-b0')
    model = model.cuda(device=opt.gpu_id)

    bar = ChargingBar('Extracted: ', max=(len(dl)))


    for index, (faces, output_paths) in enumerate(dl):
        faces = faces.cuda(device=opt.gpu_id)
        features = model.extract_features(faces)

        for i in range(len(faces)):
            os.makedirs(os.path.dirname(output_paths[i]), exist_ok = True)
            torch.save(features[i], output_paths[i])
        
        bar.next()


