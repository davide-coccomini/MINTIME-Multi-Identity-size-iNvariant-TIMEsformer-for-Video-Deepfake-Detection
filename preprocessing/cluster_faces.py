# Since several subjects can be found within a video, it is necessary to cluster them into groups based on similarity. 
# This operation is carried out in the following code with additional attention to maintaining the temporal coherence of faces.
# The extracted faces are reorganised into consecutive sequences of similar faces so as to be more suitable for network processing.


import argparse
import os
import glob
import torch
import numpy as np
import pandas as pd
import shutil
from numpy.linalg import norm
from PIL import Image
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1, fixed_image_standardization
from collections import OrderedDict
from sklearn.cluster import KMeans
from torch.utils.data.dataloader import DataLoader
from progress.bar import ChargingBar


seed = 42
END_TUPLE = ("N/A", np.inf)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--faces_path', default="../../datasets/ForgeryNet/faces", type=str,
                        help='Path of folder containing train/val/test with extracted cropped faces to be clustered.')
    parser.add_argument('--gpu_id', default=0, type=int,
                        help='ID of GPU to be used.')
    parser.add_argument('--similarity_threshold', default=0.8, type=float,
                        help='Threshold to discard faces with high distance.')
    parser.add_argument('--min_faces_number_per_sequence', default=3, type=int,
                        help='Minimum number of faces per sequence to be considered.')
    parser.add_argument('--workers', default=40, type=int,
                        help='Number of data loader workers.')

    opt = parser.parse_args()
    print(opt)

    # Get all the paths of the videos to be clustered 
    for dataset in os.listdir(opt.faces_path):
        dataset_path = os.path.join(opt.faces_path, dataset)
        if not os.path.isdir(dataset_path):
            continue

        print()
        print("Clustering videos in ", dataset_path)
        set_paths = glob.glob(f'{dataset_path}/*/**/*.mp4', recursive=True)
    

        # Initialize the features extractor
        embeddings_extractor = InceptionResnetV1(pretrained='vggface2').eval().to(opt.gpu_id)
        transformation = transforms.Compose([
            transforms.Resize((299, 299)),
            np.float32,
            fixed_image_standardization
        ])
        
        # For each video in each set, perform faces clustering
        bar = ChargingBar('Clustered videos', max=(len(set_paths)))
        for path in set_paths:
            # Read all faces, load them into a dictionary 
            faces = []
            faces_ids = []
            faces_files = [face_file for face_file in os.listdir(path) if not os.path.isdir(os.path.join(path, face_file))]
            faces_files = sorted(faces_files, key=lambda x:(int(x.split("_")[0]), int(os.path.splitext(x)[0].split("_")[1])))
            for face_file in faces_files:
                faces_ids.append(face_file)
                face_path = os.path.join(path, face_file)
                face_image = Image.open(face_path)
                frame_number = int(os.path.splitext(face_file)[0].split("_")[0])
                face = np.transpose(transformation(face_image), (2, 0, 1))
                faces.append(np.asarray(face))
               


            # Extract the embeddings
            embeddings = []
            loader = DataLoader(faces, shuffle=False, num_workers=opt.workers, batch_size=8, collate_fn=lambda x: x)

            for faces in loader:
                embeddings.extend(embeddings_extractor(torch.tensor(np.asarray(faces)).to(opt.gpu_id)).detach().cpu())


            # Calculate distances and cluster the images
            # couples = {
            #             frame_0: {
            #                   face_0_0: {(face_1_0, distance), (face_1_1, distance) ...}
            #                   face_0_1: {(face_1_0, distance), (face_1_1, distance) ...}
            #             },
            #             frame_1: {
            #                   face_1_0: {(face_2_0, distance), (face_2_1, distance) ...}
            #                   face_1_1: {(face_2_0, distance), (face_2_1, distance) ...}
            #             },
            #             ....
            #             frame_N-2: {
            #                   face_N-2_0: {(face_N-1_0, distance), (face_N-1_1, distance) ...}
            #                   face_N-2_1: {(face_N-1_0, distance), (face_N-1_1, distance) ...}
            #             }
            # }


            couples = {}
            for i in range(len(embeddings)): 
                face_id = faces_ids[i]
                frame_number = int(os.path.splitext(face_id)[0].split("_")[0])
                for j in range(i+1, len(embeddings)): # For each face in the future of the video
                    next_face_id = faces_ids[j]
                    next_frame_number = int(os.path.splitext(next_face_id)[0].split("_")[0])
                    if frame_number == next_frame_number+1:
                        continue
                    if next_frame_number > frame_number + 1:
                        break
                    dist = (embeddings[i] - embeddings[j]).norm().item()

                    if frame_number not in couples:
                        couples[frame_number] = {}

                    if face_id in couples[frame_number]:
                        couples[frame_number][face_id].append((next_face_id, dist))
                    else:
                        couples[frame_number][face_id] = [(next_face_id, dist)]

            # Get the best match for each face, also identify the collisions and consider the best only to remove false detection
            for frame in couples:
                already_used_faces = {}
                
                for face in couples[frame]:
                    best_match_face = min(couples[frame][face], key=lambda x:x[1])
                    if best_match_face[1] > opt.similarity_threshold:
                        couples[frame][face] = END_TUPLE
                    else:
                        if best_match_face[0] in already_used_faces:
                            if already_used_faces[best_match_face[0]][1] > best_match_face[1]:
                                couples[frame][already_used_faces[best_match_face[0]][0]] = END_TUPLE
                            else: 
                                couples[frame][face] = END_TUPLE
                                continue
                        
                        couples[frame][face] = best_match_face
                        already_used_faces[best_match_face[0]] = (face, best_match_face[1])
                        
          
            # Convert from couples to sequences
            sequences = []
            discarded_sequences = []
            already_explored_faces = []
            frames_keys = list(couples.keys())
            for index, frame in enumerate(couples):
                for face_id in couples[frame]:
                    sequence = []
                    if face_id not in already_explored_faces:
                        already_explored_faces.append(face_id)
                        sequence.append(face_id)
                        next_face_id = couples[frame][face_id][0]
                        j = 1
                        while index + j < len(frames_keys):
                            next_frame_couples = couples[frames_keys[index + j]]
                            if next_face_id == 'N/A':
                                break
                            if next_face_id not in already_explored_faces:
                                already_explored_faces.append(next_face_id)
                                sequence.append(next_face_id)

                            if next_face_id not in next_frame_couples:
                                break

                            next_face_id = next_frame_couples[next_face_id][0]
                            j += 1
                    if len(sequence) >= opt.min_faces_number_per_sequence:
                        sequences.append(sequence)
                    else:
                        discarded_sequences.append(sequence)

            # Organize the clusters inside the folder
            for identity_index, sequence in enumerate(sequences):
                for image in sequence:
                    src_path = os.path.join(path, image)
                    dst_path = os.path.join(path, str(identity_index), image)
                    #print(src_path, dst_path)
                    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                    shutil.move(src_path, dst_path)

            # Add additional folder for discarded images
            for identity_index, sequence in enumerate(discarded_sequences):
                for image in sequence:
                    src_path = os.path.join(path, image)
                    dst_path = os.path.join(path, "discarded", str(identity_index), image)
                    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                    #print(src_path, dst_path)
                    shutil.move(src_path, dst_path)

            # Clean any orphan image
            for file_name in os.listdir(path):
                file_path = os.path.join(path, file_name)
                if not os.path.isdir(file_path):
                    dst_path = os.path.join(path, "discarded/orphans", file_name)
                    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                    shutil.move(file_path, dst_path)
            bar.next()
        
        print()
