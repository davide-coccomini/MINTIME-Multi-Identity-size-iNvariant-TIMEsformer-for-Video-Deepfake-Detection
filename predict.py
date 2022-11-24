
import argparse
import cv2
import numpy as np
import yaml
import random

from typing import Type
import preprocessing.face_detector as face_detector
from preprocessing.face_detector import VideoDataset, VideoFaceDetector
from torch.utils.data.dataloader import DataLoader

from PIL import Image

import torch
from preprocessing.utils import preprocess_images, _generate_connected_components
from facenet_pytorch import InceptionResnetV1, fixed_image_standardization

from statistics import mean

from albumentations import Compose, RandomBrightnessContrast, HorizontalFlip, FancyPCA, HueSaturationValue, OneOf, ToGray, ShiftScaleRotate, ImageCompression, PadIfNeeded, GaussNoise, GaussianBlur, Rotate, Normalize, Resize
from transforms.albu import IsotropicResize

from models.size_invariant_timesformer import SizeInvariantTimeSformer
from models.efficientnet.efficientnet_pytorch import EfficientNet
from models.baseline import Baseline
import os
from einops import rearrange
from utils import aggregate_attentions, draw_border, save_attention_plots
from models.xception import xception



RANGE_SIZE = 5
SIZE_EMB_DICT = [(1+i*RANGE_SIZE, (i+1)*RANGE_SIZE) if i != 0 else (0, RANGE_SIZE) for i in range(20)]

def detect_faces(video_path, detector_cls: Type[VideoFaceDetector], opt):
    # Init the face detector
    detector = face_detector.__dict__[detector_cls](device=opt.gpu_id)

    # Read the video and its information
    dataset = VideoDataset([video_path])
    loader = DataLoader(dataset, shuffle=False, num_workers=opt.workers, batch_size=1, collate_fn=lambda x: x)
    
    # Detect the faces
    for item in loader: 
        bboxes = {}
        video, indices, fps, frames = item[0]
        bboxes.update({i : b for i, b in zip(indices, detector._detect_faces(frames))})
        found_faces = False
        for key in bboxes:
            if type(bboxes[key]) == list:
                found_faces = True
                break

        if not found_faces:
            raise Exception("No faces found.")

    return bboxes

def extract_crops(video_path, bboxes_dict):

    # Read video frames
    frames = []
    
    capture = cv2.VideoCapture(video_path)
    frames_num = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(capture.get(5))

    for i in range(frames_num):
        capture.grab()
        success, frame = capture.retrieve()
        if not success:
            continue
        frames.append(frame)

    # Extract the faces crops
    explored_indexes = []
    crops = []

    for i in range(0, len(frames), fps):
        while str(i) not in bboxes_dict:
            if i == frames_num - 1:
                i -= 1
            if i in explored_indexes:
                break
            else:
                explored_indexes.append(i)

        frame = frames[i]
        index = i
        limit = i + fps - 1
        keys = [int(x) for x in list(bboxes_dict.keys())]

        while index < limit:
            index += 1
            if index in keys and bboxes_dict[index] is not None:
                break
        if index == limit:
            continue

        bboxes = bboxes_dict[index]
        for bbox in bboxes:
            xmin, ymin, xmax, ymax = [int(b * 2) for b in bbox]
            w = xmax - xmin
            h = ymax - ymin

            # Add some padding to catch background too
            p_h = h // 3
            p_w = w // 3
            
            crop_h = (ymax + p_h) - max(ymin - p_h, 0)
            crop_w = (xmax + p_w) - max(xmin - p_w, 0)

            # Make the image square
            if crop_h > crop_w:
                p_h -= int(((crop_h - crop_w)/2))
            else:
                p_w -= int(((crop_w - crop_h)/2))

            # Extract the face from the frame
            crop = frame[max(ymin - p_h, 0):ymax + p_h, max(xmin - p_w, 0):xmax + p_w]
            
            # Check if out of bound and correct
            h, w = crop.shape[:2]
            if h > w:
                diff = int((h - w)/2)
                if diff > 0:         
                    crop = crop[diff:-diff,:]
                else:
                    crop = crop[1:,:]
            elif h < w:
                diff = int((w - h)/2)
                if diff > 0:
                    crop = crop[:,diff:-diff]
                else:
                    crop = crop[:,:-1]

            # Add the extracted face to the list
            crops.append((i, Image.fromarray(crop), bbox))

    return crops

def cluster_faces(crops, valid_cluster_size_ratio = 0.20, similarity_threshold = 0.45):

    # Convert crops to PIL images
    crops_images = [row[1] for row in crops]
    
    # Extract the embeddings
    embeddings_extractor = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    faces = [preprocess_images(face) for face in crops_images]
    faces = np.stack([np.uint8(face) for face in faces])
    faces = torch.as_tensor(faces)
    faces = faces.permute(0, 3, 1, 2).float()
    faces = fixed_image_standardization(faces)
    face_recognition_input = faces.cuda()
    embeddings = []
    embeddings = embeddings_extractor(face_recognition_input).detach().cpu().numpy()

    # Clustering
    valid_cluster_size = int(len(faces) * valid_cluster_size_ratio)
    similarities = np.dot(np.array(embeddings), np.array(embeddings).T)
    
    components = _generate_connected_components(
        similarities, similarity_threshold=similarity_threshold
    )
    components = [sorted(component) for component in components]

    clustered_faces = {}
    for identity_index, component in enumerate(components):
        for index, face_index in enumerate(component):
            component[index] = crops[face_index]
        
        clustered_faces[identity_index] = component

    return clustered_faces

def get_identity_information(identity, faces):
    mean_side = mean([row[1].size[0] for row in faces])   
    number_of_faces = len(faces)
    return [identity, mean_side, number_of_faces, faces]

def get_sorted_identities(identities, discarded_faces, max_identities = 2, num_frames = 16):
    sorted_identities = []
    discarded_faces = []
    for identity in identities:
        sorted_identities.append(get_identity_information(identity, identities[identity]))

    '''
    # If no faces have been found, use the discarded faces
    if len(sorted_identities) == 0:
        sorted_identities.append(self.get_identity_information(identities))
        discarded_faces = []
    '''

    # Sort identities based on faces size
    sorted_identities = sorted(sorted_identities, key=lambda x:x[1], reverse=True)

    if len(sorted_identities) > max_identities:
        sorted_identities = sorted_identities[:max_identities]

    # Adjust the identities list faces number
    identities_number = len(sorted_identities)
    available_additional_faces = []
    if identities_number > 1:
        max_faces_per_identity = {1: [num_frames], 
                  2:  [int(num_frames/2), int(num_frames/2)],
                  3:  [int(num_frames/3), int(num_frames/3), int(num_frames/4)],
                  4:  [int(num_frames/3), int(num_frames/3), int(num_frames/8), int(num_frames/8)]}

        max_faces_per_identity = max_faces_per_identity[identities_number]
        for i in range(identities_number):
            if sorted_identities[i][2] < max_faces_per_identity[i] and i < identities_number - 1:
                sorted_identities[i+1][2] += max_faces_per_identity[i] - sorted_identities[i][2] 
                available_additional_faces.append(0)
            elif sorted_identities[i][2] > max_faces_per_identity[i]:
                available_additional_faces.append(sorted_identities[i][2] - max_faces_per_identity[i])
                sorted_identities[i][2] = max_faces_per_identity[i]
            else:
                available_additional_faces.append(0)

    else: # If only one identity is in the video, all the frames are assigned to this identity
        sorted_identities[0][2] = num_frames
        available_additional_faces.append(0)


    # Check if we found enough faces to fullfill the input sequence, otherwise go back and add some faces from previous identities
    input_sequence_length = sum(faces_number for _, _, faces_number, _ in sorted_identities)
    if input_sequence_length < num_frames:
        for i in range(identities_number):
            needed_faces = num_frames - input_sequence_length
            if available_additional_faces[i] > 0:
                added_faces = min(available_additional_faces[i], needed_faces)
                sorted_identities[i][2] += added_faces
                input_sequence_length += added_faces
                if input_sequence_length == num_frames:
                    break
        # If not enough faces have been found, add some "dummy" images in the last identity
        if input_sequence_length < num_frames:
            needed_faces = num_frames - input_sequence_length
            sorted_identities[-1][2] += needed_faces
            input_sequence_length += needed_faces
    
    return sorted_identities, discarded_faces

def create_val_transform(size, additional_targets):
    return Compose([
        IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
        PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
        Resize(height=size, width=size)
    ],  additional_targets = additional_targets
    )

def generate_masks(video_path, identities, discarded_faces, num_frames, image_size, num_patches):
    mask = []
    last_range_end = 0
    sequence = []
    size_embeddings = []
    
    images_frames = []
    for identity_index, identity in enumerate(identities):
        max_faces = identity[2]
        identity_images = identity[3]
        '''
        # If no faces were considered for a frame during clustering, probably it is inside the discarded faces
        if identity_index == 0 and len(discarded_faces) > 0:
            frames = [int(os.path.basename(image_path).split("_")[0]) for image_path in identity_faces]
            discarded_frames = [int(os.path.basename(image_path).split("_")[0]) for image_path in discarded_faces]
            missing_frames = list(set(discarded_frames) - set(frames))
            missing_faces = [discarded_faces[discarded_frames.index(missing_frame)] for missing_frame in missing_frames]
            
            if len(missing_faces) > 0:
                identity_faces = identity_faces + missing_faces # Add the missing faces to the identity
        '''


        # Select uniformly the frames in an alternate way
        if len(identity_images) > max_faces:
            idx = np.round(np.linspace(0, len(identity_images) - 2, max_faces)).astype(int)
            identity_images = np.asarray(identity_images)[idx]
            
        images_frames.extend(identity_image[0] for identity_image in identity_images)
        identity_images = [identity_image[1] for identity_image in identity_images]

        # Generate size embeddings
        capture = cv2.VideoCapture(video_path)
        width  = capture.get(3)  
        height = capture.get(4) 
        video_area = width*height/2
        identity_size_embeddings = []
        
        for image_index, image in enumerate(identity_images):
            # Get face-frame area ratio for size embedding
            face_area = image.size[0] * image.size[1]
            ratio = int(face_area * 100 / video_area)
            side_ranges = list(map(lambda a_: ratio in range(a_[0], a_[1] + 1), SIZE_EMB_DICT))
            identity_size_embeddings.append(np.where(side_ranges)[0][0]+1)
      

        # If the readed faces are less than max_faces we need to add empty images and generate the mask
        if len(identity_images) < max_faces: 
            diff = max_faces - len(identity_size_embeddings)
            identity_size_embeddings = np.concatenate((identity_size_embeddings, np.zeros(diff)))
            identity_images.extend([np.zeros((image_size, image_size, 3), dtype=np.uint8) for i in range(diff)])
            mask.extend([1 if i < max_faces - diff else 0 for i in range(max_faces)])
            images_frames.extend([max(images_frames) for i in range(diff)])
        else: # Otherwise all the faces are valid
            mask.extend([1 for i in range(max_faces)])

        # Compose the size_embedding and sequence list
        size_embeddings.extend(identity_size_embeddings)
        sequence.extend(identity_images)

    # Transform the images, the same transformation is applied to all the faces in the same video
    sequence = [np.asarray(image) for image in sequence]
    additional_targets_keys = ["image" + str(i) for i in range(num_frames)]
    additional_targets_values = ["image" for i in range(num_frames)]
    additional_targets = dict(zip(additional_targets_keys, additional_targets_values))

 
    transform = create_val_transform(image_size, additional_targets)  
    if len(sequence) == 8:
        transformed_images = transform(image=sequence[0], image1=sequence[1], image2=sequence[2], image3=sequence[3], image4=sequence[4], image5=sequence[5], image6=sequence[6], image7=sequence[7])
    elif len(sequence) == 16:
        transformed_images = transform(image=sequence[0], image1=sequence[1], image2=sequence[2], image3=sequence[3], image4=sequence[4], image5=sequence[5], image6=sequence[6], image7=sequence[7], image8=sequence[8], image9=sequence[9], image10=sequence[10], image11=sequence[11], image12=sequence[12], image13=sequence[13], image14=sequence[14], image15=sequence[15])
    else:
        raise Exception("Invalid number of frames.")

    sequence = [transformed_images[key] for key in transformed_images]
        
    # Generate the identities_mask telling to the model which faces attend to an identity and which to another one
    identities_mask = []
    last_range_end = 0
    for identity_index in range(len(identities)):
        identity_mask = [True if i >= last_range_end and i < last_range_end + identities[identity_index][2] else False for i in range(0, num_frames)]
        for k in range(identities[identity_index][2]):
            identities_mask.append(identity_mask)
        last_range_end += identities[identity_index][2]

    # Generate coherent temporal-positional embedding
    images_frames_positions = {k: v+1 for v, k in enumerate(sorted(set(images_frames)))}
    frame_positions = [images_frames_positions[frame] for frame in images_frames]   
    if num_patches != None: 
        positions = [[i+1 for i in range(((frame_position-1)*num_patches), num_patches*(frame_position))] for frame_position in frame_positions]
        positions = sum(positions, []) # Merge the lists
        positions.insert(0,0) # Add CLS
    else:
        positions = []

    tokens_per_identity = [(identities[i][0], identities[i][2]*num_patches + identities[i-1][2]*num_patches) if i > 0 else (identities[i][0], identities[i][2]*num_patches) for i in range(len(identities))]     

    return torch.tensor([sequence]).float(), torch.tensor([size_embeddings]).int(), torch.tensor([mask]).bool(), torch.tensor([identities_mask]).bool(), torch.tensor([positions]), tokens_per_identity


def predict(video_path, clustered_faces, config, opt, discarded_faces = None):

    # Load required weights for feature extractor
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



    # Init the model
    model = SizeInvariantTimeSformer(config=config, require_attention=True)
    num_patches = config['model']['num-patches']

    
    features_extractor = torch.nn.DataParallel(features_extractor)
    model = torch.nn.DataParallel(model)

    # Move into GPU
    features_extractor = features_extractor.to(device)    
    model = model.to(device)
    features_extractor.eval()
    model.eval()

    if os.path.exists(opt.model_weights):
        model.load_state_dict(torch.load(opt.model_weights))
    else:
        raise Exception("No checkpoint loaded for the model.")    
    
    identities, discarded_faces  = get_sorted_identities(clustered_faces, discarded_faces)
    videos, size_embeddings, mask, identities_mask, positions, tokens_per_identity = generate_masks(video_path, identities, discarded_faces, config["model"]["num-frames"], config["model"]["image-size"], config["model"]["num-patches"])
    b, f, h, w, c = videos.shape
    videos = videos.to(device)    
    identities_mask = identities_mask.to(device)
    mask = mask.to(device)
    positions = positions.to(device)
    

    with torch.no_grad():
        video = rearrange(videos, "b f h w c -> (b f) c h w")
        features = features_extractor(video)  

        features = rearrange(features, '(b f) c h w -> b f c h w', b = b, f = f)   
        test_pred, attentions = model(features, mask=mask, size_embedding=size_embeddings, identities_mask=identities_mask, positions=positions)
        
        identity_names = [row[0] for row in tokens_per_identity]
        frames_per_identity = [int(row[1] / config["model"]["num-patches"]) for row in tokens_per_identity]
        
        if opt.save_attentions:
            aggregated_attentions, identity_attentions = aggregate_attentions(attentions, config['model']['heads'], config['model']['num-frames'], frames_per_identity)
            save_attention_plots(aggregated_attentions, identity_names, frames_per_identity, config['model']['num-frames'], os.path.basename(video_path))
        else:
            identity_attentions = []
            aggregated_attentions = []
        return torch.sigmoid(test_pred[0]).item(), identity_attentions, aggregated_attentions, identities, frames_per_identity

def get_identities_bboxes(identities):
    identities_bboxes = {}
    for row in identities:
        identity = row[3]
        for face in identity:
            frame = face[0]
            if frame in identities_bboxes:
                identities_bboxes[frame].append(face[2])
            else:
                identities_bboxes[frame] = [face[2]]
    return identities_bboxes


def generate_output_video(video_path, pred, identity_attentions, aggregated_attentions, identities, frames_per_identity):

    identities_bboxes = get_identities_bboxes(identities)
    available_frames_keys = [frame for frame in identities_bboxes]

    cap = cv2.VideoCapture(video_path)
    width  = cap.get(3)  
    height = cap.get(4) 
    fps = int(cap.get(5))
    fourcc = hex(int(cap.get(cv2.CAP_PROP_FOURCC)))
    output = cv2.VideoWriter("examples/preds/"+str(os.path.basename(video_path).replace(".mp4", ".avi")), cv2.VideoWriter_fourcc("X", "V", "I", "D"), fps, (int(width), int(height)))
    frame_index = 0
    while True:
        ret, frame = cap.read()
        if ret:
            nearest_frame_index = min(available_frames_keys, key=lambda x:abs(x - frame_index))
            if nearest_frame_index - frame_index > fps: 
                continue

            bbox = identities_bboxes[nearest_frame_index] 
                
            for identity_index, identity_bbox in enumerate(bbox):
                
                xmin, ymin, xmax, ymax = [int(b * 2) for b in identity_bbox]
                if pred > 0.5:
                    red = 255 * identity_attentions[identity_index]
                    green = 255 - red

                    if red > green:
                        text = 'Fake ' + str(round(pred*100,2)) + "%"
                    else:
                        text = 'Pristine'
                else:
                    green = int(255 * (1 - pred))
                    red = 255 - green
                    text = 'Pristine '  + str(round((1-pred)*100,2)) + "%"

                color = (0, green, red)
                frame = draw_border(frame, (xmin,ymin), (xmax,ymax), color, 2, 10, 20)
                cv2.putText(frame, text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            output.write(frame)        
        else:
            break
        
        frame_index += 1
    output.release()
    cap.release()

        


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str,
                        help='Path to the video file')
    parser.add_argument("--detector_type", help="type of the detector", default="FacenetDetector",
                        choices=["FacenetDetector"])
    parser.add_argument('--random_state', default=42, type=int,
                        help='Random state value')
    parser.add_argument('--gpu_id', default=0, type=int,
                        help='ID of GPU to be used')
    parser.add_argument('--workers', default=1, type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--config', type=str, 
                        help="Which configuration to use. See into 'config' folder.")
    parser.add_argument('--model_weights', type=str,
                        help='Model weights.')
    parser.add_argument('--extractor_model', type=int, default=0, 
                        help="Which model use for features extraction (0: EfficientNet; 1: XceptionNet).")
    parser.add_argument('--extractor_weights', default='ImageNet', type=str,
                        help='Path to extractor weights or "imagenet".')
    parser.add_argument('--output_type', default=0, type=int,
                        help='Specify which type of output is requested (0: Prediction; 1: Video)".')
    parser.add_argument('--save_attentions', default=False, action="store_true",
                        help='Save attentions plots.')

    opt = parser.parse_args()
    print(opt)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(opt.config, 'r') as ymlfile:
        config = yaml.safe_load(ymlfile)

    # Check for integrity
    if config['model']['num-frames'] != 8 and config['model']['num-frames'] != 16:
        raise Exception("Invalid number of frames.")
        
    if not os.path.exists(opt.video_path):
        raise Exception("Invalid video path.")
 

    # Setup CUDA settings
    torch.cuda.set_device(opt.gpu_id) 
    torch.backends.cudnn.deterministic = True
    random.seed(opt.random_state)
    torch.manual_seed(opt.random_state)
    torch.cuda.manual_seed(opt.random_state)
    np.random.seed(opt.random_state)


    print("Detecting faces...")
    bboxes_dict = detect_faces(opt.video_path, opt.detector_type, opt)
    print("Face detection completed.")


    print("Cropping faces from the video...")
    crops = extract_crops(opt.video_path, bboxes_dict)
    print("Faces cropping completed.")

    '''
    for j, crop in enumerate(crops):
        cv2.imwrite("outputs/faces/face_{}.png".format(j), np.asarray(crop[1]))
    '''
    
    print("Clustering faces...")
    clustered_faces = cluster_faces(crops)
    print("Faces clustering completed.")


    print("Searching for fakes in the video...")
    pred, identity_attentions, aggregated_attentions, identities, frames_per_identity = predict(opt.video_path, clustered_faces, config, opt)
    if pred > 0.5:
        print("The video is fake ("+str(round(pred*100,2)) + "%), showing video result...")
    else:
        print("The video is pristine ("+str(round((1-pred)*100,2)) + "%), showing video result...")
    if opt.output_type == 0:
        print("Prediction", pred)
    else:
        generate_output_video(opt.video_path, pred, identity_attentions, aggregated_attentions, identities, frames_per_identity)
