# MINTIME-DF Multi-Identity size-iNvariant TIMEsformer for Video Deepfake Detection

![Header](images/header.gif)
## Motivations behind this study
The continuing advancement of deepfake generation techniques and the increasingly credible results obtained through these, makes it increasingly urgent to develop new techniques to distinguish a manipulated video from a real one. This is, however, a far from simple task that introduces multiple challenges to be overcome, challenges that form the basis of this research work. 
- <b>Generalization of the Deepfake concept</b>:  Deepfake generation methods tend to introduce specific anomalies within images and videos. Deepfake detection models often tend to learn to recognise these specific anomalies and are therefore ineffective in the real world when dealing with unseen manipulations. Our previous studies in this area suggest a greater capacity for generalisation by Vision Transformers than by Convolutional Neural Networks [<a href="https://arxiv.org/abs/2206.13829">Coccomini et al, 2022</a>];
- <b>Ability to pick up both spatial and temporal anomalies within a video</b>: Very often the anomalies that are searched for by deepfake detectors are exclusively spatial with frame-by-frame classifications. However, some important anomalies lie precisely in the variation of the face over time, which can be unnatural and thus allow a manipulated video to be identified;
- <b>Handling of multiple faces within the same video</b>: A specific situation that can be exploited by an attacker to deceive a deepfake detection system is found in the case of videos or images with multiple faces (identities). An attacker could in fact decide to manipulate only one of the people in the video. However, if the detection is carried out en bloc for all the faces in the video, the negative contribution to the final prediction made by the fake faces could be 'masked' by the non-manipulated ones, thus deceiving the system.
- <b>Management of different face-frame area ratios</b>: Typically in deepfake detection systems, the person's face is extracted from the video or image to be classified and before being given as input to a neural model it is rescaled to be uniform with all the others. This results in an important loss of information, namely the ratio of the area of the subject's face to the rest of the scene.

To solve all these problems, we propose a Size-Invariant Multi-Identity Transformer-based architecture that exploits Divided Space-Time attention. 
The special features of our proposal are as follows:
- Application of preprocessing and data augmentation techniques to increase the coherence of the video tracks given as input to the network and its generalisation capability;
- Exploitation of an EfficientNet B0 as a pre-trained patch extractor on Deepfake Detection tasks. This module provides a trainable and efficient method of patch extraction compared to the traditional input image split used in Vision Transformers;
- Using a TimeSformer as a Transformer module in order to also capture time variations within a video;
- Introduction of a new embedding technique, namely size-embedding, to induce face-frame area ratio information to the TimeSformer;
- Use of novels positional embedding, attention calculation and input sequence generation to enable a TimeSformer to handle multiple identities in the same video.


## Setup
Clone the repository and move into it:

```
git clone https://github.com/davide-coccomini/MINTIME-Multi-Identity-size-iNvariant-TIMEsformer-for-Video-Deepfake-Detection.git

cd MINTIME-Multi-Identity-size-iNvariant-TIMEsformer-for-Video-Deepfake-Detection
```

Setup Python environment using conda:

```
conda env create --file environment.yml
conda activate deepfakes
export PYTHONPATH=.
```


## Run Deepfake Detection on a video
If you want to directly classify a video using pre-trained models, you can download the weights from the model zoo and use the following command:

```
python3 predict.py --video_path path/to/video.mp4 --model_weights path/to/model_weights --extractor_weights path/to/extractor_weights --config config/size_invariant_timesformer.yaml
```

The output video will be stored in the examples/preds folder:
![Prediction Example](images/example_detection.gif)

For purposes of explainability the attention maps on the various slots of the input sequence are also saved. These are used to discover, in the multi-identity case, which identity is fake in each frame.

In the following example, the 16 slots are distributed between the two identities according to the number of available faces, the first 6 for identity 0 (the man), the second 6 for identity 1 (the woman). The remaining 4 slots are ignored as there are no additional faces to fill them. 
The attention values extracted from the various heads are combined considering the maximum values for each tokens. The tokens are then grouped according to the frame and identity they refer to, resulting in 16 attention values (of which 4 are null). The spatial and temporal attention is combined to obtain the final value via the average function. Finally, the softmax function is applied to emphasise the differences between the attention placed on one face rather than another.

In this case, the attention in frame 20 of the second identity is particularly high, which indicates that there is an anomaly there.

![Prediction Example](images/attention_analysis.gif)

## Model ZOO

### Models comparison
| Model | Identities | Training Dataset | Test Dataset | Accuracy | AUC  |  Weights |
| --------------- | --------------- | --------------- |  --------------- | --------------- | --------------- | --------------- |
| SlowFast R-50 | 1 | ForgeryNet | ForgeryNet | 88.78 | 93.88 | N/A |
| TSM | 1 | ForgeryNet | ForgeryNet | 88.04 | 93.05 | N/A |
| MINTIME | 1 | ForgeryNet | ForgeryNet | 81.92 | 90.13 | LINK |
| MINTIME | 2 | ForgeryNet | ForgeryNet | 82.28 | 90.45 | LINK |
| MINTIME | 3 | ForgeryNet | ForgeryNet | 82.05 | 90.28 | LINK |
| EfficientNet-B0 + MLP | 1 | ForgeryNet | ForgeryNet | 65.33 | 71.42 | LINK |
| EfficientNet-B0 + MLP | 2 | ForgeryNet | ForgeryNet | 67.03 | 71.05 | LINK |
| EfficientNet-B0 + MLP | 3 | ForgeryNet | ForgeryNet | 66.89 | 70.92 | LINK |


### Cross-Forgery Analysis
|                 |                 | ID-replaced     | ID-remained     | Identities      |
| --------------- | --------------- | --------------- | --------------- | --------------- | 
|                 |                 | Accuracy & AUC  | Accuracy & AUC  |                 |
| X3D-M           | ID-replaced     |  87.92   & 92.91| 55.25    & 65.59|       1         |
|                 | ID-remained     |  55.93   & 62.87| 88.85    & 95.40|                 |
| SlowFast        | ID-replaced     |  88.26   & 92.88| 52.64    & 64.83|       1         |
|                 | ID-remained     |  52.70   & 61.50| 87.96    & 95.47|                 |
| MINTIME         | ID-replaced     |  80.18   & 83.86| 79.03    & 86.98|       2         |
|                 | ID-remained     |  63.13   & 66.26| 89.22    & 95.02|                 |


## Dataset
In order to conduct our research, it was necessary to analyse the various datasets in circulation in order to identify the one with the following characteristics:
- Containing a sufficient number of videos for effective training;
- Presence of multi-faces videos;
- Multiple face-frame area ratios present in the videos;
- Large variety of subjects, scenes, perturbations, manipulation techniques. 

For these reasons, ForgeryNet was chosen as the dataset for our experiments. It is in fact characterised by a number of videos equal to 221,247 divided into 99,630 pristine videos and 121,617 manipulated ones with a frame rate between 20 and 30 FPS and variable duration.
From an analysis conducted during our research, we identified the presence of 11,785 video multi-faces with a maximum of 23 faces per video. The face-frame area ratio also appears to be highly distributed with videos containing faces covering an area up to almost, in some rare cases, even 100% of the entire image.

![ForgeryNet face/frame ratio distribution](images/forgery_net_ratios.png)

Furthermore, the EfficientNet B0 used as a patch extraction backbone was trained in <a href="https://arxiv.org/abs/2107.02612">our previous research work</a> on the DFDC and FaceForensics++ datasets.

The datasets can be downloaded at the following links:
- ForgeryNet: https://yinanhe.github.io/projects/forgerynet.html#download
- DFDC: https://dfdc.ai/
- FaceForensics++: https://github.com/ondyari/FaceForensics/blob/master/dataset/


## Preprocessing
In order to use the proposed model, some preprocessing steps are required to convert the ForgeryNet into the desired format.

In case you want to retrain the convolutional backbone patch extraction the preprocessing of DFDC and FaceForensics++ datasets, the procedure is described in <a href=https://github.com/davide-coccomini/Combining-EfficientNet-and-Vision-Transformers-for-Video-Deepfake-Detection>this repository</a>. Otherwise you can directly use the pretrained model as explained in the training section.

### Face Detection and Extraction
To perform deepfake detection it is necessary to first identify and extract faces from all the videos in the dataset.
Detect the faces inside the videos:
```
cd preprocessing
python3 detect_faces.py --data_path "path/to/videos"
```

The extracted boxes will be saved inside the "path/to/videos/boxes" folder.
In order to get the best possible result, make sure that at least one face is identified in each video. If not, you can reduce the threshold values of the MTCNN on line 38 of face_detector.py and run the command again until at least one detection occurs.
At the end of the execution of face_detector.py an error message will appear if the detector was unable to find faces inside some videos.

If you want to manually check that at least one face has been identified in each video, make sure that the number of files in the "boxes" folder is equal to the number of videos. To count the files in the folder use:
```
cd path/to/videos/boxes
ls | wc -l
```

Extract the detected faces obtaining the images:
```
python3 extract_crops.py --data_path "path/to/videos" --output_path "path/to/output"
```

Repeat detection and extraction for all the different parts of your dataset.

After extracting all the faces from the videos in your dataset, organise the "dataset" folder as follows:
```
- ForgeryNet
    - Training
        - crops
            - train_video_release
                - 1
                    - video_name_0
                        0_0.png
                        1_0.png
                        2_0.png
                        ...
                        N_0.png
                    ...
                    - video_name_K
                        0_0.png
                        1_0.png
                        2_0.png
                        ...
                        M_0.png
                ...
                - 19
                    - video_name_0
                        0_0.png
                        1_0.png
                        2_0.png
                        ...
                        S_0.png
                    ...
                    - video_name_Y
                        0_0.png
                        1_0.png
                        2_0.png
                        ...
                        J_0.png
    - Validation
        - crops
            - val_video_release
                ...
                    ...
                        ...
                        ...
                        ...
```

We suggest to exploit the --output_path parameter when executing extract_crops.py to build the folders structure properly.


### Split the Dataset
Since the labels of the ForgeryNet test set were not made public at the time of the study, the Validation Set will be used as our Test Set while our Validation Set is obtained through a customised split on the distribution of the training set.
The CSV files containing the videos belonging to each set are available in the "splits" folder, however, should you wish to redo the process of splitting the dataset, you can follow the steps below.
```
cd preprocessing
python3 split_dataset.py --train_list_file path/to/training_list_file.txt --validation_list_file path/to/validation_list_file.txt 
```
The script will analyse the distribution of deepfake generation methods in the training set and move the videos within three separate folders train, val and test accordingly inside the "faces" folder.

![ForgeryNet custom split deepfake generation methods distribution](images/resulting_distribution.png)


The dataset at the end of this process will have the following structure:

```
- ForgeryNet
    - faces
        - train
            - 1
                - video_name_0
                    0_0.png
                    1_0.png
                    2_0.png
                    ...
                    N_0.png
                ...
                - video_name_K
                    0_0.png
                    1_0.png
                    2_0.png
                    ...
                    M_0.png
            ...
            - 19
                - video_name_0
                    0_0.png
                    1_0.png
                    2_0.png
                    ...
                    S_0.png
                ...
                - video_name_Y
                    0_0.png
                    1_0.png
                    2_0.png
                    ...
                    J_0.png
        - val
            ...
                ...
                    ...

        - test
            ...
                ...
                    ...

```

### Identity Clustering
Having to manage multi-face videos and wanting to detect temporal and not just spatial anomalies, it is necessary to clustered the faces in each video on the basis of their similarity and maintaining the temporal order of their appearance in the frames. To do this, a clustering algorithm was developed that groups the faces extracted from the videos into sequences. 

To run the clustering split use the following commands:
```
cd preprocessing
python3 cluster_faces.py --faces_path path/to/faces
```

The algorithm is structured as follows:
- The features of each face are extracted via an InceptionResnetV1 pretrained on FaceNet;
- The distance between each face and all faces identified in the video is calculated;
- A graph is constructed with hard connection if the similarity is higher than the threshold;
- Clusters are obtained based on the graph and small clusters are discarded;
- The faces inside the clusters are temporally reordered;
- The clusters are enumerated based on mean faces size during data loading. 

![Clustering algorithm for multi-faces videos](images/clustering_deepfake_global.gif)


The following parameters can be changed as desired to achieve different clustering:
- --similarity_threshold: Threshold used to discard faces with high distance value (default 0.8);
- --valid_cluster_size_ratio: Valid cluster size percentage (default: 0.2)

The dataset at the end of this process will have the following structure:
```
- ForgeryNet
    - faces
        - train
            - 1
                - video_name_0
                    - identity_0
                        0_0.png
                        1_0.png
                        2_0.png
                        ...
                        D_0.png
                    ...
                    - identity_U
                        0_0.png
                        1_0.png
                        2_0.png
                        ...
                        T_0.png
                ...
                - video_name_K
                    - identity_0
                        0_0.png
                        1_0.png
                        2_0.png
                        ...
                        P_0.png
                    ...
                    - identity_X
                        0_0.png
                        1_0.png
                        2_0.png
                        ...
                        R_0.png
            ...
            - 19
                ...
                    ...
                        ...
        - val
            ...
                ...
                    ...
                        ...

        - test
            ...
                ...
                    ...
                        ...
```

## Training
After transforming each video in the dataset into temporally and spatially coherent sequences, one can move on to the training phase of the model.

To download the pretrained weights of the models you can run the following commands:
```
mkdir weights
cd weights
wget ...
wget ...
wget ...
```


If you are unable to use the previous urls you can download the weights from [Google Drive](https://drive.google.com/drive/folders/19bNOs8_rZ7LmPP3boDS3XvZcR1iryHR1?usp=sharing).


The network is trained to perform pristine/fake binary classification. The features are extracted from a pertrained EfficientNet B0 and the training of the TimeSformer is also influenced by the presence of an additional embedding, namely the size embedding. It is calculated from the face-frame area ratio for each face of the video and concatenated to each token obtained from it. 

![Size Invariant TimeSformer](images/size_invariant_timesformer.gif)

### Masking and Sampling
The number of frames per video, and thus consecutive faces to be considered for classification, is set via the num-frames parameter in the configuration file. In the event that there are fewer faces in the considered identity than necessary, more empty ones are added and then a mask is used to drive the calculation of attention properly. In the case of longer sequences, however, uniform sampling is performed. Like a kind of data augmentation, this uniform sampling is performed by alternating various combinations of frames as shown in figure.

![Mask Generation and Sequence Sampling](images/masking.png)
 

### Adaptive Input Sequence Assignment
To enable the model to handle multiple identities within one video, the number of available frames is divided among the identities of the video.
The maximum number of identities per video is set via the max-identities parameter in the configuration file.

The identities are reordered according to the size of the faces within them, and the most important identities are given a higher number of frames to be exploited in the input sequence, in order to give more importance to faces that cover a larger area and are therefore likely to be more relevant in the video, as opposed to smaller faces.
![Input Sequence Assignemnt Example 1](images/sequence_generation_0.gif)


In the event that an identity does not have enough faces to satisfy the number of slots allocated to it, the remaining slots are inherited by the next identity. 
![Input Sequence Assignemnt Example 2](images/sequence_generation_1.gif)


### Temporal Coherent Positional Embedding
Classical positional embedding was then evolved to ensure temporal consistency between frames as well as spatial consistency between tokens. 
Tokens are numbered in such a way that two faces, of different identities but belonging to the same frame, have the same numbering. 
Temporal coherence is maintained both locally by having an increasing numbering sequence as well as the frames from which the faces originate and globally by being generated on the basis of the global distribution of frames of all identities in the video.
In this first example, the two sequences are of the same length and have the same frame numbering. Therefore, the tokens are also numbered in the same way.
![Temporal Positional Embedding Example 1](images/temporal_positional_embedding_0.gif)

In example number two, however, although the two identities have the same number of faces, they are extracted from different frames. The numbering of the tokens therefore in this case, in addition to being generated by taking the sequentiality locally for each identity into account, is also assigned on the basis of the global distribution of frames.
![Temporal Positional Embedding Example 2](images/temporal_positional_embedding_1.gif)

### Identity-based Attention Calculation
For our TimeSformer we apply the version of attention that was most effective in the original paper, namely Divided Space-Time Attention. Attention is calculated spatially between all patches in the same frame, but is then also calculated between the corresponding patches in the next and previous frames using a moving window. 

![Divided Space-Time Attention](images/divided_space_time_attention.gif)

As far as spatial attention is concerned, no further effort is required for this to be applied to our case.  
Not being interested in capturing the relationships between faces of different identities, the calculation of temporal attention in our case is carried out exclusively between faces belonging to the same identity. 

![Multi-Face Identity-based Attention Calculation](images/identity_attention.gif)

All faces, however, influence the CLS that is global and unique for all identities. 
In the animation below, it is shown how attention is calculated exclusively by tokens referring to identity 0 faces (green), ignoring those referring to identity 1 faces (red) and vice versa. While all refer to the global CLS.

![Multi-Face Identity-based Attention Calculation](images/attention_calculations_multi_face.gif)

### Multi-Face Size-Invariant TimeSformer

![Multi-Face Size-Invariant TimeSformer](images/multi_face_size_invariant_timesformer.gif)

To run the training process use the following commands:
```
python3 train.py --config config/size_invariant_timesformer.yaml --model 1 --train_list_file path/to/training_list_file.txt --validation_list_file path/to/validation_list_file.txt --extractor_weights path/to/backbone_weights
```

The following parameters can be changed as desired to perform different training:
- --num_epochs: Number of training epochs (default: 300);
- --resume: Path to latest checkpoint (default: none);
- --restore_epoch: Restart from the checkpoint's epoch if --resume option specified (default: False)
- --freeze_backbone: Maintain the network freezed or train it (default: False);
- --extractor_unfreeze_blocks: Number of blocks to train in the backbone (default: All);
- --max_videos: Maximum number of videos to use for training (default: all);
- --patience: How many epochs wait before stopping for validation loss not improving (default: 5);
- --logger_name: Path to the folder for tensorboard logging (default: runs/train);


### Baseline 
To validate the real effectiveness of the implementation choices made on the presented architecture, we also conducted some alternative architecture training. In particular, the simplest of the two consists of a freezed EfficientNet-B0 pre-trained on the DFDC and FaceForensics++ datasets but whose output features, instead of going into a Transformer as in the original architecture, are given as input directly to a simple MLP. 
![Baseline](images/baseline.gif)

The MLP performs frame-by-frame classification for each face of the video and the predictions are then averaged and evaluated against a fixed threshold. 


## Inference
To run the evaluation process of a trained model on a test set, use the following command:
```
test.py --model_weights path/to/model  --extractor_weights path/to/model --video_path path/to/videos --data_path path/to/faces --test_list_file path/to/test.csv --model model_type --config path/to/config
```

You can also use the option --save_attentions to save space, time and combined attention plots.


## Additional Parameters
In almost all the scripts the following parameters can be also customized:

- --gpu_id: ID of GPU to use for processing or -1 to use multi-gpu only for the training (default: 0);
- --workers: Number of data loader workers (default: 8);
- --random_state: Random state number for reproducibility (default: 42)

# Reference
TODO
