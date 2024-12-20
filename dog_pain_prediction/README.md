﻿# Dog pain detection algorihtm

This is a video-based dog's pain behavior recognition algorithm. This project is based on https://github.com/s04240051/pain_detection .
The thesis can be found at https://studenttheses.uu.nl/handle/20.500.12932/47308 .
We implement a hierarchical method that can localize the position of the dog within the image frames, extract the pose information, treat missing data problems and normalize the keypoint coordinates. 
A three-stream ConvLSTM, LSTM and random forest/linear layer model is applied to detect dogs' pain action from the RGB video frames and corresponding keypoints sequence and adds a one-hot encoded vector with correspondings behaviors in a particular frame.

<div align=center><img src=pipeline.png width="300" height="300" alt="pipeline"/></div>  

## Data preprocess
### Video frames --> croped images ([bbox_keypoints.py](dataset/bbox_keypoints.py))
* We extract video frames from our own OFT dog pain video dataset. Applying pretrained [YOLOv5](https://github.com/ultralytics/yolov5) algorithm to get bounding box of dogs, and crop the image frame according to the bbox, make sure the dogs' body take a majority of area in the image. All the images are re-scaled to (112X112X3). 
### Croped images --> body keypoints ([kp_process.py](dataset/kp_process.py))
* We apply pretrained [HRNet](https://mmpose.readthedocs.io/en/latest/papers/backbones.html#hrnet-cvpr-2019) algorithm to extract 17 keypoints within the bbox.
* Applying our own missing data treatment algorihtm to filter and complement the keypoints graph. 
## Model
### ConvLSTM
* ConvLSTM is used to extract spatial information from the RGB video frame. Input data of the ConvLSTM is a video clip consist of a stack of RGN frames. The input shape is (N, 112, 112,3), N is length of video clip
### LSTM 
* LSTM with attention mechenism is applied to process the keypoints graph. Input data of the LSTM is a stack of keypoints 2-D coordinates. The input shape is (N, 2X17), N is length of video clip, is the number of keypoints.
### Linear layer and Random Forest classifier
* Processes the behaviors extracted from MSQNet. Subsequently aids in the explainability of the model in a diverse set of methods.
## Training & testing
### Training on Dog-pain dataset
All code details can be found in [./tool/train.py](tool/train.py). See all the configs file in ./configs
```
python test.py --cfg configs/two_stream.yaml \
KEYPOINT_FILE path to_your_keypoint_file \
TRAIN_TEST_SPLIT path_to_your_label_file \ 
CROP_IMAGE path_to_your_image_file \
CHECKPOINTS_FOLD filename_of_checkpoint_file\
ENABLE_TRAIN True
```
### Test on Dog-pain dataset
All code details can be found in [./tool/test.py](tool/test.py).
video clips are sampled without overlapping. Enable output model prediction result by setting `SAVE_PREDS = True`.
```
python test.py --cfg configs/two_stream.yaml \
KEYPOINT_FILE path to_your_keypoint_file \
TRAIN_TEST_SPLIT path_to_your_label_file \ 
CROP_IMAGE path_to_your_image_file \
TEST_INITIAL_WEIGHT path_to_your_checkpoint_file\
CHECKPOINTS_FOLD filename_of_checkpoint_folder\
ENABLE_TEST True
```
## Install
1. Clone this repository
```
https://github.com/s04240051/pain_detection.git
```
2. Install PyTorch>=1.6 and torchvision>=0.7 from the PyTorch [official website](https://pytorch.org/get-started/locally/)
```
pip install -r requirements.txt
```



