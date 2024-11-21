import os
import random
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from datasets.transforms_ss import *
from torchvision.transforms import Compose

# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, annotation_files, frames_paths, dog_names, prompts):
        self.annotation_files = annotation_files
        self.frames_paths = frames_paths
        self.dog_names = dog_names
        self.texts = []
        self.labels = []
        self.image_paths = []
        self.prompts = prompts

        # Process annotations
        for idx, annotation_file in enumerate(annotation_files):
            dog_name = self.dog_names[idx]
            frames_path = self.frames_paths[idx]
            for index, row in annotation_file.iterrows():
                behavior = row[0]
                description = self.generate_prompt(str(behavior).lower(), self.prompts)
                start_frame = int(row[1])
                end_frame = int(row[2])
                nr_frames = end_frame - (start_frame-1)
                #TODO: check labels, texts and image_paths
                for frame in range(1, nr_frames+1):
                    idx = str(frame).zfill(6)
                    self.texts.append(description)
                    self.labels.append(str(behavior).lower())
                    #dog_name = self.frames_path.split('\\')[-1]
                    image_path = os.path.join(frames_path, f"{dog_name}_{idx}.jpg")
                    self.image_paths.append(image_path)

    def generate_prompt(self, behavior, prompts):
        prompt = ''
        prompt = random.choice(prompts[behavior])
        idx = prompts[behavior].index(prompt)
        return prompt

    # def get_train_transforms(self):
    #     """Returns the training torchvision transformations for each dataset/method.
    #        If a new method or dataset is added, this file should by modified
    #        accordingly.
    #     Args:
    #       method: The name of the method.
    #     Returns:
    #       train_transform: An object of type torchvision.transforms.
    #     """
    #     input_mean = [0.48145466, 0.4578275, 0.40821073]
    #     input_std = [0.26862954, 0.26130258, 0.27577711]
    #     input_size = 224
    #     unique = Compose([GroupMultiScaleCrop(input_size, [1, .875, .75, .66]),
    #                         GroupRandomHorizontalFlip(True),
    #                         GroupRandomColorJitter(p=0.8, brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
    #                         GroupRandomGrayscale(p=0.2),
    #                         GroupGaussianBlur(p=0.0),
    #                         GroupSolarization(p=0.0)])
    #     common = Compose([Stack(roll=False),
    #                     ToTorchFormatTensor(div=True),
    #                     GroupNormalize(input_mean, input_std)])
    #     transforms = Compose([unique, common])
    #     return transforms
    
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        image_path = self.image_paths[idx]
        return image_path, text, label

        # images = list()
        # transforms = self.get_train_transforms()
        # for img in self.image_paths:
        #     try:
        #         img = [Image.open(img).convert("RGB")]
        #     except:
        #         print('ERROR: Could not read image')
        #         raise
        #     images.extend(img)
        #     process_data = transforms(images)
        #     process_data = process_data.view((10, -1) + process_data.size()[-2:])
        #     label = np.zeros(self.num_classes)  # need to fix this hard number
        #     label[label] = 1.0
        #     return process_data, label


class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return self._data[2]
    
class DogVideoRecord(object):
    def __init__(self, row):
        '''
        The row consists of [path(string), num_frames(int), start_frame(int), end_frame(int), label(list)]
        '''
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])
    
    @property
    def start_frame(self):
        return int(self._data[2])
    
    @property
    def end_frame(self):
        return int(self._data[3])

    @property
    def label(self):
        return self._data[4]
    

class ActionDataset(Dataset):
    def __init__(self, total_length):
        self.total_length = total_length
        self.video_list = []
        self.random_shift = False

    def _sample_indices(self, num_frames):
        if num_frames <= self.total_length:
            indices = np.linspace(0, num_frames - 1, self.total_length, dtype=int)
        else:
            ticks = np.linspace(0, num_frames, self.total_length + 1, dtype=int)
            if self.random_shift:
                indices = ticks[:-1] + np.random.randint(ticks[1:] - ticks[:-1])
            else:
                indices = ticks[:-1] + (ticks[1:] - ticks[:-1]) // 2
        return indices
    
class Dog(ActionDataset):
    def __init__(self, path, act_dict, total_length=12, transform=None, random_shift=False, mode='train'):
        self.path = path
        self.total_length = total_length
        self.transform = transform
        self.random_shift = random_shift
        self.mode = mode
        self.anno_path = os.path.join(self.path, 'dog_pain', 'annotations_' + mode + '.csv')
        self.act_dict = act_dict
        self.num_classes = len(act_dict)
        try:
            self.video_list, self.file_list = self._parse_annotations()
        except OSError:
            print('ERROR: Could not read annotation file "{}"'.format(self.anno_path))
            raise

    def _parse_annotations(self):
        """
        returns a list of the videos and a list of the frames
        """
        image_folder = r"/home/albert/Lisanne/frames/pain/frames/spatial"
        annotation_folder = r"/home/albert/Lisanne/frames/pain/annotations"
        video_list = []
        file_list = []
        dog_folders = ['unal_front_Trim', 'unal_side_trim']
        #dog_folders = os.listdir(image_folder)
        for dog in dog_folders:

            annotation_file_path = os.path.join(annotation_folder, f"annotations-{dog}.csv")
            annotation_file = pd.read_csv(annotation_file_path, header=None, skiprows=1)
            image_path = os.path.join(image_folder, dog)
            files = sorted(os.listdir(image_path))

            for index, row in annotation_file.iterrows():
                label = row[0]
                label_int = self.act_dict[label.lower()]
                start_frame = int(row[1])
                end_frame = int(row[2])
                num_frames = end_frame - (start_frame-1)
                record_files = files[start_frame-1:end_frame] #this is indexed exluding the endframe number
                file_list += [record_files]
                video_list.append(DogVideoRecord([image_path, num_frames, start_frame, end_frame, [label_int]]))
                                  
        return video_list, file_list