import os
import csv
import glob
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils import data
from itertools import compress


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
    

class ActionDataset(data.Dataset):
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
    
    @staticmethod
    def _load_image(directory, image_name):
        return [Image.open(os.path.join(directory, image_name)).convert('RGB')]
    
    def __getitem__(self, index):
        record = self.video_list[index]
        image_names = self.file_list[index]
        indices = self._sample_indices(record.num_frames)
        return self._get(record, image_names, indices)
    
    def __len__(self):
        return len(self.video_list)
        

class AnimalKingdom(ActionDataset):
    def __init__(self, path, act_dict, total_length=12, transform=None, random_shift=False, mode='train'):
        self.path = path
        self.total_length = total_length
        self.transform = transform
        self.random_shift = random_shift
        self.mode = mode
        self.anno_path = os.path.join(self.path, 'action_recognition', 'annotation', mode + '_light.csv')
        self.act_dict = act_dict
        self.num_classes = len(act_dict)
        try:
            self.video_list, self.file_list = self._parse_annotations()
            print("video list", len(self.video_list))
            print("file list", len(self.file_list))
            print("file list entry", self.file_list[0])
        except OSError:
            print('ERROR: Could not read annotation file "{}"'.format(self.anno_path))
            raise

    def _parse_annotations(self):
        video_list = []
        file_list = []
        with open(self.anno_path) as f:
            reader = csv.DictReader(f, delimiter=';')
            for row in reader:
                ovid = row['video_id']
                labels = row['labels']
                path = os.path.join(self.path, 'action_recognition', 'dataset', 'image', ovid)
                files = sorted(os.listdir(path))
                file_list += [files]
                count = len(files)
                labels = [int(l) for l in labels.split(',')]
                video_list += [VideoRecord([path, count, labels])]
        return video_list, file_list

    def _get(self, record, image_names, indices):
        images = list()
        for idx in indices:
            try:
                img = self._load_image(record.path, image_names[idx])
            except:
                print('ERROR: Could not read image "{}"'.format(os.path.join(record.path, image_names[idx])))
                print('invalid indices: {}'.format(indices))
                raise
            images.extend(img)
        
        #print("images", len(images)) #8 images
        process_data = self.transform(images)
        #print("img after transform", process_data.size()) #size [30, 224, 224]
        process_data = process_data.view((self.total_length, -1) + process_data.size()[-2:])
        #print("img after view", process_data.size()) #size [10, 3, 224, 224]
        label = np.zeros(self.num_classes)  # need to fix this hard number
        label[record.label] = 1.0
        print("record.label in _get function", record.label)
        print("label in _get function", label)
        return process_data, label
        
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
        # image_folder = r"C:\Thesis\deepaction_dog\dog_annotation2\frames\spatial"
        # annotation_folder = r"C:\Thesis\deepaction_dog\dog_annotation2\results\annotations"
        video_list = []
        file_list = []

        if self.mode == 'train':
            dog_folders = ['binnur_side_Trim', 'binnur_front_trim', 'Arwen -OFT', 'unal_front_Trim', 'unal_side_trim',
                    'Akrep_OFT_Cam1_Trim', 'Alfa_OFT_Cam1_Trim', 'Alisa_OFT_Cam1_Trim', 'Aria_OFT_ Cam1_Trim']
            # dog_folders = ['binnur_side_Trim', 'binnur_front_trim', 'Arwen -OFT', 'unal_front_Trim', 'unal_side_trim']
        elif self.mode == 'test':
            dog_folders = ['OZTUNCA_SIDE_Trim']
        else:
            print(f"does not recognize mode: {self.mode}")
        #dog_folders = os.listdir(image_folder)
        for dog in dog_folders:
            print(dog)
            annotation_file_path = os.path.join(annotation_folder, f"annotations-{dog}.csv")
            annotation_file = pd.read_csv(annotation_file_path, header=None, skiprows=1)
            image_path = os.path.join(image_folder, dog)
            files = sorted(os.listdir(image_path))

            # TODO:
            # if more than 150 frames, split into smaller records
            # find a way to ensure that there are not record of just 1 frame or something
            for index, row in annotation_file.iterrows():
                label = row[0]
                label_int = self.act_dict[label.lower()]
                start_frame = int(row[1])
                end_frame = int(row[2])
                num_frames = end_frame - (start_frame-1)

                if num_frames > 150:
                    print("too many frames, splitting into smaller records")
                    # Splitting into smaller records
                    rem_frames = num_frames
                    curr_start_frame = start_frame
                    while rem_frames > 0: #ensures that not an empty record is added
                        # Creates a record size with a maximum size of 150
                        curr_size = min(rem_frames, 150)

                        # Extract files for the current record
                        curr_end_frame = curr_start_frame + curr_size - 1
                        curr_record_files = files[curr_start_frame - 1:curr_end_frame]

                        print(f"remaining frames is: {rem_frames}")
                        print(f"current size is: {curr_size}")
                        print(f"current start frames is: {curr_start_frame}")
                        print(f"current end frame is: {curr_end_frame}")
                        
                        # Append the current record to the file list
                        file_list += [curr_record_files]
                        video_list.append(DogVideoRecord([image_path, curr_size, curr_start_frame, curr_end_frame, [label_int]]))
                        print(f"appended record with image path: {image_path}, curr size: {curr_size}, curr_start: {curr_start_frame} \
                              , curr end: {curr_end_frame} and label: {[label_int]}")
                        # Update remaining frames and start frame for the next iteration
                        rem_frames -= curr_size
                        curr_start_frame += curr_size
                else:
                    record_files = files[start_frame-1:end_frame] #this is indexed exluding the endframe number
                    file_list += [record_files]
                    video_list.append(DogVideoRecord([image_path, num_frames, start_frame, end_frame, [label_int]]))
                                  
        return video_list, file_list
    
    def _get(self, record, image_names, indices): #TODO: implement this for dogpain dataset
        images = list()
        for idx in indices:
            try:
                if(idx>=len(image_names)):
                    print("idx is out of bounds")
                    print("record.path", record.path)
                    print("record.num_frames", record.num_frames)
                    print("record.start_frame", record.start_frame)
                    print("record.end_frame", record.end_frame)
                    print("record.label", record.label)
                    print("images_names length", len(image_names))
                    print("idx", idx)
                    print("indices", indices)
                img = self._load_image(record.path, image_names[idx])
            except:
                print('ERROR: Could not read image')
                # print('invalid indices: {}'.format(indices))
                raise
            images.extend(img)
        
        #print("images", len(images)) #8 images
        process_data = self.transform(images)
        #print("img after transform", process_data.size()) #size [30, 224, 224]
        process_data = process_data.view((self.total_length, -1) + process_data.size()[-2:])
        #print("img after view", process_data.size()) #size [10, 3, 224, 224]
        label = np.zeros(self.num_classes)  # need to fix this hard number
        label[record.label] = 1.0
        # print("record.label in _get function", record.label)
        # print("label in _get function", label)
        tuple = (record.path, record.start_frame, record.end_frame)
        return process_data, label, tuple

class Charades(ActionDataset):
    def __init__(self, path, act_dict, total_length=12, transform=None, random_shift=False, mode='train'):
        self.path = path
        self.total_length = total_length
        self.transform = transform
        self.random_shift = random_shift
        self.mode = mode
        self.anno_path = os.path.join(self.path, 'Charades', 'Charades_v1_' + mode + '.csv')
        self.act_dict = act_dict
        self.num_classes = len(act_dict)
        try:
            self.video_list, self.file_list = self._parse_annotations()
        except OSError:
            print('ERROR: Could not read annotation file "{}"'.format(self.anno_path))
            raise

    @staticmethod
    def _cls2int(x):
        return int(x[1:])

    def _parse_annotations(self):
        video_list = []
        file_list = []
        with open(self.anno_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                actions = row['actions']
                if actions == '': continue
                vid = row['id']
                path = os.path.join(self.path, 'Charades_v1_rgb', vid)
                files = sorted(os.listdir(path))
                num_frames = len(files)
                fps = num_frames / float(row['length'])
                labels = np.zeros((num_frames, self.num_classes), dtype=bool)
                actions = [[self._cls2int(c), float(s), float(e)] for c, s, e in [a.split(' ') for a in actions.split(';')]]
                for frame in range(num_frames):
                    for ann in actions:
                        if frame/fps > ann[1] and frame/fps < ann[2]: labels[frame, ann[0]] = 1
                idx = labels.any(1)
                num_frames = idx.sum()
                file_list += [list(compress(files, idx.tolist()))]
                video_list += [VideoRecord([path, num_frames, labels[idx]])]
        return video_list, file_list

    def _get(self, record, image_names, indices):
        images = list()
        for idx in indices:
            try:
                img = self._load_image(record.path, image_names[idx])
            except OSError:
                print('ERROR: Could not read image "{}"'.format(record.path))
                print('invalid indices: {}'.format(indices))
                raise
            images.extend(img)
        process_data = self.transform(images)
        process_data = process_data.view((self.total_length, -1) + process_data.size()[-2:])
        label = record.label[indices].any(0).astype(np.float32)
        return process_data, label


class Hockey(ActionDataset):
    def __init__(self, path, act_dict, total_length=12, transform=None, random_shift=False, mode='train', test_part=6, stride=10):
        self.path = path
        self.total_length = total_length
        self.transform = transform
        self.random_shift = random_shift
        self.mode = mode
        self.stride = stride
        self.clip_length = 30
        self.act_dict = act_dict
        self.num_classes = len(act_dict)
        self._parse_annotations(test_part, mode)

    def _parse_annotations(self, test_part, mode):
        all_dirs = glob.glob(os.path.join(self.path, 'period*-gray'))
        test_dirs = glob.glob(os.path.join(self.path, 'period*-' + str(test_part) + '-gray'))
        train_dirs = list(set(all_dirs) - set(test_dirs))
        if mode == 'train':
            self.video_list, self.file_list = self._files_labels(train_dirs)
        else:
            self.video_list, self.file_list = self._files_labels(test_dirs)

    def _files_labels(self, dirs):
        video_list = []
        file_list = []
        for dir in dirs:
            files = sorted(os.listdir(dir))
            with open(dir + '-label.txt') as f: labels = [[int(x) for x in line.split(',')] for line in f]
            for j in range(0, (len(files) - self.clip_length) + 1, self.stride):
                file_list += [files[j : j + self.clip_length]]
                video_list += [VideoRecord([dir, self.clip_length, np.array(labels[j : j + self.clip_length], dtype=bool)])]
        return video_list, file_list
    
    def _get(self, record, image_names, indices):
        images = list()
        for idx in indices:
            try:
                img = self._load_image(record.path, image_names[idx])
            except OSError:
                print('ERROR: Could not read image "{}"'.format(record.path))
                print('invalid indices: {}'.format(indices))
                raise
            images.extend(img)
        process_data = self.transform(images)
        process_data = process_data.view((self.total_length, -1) + process_data.size()[-2:])
        label = record.label[indices].any(0).astype(np.float32)
        return process_data, label


class Thumos14(ActionDataset):
    def __init__(self, path, act_dict, total_length=12, transform=None, random_shift=False, mode='train'):
        self.path = path
        self.total_length = total_length
        self.transform = transform
        self.random_shift = random_shift
        self.mode = mode
        self.act_dict = act_dict
        self.num_classes = len(act_dict)
        if mode == 'train':
            self.video_list, self.file_list = self._parse_train_annotations()
        else:
            self.video_list, self.file_list = self._parse_testval_annotations(mode)

    def _parse_train_annotations(self):
        path_frames = os.path.join(self.path, 'frames', 'train')
        paths_videos = glob.glob(os.path.join(path_frames, '*/*'))
        file_list = []
        video_list = []
        for path in paths_videos:
            files = sorted(os.listdir(path))
            num_frames = len(files)
            cls = self.act_dict.get(path.split('/')[8])
            file_list += [files]
            video_list += [VideoRecord([path, num_frames, cls])]
        return video_list, file_list

    def _parse_testval_annotations(self, mode):
        path_frames = os.path.join(self.path, 'frames', mode)
        paths_videos = sorted(glob.glob(os.path.join(path_frames, '*')))
        path_ants = os.path.join(self.path, 'annotations', mode)

        # consider the fps from the meta data
        from scipy.io import loadmat
        if mode == 'val':
            file_meta_data = os.path.join(path_ants, 'validation_set.mat')
            meta_key = 'validation_videos'
        elif mode == 'test':
            file_meta_data = os.path.join(path_ants, 'test_set_meta.mat')      
            meta_key = 'test_videos'
        fps = loadmat(file_meta_data)[meta_key][0]['frame_rate_FPS'].astype(int)

        video_fps = {}
        video_frames = {}
        video_num_frames = {}
        for i, path in enumerate(paths_videos):
            vid = path.split('/')[-1]
            files = sorted(os.listdir(path))
            video_fps[vid] = fps[i]
            video_frames[vid] = files
            num_frames = len(files)
            video_num_frames[vid] = num_frames
        file_list = []
        video_list = [] 
                
        for cls in self.act_dict.keys():
            path_ants_cls = os.path.join(path_ants, cls + '_' + mode + '.txt')
            with open(path_ants_cls, 'r') as f:
                lines = f.read().splitlines()
                for lin in lines:
                    vid, _, strt_sec, end_sec = lin.split(' ')
                    strt_frme = np.ceil(float(strt_sec) * video_fps[vid]).astype(int)
                    end_frme = np.floor(float(end_sec) * video_fps[vid]).astype(int)
                    frames_ = video_frames[vid][strt_frme:end_frme + 1]
                    num_frames = end_frme - strt_frme + 1
                    if len(frames_) != num_frames:
                        continue
                        # breakpoint()
                    file_list += [frames_]
                    path = os.path.join(path_frames, vid)
                    video_list += [VideoRecord([path, num_frames, self.act_dict.get(cls)])]
        return video_list, file_list

    def _get(self, record, image_names, indices):
        images = list()
        for idx in indices:
            try:
                img = self._load_image(record.path, image_names[idx])
            except OSError:
                print('ERROR: Could not read image "{}"'.format(record.path))
                print('invalid indices: {}'.format(indices))
                raise
            images.extend(img)
        process_data = self.transform(images)
        process_data = process_data.view((self.total_length, -1) + process_data.size()[-2:])
        return process_data, record.label
    

class Volleyball(ActionDataset):
    def __init__(self, path, act_dict, total_length=12, transform=None, random_shift=False, mode='train'):
        self.path = path
        self.total_length = total_length
        self.transform = transform
        self.random_shift = random_shift
        self.mode = mode
        self.act_dict = act_dict
        self.num_classes = len(act_dict)
        
        # dataset split
        train_set = [1, 3, 6, 7, 10, 13, 15, 16, 18, 22, 23, 31, 32, 36, 38, 39, 40, 41, 42, 48, 50, 52, 53, 54]
        val_set = [0, 2, 8, 12, 17, 19, 24, 26, 27, 28, 30, 33, 46, 49, 51]
        test_set = [4, 5, 9, 11, 14, 20, 21, 25, 29, 34, 35, 37, 43, 44, 45, 47]

        # clip_length = 41
        self.clip_length = 41

        # set the split
        if self.mode == 'train':
            self.set = train_set
        elif self.mode == 'val':
            self.set = val_set
        elif self.mode == 'test':
            self.set = test_set
        else:
            print('ERROR: invalid split')

        self.video_list, self.file_list = self._parse_annotations(self.path)

    def _parse_annotations(self, path):        
        video_list = []
        file_list = []
        for c in self.set:
            file = os.path.join(path, 'volleyball', str(c), 'annotations.txt')
            with open(file, 'r') as f: lines = f.read().splitlines()
            for line in lines:
                video_path = os.path.join(path, 'volleyball', str(c), line.split()[0].split('.')[0])
                labels = [self.act_dict.get(c.capitalize()) for c in list(set(line.split()[6::5]))]
                video_list += [VideoRecord([video_path, self.clip_length, labels])]
                file_list += [sorted(os.listdir(video_path))]
        return video_list, file_list

    def _get(self, record, image_names, indices):
        images = list()
        for idx in indices:
            try:
                img = self._load_image(record.path, image_names[idx])
            except OSError:
                print('ERROR: Could not read image "{}"'.format(record.path))
                print('invalid indices: {}'.format(indices))
                raise
            images.extend(img)
        process_data = self.transform(images)
        process_data = process_data.view((self.total_length, -1) + process_data.size()[-2:])
        label = np.zeros(self.num_classes)
        label[record.label] = 1.0
        return process_data, label
        
    

if __name__ == '__main__':
    path = '/home/adutta/Workspace/Datasets/Volleyball'
    volleyball_dataset = Volleyball(path, total_length=12, random_shift=True, mode='train')
    # import torchvision.transforms as transforms
    # transform = transforms.ToTensor()
    # thumos_dataset = Thumos14(path, total_length=15, random_shift=True, mode='train')
    # ind = thumos_dataset._sample_indices(150)
    # print(ind)
    # print(len(thumos_dataset))

    # charades_dataset.__getitem__(0)