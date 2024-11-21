from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
import numpy as np
import pandas as pd
import os
from lib.utils.logging import get_logger
from . import data_utils as utils
logger = get_logger(__name__)


class Data_loader:
    def __init__(self, cfg):
        print("init dataloader")
        if cfg.FORMAT_DATASET and cfg.DATASET_PATH != "":
            data_path = cfg.DATASET_PATH #TODO figure out where exactly this needs to point to, might be the answer if it still goes wrong
            ann_path = os.path.join(data_path, "annotation")
            self.label_file = os.path.join(ann_path, "split")
            print(f"TRUE: label file is {self.label_file}")
        else:
            self.label_file = cfg.TRAIN_TEST_SPLIT
            print(f"FALSE: label file is {cfg.TRAIN_TEST_SPLIT}")
            
        self.model_type = cfg.MODEL.TYPE
        assert self.model_type in ["two_stream",
                                   "rgb", "kp", "flow", "ethogram", "ethogram_only"], f"model type not support"
        self.img_size = cfg.MODEL.IMG_SIZE
        self.clip_len = cfg.DATA.CLIP_LENGTH
        self.ethogram_path = cfg.ETHOGRAM_FILE
        self.ethogram_dim = cfg.MODEL.ETHOGRAM_EMBEDDING_DIM
        self.cfg = cfg

    def label_prepare(self, split):
        file = os.path.join(self.label_file, split+".h5")
        step = self.clip_len if split == "test" else None
        container = []

        df = pd.read_hdf(file, "df_with_missing") #this is the train/test split file with the starts and ends
        df = df.applymap(lambda item: int(item))
        data_meter = utils.Dataset_inf(self.cfg)
        for i in range(len(df)):
            line = df.iloc[i] #probably add the ethogram to this line
            extra_label = data_meter.extra_label(line)
            #ethogram = []
            if self.model_type == "ethogram" or self.model_type == "ethogram_only":
                #print(f"the model is ethogram, extracting ethograms now")
                ethogram = self.extract_ethogram(line)
            else:
                print(f"the model is not ethogram, so not extracting ethograms.")
                ethogram = []
            clip_set = self.clip_extract(line, extra_label, ethogram, step)
            #print(f"clip_set: {clip_set}")
            data_meter.label_update(len(clip_set), line["label"], extra_label)
            container.extend(clip_set)
        data_meter.display(logger, split, len(container))
        #print(f"this is the container: {container}")
        return container
    
    def extract_ethogram(self, line):
        name_split = (line.name).split(".")[0]
        file_name = f"{name_split}-one_hot_annotations.csv"
        file_path = os.path.join(self.ethogram_path, file_name)
        etho_df = pd.read_csv(file_path)
        starts = line.starts
        ends = line.ends
        ethogram = np.zeros(self.ethogram_dim)
        print(f"this is line information: {line}")
        for i, row in etho_df.iterrows():
            if i < starts:
                continue
            elif i > ends:
                continue
            else:
                temp_ethogram = self.string_to_float_list(row.iloc[1])
                # print(f"index: {i}")
                # print(f"curr ethogram: {ethogram}")
                # print(f"temp ethogram: {temp_ethogram}")
                #if sitting abornammy is present, change it to sitting
                # if temp_ethogram[6] > 0:
                #     temp_ethogram[5] += temp_ethogram[6]
                #     temp_ethogram[6] = 0
                # if temp_ethogram[8] > 0:
                #     temp_ethogram[8] = 0
            ethogram = np.add(ethogram, temp_ethogram)
        #print(f"ethogram in extract ethogram: {ethogram}")
        #ethogram[ethogram > 1] = 1
        ethogram[8] = 0
        ethogram[9] = 0
        print(f"ethogram before: {ethogram}")
        #filtered_ethogram_data = self.filter_max_values(ethogram)
        #print(f"ethogram after: {filtered_ethogram_data}")
        return ethogram

    def filter_max_values(self, arr):
        max_value = np.max(arr)
        if max_value == 0:
            return arr
        filtered_arr = np.where(arr == max_value, arr, 0)
        #keep the decreased weight bearing
        # if arr[-1] > 0:
        #     filtered_arr[-1] = arr[-1]
        return filtered_arr

    def string_to_float_list(self, array):
        stripped = array.strip('[]"')
        return [float(x) for x in stripped.split()]

    def label_update(self, label_info, num_clip, label):
        name_mp = ["not_pain", "pain"]
        key = name_mp[label]
        label_info[key] = label_info[key] + num_clip

    def clip_extract(self, line, extra_label, ethogram, step=None):
        """returns a container which looks like
        [dog_name, start_index, end_index, label, ethogram]
        """
        if not step:
            step = 1 if line["length"] <= 2*self.clip_len else 2
        if self.model_type == "flow" and line["length"] > self.clip_len:
            return  [[line.name, i, i+self.clip_len, line["label"], ethogram]+extra_label for i in range(line["starts"]+1, line["ends"]-self.clip_len, step)]
        else:
            return [[line.name, i, i+self.clip_len, line["label"], ethogram]+extra_label for i in range(line["starts"], line["ends"]-self.clip_len, step)]

    def construct_loader(self, split):
        data_type = self.cfg.DATA.DATA_TYPE
        balance_policy = self.cfg.DATA.BALANCE_POLICY
        if split == "train":
            shuffle = True 
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        clip_df_set = self.label_prepare(split) #in here is the ethogram
        
        sampler = utils.weightedsample(
            clip_df_set, data_type) if balance_policy == 2 else None
        loss_w = [None]*2 if balance_policy == 0 else utils.loss_weight(
            clip_df_set, data_type, balance_policy)
        aug = self.cfg.DATA.AUG if split == "train" else False

        dataset = Pain_dataset(
            clip_df_set,
            self.img_size,
            self.model_type,
            self.cfg,
            aug,
        )
        loader = DataLoader(
            dataset,
            self.cfg.RUN.TRAIN_BATCH_SIZE,
            shuffle=(False if sampler else shuffle),
            sampler=sampler,
            num_workers=self.cfg.RUN.NUM_WORKS,
            pin_memory=True,
            drop_last=drop_last,

        )
        return loader, loss_w


class Pain_dataset(Dataset):
    def __init__(
        self,
        label_list,
        img_size,
        model_type,
        cfg,
        aug
    ):

        if cfg.FORMAT_DATASET and cfg.DATASET_PATH != "":
            data_path = cfg.DATASET_PATH
            ann_path = os.path.join(data_path, "annotation")
            self.crop_img = os.path.join(data_path, "crop_img")
            self.video_file = os.path.join(ann_path, "kp_valid")
            self.flow_img = os.path.join(data_path, "optical_flow")
            self.ethogram = os.path.join(data_path, "ethogram")
        else:
            self.crop_img = cfg.CROP_IMAGE
            self.video_file = cfg.KEYPOINT_FILE
            self.flow_img = cfg.FLOW_IMAGE
            self.ethogram = cfg.ETHOGRAM_FILE
        self.label_list = label_list
        #print(f"this is the label list: {self.label_list}")
        self.img_size = img_size
        self.data_dict = {}
        self.bbox_col = ["x", "y", "w", "h"]
        self.model_type = model_type
        self.cfg = cfg
        self.aug = aug
        #print(f"this is the label list: {label_list}")

    def __len__(self):
        return len(self.label_list)
    
    def _get_entire_ethogram_set(self):
        ethogram_list = []
        labels = []
        for item in self.label_list:
            ethogram = item[-1]
            label = item[-2]
            ethogram_list.append(ethogram)
            labels.append(label)
        return ethogram_list, labels

    def __getitem__(self, index):
        clip = self.label_list[index]
        #print(f"this should be the ethogram: {clip[-1]}")
        file_name = clip[0]
        if file_name in self.data_dict:
            video_df = self.data_dict[file_name]

        else:
            video_df = pd.read_hdf(
                os.path.join(self.video_file, file_name), "df_with_missing"
            )
            self.data_dict[file_name] = video_df

        input_data = []
        video_name = os.path.join(self.crop_img, file_name.split(".")[0])
        clip_inf = video_df.iloc[clip[1]:clip[2]] #this looks at all the frames from the start until the end frame
        aug_para = utils.aug_param_builder(
            clip_inf, video_name) if self.aug else []

        if self.model_type in ["two_stream", "rgb", "flow", "ethogram"]:
            frames = self.frames_builder(
                clip_inf[self.bbox_col], video_name, aug_para, stream="rgb"
            )
            input_data.append(frames)

        if self.model_type in ["two_stream", "kp", "ethogram"]:
            feature = clip_inf.iloc[:, :-4].values
            kps = self.keypoints_builder(
                feature, aug_para
            )
            input_data.append(kps)
        if self.model_type in ["flow"]:
            flow_video_name = os.path.join(self.flow_img, file_name.split(".")[0])
            frames = self.frames_builder(
                clip_inf[self.bbox_col], flow_video_name, aug_para, stream="flow"
            )
            input_data.append(frames)
        start_name = clip_inf.index[0]

        # print(f"this is the start name: {start_name}") #--> the name of the img file
        # print(f"this is the clip_inf: {clip_inf}")
        # print(f"this is the clip_inf shape: {clip_inf.shape}")
        # print(f"these are the keypoints of one image: {clip_inf.iloc[0,:-4]}")


        if self.cfg.DATA.DATA_TYPE in ["simple", "diff"]:
            return input_data, int(clip[-2]), clip[-1], start_name #frame and keypoint, label, ethogram, name
        elif self.cfg.DATA.DATA_TYPE == "aux":
            return input_data, [int(clip[-2]), int(clip[-1])], start_name #TODO change the indices here if the ethogram is added if you want to make auxiliarly learning work again

    def new_crop(self, img, crop_x, crop_y, size, percentage):
        # Calculate the new crop size (10% smaller)
        new_size = size * percentage

        # Calculate the center point of the original crop box
        center_x = crop_x + size / 2
        center_y = crop_y + size / 2

        # Calculate the new top-left corner coordinates
        new_crop_x = center_x - new_size / 2
        new_crop_y = center_y - new_size / 2

        # Calculate the new bottom-right corner coordinates
        new_crop_x_end = new_crop_x + new_size
        new_crop_y_end = new_crop_y + new_size

        # Crop the image with the new coordinates
        img = img.crop((new_crop_x, new_crop_y, new_crop_x_end, new_crop_y_end))

        return img

    def keypoints_builder(self, kp_inf, aug_para):
        if not aug_para:

            return torch.from_numpy(kp_inf.copy()).float()
        kp_inf = utils.kp_horizontal_flip(kp_inf, aug_para[2], aug_para[3])
        kp_inf = utils.kp_rotate(kp_inf, aug_para)
        kp_inf = utils.kp_normal(kp_inf)
        return torch.from_numpy(kp_inf.copy()).float()

    def frames_builder(self, frame_inf, video_name, aug_para, stream="rgb"):
        crop_threshold = self.cfg.DATA.CROP_THRESHOLD
        frames = []
        mean = self.cfg.DATA.MEAN if stream == "rgb" else self.cfg.DATA.MEAN_FLOW
        std = self.cfg.DATA.STD if stream == "rgb" else self.cfg.DATA.STD_FLOW
        for img_index in frame_inf.index:
            line = frame_inf.loc[img_index]
            path = os.path.join(video_name, img_index)
            img = Image.open(path)
            img_w, img_h = img.size

            crop_size = np.ceil(
                max(line[["w", "h"]])/10
            )*10
            bound = max(self.img_size)
            if crop_size <= bound:
                crop_size = bound
            crop_x = np.clip(
                        line["x"] - (crop_size-line["w"]) /
                        2, 0, img_w - crop_size
                    )
            crop_y = np.clip(
                line["y"] - (crop_size-line["h"]) /
                2, 0, img_h - crop_size
            )
            # img = img.crop(
            #     (crop_x, crop_y, crop_x+crop_size, crop_y+crop_size)
            # )
            # img = self.new_crop(img, crop_x, crop_y, crop_size, 0.85)
            '''
            for crop_size in crop_threshold:
                if (line[["w", "h"]] < crop_size).all() and img_w != crop_size:
                    crop_x = np.clip(
                        line["x"] - (crop_size-line["w"]) /
                        2, 0, img_w - crop_size
                    )
                    crop_y = np.clip(
                        line["y"] - (crop_size-line["h"]) /
                        2, 0, img_h - crop_size
                    )
                    img = img.crop(
                        (crop_x, crop_y, crop_x+crop_size, crop_y+crop_size)
                    )
                    break
            '''
            img = img.resize(self.img_size, Image.LANCZOS)  #note: changes from ANTIALIAS to LANCZOS because ANTIALIAS is deprecated
            frames.append(img)

        frames = torch.as_tensor(np.stack(frames))
        frames = utils.tensor_normalize(frames, mean, std)
        frames = frames.permute(0, 3, 1, 2)

        if not aug_para:
            return frames
        else:
            return utils.frame_augmentation(frames, aug_para)


class Dog_pain(torch.utils.data.Dataset):
    def __init__(self, cfg, mode, num_retries=10):
        assert mode in [
            "train",
            "val",
            "test",
        ], "Split '{}' not supported for Kinetics".format(mode)
        self.mode = mode
        self.cfg = cfg

        self._video_meta = {}
        self._num_retries = num_retries
        # For training or validation mode, one single clip is sampled from every
        # video. For testing, NUM_ENSEMBLE_VIEWS clips are sampled from every
        # video. For every clip, NUM_SPATIAL_CROPS is cropped spatially from
        # the frames.
        '''
        if self.mode in ["train", "val"]:
            self._num_clips = 1
        elif self.mode in ["test"]:
            self._num_clips = (
                cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS
            )
        '''
        logger.info("Constructing Kinetics {}...".format(mode))
        self._construct_loader()
        self.aug = False
        self.rand_erase = False
        self.use_temporal_gradient = False
        self.temporal_gradient_rate = 0.0
        self.clip_len = self.cfg.DATA.NUM_FRAMES
        if self.mode == "train" and self.cfg.AUG.ENABLE:
            self.aug = True
            if self.cfg.AUG.RE_PROB > 0:
                self.rand_erase = True
    
    def _construct_loader(self):
        data_path = self.cfg.DATA.PATH_TO_DATA_DIR
        assert os.path.isdir(data_path), f"{data_path} not found"
        ann_path = os.path.join(data_path, "annotation")
        self.crop_img = os.path.join(data_path, "crop_img")
        self.video_file = os.path.join(ann_path, "kp_valid")
        self.label_file = os.path.join(ann_path, "split")
        
        self.data_dict = {}
        self.bbox_col = ["x", "y", "w", "h"]
        self.clip_set = self.label_prepare()
    
    def label_prepare(self):
        file = os.path.join(self.label_file, self.mode+".h5")
        step = self.clip_len if self.mode == "test" else None
        container = []

        df = pd.read_hdf(file, "df_with_missing")
        df = df.applymap(lambda item: int(item))
        
        for i in range(len(df)):
            line = df.iloc[i]
            clip_set = self.clip_extract(line, step)
            container.extend(clip_set)
        return container
    
    def clip_extract(self, line, step=None):
        if not step:
            step = 1 if line["length"] <= 2*self.clip_len else 2
        return [[line.name, i, i+self.clip_len, line["label"]] for i in range(line["starts"], line["ends"]-self.clip_len, step)]
        
    def __len__(self):
        return len(self.clip_set)
    
    def __getitem__(self, index):

        short_cycle_idx = None
        # When short cycle is used, input index is a tupple.
        if isinstance(index, tuple):
            index, short_cycle_idx = index

        if self.mode in ["train", "val"]:
            # -1 indicates random sampling.
            
            spatial_sample_index = -1
            min_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[0]
            max_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[1]
            crop_size = self.cfg.DATA.TRAIN_CROP_SIZE
            if short_cycle_idx in [0, 1]:
                crop_size = int(
                    round(
                        self.cfg.MULTIGRID.SHORT_CYCLE_FACTORS[short_cycle_idx]
                        * self.cfg.MULTIGRID.DEFAULT_S
                    )
                )
        elif self.mode in ["test"]:
            
            # spatial_sample_index is in [0, 1, 2]. Corresponding to left,
            # center, or right if width is larger than height, and top, middle,
            # or bottom if height is larger than width.
            spatial_sample_index = (
                (
                    self._spatial_temporal_idx[index]
                    % self.cfg.TEST.NUM_SPATIAL_CROPS
                )
                if self.cfg.TEST.NUM_SPATIAL_CROPS > 1
                else 1
            )
            min_scale, max_scale, crop_size = (
                [self.cfg.DATA.TEST_CROP_SIZE] * 3
                if self.cfg.TEST.NUM_SPATIAL_CROPS > 1
                else [self.cfg.DATA.TRAIN_JITTER_SCALES[0]] * 2
                + [self.cfg.DATA.TEST_CROP_SIZE]
            )
            # The testing is deterministic and no jitter should be performed.
            # min_scale, max_scale, and crop_size are expect to be the same.
            assert len({min_scale, max_scale}) == 1
        else:
            raise NotImplementedError(
                "Does not support {} mode".format(self.mode)
            )
        
        clip = self.clip_set[index]
        file_name = clip[0]
        if file_name in self.data_dict:
            video_df = self.data_dict[file_name]

        else:
            video_df = pd.read_hdf(
                os.path.join(self.video_file, file_name), "df_with_missing"
            )
            self.data_dict[file_name] = video_df

        video_name = os.path.join(self.crop_img, file_name.split(".")[0])
        clip_inf = video_df.iloc[clip[1]:clip[2]]
        frames = self.frames_builder(
                clip_inf[self.bbox_col], video_name
            )
        frames = utils.tensor_normalize(
                    frames, self.cfg.DATA.MEAN, self.cfg.DATA.STD
                )
                # T H W C -> C T H W.
        frames = frames.permute(3, 0, 1, 2)
        if self.aug:
            frames = utils.spatial_sampling(
                    frames,
                    spatial_idx=spatial_sample_index,
                    min_scale=min_scale,
                    max_scale=max_scale,
                    crop_size=crop_size,
                    random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP,
                    inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE,
                )
        label = int(clip[-2]) #-1 before ethogram
        frames = utils.pack_pathway_output(self.cfg, frames)
        return frames, label, index, {}

    def frames_builder(self, frame_inf, video_name):
        crop_threshold = self.cfg.DATA.CROP_THRESHOLD
        frames = []
        
        for img_index in frame_inf.index:
            line = frame_inf.loc[img_index]
            path = os.path.join(video_name, img_index)
            img = Image.open(path)
            img_w, img_h = img.size

            for crop_size in crop_threshold:
                if (line[["w", "h"]] < crop_size).all() and img_w != crop_size:
                    crop_x = np.clip(
                        line["x"] - (crop_size-line["w"]) /
                        2, 0, img_w - crop_size
                    )
                    crop_y = np.clip(
                        line["y"] - (crop_size-line["h"]) /
                        2, 0, img_h - crop_size
                    )
                    img = img.crop(
                        (crop_x, crop_y, crop_x+crop_size, crop_y+crop_size)
                    )
                    break
            img = img.resize(self.img_size, Image.ANTIALIAS)
            frames.append(img)

        frames = torch.as_tensor(np.stack(frames))
        return frames

        
        
       
