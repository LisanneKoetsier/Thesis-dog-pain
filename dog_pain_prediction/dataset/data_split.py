import os
import json
import itertools
from matplotlib.font_manager import json_load
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import KFold

class Dataset_builder:
    def __init__(
        self,
        hdf_file,
        save_file=None,
        video_file=r"D:\Uni\Thesis\data_hongyi_og\videos",
        annotation=r"C:\Thesis\dog_pain_lisanne\raw_frames\not_pain",
        split_file=r"D:\Uni\Thesis\data\splits\side_split",
        num_kp=17,
        
    ):
        assert os.path.isdir(hdf_file)
        self.hdf = [item for item in os.listdir(
            hdf_file) if item.endswith(".h5")]
        self.dir = hdf_file
        self.out = save_file
        self.crop_file = os.path.join(annotation, "crop_df_side")
        
        self.split_file = split_file

        #extract dog names from the video files
        self.painset = [item.split(".")[0]+".h5"
                        for item in os.listdir(video_file+"\\pain")]
        self.notpainset = [item.split(".")[0]+".h5"
                           for item in os.listdir(video_file+"\\not_pain")]
        
        self.img_inf_col = ["x", "y", "w", "h"]
        self.data_col = [f"kp{i}" for i in range(2*num_kp)]

    def run(self, gap_len=9, clip_len=5, k_fold=None, default_split= None, split_test=False):
        """
            Main function that runs all the stuf that has to be ran

            Parameters:
                gap_len:
                clip_len: 
                k_fold: indicates whether k_fold needs to be done
                default_split:
                split_test:
        """
        if k_fold is not None and isinstance(k_fold, int):
            self.k_fold = k_fold
            self.kfold_path = [os.path.join(
                self.split_file, f"split{i}") for i in range(self.k_fold)]
            for path in self.kfold_path:
                if not os.path.isdir(path):
                    os.mkdir(path)
            if default_split is not None:
                train_val, test_set = self.default_kfold(default_split, split_test) 
            else:
                train_val, test_set = self.train_test_split_kfold(split_test)
            if test_set:
                self.pipeline_pack(test_set, gap_len, clip_len, save_file=[self.split_file],process_test=True)
            self.pipeline_pack(train_val, gap_len, clip_len, save_file=self.kfold_path)
        else:
            self.pipeline(gap_len, clip_len)
    
    def default_kfold(self, default_file, split_test=False):
        out_test = []
        if split_test:
            js_path = os.path.join(default_file, "video_split.json")
            assert os.path.isfile(js_path), f"json file {js_path} not exist"
            with open(js_path, "r") as f:
                fold = json.load(f)
            out_test = [fold.values()]
        kfold_path = [os.path.join(
                default_file, f"split{i}") for i in range(self.k_fold)]
        out_set = []
        for dir in kfold_path:
            js_path = os.path.join(dir, "video_split.json")
            assert os.path.isfile(js_path), f"json file {js_path} not exist"
            with open(js_path, "r") as f:
                fold = json.load(f)
            out_set.append(fold.values())
        return out_set, out_test
    
    def pipeline(self, gap_len=9, clip_len=5):
        """
            Parameters:
                gap_len:
                clip_len:
        """
        split_set = self.train_test_split()
        split_df_set = []

        # for all keypoint files in split_set
        for i, hdfs in enumerate(split_set):
            df_temp = []
            is_pain = True if i % 2 == 0 else False
            for file in hdfs:
                test_hdf = pd.read_hdf(
                    os.path.join(self.dir, file), "df_with_missing"
                )
                label, data = self.range_select(
                    test_hdf, file, is_pain, gap_len, clip_len)
                df_temp.append(label)
                self.save_data(data, file)
            split_df = pd.concat(df_temp, axis=0)
            split_df_set.append(split_df)

        inf = zip(
            ("train", "val"), (split_df_set[:2], split_df_set[2:])
        )
        self.save_label(inf, self.split_file)

    def pipeline_pack(self, split_set_kfold, gap_len, clip_len,save_file, process_test=False):

        for k, fold in enumerate(split_set_kfold):
            split_df_set = []

            for i, hdfs in enumerate(fold):
                df_temp = []
                is_pain = True if i % 2 == 0 else False
                for file in hdfs:
                    test_hdf = pd.read_hdf(
                        os.path.join(self.dir, file), "df_with_missing"
                    )
                    label, data = self.range_select(
                        test_hdf, file, is_pain, gap_len, clip_len)
                    if label.empty:
                        continue
                    df_temp.append(label)
                    if k == 0 and self.out is not None:
                        self.save_data(data, file)
                split_df = pd.concat(df_temp, axis=0)
                split_df_set.append(split_df)
            if process_test:
                inf = [("test", split_df_set)]
            else:
                inf = zip(
                    ("train", "val"), (split_df_set[:2], split_df_set[2:])
                )
            self.save_label(inf, save_file[k])

    def save_label(self, inf, split_file):

        for name, df in inf:
            df = pd.concat(df, axis=0)
            df.to_hdf(
                os.path.join(split_file, name+".h5"), "df_with_missing", format="table", mode="w"
            )
            df.to_csv(os.path.join(split_file, name+".csv"))

    def save_data(self, video_data, name):

        video_data.to_hdf(
            os.path.join(self.out, name), "df_with_missing", format="table", mode="w"
        )
        video_data.to_csv(os.path.join(
            self.out, name.strip(".h5")+".csv"))
        
    def check_test_overlap(self, train_dogs, test_dogs):
        """
        this function checks whether there is overlap between the train and test set
            Parameters:
                train_dogs: a list of dogs in the train set
                test_dogs: a list of dogs in the test set
            
            Returns:
                new_train_dogs: a list of non-overlapping dogs for the train set
                new_test_dogs: a list of non-overlapping dogs for the test set
        """
        new_train_dogs = []
        new_test_dogs = []
        swap_dog_names = [] #dog names that need to be swapped with another dog from the train set
        front_side_occurences = {}

        for dog_video in test_dogs:
            split_string = dog_video.split("_")
            #dog_name = split_string[0].lower()
            lower_case_video = dog_video.lower()
            if 'front' in lower_case_video:
                front_side_occurences[dog_video] = front_side_occurences.get(dog_video,0) + 1
            elif 'side' in lower_case_video:
                front_side_occurences[dog_video] = front_side_occurences.get(dog_video,0) + 1

        # print("front side", front_side_occurences)
        for dog_video, count in front_side_occurences.items():
            if count == 1:
                swap_dog_names.append(dog_video)

        # print("test dogs before swapping", test_dogs)
        # print("test swap", swap_dog_names)

        candidate_test_dogs = [dog_name for dog_name in train_dogs if 'side' not in dog_name.lower() 
                               and 'front' not in dog_name.lower()]
        
        # print("train dogs before adding candidates", train_dogs)
        # print("candidates", candidate_test_dogs)

        for dog in swap_dog_names:
            #randomly choose a new test dog from the candidates
            swapped_dog = random.choice(candidate_test_dogs)
            # print(f"candidate dog for {dog} is {swapped_dog}")
            # delete dog that will be swapped to test set from train set, 
            # and dog that will be swapped from test to train
            train_dogs = np.delete(train_dogs, np.where(train_dogs == swapped_dog))
            test_dogs = np.delete(test_dogs, np.where(test_dogs == dog))
            #add new dog to test set, add new dog to train set
            train_dogs = np.append(train_dogs, dog)
            test_dogs = np.append(test_dogs, swapped_dog)

        new_train_dogs = train_dogs
        new_test_dogs = test_dogs

        # print("new test dogs", new_test_dogs)
        # print("new train dogs", new_train_dogs)

        return new_train_dogs, new_test_dogs

    def train_test_split(self, rate=0.2):
        """
            This function makes a regular train/test split without kfold

            Parameters:
                rate: the number of test instances on a scale of 1

            Returns:
                A list with a training set of dogs with and without pain, and a test set of dogs with and without pain
        """
        pain = list(set(self.hdf) & set(self.painset))
        notpain = list(set(self.hdf) & set(self.notpainset))
        test_pain = np.random.choice(pain, round(
            len(pain)*rate), replace=False).tolist()
        train_pain = list(set(pain) - set(test_pain))
        test_not = np.random.choice(notpain, round(
            len(notpain)*rate), replace=False).tolist()
        train_not = list(set(notpain) - set(test_not))

        return [train_pain, train_not, test_pain, test_not]

    def train_test_split_kfold(self, split_test):
        """
            This function makes a train/test split with kfold

            Parameters:
                split_test: ??? #TODO find out why this exists

            Returns:
                out: a list of quadruples (train with pain, train without pain, test with pain, test without pain)
                out_test: an array with the test values of dogs with and without pain
                
        """
        pain = np.array(list(set(self.hdf) & set(self.painset)))
        notpain = np.array(list(set(self.hdf) & set(self.notpainset)))
        out_test = []
        if split_test:
            pain, notpain, out_test = self.test_split(pain, notpain)
        kf = KFold(self.k_fold, shuffle=True, random_state=1)
        train_pain, test_pain = [], []
        train_not, test_not = [], []

        # print("out test", out_test)
        # print("pain", pain)
        # print("notpain", notpain)
        for train_index, test_index in kf.split(pain):
            train_pain_dogs, test_pain_dogs = self.check_test_overlap(pain[train_index], pain[test_index])
            train_pain.append(train_pain_dogs)
            test_pain.append(test_pain_dogs)
        for train_index, test_index in kf.split(notpain):
            train_not_dogs, test_not_dogs = self.check_test_overlap(notpain[train_index], notpain[test_index])
            train_not.append(train_not_dogs)
            test_not.append(test_not_dogs)
        out = list(zip(train_pain, train_not, test_pain, test_not))

        self.split_record(out)

        return out, out_test
    
    def test_split(self, painset, not_pain):
        """
            This function makes a test split based on the next kfold

            Parameters:
                painset: an array with the instances of dogs with pain
                not_pain: an array with the instances of dogs without pain

            Returns:
                pain: the train instances of dogs with pain
                notpain: the train instances of dogs without pain
                out: an array with the test values of dogs with and without pain
                
        """
        kf = KFold(self.k_fold+1, shuffle=True, random_state=1)
        train_index, test_index = next(kf.split(painset))
        pain_test = painset[test_index]
        pain = painset[train_index]
        new_pain, new_pain_test = self.check_test_overlap(pain, pain_test)

        # print("old pain train", pain)
        # print("old pain test", pain_test)
        # print("new pain train", new_pain)
        # print("new pain test", new_pain_test)
        
        train_index, test_index = next(kf.split(not_pain))
        notpain_test = not_pain[test_index]
        notpain = not_pain[train_index]
        new_notpain, new_notpain_test = self.check_test_overlap(notpain, notpain_test)  
        #est_array = np.concatenate(pain_test, notpain_test, axis=0)
       
        out_dict = {"test_pain": new_pain_test.tolist(), "test_not":new_notpain_test.tolist()}
        file_path = os.path.join(self.split_file, "video_split.json")
        with open(file_path, "w") as f:
            json.dump(out_dict, f)
        out = [out_dict.values()]
        
        return new_pain, new_notpain, out
    
    def split_record(self, out):
        """
            Records how the data is split

            Parameters:
                out: a list of quadruples (train with pain, train without pain, test with pain, test without pain)

        """
        keys = ["train_pain", "train_not", "val_pain", "val_not"]
        for i, fold in enumerate(out):

            fold_dict = {keys[j]: fold[j].tolist() for j in range(4)}
            with open(os.path.join(self.kfold_path[i], "video_split.json"), "w") as f:
                json.dump(fold_dict, f)

    def range_select(self, test_hdf, name, is_pain, gap_len=9, clip_len=5):
        """
            Parameters:
                test_hdf: keypoints of the test instances
                name: 
                is_pain: boolean that indicates whether the clip 
                gap_len:
                clip_len: 

            Returns:
                clip_df:
                dataframe: 
        """
        order_list = (pd.Series(test_hdf.index)).apply(
                lambda item: int(item.split("_")[-1].split(".")[0]) if item.split("_")[-1].strip("0") != '' else 0)
        gap = np.append(order_list.values[1:], 0)-order_list.values
        gap[-1] = -gap[-1]
        end_point = np.arange(len(gap))[gap > gap_len]
        start_point = np.append([0], end_point[:-1])
        ends = end_point[(end_point-start_point) > clip_len] + 1
        starts = start_point[(end_point-start_point) > clip_len] + 1
        if not starts.any():
            return pd.DataFrame(), pd.DataFrame()

        starts[0] = 0 if starts[0] == 1 else starts[0]
        length = ends-starts

        clip_index = np.concatenate([list(range(s, e))
                                    for s, e in zip(starts, ends)])
        index_list = test_hdf.index[clip_index]
        crop_inf = pd.read_hdf(os.path.join(self.crop_file, name))
        crop_inf = (crop_inf.loc[index_list])[self.img_inf_col]

        clip_df = test_hdf.iloc[clip_index]
        clip_df.columns = self.data_col
        clip_df = pd.concat([clip_df, crop_inf], axis=1)
        end_new = list(itertools.accumulate(length))
        start_new = [0] + end_new[:-1]
        index = [name]*len(start_new)
        label = [1]*len(start_new) if is_pain else [0]*len(start_new)

        return pd.DataFrame({"starts": start_new, "ends": end_new, "length": length, "label": label}, index=index), clip_df

if __name__ == "__main__":
    hdf_file = r"C:\Thesis\dog_pain_lisanne\raw_frames\not_pain\keypoints_fix_combined"  #r"D:\Uni\Thesis\data_hongyi_og\annotation\fixed_kp" 
    save_file = r"D:\Uni\Thesis\data\splits\side_kp" 
    data_model = Dataset_builder(hdf_file, save_file = save_file, split_file=r"D:\Uni\Thesis\data\splits\side_split")
    data_model.run(gap_len=9, clip_len=12,k_fold=None,default_split=None, split_test=True)

    # hdf_file = r"D:\Uni\Thesis\data_hongyi_og\annotation\fixed_kp"  #r"D:\Uni\Thesis\data_hongyi_og\annotation\fixed_kp" 
    # save_file = r"D:\Uni\Thesis\data_hongyi_og\annotation\no_side_kp" 
    # data_model = Dataset_builder(hdf_file, save_file = save_file, split_file=r"D:\Uni\Thesis\data_hongyi_og\annotation\no_side_split")
    # data_model.run(gap_len=9, clip_len=12,k_fold=None,default_split=None, split_test=True)
    '''
    hdf_file: key point file
    save_file: The folder where the data of synthesized key points and bbox is stored. 
                If you simply use different divisions, fill in None. Use clips of different lengths to regenerate.
    split_file: The label folder is stored after the data set is split.
    default_split: existing split results
    '''