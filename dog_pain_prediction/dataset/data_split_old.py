import os
import json 
import re
import itertools
from matplotlib.font_manager import json_load
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

class Dataset_builder:
    def __init__(
        self,
        hdf_file,
        crop_file,
        save_file,
        video_file, #TODO: Do I need this? Otherwise I need to ask Yasemin
        split_file="",
        num_kp=17,
        
    ):
        print(f"dir: {hdf_file}")
        assert os.path.isdir(hdf_file)
        self.hdf = [item for item in os.listdir(
            hdf_file) if item.endswith(".h5")]
        self.dir = hdf_file
        self.out = save_file
        self.crop_file = crop_file
        
        self.split_file = split_file
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
    
    def extract_number(self, filename):
        match = re.search(r'_(\d+).jpg$', filename)
        print(f"match: {int(match.group(1))}")
        if match:
            return int(match.group(1))
        else:
            return None
    
    def pipeline(self, gap_len=9, clip_len=5):
        """


            Parameters:
                gap_len:
                clip_len:
        """
        split_set = self.train_test_split()
        split_df_set = []
        print(f"split_set: {split_set}")

        # for all keypoint files in split_set
        for i, hdfs in enumerate(split_set):
            df_temp = []
            is_pain = True if i % 2 == 0 else False
            for file in hdfs:
                print(f"currently looking at: {file}")
                test_hdf = pd.read_hdf(
                    os.path.join(self.dir, file), "df_with_missing"
                )
                print(f"test hdf: {test_hdf}")
                numeric_parts = test_hdf.index.map(lambda filename: self.extract_number(filename))
                test_hdf['numeric_part'] = numeric_parts
                test_hdf_sorted = test_hdf.sort_values(by='numeric_part').drop(columns='numeric_part')
                label, data = self.range_select(
                    test_hdf_sorted, file, is_pain, gap_len, clip_len)
                
                #print(f"label: {label}")
                #print(f"data: {data}")
                df_temp.append(label)
                self.save_data(data, file)
            split_df = pd.concat(df_temp, axis=0)
            split_df_set.append(split_df)
            #print(f"split_df_set: {split_df_set}")

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
                    print(f"test hdf: {test_hdf}")
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

    def train_test_split(self, rate=0.2):
        """
            This function makes a regular train/test split without kfold

            Parameters:
                rate: the number of test instances on a scale of 1

            Returns:
                A list with a training set of dogs with and without pain, and a test set of dogs with and without pain
        """
        painset= [dog for dog in self.painset]
        notpainset = [dog for dog in self.notpainset]
        pain = list(set(self.hdf) & set(painset))
        notpain = list(set(self.hdf) & set(notpainset))

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
                split_test: Whether to create separate test set or only  train/validation set

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

        for train_index, test_index in kf.split(pain):
            train_pain.append(pain[train_index])
            test_pain.append(pain[test_index])
        for train_index, test_index in kf.split(notpain):
            train_not.append(notpain[train_index])
            test_not.append(notpain[test_index])
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
        
        train_index, test_index = next(kf.split(not_pain))
        notpain_test = not_pain[test_index]
        notpain = not_pain[train_index]  
        #est_array = np.concatenate(pain_test, notpain_test, axis=0)
       
        out_dict = {"test_pain": pain_test.tolist(), "test_not":notpain_test.tolist()}
        file_path = os.path.join(self.split_file, "video_split.json")
        with open(file_path, "w") as f:
            json.dump(out_dict, f)
        out = [out_dict.values()]
        
        return pain, notpain, out
    
    def split_record(self, out):
        """
            Records how the data is split

            Parameters:
                out: a list of quadruples (train with pain, train without pain, test with pain, test without pain

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
        print(f"yes were in range select for {name}")
        order_list = (pd.Series(test_hdf.index)).apply(
                lambda item: int(item.split("_")[-1].split(".")[0]) if item.split("_")[-1].strip("0") != '' else 0)
        print(f"order list values: {order_list.values}")
        gap = np.append(order_list.values[1:], 0)-order_list.values
        gap[-1] = -gap[-1]
        end_point = np.arange(len(gap))[gap > gap_len]
        start_point = np.append([0], end_point[:-1])
        ends = end_point[(end_point-start_point) > clip_len] + 1
        starts = start_point[(end_point-start_point) > clip_len] + 1

        # print(f"orderlist: {order_list}")
        # print(f"gap: {gap}")
        # print(f"gap[-1]: {gap[-1]}")
        # print(f"end_point: {end_point}")
        # print(f"start_point: {start_point}")
        # print(f"ends: {ends}")
        # print(f"starts: {starts}")
        # print(f"test_hdf index: {test_hdf.index}")

        if not starts.any():
            return pd.DataFrame(), pd.DataFrame()

        starts[0] = 0 if starts[0] == 1 else starts[0]
        length = ends-starts

        clip_index = np.concatenate([list(range(s, e))
                                    for s, e in zip(starts, ends)])
        index_list = test_hdf.index[clip_index]
        crop_inf = pd.read_hdf(os.path.join(self.crop_file, name))
        print(f"clip_index: {clip_index}")
        print(f"index_list: {index_list}")
        print(f"crop_inf: {crop_inf}")
        print(f"self.img_inf_col: {self.img_inf_col}")
        valid_indices = [idx for idx in index_list if idx in crop_inf.index]
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
    hdf_file = r"C:\Thesis\dog_pain_lisanne\raw_frames\not_pain\keypoints_fix_side" #this should be the keypoints fixed from kp_process.py
    save_file = r"D:\Uni\Thesis\data\splits\side_kp" 
    crop_file = r"C:\Thesis\dog_pain_lisanne\raw_frames\not_pain\crop_df_side"
    split_file = r"D:\Uni\Thesis\data\splits\side_split"
    video_file = r"D:\Uni\Thesis\data_hongyi_og\videos"

    # hdf_file = r"D:\Uni\Thesis\data_hongyi_og\annotation\fixed_kp" 
    # save_file = r"D:\Uni\Thesis\data_hongyi_og\annotation\kp_debug" 
    # crop_file = r"D:\Uni\Thesis\data_hongyi_og\annotation\crop"
    # split_file = r"D:\Uni\Thesis\data_hongyi_og\annotation\debug_split"
    # video_file = r"D:\Uni\Thesis\data_hongyi_og\videos"
    data_model = Dataset_builder(hdf_file, crop_file=crop_file, save_file=save_file, video_file=video_file, split_file=split_file)
    data_model.run(gap_len=90, clip_len=10, default_split=None, split_test=True)
    '''
    hdf_file: key point file
    save_file: The folder where the data of synthesized key points and bbox is stored. 
                If you simply use different divisions, fill in None. Use clips of different lengths to regenerate.
    split_file: The label folder is stored after the data set is split.
    default_split: existing split results
    '''