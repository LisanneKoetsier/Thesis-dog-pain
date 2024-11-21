import pandas as pd
import numpy as np
import itertools
import json
import os
from tqdm import tqdm
from collections import defaultdict
from sklearn.model_selection import KFold

head_set = [
    "L_Eye",
    "R_Eye",
    "L_EarBase",
    "R_EarBase",
    "Nose",
    "Throat",
]
back_set = [
    "TailBase",
    "Withers",
]
leg_set = [
    "L_F_Elbow",
    "L_B_Elbow",
    "R_B_Elbow",
    "R_F_Elbow",
    "L_F_Knee",
    "L_B_Knee",
    "R_B_Knee",
    "R_F_Knee",
    "L_F_Paw",
    "L_B_Paw",
    "R_B_Paw",
    "R_F_Paw",
]

kp_class = (head_set, back_set, leg_set)


def _isArrayLike(obj):

    return not type(obj) == str and hasattr(obj, '__iter__') and hasattr(obj, '__len__')

class keypoints_fix:
    def __init__(self, hdf_file, crop_file, save_file=None, num_kp=17, need_normal=True):
        if _isArrayLike(hdf_file):
            self.hdf = hdf_file
        elif os.path.isdir(hdf_file):
            self.hdf = [os.path.join(hdf_file, item) for item in os.listdir(
                hdf_file) if item.endswith(".h5")]
        else:
            self.hdf = [hdf_file]

        if _isArrayLike(crop_file):
            self.crop = crop_file
        elif os.path.isdir(crop_file):
            self.crop = [os.path.join(crop_file, item) for item in os.listdir(
                crop_file) if item.endswith(".h5")]
        else:
            self.crop = [crop_file]

        print(f"self.crop: {self.crop}")
        print(f"size self.crop: {len(self.crop)}")
        print(f"self.hdf: {self.hdf}")
        print(f"size self.hdf: {len(self.hdf)}")
        self.out = save_file
        self.head_set, self.back_set, self.leg_set = kp_class
        self.four_legs = [self.leg_set[i::4] for i in range(4)] #gets first 4 keypoints to check if there are four legs
        self.save_file = save_file
        self.need_normal = need_normal #Indicates if it needs normalization
        if need_normal:
            self.new_index = pd.MultiIndex.from_product(
                [
                    ["head1", "head2", "head3"] + self.back_set + self.leg_set,
                    ["x", "y"],
                ],
                names=["bodypart", "coord"]
            )
        else:
            self.new_index = [f"kp{i}" for i in range(2*num_kp)] 

    def pipeline(self, pre_filter=False, threshold=0.3, frame_gap=3):
        """
            The pipeline function for processing the keypoints

            Parameters:
                pre_filter:
                threshold:
                frame_gap: 
        """
        for index, file in tqdm(enumerate(self.hdf)):
            print(f"currently processing: {file}")
            file_name = file.split("\\")[-1] #of the form dog_name.h5 probably
            crop_file = self.crop[index]
            if self.save_file:
                out_path = os.path.join(self.save_file, file_name) #where the editted keypoints will be saved
                if os.path.isfile(out_path):
                    continue
                
            kp_df = pd.read_hdf(file, "df_with_missing")
            crop_df = pd.read_hdf(crop_file, "df_with_missing")

            #filter keypoints on useless keypoints
            kp_dfp = self.df_filter(kp_df, pre_filter, threshold, frame_gap)
            out_container = []
            index = kp_dfp.index.values
            line_process_func = self.line_process if self.need_normal else self.line_process_off

            #fill missing keypoints if possible
            for i in range(len(kp_dfp)):
                out_array = self.kp_fill(kp_dfp.iloc[i], line_process_func)
                if out_array.any():
                    out_container.append(out_array)
                else:
                    index = np.delete(index, i)

            # print("out container", len(out_container))
            # print("out container 0", out_container[0])
            # print("out container 0", len(out_container[0]))
            # print("self.new_index", self.new_index)
            # print("self.new_index len", len(self.new_index))

            #these are the processed keypoints
            out_df = pd.DataFrame(
                out_container, columns=self.new_index, index=index)
            
            #this function processes the keypoints to the correct format, including the crop values
            # valid_out_df = self.convert_to_kp_valid(out_df, crop_df)

            if self.save_file:
                self.save_hdf(out_df, file_name)

    def save_hdf(self, out_df, file_name):
        """
            Save the hdf file to the corresponding directory

            Parameters:
                out_df:
                file_name:
        """
        hdf_path = os.path.join(self.save_file, file_name)
        csv_file = os.path.join(self.save_file, file_name.strip(".h5")+".csv")

        out_df.to_hdf(
            hdf_path, "df_with_missing", format="table", mode="w"
        )
        out_df.to_csv(csv_file)
    
    def convert_to_kp_valid(self, in_df, crop_df):
        in_df.columns = in_df.columns.droplevel(0)
        new_column_names = ['kp0', 'kp1', 'kp2', 'kp3', 'kp4', 'kp5', 'kp6', 'kp7', 'kp8', 'kp9', 'kp10', 'kp11', 'kp12', 'kp13',
                            'kp14', 'kp15', 'kp16', 'kp17', 'kp18', 'kp19', 'kp20', 'kp21', 'kp22', 'kp23', 'kp24', 'kp25', 'kp26',
                            'kp27', 'kp28', 'kp29', 'kp30', 'kp31', 'kp32', 'kp33']
        in_df.columns = new_column_names

        col_to_concat = crop_df[['x','y','w','h']]
        df_kp_fixed_combined = in_df.join(col_to_concat)
        
        return df_kp_fixed_combined

    def df_filter(self, kp_df, pre_filter, threshold, frame_gap):
        """
            Parameters:
                kp_df: the dataframe with the keypoints of a dog
                pre_filter: boolean that indicates whether a first filter needs to be applied before the actual filter
                threshold: threshold value that indicates
                frame_gap: a value that indicates the gap between frames

            Returns:
                kp_dfp: a filtered keypoint dataframe
        """
        #filter only the keypoints that are above a certain score
        mask = kp_df.xs("score", level=1, axis=1) > threshold
        kp_dfp = kp_df[mask]

        if pre_filter:
            # creates an order list, which indicates the order of the keypoints in terms of frames
            order_list = (pd.Series(kp_dfp.index)).apply(
                lambda item: int(item.split("_")[-1].split(".")[0]) if item.split("_")[-1].strip("0") != '' else 0)
            kp_dfp = self.first_filter(kp_dfp, order_list, frame_gap) #this function handles null values I think
        delet_set = []

        for i in range(len(kp_df)):
            mask_i = mask.iloc[i]
            # if there are less than 11 keypoints in total, or not two keypoints in total for the backset
            # or less than three keypoints for the head set, or less than 6 for the legs
            # delete the keypoint set
            if mask_i.sum() < 11 or mask_i[self.back_set].sum() != 2 or \
                    mask_i[self.head_set].sum() < 3 or \
                    mask_i[self.leg_set].sum() < 6:
                delet_set.append(i)
        kp_dfp = kp_dfp.drop(kp_df.index[delet_set])

        return kp_dfp

    def first_filter(self, first_f, order_list, frame_gap):
        # Complete adjacent frames first
        full_gap = 2*frame_gap
        for i in range(len(first_f)):
            null_index = np.where(first_f.iloc[i].xs("x", level=1).isnull())[0]
            clip_start = np.clip([i-frame_gap], 0, len(first_f)-full_gap)[0]
            frame_order = order_list[i]

            if null_index.any():
                for index in null_index:
                    item_deter = ~(
                        first_f.iloc[clip_start:clip_start+full_gap, 3*index].isnull()).values

                    frame_deter = (order_list[clip_start:clip_start+full_gap].values >= (frame_order-frame_gap))*(
                        order_list[clip_start:clip_start+full_gap].values <= (frame_order+frame_gap))
                    deter_list = np.where(item_deter * frame_deter)[0]

                    if deter_list.any():
                        deter_index = deter_list[len(deter_list) // 2]
                        first_f.iloc[i, 3*index] = first_f.iloc[
                            clip_start + deter_index, 3*index
                        ]

                        first_f.iloc[i, 3*index+1] = first_f.iloc[
                            clip_start + deter_index, 3*index+1
                        ]
        return first_f

    def kp_fill(self, example_line, line_process):
        # leg fillers

        """
            Parameters:
                example_line:
                line_process:
            
            Returns:
                ...
        """
        line_x = example_line.xs("x", level=1)
        line_y = example_line.xs("y", level=1)
        temp_set = defaultdict(list)
        for leg in self.four_legs:
            notnan = (~np.isnan(line_x[leg])).sum()
            temp_set[notnan].append(leg)


        #if there are 4 keypoints, then do the line_process functionality
        if len(temp_set[3]) == 4:
            return line_process((line_x, line_y))
        if temp_set[2]:
            for leg in temp_set[2]:
                line_x[leg], line_y[leg] = self.fix_two(
                    line_x, line_y, leg)

        case3 = temp_set[0]+temp_set[1]
        if case3:
            for leg in case3:
                leg_index = self.four_legs.index(leg)
                vertical_pare = 1-leg_index if leg_index < 1.5 else 5-leg_index
                hori_pare = 3-leg_index
                if temp_set[3]:
                    if self.four_legs[hori_pare] in temp_set[3]:
                        line_x[leg], line_y[leg] = self.hori_shift(
                            (line_x, line_y), self.four_legs[hori_pare])
                    elif self.four_legs[vertical_pare] in temp_set[3]+temp_set[2]:
                        line_x[leg], line_y[leg] = self.vertical_shift(
                            (line_x, line_y), leg, self.four_legs[vertical_pare], self.back_set)
                    else:
                        line_x[leg], line_y[leg] = self.vertical_shift(
                            (line_x, line_y), leg, temp_set[3][0], self.back_set)
                else:
                    if self.four_legs[hori_pare] in temp_set[2]:
                        line_x[leg], line_y[leg] = self.hori_shift(
                            (line_x, line_y), self.four_legs[hori_pare])
                    else:
                        line_x[leg], line_y[leg] = self.vertical_shift(
                            (line_x, line_y), leg, temp_set[2][0], self.back_set)

        return line_process((line_x, line_y))

    def line_process(self, linexy):
        """
            Calculates the coordinates of a line relative to the midpoint of the spine and normalizes this by divivding it
            by the spine length

            Parameters:
                linexy: a tuple of coordinates ((x,x), (y,y)) which can constuct a line
            
            Returns:
                out_contain: a list of coordinates from linexy relative to the midpoint of the spine and normalized by spine length
        """
        # Taking the midpoint of the spine as the coordinate origin, 
        # the relative coordinates are obtained. normalized divided by spine length
        out_contain = []
        linex, liney = linexy

        # calculate the length of the spine
        spine_len = np.sqrt(
            (linex[back_set][0]-linex[back_set][1])**2 +
            (liney[back_set][0]-liney[back_set][1])**2
        )

        # if there is no spine, return an array with an empty list
        if spine_len == 0:
            return np.array(out_contain)
        for line in linexy:
            # obtain coordinates relative to the midpoint of the spine and normalize by spine length
            # do this for each coordinate tuple in linexy
            root = (line[back_set][0]+line[back_set][1])/2
            line = line.apply(lambda item: item-root)
            line = (line[~np.isnan(line)].values)/spine_len
            out_contain.append(np.append(line[:3], line[-14:]))
        out_contain = np.array(out_contain)
        return out_contain.flatten(order="F")

    def line_process_off(self, linexy):
        """
            Does not process the relative coordinates and does not apply normalization.
            
            Parameters:
                linexy: a tuple of coordinates ((x,y), (x,y)) which can constuct a line
            
            Returns:
                out_contain: a list of coordinates from linexy, filtered on nan values 
        """
        # Does not process relative coordinates and standardization, 
        # only filters, records spine length, and returns one more column of data
        out_contain = []
        linex, liney = linexy

        # Calculate spine length
        spine_len = np.sqrt(
            (linex[back_set][0]-linex[back_set][1])**2 +
            (liney[back_set][0]-liney[back_set][1])**2
        )

        # If there is no spine, return an array with an empty list
        if spine_len == 0:
            return np.array(out_contain)
        for line in linexy:

            line = line[~np.isnan(line)].values
            out_contain.append(np.append(line[:3], line[-14:])) #TODO what do the :3 and -14: numbers represent?
            #out_contain.append(line)
        out_contain = np.array(out_contain).flatten(order="F")
        return out_contain

    def fix_two(self, line_x, line_y, leg):
        """
            If there are no NaN values:
            If there is 1 NaN value:
            If there are >1 NaN values:

            Parameters:
                line_x:
                line_y: 
                leg: a coordinate of an elbow keypoint from the leg_set (first 4 instances of leg_set)

            Returns:
                leg_x, leg_y: x and y coordinates of the leg respectively
        """
        # Missing one point to complete #TODO: figure out which point he means
        leg_x, leg_y, backx, backy = line_x[leg], line_y[leg], line_x[self.back_set], line_y[self.back_set]
        nan_index = np.where(np.isnan(leg_x))[0][0]
        xb, yb = (backx[0], backy[0]) if leg[0].split(
            "_")[1] == "B" else (backx[1], backy[1])
        if nan_index == 0:
            leg_x[0] = (leg_x[1] + xb)/2
            leg_y[0] = (leg_y[1]+yb)/2

        elif nan_index == 1:
            leg_x[1] = (leg_x[0] + leg_x[2])/2
            leg_y[1] = (leg_y[0] + leg_y[2])/2

        else:
            leg_x[2] = 2*leg_x[1]-leg_x[0]
            leg_y[2] = 2*leg_y[1]-leg_y[0]
        return leg_x, leg_y

    def hori_shift(self, xy, leg_shift):
        # Horizontal occlusion completion
        linex, liney = xy

        return ((linex[leg_shift]+np.random.randint(10, 20)).to_list(),
                (liney[leg_shift]+np.random.randint(10, 20)).to_list())

    def vertical_shift(self, xy, leg, leg_shift, back_set):
        # Vertical occlusion completion
        linex, liney = xy

        F = True if leg[0].split("_")[1] == "F" else False
        if F:
            shift_vector = (linex[back_set][1]-linex[back_set]
                            [0], liney[back_set][1]-liney[back_set][0])
            return ((linex[leg_shift]+shift_vector[0]*0.75).to_list(),
                    (liney[leg_shift]+shift_vector[1]*0.75).to_list())
        else:
            shift_vector = (linex[back_set][0]-linex[back_set]
                            [1], liney[back_set][0]-liney[back_set][1])
            return ((linex[leg_shift]+shift_vector[0]*0.75).to_list(),
                    (liney[leg_shift]+shift_vector[1]*0.75).to_list())


if __name__ == "__main__":
    
    hdf_file = r"C:\Thesis\dog_pain_lisanne\raw_frames\not_pain\pose_files_side"
    crop_file = r"C:\Thesis\dog_pain_lisanne\raw_frames\not_pain\crop_df_side"
    save_file = r"C:\Thesis\dog_pain_lisanne\raw_frames\not_pain\keypoints_fix_side"
    data_model = keypoints_fix(hdf_file, crop_file, save_file, num_kp=20)
    data_model.pipeline(pre_filter=True)

    #this file output the fixed kp
    #the kp valid is processed in the data_split file
    '''
    hdf_file = r"D:\pose\pain\data\pain_data\annotation\fixed_kp_raw"
    save_file = r"D:\pose\pain\data\pain_data\annotation\kp_valid_raw"
    data_model = Dataset_builder(hdf_file, save_file = save_file, split_file=r"D:\pose\pain\data\pain_data\annotation\split_nonormal")
    data_model.run(gap_len=9, clip_len=8,k_fold=5,default_split=r"D:\pose\pain\data\pain_data\annotation\split")
    '''