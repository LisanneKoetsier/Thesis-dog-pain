import torch
import cv2
import os
from PIL import Image
import pandas as pd
import numpy as np
import subprocess
from tqdm import tqdm
from ultralytics import YOLO

def _isArrayLike(obj):

    return not type(obj) == str and hasattr(obj, '__iter__') and hasattr(obj, '__len__')


class Databuilder:
    def __init__(self, img_file, save_root, width, height, yolov8=False):
        self.img = img_file
        self.save_root = save_root
        self.yolov8 = yolov8
        self.crop_w = width/2
        self.crop_h = height/2
        self.setup()

    def setup(self):
        self.crop_save = os.path.join(self.save_root, "crop_df_missing")
        self.keypoints_save = os.path.join(
            self.save_root, "annotation\\keypoints")
        self.img_save = os.path.join(self.save_root, "crop_img_missing")
        self.keypoints_vis = os.path.join(self.save_root, "keypoint_img")

        # load yolov5 here for bounding box detection in the video frames
        if self.yolov8:
            #model = torch.hub.load('ultralytics/yolov5', 'yolov5x')
            model = YOLO("yolov8n.pt")
            model.conf = 0.01  
            model.iou = 0.45 
            model.classes = [16] #16 is the dog class, 0 is the person class
            model.multi_label = False
            self.model = model
            print("yes yolov8")
        else:
            model = torch.hub.load('ultralytics/yolov5', 'yolov5x')
            model.conf = 0.01  
            model.iou = 0.45 
            model.classes = [16] #16 is the dog class, 0 is the person class
            model.multi_label = False
            self.model = model
            print("yes yolov5")

    def bbox_crop(self, img_dir):
        """
        crop image according to bounding box
        """
        img_list = [os.path.join(img_dir, img) for img in os.listdir(
            img_dir) if img.endswith(".jpg")]
        dog_name = img_dir.split("\\")[-1]

        bbox = pd.DataFrame(
            columns=["xmin", "ymin", "xmax", "ymax", "confidence", "class", "name"])
        empty_name = [] #list with all instances that have an empty bounding box

        print(f"Currently processing {dog_name}")
        for i, img in enumerate(tqdm(img_list)):
            
            if self.yolov8:
                im = Image.open(img)
                result = self.model.predict(source=im, classes=16)
                cls_id = result[0].boxes.cls
                
                #print("numel", cls_id.numel())

                if cls_id.numel() == 0:
                    #print("tensor empty!")
                    empty_name.append(i)
                else:
                    # Find all indices where class id equals 16: dog
                    dog_class = 16
                    dog_indices = [idx for idx, val in enumerate(cls_id) if val == dog_class]

                    # Find indices where anything else except a dog is detected
                    non_dog_indices = [idx for idx, val in enumerate(cls_id) if val != dog_class]

                    # Extract bounding box from dog objects
                    if dog_indices:
                        # print("yay we got a dog!")
                        # print(f"the dog indices are: {dog_indices}")
                        confidence = result[0].boxes.conf
                        dog_confidences = confidence[dog_indices]
                        dog_boxes = result[0].boxes.xyxy[dog_indices]
                        dog_cls_ids = cls_id[dog_indices]
                        max_conf_idx = dog_confidences.argmax()
                        max_cls_idx = int(dog_cls_ids[max_conf_idx])
                        cls_names = result[0].names
                        # print(f"class names: {cls_names}")
                        # print(f"confidences: {confidence}")
                        # print(f"max conf idx: {max_conf_idx}")
                        # print(f"max cls_id: {max_cls_idx}")
                        # print(f"max class: {cls_names[max_cls_idx]}")
                        cls = cls_names[max_cls_idx]
                        box_coordinates = dog_boxes[max_conf_idx]
                    
                    #if anything else than a dog is detected, append index to delete later
                    if non_dog_indices and not dog_indices:
                        # print("we got only non-dogs!")
                        empty_name.append(i)

                    # print(f"box coordinates before: {box_coordinates}")
                    box_coordinates = box_coordinates.cpu().numpy().tolist()
                    row_to_append = {
                        "xmin": box_coordinates[0],
                        "ymin": box_coordinates[1],
                        "xmax": box_coordinates[2],
                        "ymax": box_coordinates[3],
                        "confidence": confidence[max_conf_idx].item(),
                        "class": max_cls_idx,
                        "name": cls
                    }

                    row_df = pd.DataFrame([row_to_append], columns=bbox.columns)
                    bbox = pd.concat([bbox, row_df], ignore_index=True)
            else:
                results = self.model(img)
                df_bb = results.pandas().xyxy[0]
                if df_bb.empty:
                    empty_name.append(i)
                else:
                    if len(df_bb) == 1:
                        row_to_append = df_bb.iloc[0]
                        row_df = pd.DataFrame([row_to_append], columns=bbox.columns)
                        bbox = pd.concat([bbox, row_df], ignore_index=True)
                    else:
                        row_to_append = df_bb.loc[df_bb["confidence"].idxmax()]
                        row_df = pd.DataFrame([row_to_append], columns=bbox.columns)
                        bbox = pd.concat([bbox, row_df], ignore_index=True)


        order_list = np.arange(len(img_list))
        order_list = np.delete(order_list, empty_name)
        order_list += 1
        img_list = np.asarray(img_list)
        img_list = np.delete(img_list, empty_name)
        
        img_name = list(
            map(
                lambda item: 
                     item.split("\\")[-1], img_list
            )
        )
        bboxcrop = []

        im = Image.open(img_list[0])
        img_w, img_h = im.size
        bbox.index = img_name
        
        bbox.to_csv("test2.csv", index=False)
        bboxcrop = self.table_process(bbox, img_w, img_h)
        bboxcrop["order"] = order_list
        bboxcrop.to_csv("crop_data.csv", index=False)

        return bboxcrop

    def table_process(self, bbox, img_w, img_h):
        # crop coordinate
        bbox_coord = bbox.iloc[:, :4]
        bbox_coord["xcrop"] = ((bbox_coord["xmin"]+bbox_coord["xmax"])/2)-self.crop_w
        bbox_coord["ycrop"] = ((bbox_coord["ymin"]+bbox_coord["ymax"])/2)-self.crop_h
        
        bbox_coord["xcrop"] = np.clip(bbox_coord["xcrop"], 0, img_w-self.crop_w)
        bbox_coord["ycrop"] = np.clip(bbox_coord["ycrop"], 0, img_h-self.crop_h)

        print("bbox_coord xcrop", bbox_coord["xcrop"].values)
        print("bbox_coord ycrop", bbox_coord["ycrop"].values)

        print("imw_w", img_w)
        print("imw_h", img_h)
        print("self crop w", self.crop_w)
        print("self crop h", self.crop_h)
        print("xcrop clip max", img_w-self.crop_w)
        print("ycrop clip max", img_h-self.crop_h)

        out_dict = {
            "x": (bbox_coord["xmin"]-bbox_coord["xcrop"]).values,
            "y": (bbox_coord["ymin"]-bbox_coord["ycrop"]).values,
            "w": (bbox_coord["xmax"]-bbox_coord["xmin"]).values,
            "h": (bbox_coord["ymax"]-bbox_coord["ymin"]).values,
        }

        # bbox expanded by 10 % to ensure the dog is within the bounding box
        out_dict_extend = {
            "x": np.clip(out_dict["x"]-out_dict["w"]*0.1/2, 0, self.crop_w*2),
            "y": np.clip(out_dict["y"]-out_dict["h"]*0.1/2, 0, self.crop_h*2),
            "w": np.clip(out_dict["w"]*1.1, 0, self.crop_w*2),
            "h": np.clip(out_dict["h"]*1.1, 0, self.crop_h*2),
        }
        out_inf_extend = pd.DataFrame(out_dict_extend)
        out_inf_extend.index = bbox_coord.index
        # combine two tables 
        result = pd.concat([bbox_coord, out_inf_extend],
                           axis=1)
        result = result.apply(lambda item: round(item), axis=0)

        print("result", result)
        return result

    def save_ann(self, bboxcrop, name):
        """ 
            Saves files to designated folder

            Parameters:
                bboxcrop:
                name: name of the file
        """
        hdf_file = os.path.join(self.crop_save, name+".h5")
        csv_file = os.path.join(self.crop_save, name+".csv")
        if not os.path.isfile(hdf_file):
            bboxcrop.to_hdf(
                hdf_file, "df_with_missing", format="table", mode="w"
            )
            bboxcrop.to_csv(csv_file)
            
    # def crop_image(self, image, xmin, ymin, xmax, ymax):
    #     return image[int(ymin):int(ymax), int(xmin):int(xmax)]

    def crop_image(self, image, xmin, ymin, size):
        crop_img = image[int(ymin):int(ymin + size), int(xmin):int(xmin + size)]
        #resized_img = cv2.resize(crop_img, (750,750))
        return crop_img

    def save_img(self, crop_df, input_dir, out_dir):
        """
            This function crops the images and saves it to the corresponding output directory

            Parameters:
                crop_df: a dataframe with all the coordinates where the corresponding images needs to be cropped
                input_dir: the directory where the original input images are stored
                out_dir: the directory where the images should be saved
        """
        bbox_size = 100

        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
            
        for i, row in crop_df.iterrows():
            img_path = os.path.join(input_dir, row.name.split("\\")[-1])
            img = cv2.imread(img_path)
            enlarge_percentage = 0.40
            xmin, ymin, xmax, ymax = row["xmin"], row["ymin"], row["xmax"], row["ymax"]
            x, y, w, h = row["x"], row["y"], row["w"], row["h"]
            xcrop, ycrop = row["xcrop"], row["ycrop"]

            # Calculate the center of the bounding box
            center_x = (xmin + xmax) / 2
            center_y = (ymin + ymax) / 2

            width = xmax - xmin
            height = ymax - ymin
            xmin_new = max(0, xmin - width * enlarge_percentage)
            ymin_new = max(0, ymin - height * enlarge_percentage)
            xmax_new = min(img.shape[1], xmax + width * enlarge_percentage)
            ymax_new = min(img.shape[0], ymax + height * enlarge_percentage)
            
            # Ensure that the new bounding box maintains the specified size
            size = max(width, height, bbox_size)
            xmin_new = max(0, center_x - size / 2)
            ymin_new = max(0, center_y - size / 2)
            xmax_new = min(img.shape[1], center_x + size / 2)
            ymax_new = min(img.shape[0], center_y + size / 2)
            
            
            cropped_img = self.crop_image(img, xmin_new, ymin_new, size)
            cv2.imwrite(os.path.join(out_dir, row.name.split("\\")[-1]), cropped_img)

    def run_pose(self, hdf_file, crop_img, pose_vis):
        """
            idk what it does yet

            Parameters:
                hdf_file:
                crop_img:
                pose_vis: a boolean which 
        """

        if os.path.isdir(crop_img) and os.path.isfile(hdf_file):
            if pose_vis:
                name = crop_img.split("\\")[-1]
                pose_img = os.path.join(self.keypoints_vis, name)
                if not os.path.isdir(pose_img):
                    os.makedirs(pose_img)
                # command += ["--out-img-root", pose_img]
            command = [
                f"python demo/topdown_img_custom.py \
                demo/mmdetection_cfg/rtmdet_m_8xb32-300e_coco.py \
                https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_m_8xb32-300e_coco/rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth \
                configs/animal_2d_keypoint/topdown_heatmap/animalpose/td-hm_hrnet-w32_8xb64-210e_animalpose-256x256.py \
                https://download.openmmlab.com/mmpose/animal/hrnet/hrnet_w32_animalpose_256x256-1aa7f075_20210426.pth \
                --img-root {crop_img} \
                --output-df-root {self.keypoints_save} \
                --det-cat-id=16 \
                --out-img-root {pose_img}"
            ]
            print("command", command)
            subprocess.run(command)

    def pipeline(self, needcrop=True, pose=True, pose_vis=True):
        """
            The general pipeline for finding the bounding box of the dogs
            First it retrieves the current directory and the images
            For each image it retrieves the path and checks if it needs cropping, and crops if it does
            It also checks for each image if it needs to calculate and visualize the pose, and does this if it has to
            All the images and poses are saved to the corresponding directories

            Parameters:
                needcrop: a boolean that decides if the images still need cropping
                pose: a boolean that decides if the pose of the dog has to be calculated
                pose_vis: a boolean that decides if the pose of the dog has to be visualized #TODO: check if this is actually true lol
        """
        print("yes2")
        images = self.img if _isArrayLike(self.img) else [self.img] #put in a list if it's a single image
        cur_dir = os.getcwd()
        for img_dir in images:
            print(f"yes img_dir: {img_dir}")
            name = img_dir.split("\\")[-1] #for windows
            #name = img_dir.split("/")[-1] #for ubuntu
            crop_img = os.path.join(self.img_save, name)
            if needcrop:
                if os.path.isdir(crop_img):
                    continue
                bboxcrop = self.bbox_crop(img_dir)
                
                self.save_ann(bboxcrop, name)
                self.save_img(bboxcrop, img_dir, crop_img)
            print("done with crop")
            if pose:
                print("pose yes!")
                hdf_file = os.path.join(self.crop_save, name+".h5")
                print("hdf file", hdf_file)
                pose_hdf = os.path.join(self.keypoints_save, name+".h5")
                print("hdf pose", pose_hdf)
                if os.path.isfile(pose_hdf):
                    continue
                if os.path.isfile(hdf_file):
                    mmpose_dir = r"C:\Thesis\dog_pain_lisanne\mmpos"
                    os.chdir(mmpose_dir)
                    self.run_pose(hdf_file, crop_img, pose_vis)
                    os.chdir(cur_dir)
            print("pose done?")

if __name__ == "__main__":
    img_file = r"D:\Uni\Thesis\data\new_sampled_pain"
    # img_file = r"/home/albert/Lisanne/data/frames/all_frames/sampled_frames"
    #img = ["cocker front"]
    img = os.listdir(img_file)
    print("img", img)
    #img_file_list = [os.path.join(img_file, item) for item in os.listdir(img_file)]
    img_file_list = [os.path.join(img_file, item) for item in img]
    save_file = r"D:\Uni\Thesis\data\crop_new_sampled_pain_v8"
    #save_crop_file = r"C:\Thesis\dog_pain_lisanne\raw_frames\not_pain\crop_df_new_pain"
    # save_kp_file = r"/home/albert/Lisanne/data/frames/all_frames"
    # save_crop_file = r"/home/albert/Lisanne/data/frames/all_frames"
    print(f"img_file_list: {img_file_list}")
    data_model = Databuilder(img_file_list, save_file, 600, 600, yolov8=True)
    data_model.pipeline(needcrop=True, pose=False)