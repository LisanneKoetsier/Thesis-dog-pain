import os



os.environ["OPENCV_FFMPEG_READ_ATTEMPTS"] = "10000"
print(os.environ["OPENCV_FFMPEG_READ_ATTEMPTS"])

import cv2
import numpy as np
from PIL import Image
from subprocess import call
from tqdm import tqdm


def _isArrayLike(obj):

    return not type(obj) == str and hasattr(obj, '__iter__') and hasattr(obj, '__len__')


def cv_frame(input_dir, video_list, out_dir, start=None, scale=0.2, sample_rate=2, sample_sec=60):

    if not start:
        start = [[30, 165, 270] for _ in range(len(video_list))]
    else:
        assert len(start) == len(video_list)
    if _isArrayLike(sample_sec):
        assert len(sample_sec) == len(video_list) 
    for i, video in enumerate(video_list):
        input_video = os.path.join(input_dir, video) #filepath to video
        name = video.split(".")[0]
        out_file = os.path.join(out_dir, name) #folder where the images will be outputted
        if not os.path.isdir(out_file):
            os.makedirs(out_file)
            cap = cv2.VideoCapture(input_video)

            fps = int(cap.get(cv2.CAP_PROP_FPS))
            start_step = list(map(lambda x: x*fps, start[i])) #retrieve start frame
            select_step = set() #initialize the set of which steps will be selected to extract frames
            step = fps//sample_rate #the number of frames between sampled frames
            if not _isArrayLike(sample_sec):
                sample_nums = [sample_sec*sample_rate]*len(start_step)
            else:
                assert len(sample_sec[i]) == len(start[i])
                sample_nums = [sec * sample_rate for sec in sample_sec[i]]
            for j, item in enumerate(start_step):
                select_step = select_step.union(
                    set(range(item, item+step*sample_nums[j], step)))

            select_len = len(select_step)
            read, img = cap.read()
            count_frame = 0

            count_sample = 0
            while read:
                if count_frame in select_step:
                    if scale:
                        img = cv2.resize(img, (541, 406))
                    target_file = os.path.join(
                        out_file, name+"_%04d.jpg" % count_sample)  # f"{name}_{count_sample}.jpg")
                    cv2.imencode('.jpg', img)[1].tofile(target_file)
                    count_sample += 1
                if count_sample >= select_len:
                    break
                read, img = cap.read()
                count_frame += 1
                cv2.waitKey(1)
            cap.release()

def cv_frame_unsampled(input_dir, video_list, out_dir, scale=0.2):
    for video in video_list:
        input_video = os.path.join(input_dir, video)
        name = video.split(".")[0]
        out_file = os.path.join(out_dir, name)
        if not os.path.isdir(out_file):
            os.makedirs(out_file)
        cap = cv2.VideoCapture(input_video)
        count_frame = 0
        while True:
            read, img = cap.read()
            if not read:
                break
            if scale:
                img = cv2.resize(img, (541, 406))
            target_file = os.path.join(out_file, f"{name}_{count_frame:04d}.jpg")
            cv2.imwrite(target_file, img)
            count_frame += 1
        cap.release()


def ffmpeg_cut(video_dir, target_dir, video_clip=None, video_list=None):
    if not video_list:
        video_list = os.listdir(video_dir)
    if video_clip:
        assert len(video_clip) == len(video_list)
        manual = True
    else:
        manual = False
    for i, video in enumerate(video_list):
        input_video = os.path.join(video_dir, video)

        name = video.split(".")[0]
        out_file = os.path.join(target_dir, name)
        if not os.path.isdir(out_file):
            os.makedirs(out_file)
            if manual:
                start, end = video_clip[i]
            else:
                start, end = get_duration(input_video)

            out_frame = os.path.join(out_file, f"{name}_%04d.jpg")
            """
            call(["ffmpeg", '-i', input_video, "-r", "2", "-s",
                "676,507", "-ss", start, "-to", end, out_frame])
            """
            call(["ffmpeg", '-i', input_video, "-r", "3",
                 "-ss", start, "-to", end, out_frame])


def get_duration(input_video):
    cap = cv2.VideoCapture(input_video)
    if cap:
        fps = cap.get(5)
        num_frames = cap.get(7)
        seconds = int(num_frames/fps)-60
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        end = "%02d:%02d:%02d" % (h, m, s)
        start = "00:00:30"
        return start, end

def delete_frames(folder, video_path, sample_rate):
    # Open a sample video to get FPS (assuming cap is an opened cv2.VideoCapture object)
    dog_folders = sorted(os.listdir(folder))
    #dog_videos = sorted(os.listdir(video_path))
    dog_videos = ['Alisa_OFT_Cam1_Trim.mp4', 'Arya_OFT_Cam1_Trim.mp4', 'Arya-oft-cam1_Trim.mp4', 'Atlas_OFT_cam1_Trim.mp4',
                  'Corek_Cam1_Fakulte_OFT_Trim.mp4', 'Deniz_OFT_Cam1_Trim.mp4', 'Dost_Cam1_OFT_trim.mp4', 'Frankie_OFT_Cam 1_Trim.mp4',
                  'Freedy- OFT-Cam1_Trim.mp4', 'Iris_Cam1_OFT_Trim.mp4', 'Komutan_OFT_Cam1_Trim.mp4', 'Mafi-oft-ilk-cam1_Trim.mp4',
                  'mia-oft-cam1_Trim.mp4', 'Miki_OFT_Cam1_Trim.mp4', 'Mocha_OFT_Cam1_Trim.mp4', 'Sansa OFT_Trim.mp4']

    for idx, dog_folder in enumerate(dog_folders):
        dog_video = dog_videos[idx]
        dog_video_path = os.path.join(video_path, dog_video)
        print(f"dog video path: {dog_video_path}")
        print(f"dog folder: {dog_folder}")
        cap = cv2.VideoCapture(dog_video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        cap.release()  # Release the video capture object
        
        step = fps // sample_rate

        folder_path = os.path.join(folder, dog_folder)
        print(f"folder path: {folder_path} to extract frames from")
        # Get list of all frame files in the folder
        frames = sorted([f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        # Iterate through frames and delete unwanted ones
        for i in tqdm(range(len(frames))):
            if (i % step) != 0:
                os.remove(os.path.join(folder_path, frames[i]))
                print(f"Deleted: {frames[i]}")
            else:
                print(f"Kept: {frames[i]}")
        
        new_frames = len(os.listdir(folder_path))
        print(f"the new length is {new_frames}")


if __name__ == "__main__":
    video_dir = r"D:\Uni\Thesis\data_hongyi_og\videos\not_pain"
    #video_dir = r"/home/albert/Lisanne/data/videos/pain"
    #video_list = os.listdir(video_dir)
    video_list = ['Dost_Cam1_OFT_trim']
    # Chi_front.mp4, Cocker_side.mp4
    # Luna_front.mp4, Luna_side.mp4,
    target_dir = r"D:\Uni\Thesis\data\videos\raw_frames\pain"
    #target_dir = r"/home/albert/Lisanne/data/frames/pain/new_sampled_frames"
    starts = [[0] for x in video_list]
    #cv_frame(video_dir, video_list, target_dir, starts)
    #cv_frame_unsampled(video_dir, video_list, target_dir)
    folder_test = r"/home/albert/Lisanne/data/own_frames/all_frames/sampled_frames_missed"
    video_test = r"/home/albert/Lisanne/data/videos/all_videos"
    delete_frames(folder_test, video_test, 3)
