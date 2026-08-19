'''
DESCRIPTION:

This script serves to extract images given a video file. 

Modes:
1. Extracting calibration videos stored under data->intri_data or extri_data->videos

    ARGUMENTS: calibrate video_path
    eg: extract_video.py calibrate data/extri_data/videos/2026-08-17_13

2. Extracting Handpose detection videos stored under 
   anipose->folder marked by current date and time -> videos-raw

    ARGUMENTS: anipose video_path
    eg: extract_video.py anipose anipose/2026-08-15_13/videos-raw    
'''

import os
from os.path import join
from glob import glob
import argparse
import cv2

'''
Expects .mp4 video files and description of the purpose of the videos (calibrate, anipose)
at the spefied path, extracts image frames using cv2.video_read() and saves them to the 
same root file in an 
images folder
'''

def extract_images(action, video_path):
    print(f'Video path = {video_path}')
    videos = sorted(sum([
        glob(join(video_path, '*.mp4'))], [])
    )

    image_out = ''
    path = video_path.split('/')

    if action=='calibrate':
        assert len(path) == 4, (f"Wrong path input for calibrate")
        date = path[-1]
        param = path[1]
        image_out = f'data/{param}/images/{date}/'
    
    elif action == 'anipose':
        assert len(path) == 3, (f"Wrong path input for anipose")
        image_out = f'anipose/{path[1]}/images_raw/'
            
    print(f"Videos = {videos}")

    for videoname in videos:
        path_comps = videoname.split('/')
        video_index = path_comps[-1].split('.')[0]
        cam_out = os.path.join(image_out, video_index)
        os.makedirs(cam_out, exist_ok=True)

        print(f"Image output directory = {cam_out}")

        video = cv2.VideoCapture(videoname)
        if not video.isOpened():
            print(f"Failed to open {videoname}")
            continue

        curr_frame = 0
        while(True):
            ret, frame = video.read()
            
            if ret:
                img_name = f"{cam_out}/{str(curr_frame)}.jpg"
                ok = cv2.imwrite(img_name, frame)
                if not ok:
                    print(f'Failed to write {img_name}')
                curr_frame += 1
            else:
                break
    
        video.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=["calibrate", "anipose"], help="purpose")
    parser.add_argument('video_path', type=str, help="videos path")
    
    args = parser.parse_args()
    
    extract_images(args.action, args.video_path)
