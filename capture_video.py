'''
DESCRIPTION:

This script serves to collect video frames from multiple cameras simultaneously.
With 3 operating modes, you can collect data for different purposes and store 
at different locations in the file structure. 

1. Calibration videos stored under data->intri_data or extri_data->videos
    ARGUMENTS: calibrate intrinsic
               calibrate extrinsic
2. Handpose detection videos stored under anipose->folder marked by current date and time -> videos-raw
    ARGUMENTS: handpose

3. Check camera feed without a timed limit
    ARGUMENTS: check

'''

import cv2
import PySpin
import EasyPySpin   
import numpy as np
import argparse
import time
from datetime import datetime
import os

fw = cv2.CAP_PROP_FRAME_WIDTH
fh = cv2.CAP_PROP_FRAME_HEIGHT


'''
Uses EasyPySpin libraries to open camera feed given their specific serial numbers 
which can be found by running the spinview app.
arrangement of serial numbers propagates through entire code and will dictate the 
sequence of cameras.
'''
def open_video_capture(action_type, param_type=None):
    '''
    camera serial numbers:
    25132909 = bottom left
    25132928 = top left
    25132908 = bottom right
    25132918 = top right
    '''
    # Alter camera sequence here. 
    cap = EasyPySpin.MultipleVideoCapture("25132928", "25132918", "25132909", "25132908")
    cap.set(cv2.CAP_PROP_EXPOSURE, -1)  # -1 sets exposure_time to auto
    cap.set(cv2.CAP_PROP_GAIN, -1)  # -1 sets gain to auto

    saving = action_type in ('calibrate', 'handpose')
    
    # Video files stored in folder marked as xxxx/2026-08-17_14/mp4* last value being the hour 
    if (saving):
        outdir = ""
        now = datetime.now()
        curr_time = f"{str(now.date())}_{str(now.hour)}"
        video_writer = cv2.VideoWriter_fourcc(*'mp4v')

        if (action_type == 'calibrate'):
            outdir = "data"
            if (param_type == "intrinsic"):
                outdir += "/intri_data/videos/"
            elif (param_type == "extrinsic"):
                outdir += "/extri_data/videos/"
            outdir += f"{curr_time}/"

        elif (action_type == 'handpose'):
            outdir = f'anipose/{curr_time}/videos-raw/'

        
        os.makedirs(outdir, exist_ok=True)

        #Alter cv2.VideoWriter()'s 3rd input to change fps
        output1 = cv2.VideoWriter(outdir+'take0-cam01.mp4', video_writer, 20.0, (1920, 1200), True)
        output2 = cv2.VideoWriter(outdir+'take0-cam02.mp4', video_writer, 20.0, (1920, 1200), True)
        output3 = cv2.VideoWriter(outdir+'take0-cam03.mp4', video_writer, 20.0, (1920, 1200), True)
        output4 = cv2.VideoWriter(outdir+'take0-cam04.mp4', video_writer, 20.0, (1920, 1200), True)

    if not all(cap.isOpened()):
        print("All cameras can't open\nexit")
        return -1

    start_time = time.time()
    
    while True:
        read_values = cap.read()
        frames = []
        frames_view = []
        all_ok = True

        for i, (ret, frame) in enumerate(read_values):
            if not ret or frame is None:
                print(f"Camera {i}: no frame")
                all_ok = False
                frame = np.zeros((1200, 1920), dtype=np.uint8)   # placeholder for display only
            else:
                frame = frame.copy()

            frame = cv2.cvtColor(frame, cv2.COLOR_BayerBG2BGR)
            frame_view = cv2.resize(frame, None, fx=0.45, fy=0.45)

            frames.append(frame)
            frames_view.append(frame_view)

        rows = []
        for i in range(0, 4, 2):
            row = np.hstack(frames_view[i:i + 2])
            rows.append(row)

        if (saving):
            if all_ok:
                output1.write(frames[0])
                output2.write(frames[1])
                output3.write(frames[2])
                output4.write(frames[3])
            else:
                print("Frame set dropped (sync preserved)")

        grid_view = np.vstack(rows)
        cv2.imshow('Real time video capture', grid_view)

        if (action_type == 'check'):
            key = cv2.waitKey(30)
            if key == ord("q"):
                break
        elif (action_type == 'calibrate'):
            cv2.waitKey(1)
            if (param_type == "intrinsic" and (time.time() - start_time) >= 80):
                print(f'Saved vides to : {outdir}')
                break
            elif (param_type == "extrinsic" and (time.time() - start_time) >= 60):
                print(f'Saved vides to : {outdir}')
                break
        else:
            cv2.waitKey(1)
            if (time.time() - start_time >= 15):
                print(f'Saved vides to : {outdir}')
                break

    cv2.destroyAllWindows()

    if(saving):
        output1.release()
        output2.release()
        output3.release()
        output4.release()
    
    cap.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Calibration
    calibration_parser = subparsers.add_parser("calibrate", help="Video for camera calibration")
    calibration_parser.add_argument("data_type", choices=["intrinsic", "extrinsic"], help="camera parameter to calibrate")

    # Handpose detection
    handpose_parser = subparsers.add_parser("handpose", help="Video for handpose detection")

    # Camera view
    check_parser = subparsers.add_parser("check", help="Check camera view")

    args = parser.parse_args()
    
    if args.command == "calibrate":
        open_video_capture(args.command, args.data_type)

    else:
        open_video_capture(args.command)
