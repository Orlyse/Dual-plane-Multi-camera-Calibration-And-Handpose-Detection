'''
Goal:
- Obtain intrinsic param matrix
- Obtain extrinsic param matrices (translation & rotation)

Steps:
- Get video
- Extract images
'''

import numpy as np
from os.path import join
import cv2
print(cv2.__version__)
print(hasattr(cv2.aruco, "calibrateCameraCharuco"))
print([x for x in dir(cv2.aruco) if "calibrate" in x.lower()])
import os
from glob import glob
from dataclasses import dataclass
import yaml
import csv

# ChAcUro board parameters
Aruco_dict = cv2.aruco.DICT_4X4_50
Aruco_rows = 10
Aruco_cols = 8
checker_size = 0.022
marker_size = 0.016 

dictionary = cv2.aruco.getPredefinedDictionary(Aruco_dict)
board = cv2.aruco.CharucoBoard((Aruco_rows, Aruco_cols), checker_size, marker_size, dictionary)


# @dataclass
# class intrinsic_params:
#     rms: float
#     mtx: np.ndarray
#     dcoeff: np.ndarray

# camera0 = intrinsic_params(rms=0.0,
#                            mtx = np.zeros(1),
#                            dcoeff=np.zeros(1))


# camera1 = intrinsic_params(rms=0.0,
#                            mtx = np.zeros(1),
#                            dcoeff=np.zeros(1))

def calibrate_and_save_parameters(image_path):
    print(f'Base path = {image_path}\n')
    sub_dirs = [] 
    for dirpath, dirnames, filenames in os.walk(image_path):
        #print(f'Dir names = {dirnames}')
        #print(f'Filenames = {filenames}')
        sub_dirs.append(dirpath)
    
    sub_dirs = sorted(sub_dirs[1:])
    
    print(f'Extracting data from {len(sub_dirs)} cameras')
    print(f'Image paths = {sub_dirs}\n')
    
    for index, image_file in enumerate(sub_dirs):
        images = [os.path.join(image_file, f) for f in os.listdir(image_file) if f.endswith(".jpg")]

        charuco_params = cv2.aruco.CharucoParameters()
        charuco_params.tryRefineMarkers = True
        detector_params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.CharucoDetector(board=board, charucoParams=charuco_params, detectorParams=detector_params)
        
        all_charucoCorners, all_charucoIds, all_markerCorners, all_markerIds = [], [], [], []
        
        image_size = (1920, 1200)

        for img in images:
            image = cv2.imread(img)

            
            charucoCorners, charucoIds, markerCorners, markerIds = (detector.detectBoard(image))

            if charucoCorners is not None and len(charucoCorners) > 10:

                all_markerCorners.append(markerCorners)
                all_markerIds.append(markerIds)
            
                all_charucoIds.append(charucoIds)
                all_charucoCorners.append(charucoCorners)

        
        print(f'Length marker corners list = {len(all_charucoCorners)}   Length ids = {len(all_charucoIds)}\n')
        print(f'Length charuco corners list = {len(all_markerCorners)}   Length ids = {len(all_markerIds)}\n')
        

        # Calibrate camera
        '''
        retval, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(charucoCorners, 
                                                                                          charucoIds, 
                                                                                          board, 
                                                                                          imageSize, 
                                                                                          cameraMatrix, 
                                                                                          distCoeffs, 
                                                                                          rvecs=None, 
                                                                                          tvecs=None, 
                                                                                          flags=None, 
                                                                                          criteria=None)
        '''
        
        retval, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(all_charucoCorners, all_charucoIds, board, image_size, None, None, flags=cv2.CALIB_FIX_ASPECT_RATIO)

        print(f"mtx = {camera_matrix}")
        print(f"dist_coeff = {dist_coeffs}")

        folders = image_file.split('/')
        new_folder_path = f"{'/'.join(folders[:2])}/params/{folders[-2]}"
        
        os.makedirs(new_folder_path, exist_ok=True)
        data_file_path = f"{new_folder_path}/intrinsic_params{folders[-1]}.yaml" 
        charuco_path = f"{new_folder_path}/charuco{folders[-1]}.csv"
        marker_path = f"{new_folder_path}/marker{folders[-1]}.csv"

        fs = cv2.FileStorage(data_file_path, cv2.FILE_STORAGE_WRITE)
        fs.write("rms", retval)
        fs.write("K", camera_matrix)
        fs.write("dist", dist_coeffs)
        fs.release()

        with open(charuco_path, "w", newline="") as f:
            writer = csv.writer(f)

            writer.writerow([
                "image_idx",
                "corner_id",
                "x",
                "y"
            ])

            for img_idx, (ids, corners) in enumerate(
                zip(all_charucoIds, all_charucoCorners)
            ):
                for corner_id, corner in zip(ids, corners):

                    x, y = corner[0]

                    writer.writerow([
                        img_idx,
                        int(corner_id[0]),
                        float(x),
                        float(y)
                    ])
        with open(marker_path, "w", newline="") as f:
            writer = csv.writer(f)

            writer.writerow([
                "image_idx",
                "marker_id",
                "corner_num",
                "x",
                "y"
            ])

            for img_idx, (ids, markers) in enumerate(
                zip(all_markerIds, all_markerCorners)
            ):
                for marker_id, marker in zip(ids, markers):

                    marker = marker.squeeze(0)   # (1,4,2) -> (4,2)

                    for corner_num, (x, y) in enumerate(marker):

                        writer.writerow([
                            img_idx,
                            int(marker_id[0]),
                            corner_num,
                            float(x),
                            float(y)
                        ])
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('img_path', type=str, help="the path of images")
    args = parser.parse_args()

    calibrate_and_save_parameters(args.img_path)