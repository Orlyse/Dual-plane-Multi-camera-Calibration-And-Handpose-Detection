
'''
DESCRIPTION:

This script serves to produce intrinsic parameters of cameras for which the images
of a charuco board are provided. Using the cv2 Charuco library, different
functions are used to detect the april IDs and return the camera's distortion coefficients, 
and camera matrix as well as the rms for the set of images used.

ARGUMENTS: image_folder
OUTPUT: .yaml file for each image folder representing a camera.
eg: intrinsic_calibrate1.py data/intri_data/images
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

'''
Creates a charuco detector using the parameters of the charuco board 
used in the videos. Detects corners and charuco_ids, then uses them to extract the 
camera matrix and dist_coeffs while removing outliers based on the error_threshold (rms).

Camera matrix: [[fx s=0 cx]
                [0  fy  cy]
                [0  0   1]]
Distortion coefficients: [k1, k2, p1, p2, k3]
'''
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
    print(f'_____________________________________________________________________________\n')

    
    for index, image_file in enumerate(sub_dirs):
        images = [os.path.join(image_file, f) for f in os.listdir(image_file) if f.endswith(".jpg")]

        charuco_params = cv2.aruco.CharucoParameters()
        charuco_params.tryRefineMarkers = True
        detector_params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.CharucoDetector(board=board, charucoParams=charuco_params, detectorParams=detector_params)
        
        all_charucoCorners, all_charucoIds, all_markerCorners, all_markerIds = [], [], [], []
        # img_indices = []  
        image_size = (1920, 1200)

        for img in images:
            image = cv2.imread(img)

            
            charucoCorners, charucoIds, markerCorners, markerIds = (detector.detectBoard(image))

            if charucoCorners is not None and len(charucoCorners) > 10:

                all_markerCorners.append(markerCorners)
                all_markerIds.append(markerIds)
            
                all_charucoIds.append(charucoIds)
                all_charucoCorners.append(charucoCorners)

        # Calibrate camera
        
        # retval, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(all_charucoCorners, all_charucoIds, board, image_size, None, None, flags=cv2.CALIB_FIX_ASPECT_RATIO)
        (retval, camera_matrix, dist_coeffs, rvecs, tvecs, stdDev_intri, stdDev_extri, perViewErrors) = cv2.aruco.calibrateCameraCharucoExtended(all_charucoCorners, all_charucoIds, board, image_size,    None, None, flags=cv2.CALIB_FIX_ASPECT_RATIO)
        print(f'Initial rms = {retval}, # views = {len(all_charucoCorners)}\n')

        # Iteratively remove outliers
        error_threshold = 0.5   # change to match desired error limit
        min_views = 100
        max_iteratiions = 10
        
        for iteration in range(max_iteratiions):
            errors_flat = perViewErrors.flatten()
            keep_mask = errors_flat <=error_threshold

            if np.all(keep_mask):
                print(f'Converged after {iteration} outlier removals')
                break

            if np.sum(keep_mask) < min_views:
                print(f'Remaining usable views = {np.sum(keep_mask)}')
                # print(f"Image indices: {[img for img, keep in zip(img_indices, keep_mask) if keep]}") 
                print(f'Could not converge in more than {min_views} views')
                break

            all_charucoCorners = [c for c, keep in zip(all_charucoCorners, keep_mask) if keep]
            all_charucoIds    = [c for c, keep in zip(all_charucoIds, keep_mask) if keep]

            (retval, camera_matrix, dist_coeffs, rvecs, tvecs, stdDev_intri, stdDev_extri, perViewErrors) = cv2.aruco.calibrateCameraCharucoExtended(all_charucoCorners, all_charucoIds, board, image_size, None, None, flags=cv2.CALIB_FIX_ASPECT_RATIO)

            print(f'Iteration {iteration}: RMS = {retval}, views left = {len(all_charucoCorners)}')

        print(f'\nFinal RMS = {retval}, Final # of views = {len(all_charucoCorners)}')
        print(f"mtx = {camera_matrix}")
        print(f"dist_coeff = {dist_coeffs}")
        print(f'_____________________________________________________________________________\n')

        folders = image_file.split('/')
        new_folder_path = f"{'/'.join(folders[:2])}/params/{folders[-2]}"
        
        os.makedirs(new_folder_path, exist_ok=True)
        data_file_path = f"{new_folder_path}/intrinsic_params{folders[-1]}.yaml" 
        # charuco_path = f"{new_folder_path}/charuco{folders[-1]}.csv"
        # marker_path = f"{new_folder_path}/marker{folders[-1]}.csv"

        fs = cv2.FileStorage(data_file_path, cv2.FILE_STORAGE_WRITE)
        fs.write("rms", retval)
        fs.write("K", camera_matrix)
        fs.write("dist", dist_coeffs)
        fs.release()

        # with open(charuco_path, "w", newline="") as f:
        #     writer = csv.writer(f)

        #     writer.writerow([
        #         "image_idx",
        #         "corner_id",
        #         "x",
        #         "y"
        #     ])

        #     for img_idx, (ids, corners) in enumerate(
        #         zip(all_charucoIds, all_charucoCorners)
        #     ):
        #         for corner_id, corner in zip(ids, corners):

        #             x, y = corner[0]

        #             writer.writerow([
        #                 img_idx,
        #                 int(corner_id[0]),
        #                 float(x),
        #                 float(y)
        #             ])
        # with open(marker_path, "w", newline="") as f:
        #     writer = csv.writer(f)

        #     writer.writerow([
        #         "image_idx",
        #         "marker_id",
        #         "corner_num",
        #         "x",
        #         "y"
        #     ])

        #     for img_idx, (ids, markers) in enumerate(
        #         zip(all_markerIds, all_markerCorners)
        #     ):
        #         for marker_id, marker in zip(ids, markers):

        #             marker = marker.squeeze(0)   # (1,4,2) -> (4,2)

        #             for corner_num, (x, y) in enumerate(marker):

        #                 writer.writerow([
        #                     img_idx,
        #                     int(marker_id[0]),
        #                     corner_num,
        #                     float(x),
        #                     float(y)
        #                 ])
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('img_path', type=str, help="the path of images")
    args = parser.parse_args()

    calibrate_and_save_parameters(args.img_path)