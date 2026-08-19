import numpy as np
from os.path import join
import cv2
import os
from glob import glob
import yaml
import csv

'''
Parameter defs:
retval = RMS
cameraMatrix = intrinsic parameter matrix
distCoeffs = distortion coefficients
rvecs = extrinsic rotation vector
tvecs = extrinsic translation vector
'''
# ChAcUro board parameters
Aruco_dict = cv2.aruco.DICT_4X4_50
Aruco_rows = 10
Aruco_cols = 8
checker_size = 0.022
marker_size = 0.016 

dictionary = cv2.aruco.getPredefinedDictionary(Aruco_dict)
board = cv2.aruco.CharucoBoard((Aruco_rows, Aruco_cols), checker_size, marker_size, dictionary)


def get_yaml_string(image_file_path):
    path = image_file_path.split('/')
    yaml_path = f"data/intri_data/params/{path[-2]}/intrinsic_params{path[-1]}.yaml"
    
    fs = cv2.FileStorage(yaml_path, cv2.FILE_STORAGE_READ)
    rms = fs.getNode("rms").real()
    K = fs.getNode("K").mat()
    dist = fs.getNode("dist").mat()

    fs.release()
    # print(f"Yaml path = {yaml_path}")

    return rms, K, dist
    
def calibrate_extrinsic(image_path):
    image_folders = sorted(sum([
        glob(join(image_path))], [])
    )

    sub_dirs = []
    for dirpath, dirnames, filenames in os.walk(image_path):
        sub_dirs.append(dirpath)
    
    sub_dirs = sorted(sub_dirs[1:])
    
    print(f'Extracting data from {len(sub_dirs)} cameras')

    charuco_corners_dict = {}
    marker_corners_dict = {}

    for camera in sub_dirs:
        image_files =  sorted([os.path.join(camera, f) for f in os.listdir(camera) if f.endswith(".jpg")])
        print(f'Image folder = {camera}')
        print(f"Total Images = {len(image_files)}")

        charuco_params = cv2.aruco.CharucoParameters()
        charuco_params.tryRefineMarkers = True
        detector_params = cv2.aruco.DetectorParameters()

        all_charucoCorners, all_charucoIds, all_markerCorners, all_markerIds, indices = [], [], [], [], []
        
        image_size = (1920, 1200)

        for index, img in enumerate(image_files):
            image = cv2.imread(img)

            detector = cv2.aruco.CharucoDetector(board=board, charucoParams=charuco_params, detectorParams=detector_params)
            charucoCorners, charucoIds, markerCorners, markerIds = (detector.detectBoard(image))

            if charucoCorners is not None:
                # print(len(charucoCorners))
                if len(charucoCorners) > 10:
                
                    all_markerCorners.append(markerCorners)
                    all_markerIds.append(markerIds)
                
                    all_charucoIds.append(charucoIds)
                    all_charucoCorners.append(charucoCorners)

                    indices.append(index)

        # print(f"{len(all_charucoCorners)}, {len(all_charucoIds)}, {len(all_markerIds)}, {len(all_markerCorners)}")
        rms, K, dist = get_yaml_string(camera)
        objectpoints = board.getChessboardCorners()
        
        all_retval, all_rvec, all_tvec = [], [], []
        all_projection_errors = []

        for id, corner in zip(all_charucoIds, all_charucoCorners):
            imagepts = corner.reshape(-1, 2)
            objectpts = objectpoints[id.flatten()]

            # compute rotation and translation of charuco board wrt to cameraX
            retval, rvec, tvec = cv2.solvePnP(objectpts, imagepts, K, dist, None, None)

            # compute best frame
            r_imagepts, _ = cv2.projectPoints(objectpts, rvec, tvec, K, dist)
            projected_imgpts = r_imagepts.reshape(-1, 2)
            err = np.mean(np.linalg.norm(imagepts - projected_imgpts, axis=1))
            
            all_projection_errors.append(err)

            all_retval.append(retval)
            all_rvec.append(rvec)
            all_tvec.append(tvec)
        
        # # Visualize 
        # img = cv2.imread(image_files[0])
        # cv2.drawFrameAxes(img, K, dist, all_rvec[0], all_tvec[0], 0.1)
        # cv2.imshow("axes", img)
        # cv2.waitKey(0)
        
        print(f"lengths all_retval = {len(all_retval)}, all_rvec = {len(all_rvec)}, all_tvec = {len(all_tvec)}\n")

        print(f'{all_retval[0]}')
        print(f'{all_rvec[0]}')
        print(f'{all_tvec[0]}')

        paths = camera.split('/')
        new_folder_path = f"{paths[0]}/{paths[1]}/params/{paths[-2]}"
        os.makedirs(new_folder_path, exist_ok=True)

        path_extrinsics = f"{new_folder_path}/extrinsic_params{paths[-1]}.csv"

        assert len(indices) == len(all_retval), "lists not equal"
        
        with open(path_extrinsics, "w", newline="") as f:
            writer = csv.writer(f)

            writer.writerow([
                "frame_idx",
                "retval",
                "rvec_x", "rvec_y", "rvec_z",
                "tvec_x", "tvec_y", "tvec_z", "p_error"
            ])

            curr_i = 0
            for (retval, rvec, tvec, err) in zip(all_retval, all_rvec, all_tvec, all_projection_errors):

                rx, ry, rz = rvec.flatten()
                tx, ty, tz = tvec.flatten()
                

                writer.writerow([
                    indices[curr_i],
                    retval,
                    float(rx), float(ry), float(rz),
                    float(tx), float(ty), float(tz), 
                    float(err)
                ])

                curr_i += 1
        

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('extri_path', type=str, help="the path of images")
    
    args = parser.parse_args()
    # cameras = [args.camera_number1,args.camera_number2]
    calibrate_extrinsic(args.extri_path) 