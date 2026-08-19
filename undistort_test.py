# import cv2
# import os
# import numpy as np

# intrinsic_matrix = np.load('camera_matrix.npy')
# distortion_vector = np.load('dist_coeffs.npy')

# def undistort(image_path):
#     # Load PNG images from folder
#     image_files = [os.path.join(image_path, f) for f in os.listdir(image_path) if f.endswith(".jpg")]
#     image_files.sort()  # Ensure files are in order

#     print(f'Found {len(image_files)} images\n')

#     for im in image_files:
#         image = cv2.imread(im)
#         undistored = cv2.undistort(image, intrinsic_matrix, distortion_vector)
#         cv2.imshow('undistorted', undistored)
#         key = cv2.waitKey(100)

#         if key == 27:
#             break
    
#     cv2.destroyAllWindows()


# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument('path', type=str, help="the path of images")
#     args = parser.parse_args()

#     undistort(args.path)

import numpy as np, glob
files = sorted(glob.glob("anipose/2026-07-27_13/detections_np/*.npz"))
valid = np.array([~np.isnan(np.load(f)["xy"][:, 0, 0]) for f in files])  # (n_cams, n_frames)
per_frame = valid.sum(axis=0)
print("cameras per frame:", {k: int((per_frame == k).sum()) for k in range(len(files) + 1)})
print("triangulatable frames (>=2):", int((per_frame >= 2).sum()), "of", per_frame.size)