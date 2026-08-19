import argparse
import os
from itertools import combinations
 
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import csv

# ---------------- Board definition (must match your calibration) ----------------
ARUCO_DICT = cv2.aruco.DICT_4X4_50
SQUARES_X = 10          # your Aruco_rows
SQUARES_Y = 8           # your Aruco_cols
CHECKER_SIZE = 0.022    # meters
MARKER_SIZE = 0.016     # meters
 
dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
board = cv2.aruco.CharucoBoard((SQUARES_X, SQUARES_Y), CHECKER_SIZE, MARKER_SIZE, dictionary)
 
# Interior corner grid of a (SQUARES_X x SQUARES_Y) board
GRID_W = SQUARES_X - 1   # corners per row (9)
GRID_H = SQUARES_Y - 1   # rows of corners (7)
 
MIN_CORNERS = 10         # per camera, per frame, to attempt solvePnP
 

def load_intrinsics(yaml_path):
    fs = cv2.FileStorage(yaml_path, cv2.FILE_STORAGE_READ)
    K = fs.getNode("K").mat()
    dist = fs.getNode("dist").mat()
    fs.release()
    if K is None or dist is None:
        raise ValueError(f"Could not read K/dist from {yaml_path}")
    return K, dist
 
 
def list_images(d):
    exts = (".jpg", ".jpeg", ".png")
    return sorted(
        os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith(exts)
    )


# ---------------- Core geometry ----------------
 
def detect_corners(image, detector):
    """Return dict {corner_id: (u, v)} of detected charuco corners."""
    ch_corners, ch_ids, _, _ = detector.detectBoard(image)
    if ch_corners is None or ch_ids is None:
        return {}
    out = {}
    for cid, corner in zip(ch_ids.flatten(), ch_corners.reshape(-1, 2)):
        out[int(cid)] = corner.astype(np.float64)
    return out

def pose_from_corners(corner_dict, K, dist, object_points):
    """solvePnP on detected charuco corners. Returns (R, T, reproj_err) or None."""
    if len(corner_dict) < MIN_CORNERS:
        return None
    ids = sorted(corner_dict.keys())
    img_pts = np.array([corner_dict[i] for i in ids], dtype=np.float64)
    obj_pts = object_points[ids]
 
    ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist,
                                  flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        return None
 
    proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
    err = float(np.mean(np.linalg.norm(img_pts - proj.reshape(-1, 2), axis=1)))
 
    R, _ = cv2.Rodrigues(rvec)
    return R, tvec.reshape(3, 1), err

def relative_pose(R0, T0, R1, T1):
    """cam0 -> cam1: X_cam1 = R_rel @ X_cam0 + T_rel"""
    R_rel = R1 @ R0.T
    T_rel = T1 - R_rel @ T0
    return R_rel, T_rel

def triangulate_shared(corners0, corners1, K0, D0, K1, D1, R_rel, T_rel):
    """Triangulate all corner IDs seen by both cameras.
    Returns dict {corner_id: XYZ in cam0/world frame}."""
    shared = sorted(set(corners0) & set(corners1))
    if len(shared) < 2:
        return {}
 
    pts0 = np.array([corners0[i] for i in shared]).reshape(-1, 1, 2)
    pts1 = np.array([corners1[i] for i in shared]).reshape(-1, 1, 2)
 
    n0 = cv2.undistortPoints(pts0, K0, D0).reshape(-1, 2)
    n1 = cv2.undistortPoints(pts1, K1, D1).reshape(-1, 2)
 
    P0 = np.hstack([np.eye(3), np.zeros((3, 1))])
    P1 = np.hstack([R_rel, T_rel.reshape(3, 1)])
 
    X_h = cv2.triangulatePoints(P0, P1, n0.T, n1.T)   # 4 x N
 
    out = {}
    for k, cid in enumerate(shared):
        w = X_h[3, k]
        if np.isclose(w, 0.0):
            continue
        out[cid] = (X_h[:3, k] / w).flatten()
    return out

def adjacent_pairs(ids):
    """Grid-adjacent (right neighbor, down neighbor) pairs among given ids."""
    idset = set(ids)
    pairs = []
    for cid in ids:
        row, col = divmod(cid, GRID_W)
        right = cid + 1
        down = cid + GRID_W
        if col + 1 < GRID_W and right in idset:
            pairs.append((cid, right))
        if row + 1 < GRID_H and down in idset:
            pairs.append((cid, down))
    return pairs


# ---------------- Main ----------------
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cam0_images")
    ap.add_argument("cam1_images")
    ap.add_argument("cam0_yaml")
    ap.add_argument("cam1_yaml")
    ap.add_argument("out_dir")
    ap.add_argument("--show", action="store_true", help="also open plot windows")
    args = ap.parse_args()
 
    os.makedirs(args.out_dir, exist_ok=True)
 
    K0, D0 = load_intrinsics(args.cam0_yaml)
    K1, D1 = load_intrinsics(args.cam1_yaml)
    print(f"K0 principal point ~ ({K0[0,2]:.0f}, {K0[1,2]:.0f})")
    print(f"K1 principal point ~ ({K1[0,2]:.0f}, {K1[1,2]:.0f})")
 
    imgs0 = list_images(args.cam0_images)
    imgs1 = list_images(args.cam1_images)
    n_frames = min(len(imgs0), len(imgs1))
    print(f"cam0: {len(imgs0)} images, cam1: {len(imgs1)} images -> using {n_frames} pairs\n")
 
    charuco_params = cv2.aruco.CharucoParameters()
    charuco_params.tryRefineMarkers = True
    detector = cv2.aruco.CharucoDetector(
        board=board, charucoParams=charuco_params,
        detectorParams=cv2.aruco.DetectorParameters())
 
    object_points = board.getChessboardCorners()  # (N, 3) true board coords
 
    # per-frame records
    frames, baselines, cam1_positions = [], [], []
    mean_spacing_err, max_spacing_err = [], []
    pnp_err0, pnp_err1 = [], []
    all_spacings = []
    best_frame = None   # (score, frame_idx, tri_points, cam1_pos)
 
    for i in range(n_frames):
        im0 = cv2.imread(imgs0[i])
        im1 = cv2.imread(imgs1[i])
        if im0 is None or im1 is None:
            continue
 
        c0 = detect_corners(im0, detector)
        c1 = detect_corners(im1, detector)
 
        p0 = pose_from_corners(c0, K0, D0, object_points)
        p1 = pose_from_corners(c1, K1, D1, object_points)
        if p0 is None or p1 is None:
            print(f"frame {i:03d}: skipped (corners cam0={len(c0)}, cam1={len(c1)})")
            continue
 
        R0, T0, e0 = p0
        R1, T1, e1 = p1
        R_rel, T_rel = relative_pose(R0, T0, R1, T1)
 
        cam1_pos = (-R_rel.T @ T_rel).flatten()
        baseline = float(np.linalg.norm(T_rel))
 
        tri = triangulate_shared(c0, c1, K0, D0, K1, D1, R_rel, T_rel)
        pairs = adjacent_pairs(list(tri.keys()))
        if not pairs:
            continue
 
        spacings = np.array([np.linalg.norm(tri[a] - tri[b]) for a, b in pairs])
        errs_mm = np.abs(spacings - CHECKER_SIZE) * 1000.0
 
        frames.append(i)
        baselines.append(baseline)
        cam1_positions.append(cam1_pos)
        mean_spacing_err.append(float(np.mean(errs_mm)))
        max_spacing_err.append(float(np.max(errs_mm)))
        pnp_err0.append(e0)
        pnp_err1.append(e1)
        all_spacings.extend(spacings * 1000.0)
 
        score = np.mean(errs_mm)
        if best_frame is None or score < best_frame[0]:
            best_frame = (score, i, tri, cam1_pos)
 
        print(f"frame {i:03d}: baseline={baseline*1000:7.1f} mm | "
              f"shared corners={len(tri):3d} | "
              f"spacing err mean={np.mean(errs_mm):5.2f} mm max={np.max(errs_mm):5.2f} mm | "
              f"pnp err {e0:.3f}/{e1:.3f} px")
 
    if not frames:
        print("\nNo usable frame pairs. Check board visibility and sync.")
        return
 
    cam1_positions = np.array(cam1_positions)
    baselines = np.array(baselines)
 
    print(f"\n{'='*60}")
    print(f"Frames used: {len(frames)}")
    print(f"Baseline: mean {baselines.mean()*1000:.1f} mm, "
          f"std {baselines.std()*1000:.2f} mm  <-- compare to tape measure")
    print(f"cam1 position spread (std, mm): "
          f"{(cam1_positions.std(axis=0)*1000).round(2)}")
    print(f"Corner spacing: mean {np.mean(all_spacings):.2f} mm "
          f"(true {CHECKER_SIZE*1000:.1f}), std {np.std(all_spacings):.2f} mm")
    print(f"{'='*60}")
 
    # -------- CSV --------
    csv_path = os.path.join(args.out_dir, "per_frame_results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "baseline_mm",
                    "cam1_x_mm", "cam1_y_mm", "cam1_z_mm",
                    "mean_spacing_err_mm", "max_spacing_err_mm",
                    "pnp_err0_px", "pnp_err1_px"])
        for k, fi in enumerate(frames):
            w.writerow([fi, baselines[k]*1000,
                        *(cam1_positions[k]*1000),
                        mean_spacing_err[k], max_spacing_err[k],
                        pnp_err0[k], pnp_err1[k]])
    print(f"Saved {csv_path}")
 
    # -------- Plots --------
    # 1. Baseline per frame
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(frames, baselines * 1000, "o-", ms=4)
    ax.axhline(baselines.mean() * 1000, color="r", ls="--",
               label=f"mean {baselines.mean()*1000:.1f} mm")
    ax.set_xlabel("frame"); ax.set_ylabel("baseline (mm)")
    ax.set_title("Camera baseline per frame (should be flat)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "baseline_per_frame.png"), dpi=150)
 
    # 2. cam1 position scatter (3 projections)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    labels = [("x", "y"), ("x", "z"), ("y", "z")]
    idx = [(0, 1), (0, 2), (1, 2)]
    for ax, (la, lb), (a, b) in zip(axes, labels, idx):
        ax.scatter(cam1_positions[:, a]*1000, cam1_positions[:, b]*1000,
                   c=frames, cmap="viridis", s=20)
        ax.set_xlabel(f"{la} (mm)"); ax.set_ylabel(f"{lb} (mm)")
        ax.grid(alpha=0.3); ax.set_aspect("equal", adjustable="datalim")
    fig.suptitle("cam1 position estimates across frames (tight cluster = good)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "cam1_position_scatter.png"), dpi=150)
 
    # 3. Spacing error per frame
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(frames, mean_spacing_err, "o-", ms=4, label="mean")
    ax.plot(frames, max_spacing_err, "s--", ms=4, alpha=0.6, label="max")
    ax.set_xlabel("frame"); ax.set_ylabel("|spacing - 22 mm|  (mm)")
    ax.set_title("Adjacent-corner spacing error per frame")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "spacing_error_per_frame.png"), dpi=150)
 
    # 4. Pooled spacing histogram
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(all_spacings, bins=60)
    ax.axvline(CHECKER_SIZE * 1000, color="r", ls="--", label="true 22 mm")
    ax.set_xlabel("triangulated adjacent spacing (mm)")
    ax.set_ylabel("count")
    ax.set_title("All adjacent-corner spacings, all frames")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "spacing_histogram.png"), dpi=150)
 
    # 5. 3D scene from the best frame
    _, bf_idx, tri, cam1_pos = best_frame
    pts = np.array(list(tri.values()))
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=12, label="board corners")
    ax.scatter([0], [0], [0], c="r", s=90, marker="^", label="cam0 (origin)")
    ax.scatter([cam1_pos[0]], [cam1_pos[1]], [cam1_pos[2]],
               c="g", s=90, marker="^", label="cam1")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
    ax.set_title(f"3D scene, frame {bf_idx} (best spacing error)")
    ax.legend()
    # equal-ish aspect
    allp = np.vstack([pts, [[0, 0, 0]], [cam1_pos]])
    ctr = allp.mean(axis=0); r = np.max(np.abs(allp - ctr)) * 1.1
    ax.set_xlim(ctr[0]-r, ctr[0]+r)
    ax.set_ylim(ctr[1]-r, ctr[1]+r)
    ax.set_zlim(ctr[2]-r, ctr[2]+r)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "scene_3d.png"), dpi=150)
 
    print(f"Saved plots to {args.out_dir}/")
    if args.show:
        plt.show()
 
if __name__ == "__main__":
    main()











# import numpy as np
# from os.path import join
# import cv2
# import os
# from glob import glob
# import yaml
# import csv
# import argparse
# import pandas as pd
# import extrinsic_calibrate as ec

# def compute_data_paths(date):
#     extrinsic_param_files = []
#     positions = []

#     # compute param files paths 
#     extri_params = f'data/extri_data/params/{date}'
#     for dirpath, dirnames, filenames in os.walk(extri_params):
#         extrinsic_param_files = filenames

#     # compute extrinsic image paths
#     extri_images = f'data/extri_data/images/{date}'
#     image_dirs = []
#     for dirpath, dirnames, filenames in os.walk(extri_images):
#         image_dirs.append(dirpath)

#     # compute intrinsic image paths
#     intri_path = f'data/intri_data/images/{date}'
#     intrinsic_image_path = []
#     for dirpath, dirnames, filenames in os.walk(intri_path):
#         intrinsic_image_path.append(dirpath)
        
#     return sorted([f"{extri_params}/{f}" for f in extrinsic_param_files]), sorted(image_dirs[1:]), sorted(intrinsic_image_path[1:])

# def camera_positions(date):
#     positions = []
    
#     extrinsic_param_files, extrinsic_image_dirs, intrinsic_image_path = compute_data_paths(date)
#     print(f"Extrinsic param files = {extrinsic_param_files}")  
#     print(f"Extrinsic image folders = {extrinsic_image_dirs}")
#     print(f"Intrinsics = {intrinsic_image_path}\n")

#     list_images = [sorted([os.path.join(camera, f) for f in os.listdir(camera) if f.endswith(".jpg")]) for camera in extrinsic_image_dirs]

#     for i, file in enumerate(extrinsic_param_files):
#         df = pd.read_csv(file)
#         print(f"Data shape = {df.shape}")

#         gdata = df[df["retval"] == True] 
#         print(f"Good data shape = {gdata.shape}")
        
#         best_row = gdata.loc[gdata["p_error"].idxmin()]
#         print(f"min err = {best_row['p_error']}")

#         img_index = best_row["frame_idx"]
#         rvec_x = best_row["rvec_x"]
#         rvec_y = best_row["rvec_y"]
#         rvec_z = best_row["rvec_z"]

#         tvec_x = best_row["tvec_x"]
#         tvec_y = best_row["tvec_y"]
#         tvec_z = best_row["tvec_z"]

#         rvec = np.array([[rvec_x], [rvec_y], [rvec_z]], dtype=np.float64)
#         tvec = np.array([[tvec_x], [tvec_y], [tvec_z]], dtype=np.float64)

#         dstr, jacobian_r = cv2.Rodrigues(rvec)

#         cam_pos = -dstr.T @ tvec
#         print(f"{cam_pos}\n")

#         positions.append(cam_pos) 

#         # visualize
#         rms, K, dist = ec.get_yaml_string(intrinsic_image_path[i])
#         img_path = list_images[i][int(img_index)]

#         print(f"image path = {img_path}")
#         print(f"data file = {file}")
#         print(f"intrinsic path = {intrinsic_image_path[i]}")

#         # Visualize 
#         img = cv2.imread(img_path)
#         cv2.drawFrameAxes(img, K, dist, rvec, tvec, 0.1)
#         small_img = cv2.resize(img, (800, 600))
#         cv2.imshow("axes", small_img)
#         cv2.waitKey(0)

#     return positions

# def compute_transfomrations(extri_params, extri_images, intri_path):

#     extrinsic_params, extrinsci_images, intrinsic_images = compute_data_paths(extri_params, extri_images, intri_path)

#     df0 = pd.read_csv(extrinsic_params[0])
#     df1 = pd.read_csv(extrinsic_params[1])
#     merged = df0.merge(df1, on="frame_idx", suffixes=("_0", "_1"))
#     merged = merged[merged["retval_0"] & merged["retval_1"]]

#     # pick the frame where the SUM of both reprojection errors is lowest
#     best = merged.loc[(merged["p_error_0"] + merged["p_error_1"]).idxmin()]

#     # Compute R matrix and T vectors for the individual frames 
#     R0, _ = cv2.Rodrigues(np.array([best.rvec_x_0, best.rvec_y_0, best.rvec_z_0]))  # point in board coordinates to cam0 coords
#     T0 = np.array([[best.tvec_x_0], [best.tvec_y_0], [best.tvec_z_0]])
#     R1, _ = cv2.Rodrigues(np.array([best.rvec_x_1, best.rvec_y_1, best.rvec_z_1]))
#     T1 = np.array([[best.tvec_x_1], [best.tvec_y_1], [best.tvec_z_1]])

#     R_rel = R1 @ R0.T              # cam0 -> cam1 rotation
#     T_rel = T1 - R_rel @ T0        # cam0 -> cam1 translation

#     return R_rel, T_rel

# def triangulate(R_rel, T_rel, K, dist):

#     pass

# def combine_cameras(date, action):

#     if action == "positions":
#         positions = camera_positions(date)
#         distance = np.linalg.norm(positions[0]-positions[1])

#         print(f"Camera poses: {[p for p in positions]}")
#         print(f"Distance btn cameras = {distance}")
    
#     elif action == "triangulate_2":
#         R_rel, T_rel = compute_transfomrations(date)
#         rms, K, dist = ec.get_yaml_string(f'data/intri_data/params/{date}')
#         triangulate(R_rel, T_rel, K, dist)



# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     # parser.add_argument('extri_params_path', type=str, help="the path of extrinsic paramters")
#     # parser.add_argument('extri_images', type=str, help="the path of extrinsic images")
#     # parser.add_argument('intri_path', type=str, help="the path of intrinsic images")
   
#     parser.add_argument('date', type=str, help='data date')
#     parser.add_argument('action', type=str, help="positions or triangulate")

#     args = parser.parse_args()

#     combine_cameras(args.date, args.action)