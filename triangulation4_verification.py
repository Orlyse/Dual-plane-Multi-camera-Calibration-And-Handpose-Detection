import argparse
import os
from collections import defaultdict, deque
from itertools import combinations

import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv
from itertools import permutations


# ---------------- Board definition (must match your calibration) ----------------
ARUCO_DICT = cv2.aruco.DICT_4X4_50
SQUARES_X = 10
SQUARES_Y = 8
CHECKER_SIZE = 0.022   # meters
MARKER_SIZE = 0.016    # meters

dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
board = cv2.aruco.CharucoBoard((SQUARES_X, SQUARES_Y), CHECKER_SIZE, MARKER_SIZE, dictionary)

GRID_W = SQUARES_X - 1
GRID_H = SQUARES_Y - 1
MIN_CORNERS = 10
MIN_EDGE_FRAMES = 10    # pair needs at least this many shared frames to form an edge

# distinct BGR colors for source cameras 0-4 
SRC_COLORS = [(255, 128, 0), (0, 0, 255), (255, 0, 255), (0, 255, 255)]
            # blue, red, pink, yellow
DET_COLOR = (0, 255, 0) # Camera's ground truth = green

def image_dirs(image_folder):
    subs = sorted(os.path.join(image_folder, d) for d in os.listdir(image_folder)
                  if os.path.isdir(os.path.join(image_folder, d)))
    if not subs:
        raise SystemExit(f"No camera subfolders in {image_folder}")
    return subs


def load_calib(npz_path, n_cams):
    data = np.load(npz_path)
    Ks, Ds, Rs, Ts = [], [], [], []
    for c in range(n_cams):
        Ks.append(data[f"K{c}"])
        Ds.append(data[f"D{c}"])
        if c == 0:
            Rs.append(np.eye(3))
            Ts.append(np.zeros((3, 1)))
        else:
            if f"R{c}" not in data:
                raise SystemExit(f"R{c}/T{c} missing from {npz_path} -- "
                                 f"camera {c} was not calibrated (disconnected?)")
            Rs.append(data[f"R{c}"])
            Ts.append(data[f"T{c}"].reshape(3, 1))
    return Ks, Ds, Rs, Ts

def list_images(d):
    exts = (".jpg", ".jpeg", ".png")
    return sorted(os.path.join(d, f) for f in os.listdir(d)
                  if f.lower().endswith(exts))
def detect_corners(image, detector):
    ch_corners, ch_ids, _, _ = detector.detectBoard(image)
    if ch_corners is None or ch_ids is None:
        return {}
    return {int(cid): c.astype(np.float64)
            for cid, c in zip(ch_ids.flatten(), ch_corners.reshape(-1, 2))}
  
def board_pose(corner_dict, K, D, object_points,
               max_reproj_px=3.0, ambiguity_ratio=2.0):
    if len(corner_dict) < MIN_CORNERS:
        return None
    ids = sorted(corner_dict.keys())
    img_pts = np.array([corner_dict[i] for i in ids], dtype=np.float64)
    obj_pts = object_points[ids]
    try:
        n_sol, rvecs, tvecs, _ = cv2.solvePnPGeneric(
            obj_pts, img_pts, K, D, flags=cv2.SOLVEPNP_IPPE)
    except cv2.error:
        return None
    if n_sol < 1:
        return None
 
    errs = []
    for rvec, tvec in zip(rvecs, tvecs):
        proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, D)
        errs.append(float(np.mean(
            np.linalg.norm(img_pts - proj.reshape(-1, 2), axis=1))))
    order = np.argsort(errs)
    best = order[0]
 
    if errs[best] > max_reproj_px:
        return None                                   # bad fit outright
    if n_sol > 1:
        second = order[1]
        if errs[second] < ambiguity_ratio * errs[best]:
            return None                               # flip indistinguishable
    if tvecs[best][2] <= 0:
        return None                                   # board behind camera
 
    Rb, _ = cv2.Rodrigues(rvecs[best])
    return Rb, tvecs[best].reshape(3, 1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image_path", help="folder of images")
    ap.add_argument("data_path", help="calibration array")
    ap.add_argument("--save-overlays", type=int, default=10)
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--outdir", default="none", help="output path if needed")
    args = ap.parse_args()
    
    image_folders = image_dirs(args.image_path)
    n_cams = len(image_folders)

    print(f"Image folders = {image_folders}")
    print(f"Reading from {n_cams} cameras")

    Ks, Ds, Rs, Ts = load_calib(args.data_path, n_cams)
    image_lists = [list_images(d) for d in image_folders]
    n_frames = min(len(l) for l in image_lists)
    print(f"Frames: {n_frames}\n")

    outpath = ''

    if args.outdir != "none":
        outpath = args.outdir
    else:
        # Create overlays folder
        path = args.image_path.split('/')
        assert len(path) == 4, (f"Wrong path input for image dir")
        date = path[-1]
        print(f"date = {date}\n")
        outpath = f'data/overlays/{date}'
    
    print(f"output path = {outpath}\n")
    os.makedirs(outpath, exist_ok=True)

    charuco_params = cv2.aruco.CharucoParameters()
    charuco_params.tryRefineMarkers = True
    detector = cv2.aruco.CharucoDetector(
        board=board, charucoParams=charuco_params,
        detectorParams=cv2.aruco.DetectorParameters())
    object_points = board.getChessboardCorners()   # (N,3) board frame
    
    errors = defaultdict(list)
    overlays_saved = 0
    writers = {}
 
    for i in range(n_frames):
        imgs, dets, poses = {}, {}, {}
        for c in range(n_cams):
            im = cv2.imread(image_lists[c][i])
            if im is None:
                continue
            imgs[c] = im
            d = detect_corners(im, detector)
            if d:
                dets[c] = d
                p = board_pose(d, Ks[c], Ds[c], object_points)
                if p is not None:
                    poses[c] = p
 
        if len(poses) < 1 or len(dets) < 2:
            continue
 
        frame_has_pair = False
        # predicted[k] = list of (src_j, {corner_id: predicted_px})
        predicted = defaultdict(list)
 
        for j, k in permutations(sorted(dets.keys()), 2):
            if j not in poses or k not in dets:
                continue
            Rb, Tb = poses[j]      # board -> cam j
 
            # board -> world:  X_w = Rj^T ( Rb X_b + Tb - Tj )
            # then world -> cam k, projected with K_k, D_k
            ids = sorted(dets[k].keys())
            obj = object_points[ids]                        # (n,3) board frame
            X_camj = (Rb @ obj.T + Tb)                      # 3 x n
            X_world = Rs[j].T @ (X_camj - Ts[j])            # 3 x n
            rvec_k, _ = cv2.Rodrigues(Rs[k])
            proj, _ = cv2.projectPoints(X_world.T, rvec_k, Ts[k], Ks[k], Ds[k])
            proj = proj.reshape(-1, 2)
 
            det_pts = np.array([dets[k][cid] for cid in ids])
            errs = np.linalg.norm(det_pts - proj, axis=1)
            errors[(j, k)].append((i, float(np.mean(errs)), len(ids)))
            predicted[k].append((j, dict(zip(ids, proj))))
            frame_has_pair = True
 
        # ---- overlays ----
        want_overlay = frame_has_pair and (
            overlays_saved < args.save_overlays or args.video)
        if want_overlay:
            for k in sorted(dets.keys()):
                im = imgs[k].copy()
                for cid, pt in dets[k].items():
                    cv2.circle(im, tuple(int(v) for v in pt), 6, DET_COLOR, 2,
                               lineType=cv2.LINE_AA)
                for (j, pred) in predicted.get(k, []):
                    col = SRC_COLORS[j % len(SRC_COLORS)]
                    for cid, pt in pred.items():
                        cv2.drawMarker(im, tuple(int(v) for v in pt), col,
                                       cv2.MARKER_CROSS, 12, 2,
                                       line_type=cv2.LINE_AA)
                legend = "green o = own detection | " + " ".join(
                    f"cam{j} x" for (j, _) in predicted.get(k, []))
                cv2.putText(im, f"frame {i} cam{k} | {legend}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 255, 255), 2)
 
                if overlays_saved < args.save_overlays:
                    cv2.imwrite(os.path.join(
                        outpath, f"{i:04d}_cam{k}.jpg"), im)
 
                if args.video:
                    if k not in writers:
                        h, w = im.shape[:2]
                        writers[k] = cv2.VideoWriter(
                            os.path.join(outpath, f"overlay_cam{k}.mp4"),
                            cv2.VideoWriter_fourcc(*"mp4v"), 15, (w, h))
                    writers[k].write(im)
            if overlays_saved < args.save_overlays:
                overlays_saved += 1
 
        if i % 25 == 0:
            print(f"frame {i:04d}: board seen by {sorted(dets.keys())}")
 
    for w in writers.values():
        w.release()
 
    if not errors:
        print("No usable source->target pairs found.")
        return
    
    # ---- summary matrix ----
    print(f"\n{'='*66}")
    print("Cross-reprojection error, mean px  (rows = source, cols = target)")
    header = "        " + "".join(f"cam{k:<7}" for k in range(n_cams))
    print(header)
    mat = np.full((n_cams, n_cams), np.nan)
    for j in range(n_cams):
        row = f"cam{j}   "
        for k in range(n_cams):
            if (j, k) in errors:
                m = np.mean([e[1] for e in errors[(j, k)]])
                mat[j, k] = m
                row += f"{m:7.2f}   "
            else:
                row += "      -   "
        print(row)
    print(f"{'='*66}")
    print("Rule of thumb: < 2 px excellent | 2-5 px usable | > 5 px extrinsics problem")
 
    # ---- CSV ----
    with open(os.path.join(outpath, "per_frame_cross_errors.csv"),
              "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "source_cam", "target_cam", "mean_px_err", "n_corners"])
        for (j, k), lst in sorted(errors.items()):
            for (fi, e, n) in lst:
                w.writerow([fi, j, k, e, n])

    # ---- plots ----
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(mat, cmap="viridis")
    ax.set_xticks(range(n_cams)); ax.set_yticks(range(n_cams))
    ax.set_xticklabels([f"cam{k}" for k in range(n_cams)])
    ax.set_yticklabels([f"cam{j}" for j in range(n_cams)])
    ax.set_xlabel("target (image being predicted)")
    ax.set_ylabel("source (camera providing board pose)")
    ax.set_title("Mean cross-reprojection error (px)")
    for j in range(n_cams):
        for k in range(n_cams):
            if not np.isnan(mat[j, k]):
                ax.text(k, j, f"{mat[j,k]:.2f}", ha="center", va="center",
                        color="white", fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(outpath, "cross_reproj_matrix.png"), dpi=150)
 
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for (j, k), lst in sorted(errors.items()):
        fr = [e[0] for e in lst]
        er = [e[1] for e in lst]
        ax.plot(fr, er, ".-", ms=3, alpha=0.7, label=f"cam{j}->cam{k}")
    ax.set_xlabel("frame"); ax.set_ylabel("mean px error")
    ax.set_title("Cross-reprojection error per frame")
    ax.legend(ncol=3, fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outpath, "cross_reproj_per_frame.png"), dpi=150)
 
    print(f"Saved outputs to {outpath}/  (overlays in {outpath}/)")

if __name__ == "__main__":
    main()
