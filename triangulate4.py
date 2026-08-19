'''
DESCRIPTION:
This script follows up on intrinsic_calibration using the computed intrinsic 
parameters to triangulate. ie: determine the distance and orientation of 
each camera relative to camera 0. 

ARGUMENTS: images directory(different from those used for intrinsic calibration),
           intrinsic parameter .yaml folder, output directory, --show
OUTPUTS: multi_camera_calib.npz, 5 data plots

multi_camera_calib.npz: numpy file containing the intrinsic parameters and extrinsic 
                        parameters. ie: the camera matrices and distortion coefficients
                        of each camera, and additionally the rotation matrix and 
                        translation vector of each camera from camera 0 which is 
                        considered the origin.
'''

import argparse
import os
from collections import defaultdict, deque
from itertools import combinations

import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv

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
MIN_EDGE_FRAMES = 20    # pair needs at least this many shared frames to form an edge
# ---------------- IO ----------------

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
    return sorted(os.path.join(d, f) for f in os.listdir(d)
                  if f.lower().endswith(exts))


# ---------------- Geometry ----------------

'''
Returns the charuco ids and corners of the input image using the detector created using
the same charuco board for intrinsic calibration. 
'''
def detect_corners(image, detector):
    ch_corners, ch_ids, _, _ = detector.detectBoard(image)
    if ch_corners is None or ch_ids is None:
        return {}
    return {int(cid): c.astype(np.float64)
            for cid, c in zip(ch_ids.flatten(), ch_corners.reshape(-1, 2))}

'''
Returns the predicted camera position (rotation and translation). This function
includes a set of safeguards/limitations that aim to make the selected images 
provide the best triangulation values. 

- Atleast 10 charuco corners (MIN_CORNERS) must be detected in the image 
- If the best reprojection solutions from solvePnPGeneric has a pixel error > max_reproj_px
discard the entire image
- If 2 consecutive images give a reprojection error with a large ambiguity ratio
discard the entire image 
'''
def pose_from_corners(corner_dict, K, dist, object_points, 
                      max_reproj_px=2.0, ambiguity_ratio=3.0):
    if len(corner_dict) < MIN_CORNERS:
        return None
    ids = sorted(corner_dict.keys())
    img_pts = np.array([corner_dict[i] for i in ids], dtype=np.float64)
    obj_pts = object_points[ids]

    try:
        n_sol, rvecs, tvecs, _ = cv2.solvePnPGeneric(
            obj_pts, img_pts, K, dist, flags=cv2.SOLVEPNP_IPPE)
    except cv2.error:
        return None
    if n_sol < 1:
        return None

    errs = []
    for rvec, tvec in zip(rvecs, tvecs):
        proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
        errs.append(float(np.mean(np.linalg.norm(img_pts - proj.reshape(-1, 2), axis=1))))
    
    order = np.argsort(errs)
    best = order[0]

    if errs[best] > max_reproj_px:
        return None                                    # bad fit outright
    if n_sol > 1:
        second = order[1]
        if errs[second] < ambiguity_ratio * errs[best]:
            return None                                # flip indistinguishable
    if tvecs[best][2] <= 0:
        return None                                     # board behind camera

    R, _ = cv2.Rodrigues(rvecs[best])
    return R, tvecs[best].reshape(3, 1), errs[best]

'''
Given the rotation and translation matrices of 2 cameras compute the relative
rotation and translation of camera B from camera A's origin.
'''
def relative_pose(Ra, Ta, Rb, Tb):
    """cam a -> cam b:  X_b = R_rel @ X_a + T_rel"""
    R_rel = Rb @ Ra.T
    T_rel = Tb - R_rel @ Ta
    return R_rel, T_rel

'''
Given a list rotation matrices, determine the average rotation by computing the
average of each element of the matrices.
'''
def average_rotations(R_list):
    """Chordal-mean rotation: average the matrices, project back to SO(3)."""
    M = np.mean(np.stack(R_list), axis=0)
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:            # reflection guard
        U[:, -1] *= -1
        R = U @ Vt
    return R

'''
Given the rotation and translation of matrix B from A's body frame and rotation and 
translation of C from B's body frame, determine camera C's rotation and translation
from camera A. 
'''
def compose(Rab, Tab, Rbc, Tbc):
    """(a->b) then (b->c) gives a->c."""
    Rac = Rbc @ Rab
    Tac = Rbc @ Tab + Tbc
    return Rac, Tac


def invert(R, T):
    """Invert a->b into b->a."""
    Ri = R.T
    return Ri, -Ri @ T

"""
Direct Linear Transfomation triangulation from more than or equal to 2 views.
obs: list of (cam_idx, normalized_xy); 
cams_R/T: pose per camera (world->cam).

Returns world point or None.
"""
def triangulate_multiview(obs, cams_R, cams_T):
    
    if len(obs) < 2:
        return None
    rows = []
    for cam_idx, (x, y) in obs:
        P = np.hstack([cams_R[cam_idx], cams_T[cam_idx].reshape(3, 1)])
        rows.append(x * P[2] - P[0])
        rows.append(y * P[2] - P[1])
    A = np.stack(rows)
    try:
        _, _, Vt = np.linalg.svd(A)
    except np.linalg.LinAlgError:
        return None
    X = Vt[-1]
    if np.isclose(X[3], 0.0):
        return None
    return X[:3] / X[3]

'''
Determines adjacent/neighbouring corners on the charuco board using the board's
grid height and grid width.
'''
def adjacent_pairs(ids):
    idset = set(ids)
    pairs = []
    for cid in ids:
        row, col = divmod(cid, GRID_W)
        if col + 1 < GRID_W and cid + 1 in idset:
            pairs.append((cid, cid + 1))
        if row + 1 < GRID_H and cid + GRID_W in idset:
            pairs.append((cid, cid + GRID_W))
    return pairs


def draw_axes(ax, R_world, origin, length=0.06, label=None):
    origin = np.asarray(origin, dtype=np.float64).flatten()
    for k, c in enumerate(["r", "g", "b"]):
        d = R_world[:, k] * length
        ax.quiver(*origin, *d, color=c, linewidth=2, arrow_length_ratio=0.25)
    if label:
        ax.text(*origin, f"  {label}", fontsize=9)


# ---------------- Main ----------------

def discover_camera_dirs(images_root):
    """Sorted list of immediate subdirectories = one per camera."""
    subs = sorted(
        os.path.join(images_root, d) for d in os.listdir(images_root)
        if os.path.isdir(os.path.join(images_root, d))
    )
    if not subs:
        raise SystemExit(f"No camera subfolders found in {images_root}")
    return subs


def discover_yamls(yamls_root):
    """Sorted list of .yaml/.yml files in the folder = one per camera."""
    files = sorted(
        os.path.join(yamls_root, f) for f in os.listdir(yamls_root)
        if f.lower().endswith((".yaml", ".yml"))
    )
    if not files:
        raise SystemExit(f"No .yaml files found in {yamls_root}")
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images_dir",
                    help="folder containing one subfolder of images per camera; "
                         "subfolders sorted by name, first = cam0 = origin")
    ap.add_argument("yamls_dir",
                    help="folder containing one intrinsics YAML per camera; "
                         "files sorted by name, must pair with subfolder order")
    ap.add_argument("--out_dir", default="multicam_out")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    image_dirs = discover_camera_dirs(args.images_dir)
    yaml_files = discover_yamls(args.yamls_dir)

    if len(image_dirs) != len(yaml_files):
        raise SystemExit(
            f"Found {len(image_dirs)} camera folders but {len(yaml_files)} YAMLs:\n"
            f"  folders: {[os.path.basename(d) for d in image_dirs]}\n"
            f"  yamls:   {[os.path.basename(f) for f in yaml_files]}")
    n_cams = len(image_dirs)
    os.makedirs(args.out_dir, exist_ok=True)

    print("Camera assignment (sorted-name pairing):")
    for c, (d, y) in enumerate(zip(image_dirs, yaml_files)):
        print(f"  cam{c}: images={d}  intrinsics={y}")
    print()

    Ks, Ds = [], []
    for c in range(n_cams):
        K, D = load_intrinsics(yaml_files[c])
        Ks.append(K); Ds.append(D)
        fx, fy = K[0, 0], K[1, 1]
        print(f"cam{c}: fx={fx:.1f} fy={fy:.1f} "
              f"(ratio {fx/fy:.4f}) pp=({K[0,2]:.0f},{K[1,2]:.0f})")
        if abs(fx / fy - 1.0) > 0.02:
            print(f"  WARNING: cam{c} fx/fy differ by >2% -- intrinsics look suspect")

    img_lists = [list_images(d) for d in image_dirs]
    n_frames = min(len(l) for l in img_lists)
    print(f"\nFrames per camera: {[len(l) for l in img_lists]} -> using {n_frames}\n")

    charuco_params = cv2.aruco.CharucoParameters()
    charuco_params.tryRefineMarkers = True
    detector = cv2.aruco.CharucoDetector(
        board=board, charucoParams=charuco_params,
        detectorParams=cv2.aruco.DetectorParameters())
    object_points = board.getChessboardCorners()

    # ---- Pass 1: per-frame poses and pairwise edge observations ----
    # edge_obs[(a,b)] = list of (frame, R_rel, T_rel) with a < b, meaning a->b
    edge_obs = defaultdict(list)
    frame_poses = {}       # frame -> {cam: (R, T, err)}
    frame_corners = {}     # frame -> {cam: corner dict}

    for i in range(n_frames):
        poses, corners = {}, {}
        for c in range(n_cams):
            im = cv2.imread(img_lists[c][i])
            if im is None:
                continue
            det = detect_corners(im, detector)
            p = pose_from_corners(det, Ks[c], Ds[c], object_points)
            if p is not None:
                poses[c] = p
                corners[c] = det

        if len(poses) >= 2:
            frame_poses[i] = poses
            frame_corners[i] = corners
            for a, b in combinations(sorted(poses), 2):
                Ra, Ta, _ = poses[a]
                Rb, Tb, _ = poses[b]
                edge_obs[(a, b)].append((i, *relative_pose(Ra, Ta, Rb, Tb)))

        seen = sorted(poses.keys())
        print(f"frame {i:03d}: board seen by cams {seen}")

    # ---- Edge averaging ----
    print(f"\n{'='*64}\nCo-visibility edges:")
    edges = {}   # (a,b) -> (R_ab, T_ab, n_frames, baseline_std_mm)
    for (a, b), obs in sorted(edge_obs.items()):
        if len(obs) < MIN_EDGE_FRAMES:
            print(f"  cam{a} <-> cam{b}: only {len(obs)} shared frames "
                  f"(< {MIN_EDGE_FRAMES}) -- edge skipped")
            continue
        R_avg = average_rotations([o[1] for o in obs])
        T_avg = np.mean(np.stack([o[2] for o in obs]), axis=0)
        baselines = np.array([np.linalg.norm(o[2]) for o in obs]) * 1000
        edges[(a, b)] = (R_avg, T_avg, len(obs), baselines.std())
        print(f"  cam{a} <-> cam{b}: {len(obs):3d} frames | "
              f"baseline {baselines.mean():8.1f} mm (std {baselines.std():.2f})")
        if baselines.std() > 5.0:
            print(f"    WARNING: baseline std > 5 mm -- check sync/intrinsics for this pair")

    # ---- Chain everything into cam0's frame (BFS over the edge graph) ----
    adj = defaultdict(list)
    for (a, b) in edges:
        adj[a].append(b)
        adj[b].append(a)

    world_R = {0: np.eye(3)}
    world_T = {0: np.zeros((3, 1))}
    parent = {0: None}
    q = deque([0])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v in world_R:
                continue
            if (u, v) in edges:
                R_uv, T_uv = edges[(u, v)][0], edges[(u, v)][1]
            else:
                R_vu, T_vu = edges[(v, u)][0], edges[(v, u)][1]
                R_uv, T_uv = invert(R_vu, T_vu)
            # world->v = (world->u) then (u->v)
            world_R[v], world_T[v] = compose(world_R[u], world_T[u], R_uv, T_uv)
            parent[v] = u
            q.append(v)

    missing = [c for c in range(n_cams) if c not in world_R]
    print(f"\nPose chain (world = cam0):")
    for c in sorted(world_R):
        pos = (-world_R[c].T @ world_T[c]).flatten() * 1000
        via = "" if parent[c] is None else f"  (linked via cam{parent[c]})"
        print(f"  cam{c}: position [{pos[0]:8.1f}, {pos[1]:8.1f}, {pos[2]:8.1f}] mm{via}")
    if missing:
        print(f"\n  !! Cameras {missing} share NO frames with the calibrated set.")
        print(f"  !! Capture frames where they see the board together with a calibrated camera.")

    # ---- Loop-closure consistency check (if graph has cycles) ----
    for (a, b), (R_ab, T_ab, _, _) in edges.items():
        if a in world_R and b in world_R:
            # predicted a->b from chained world poses
            R_pred, T_pred = relative_pose(world_R[a], world_T[a], world_R[b], world_T[b])
            dT = np.linalg.norm(T_pred - T_ab) * 1000
            dR = np.degrees(np.arccos(np.clip((np.trace(R_pred.T @ R_ab) - 1) / 2, -1, 1)))
            tag = " <-- inconsistent!" if (dT > 3.0 or dR > 0.3) else ""
            print(f"  consistency cam{a}->cam{b}: dT={dT:5.2f} mm, dR={dR:5.3f} deg{tag}")

    # ---- Save .npz in reference format ----
    npz_path = os.path.join(args.out_dir, "multi_camera_calib.npz")
    payload = {}
    for c in range(n_cams):
        payload[f"K{c}"] = Ks[c]
        payload[f"D{c}"] = Ds[c]
        if c == 0:
            continue
        if c in world_R:
            payload[f"R{c}"] = world_R[c]
            payload[f"T{c}"] = world_T[c]
    np.savez(npz_path, **payload)
    print(f"\nSaved calibration to {npz_path}")

    # ---- Pass 2: multi-view triangulation validation ----
    frames_v, mean_err_v, max_err_v, ncams_v = [], [], [], []
    all_spacings = []
    best_frame = None

    for i, corners in frame_corners.items():
        usable = [c for c in corners if c in world_R]
        if len(usable) < 2:
            continue

        # normalized coordinates per camera
        norm = {}
        for c in usable:
            ids = sorted(corners[c].keys())
            pts = np.array([corners[c][k] for k in ids]).reshape(-1, 1, 2)
            und = cv2.undistortPoints(pts, Ks[c], Ds[c]).reshape(-1, 2)
            norm[c] = dict(zip(ids, und))

        # gather observations per corner id
        obs_by_id = defaultdict(list)
        for c in usable:
            for cid, xy in norm[c].items():
                obs_by_id[cid].append((c, xy))

        tri = {}
        for cid, obs in obs_by_id.items():
            if len(obs) < 2:
                continue
            X = triangulate_multiview(obs, world_R, world_T)
            if X is not None:
                tri[cid] = X

        pairs = adjacent_pairs(list(tri.keys()))
        if not pairs:
            continue
        spac = np.array([np.linalg.norm(tri[a] - tri[b]) for a, b in pairs])
        errs = np.abs(spac - CHECKER_SIZE) * 1000

        frames_v.append(i)
        mean_err_v.append(float(errs.mean()))
        max_err_v.append(float(errs.max()))
        ncams_v.append(len(usable))
        all_spacings.extend(spac * 1000)

        if best_frame is None or errs.mean() < best_frame[0]:
            best_frame = (errs.mean(), i, tri)

    print(f"\n{'='*64}")
    print(f"Validation over {len(frames_v)} frames:")
    if all_spacings:
        print(f"  spacing mean {np.mean(all_spacings):.3f} mm "
              f"(true {CHECKER_SIZE*1000:.1f}), std {np.std(all_spacings):.3f} mm")
    print(f"{'='*64}")

    # ---- CSV ----
    with open(os.path.join(args.out_dir, "per_frame_results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "n_cams_used", "mean_spacing_err_mm", "max_spacing_err_mm"])
        for k in range(len(frames_v)):
            w.writerow([frames_v[k], ncams_v[k], mean_err_v[k], max_err_v[k]])

    # ---- Plots ----
    # 1. per-edge baseline stability
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for (a, b), obs in sorted(edge_obs.items()):
        if (a, b) not in edges:
            continue
        fr = [o[0] for o in obs]
        bl = [np.linalg.norm(o[2]) * 1000 for o in obs]
        ax.plot(fr, bl, "o-", ms=3, label=f"cam{a}-cam{b}")
    ax.set_xlabel("frame"); ax.set_ylabel("baseline (mm)")
    ax.set_title("Per-pair baseline across frames (each line should be flat)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "edge_baselines.png"), dpi=150)

    # 2. spacing error per frame, colored by camera count
    fig, ax = plt.subplots(figsize=(10, 4.5))
    sc = ax.scatter(frames_v, mean_err_v, c=ncams_v, cmap="viridis", s=22)
    plt.colorbar(sc, ax=ax, label="# cameras used")
    ax.set_xlabel("frame"); ax.set_ylabel("mean |spacing - 22 mm| (mm)")
    ax.set_title("Multi-view spacing error per frame")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "spacing_error_per_frame.png"), dpi=150)

    # 3. pooled histogram
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(all_spacings, bins=60)
    ax.axvline(CHECKER_SIZE * 1000, color="r", ls="--", label="true 22 mm")
    ax.set_xlabel("triangulated adjacent spacing (mm)"); ax.set_ylabel("count")
    ax.set_title("All adjacent spacings, all frames (multi-view)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "spacing_histogram.png"), dpi=150)

    # 4. 3D scene: all cameras + best-frame board
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    cam_pts = []
    for c in sorted(world_R):
        pos = (-world_R[c].T @ world_T[c]).flatten()
        cam_pts.append(pos)
        ax.scatter(*pos, s=90, marker="^", label=f"cam{c}")
        draw_axes(ax, world_R[c].T, pos, length=0.08, label=f"cam{c}")
    if best_frame is not None:
        _, bf_idx, tri = best_frame
        pts = np.array(list(tri.values()))
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=10, label=f"board (frame {bf_idx})")
        cam_pts.append(pts.mean(axis=0))
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
    ax.set_title("Camera rig in cam0's frame")
    ax.legend()
    allp = np.array(cam_pts)
    ctr = allp.mean(axis=0); r = max(np.max(np.abs(allp - ctr)) * 1.2, 0.3)
    ax.set_xlim(ctr[0]-r, ctr[0]+r); ax.set_ylim(ctr[1]-r, ctr[1]+r)
    ax.set_zlim(ctr[2]-r, ctr[2]+r)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "rig_3d.png"), dpi=150)

    print(f"Saved plots to {args.out_dir}/")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()