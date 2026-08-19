"""
Replace DeepLabCut with MediaPipe Hands as the 2D detector for anipose.

Two subcommands, because they belong in different Python environments:

  detect    (run in a MODERN python env with mediapipe installed)
            Runs MediaPipe Hands on each video, saves keypoints to .npz
  write-h5  (run INSIDE the anipose venv, which has pandas/pytables)
            Converts the .npz files into DLC-format .h5 in pose-2d/

Usage:
  # modern env:  pip install mediapipe opencv-python numpy
  python mediapipe_to_anipose.py detect SESSION/videos-raw --out detections_np

  # anipose venv:
  python mediapipe_to_anipose.py write-h5 detections_np --out SESSION/pose-2d

After write-h5, continue the anipose pipeline as usual:
  anipose filter
  anipose triangulate
  anipose label-2d-filter / label-3d / label-combined

Keypoint mapping (MediaPipe index -> demo bodypart name):
  0 wrist -> base
  thumb  1,2,3,4   -> MCP1, PIP1, DIP1, tip1
  index  5,6,7,8   -> MCP2, PIP2, DIP2, tip2
  middle 9,10,11,12-> MCP3, PIP3, DIP3, tip3
  ring   13,14,15,16-> MCP4, PIP4, DIP4, tip4
  pinky  17,18,19,20-> MCP5, PIP5, DIP5, tip5
(DIP1 is extra relative to the demo scheme; anipose ignores bodyparts
 not referenced in config constraints/scheme, so it is harmless.)
"""

import argparse
import os
import sys
from glob import glob

import numpy as np

BODYPARTS = (
    ["base"]
    + [f"{part}{finger}" for finger in range(1, 6)
       for part in ("MCP", "PIP", "DIP", "tip")]
)
# order above yields: base, MCP1, PIP1, DIP1, tip1, MCP2, ... tip5
# which matches MediaPipe landmark order 0..20 exactly.

SCORER = "mediapipe_hands"


ROT_CODES = {0: None, 90: None, 180: None, 270: None}   # filled in cmd_detect


def _rotate(frame, deg, cv2):
    if deg == 0:
        return frame
    if deg == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if deg == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)


def _unrotate_points(xy, deg, W, H):
    """Map points detected in a rotated image back to original pixel coords.
    W, H are the ORIGINAL image width/height."""
    if deg == 0:
        return xy
    x, y = xy[:, 0].copy(), xy[:, 1].copy()
    if deg == 90:      # clockwise: x_rot = H-1-y, y_rot = x
        return np.stack([y, (H - 1) - x], axis=1)
    if deg == 180:
        return np.stack([(W - 1) - x, (H - 1) - y], axis=1)
    # 270 (counter-clockwise): x_rot = y, y_rot = W-1-x
    return np.stack([(W - 1) - y, x], axis=1)


def _enhance(frame, cv2):
    """CLAHE on the luminance channel -- helps low-contrast IR/grayscale."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _read_toml(path):
    """Load a TOML file across Python versions / available packages."""
    try:
        import tomllib                      # Python 3.11+
        with open(path, "rb") as f:
            return tomllib.load(f)
    except ImportError:
        pass
    try:
        import toml                         # pip install toml
        return toml.load(path)
    except ImportError:
        pass
    try:
        import tomli                        # pip install tomli
        with open(path, "rb") as f:
            return tomli.load(f)
    except ImportError:
        sys.exit("Cannot read TOML: install a reader with 'pip install toml' "
                 "(or run on Python 3.11+, which has tomllib built in).")


def _load_calib_toml(path, video_stems):
    """Parse an anipose calibration.toml into [(K, D), ...] aligned to videos.

    Cameras are matched to videos by the `name` field when possible (robust
    against ordering mistakes), otherwise by cam_N index order.
    Returns (list_of_KD, how_matched, names).
    """
    doc = _read_toml(path)
    cams = []
    for key, val in doc.items():
        if not key.startswith("cam_") or not isinstance(val, dict):
            continue
        if val.get("fisheye", False):
            sys.exit(
                f"{path}: [{key}] has fisheye = true, but this script's "
                "undistortion uses OpenCV's standard (Brown-Conrady) model. "
                "Using the wrong model would silently misplace keypoints. "
                "Set fisheye = false if the calibration used the standard "
                "model, or omit --calib to skip undistortion.")
        try:
            idx = int(key.split("_", 1)[1])
        except ValueError:
            continue
        K = np.array(val["matrix"], dtype=np.float64)
        D = np.array(val["distortions"], dtype=np.float64).reshape(1, -1)
        cams.append((idx, str(val.get("name", idx)), K, D))

    if not cams:
        sys.exit(f"No [cam_N] sections found in {path}")
    cams.sort(key=lambda t: t[0])
    names = [c[1] for c in cams]

    by_name = {c[1]: (c[2], c[3]) for c in cams}
    if all(stem in by_name for stem in video_stems):
        return [by_name[stem] for stem in video_stems], "name", names

    if len(cams) != len(video_stems):
        sys.exit(f"{path} has {len(cams)} cameras (names {names}) but "
                 f"{len(video_stems)} videos ({video_stems}); cannot pair them.")
    return [(c[2], c[3]) for c in cams], "order", names


def _load_calib_npz(path, n_cams):
    """Return [(K, D), ...] per camera index from multi_camera_calib.npz."""
    data = np.load(path)
    out = []
    for c in range(n_cams):
        if f"K{c}" not in data:
            raise SystemExit(f"{path} has no K{c} (needs {n_cams} cameras)")
        out.append((data[f"K{c}"].astype(np.float64),
                    np.asarray(data[f"D{c}"], dtype=np.float64).reshape(1, -1)))
    return out


def _redistort_points(xy_und, K_new, K, D, cv2):
    """Map points from UNDISTORTED image coords back to ORIGINAL (distorted)
    pixel coords, so saved keypoints stay consistent with a calibration that
    still carries distortion coefficients.

    Undistorted pixel -> normalized ray (via K_new) -> project through the
    real lens model (K, D) -> original pixel.
    """
    x = (xy_und[:, 0] - K_new[0, 2]) / K_new[0, 0]
    y = (xy_und[:, 1] - K_new[1, 2]) / K_new[1, 1]
    rays = np.stack([x, y, np.ones_like(x)], axis=1).astype(np.float64)
    proj, _ = cv2.projectPoints(rays, np.zeros(3), np.zeros(3), K, D)
    return proj.reshape(-1, 2)


PROBE_NOISE_FLOOR = 3       # fewer hits than this = indistinguishable from noise
PROBE_MARGIN = 1.5          # best must beat the default by this factor to be used


def cmd_detect(args):
    import cv2
    import mediapipe as mp

    os.makedirs(args.out, exist_ok=True)
    videos = sorted(
        p for p in glob(os.path.join(args.videos_dir, "*"))
        if p.lower().endswith((".mp4", ".avi", ".mov"))
    )
    if not videos:
        sys.exit(f"No videos found in {args.videos_dir}")

    # STATIC detector: stateless, one independent detection per image.
    # Used for probing (non-sequential frames) and for retry attempts, so
    # neither can corrupt the sequential tracker's state.
    hands_static = mp.solutions.hands.Hands(
        static_image_mode=True, max_num_hands=1, model_complexity=1,
        min_detection_confidence=args.min_conf)

    def run_detector(det, frame, deg, W, H, use_clahe):
        """Detect on frame (optionally CLAHE'd, rotated by deg).
        Returns (xy_in_original_coords, score) or None."""
        img = _enhance(frame, cv2) if use_clahe else frame
        img = _rotate(img, deg, cv2)
        rh, rw = img.shape[:2]
        res = det.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not res.multi_hand_landmarks:
            return None
        lm = res.multi_hand_landmarks[0].landmark
        xy_rot = np.array([[p.x * rw, p.y * rh] for p in lm])
        score = 1.0
        if res.multi_handedness:
            score = float(res.multi_handedness[0].classification[0].score)
        return _unrotate_points(xy_rot, deg, W, H), score

    rot_candidates = ([0, 90, 180, 270] if args.rotate == "auto"
                      else [int(args.rotate)])
    clahe_candidates = {"off": [False], "on": [True],
                        "auto": [False, True]}[args.clahe]
    default_combo = (rot_candidates[0], clahe_candidates[0])

    calib = None
    if args.calib:
        stems = [os.path.splitext(os.path.basename(v))[0] for v in videos]
        if args.calib.lower().endswith((".toml", ".tml")):
            calib, how, names = _load_calib_toml(args.calib, stems)
            print(f"Loaded intrinsics from {args.calib} "
                  f"(cameras {names}, matched by {how})")
            for stem, (K, _) in zip(stems, calib):
                print(f"  {stem}: fx={K[0,0]:.1f} pp=({K[0,2]:.0f},{K[1,2]:.0f})")
        else:
            calib = _load_calib_npz(args.calib, len(videos))
            print(f"Loaded intrinsics from {args.calib} (matched by order)")
        print(f"Undistorting before detection (alpha={args.undistort_alpha}); "
              f"keypoints are mapped back to ORIGINAL pixel coords, so the "
              f"calibration stays valid downstream.\n")

    for cam_idx, vid in enumerate(videos):
        stem = os.path.splitext(os.path.basename(vid))[0]
        cap = cv2.VideoCapture(vid)
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # ---------- optional undistortion ----------
        if calib is not None:
            K, D = calib[cam_idx]
            K_new, _ = cv2.getOptimalNewCameraMatrix(
                K, D, (W, H), args.undistort_alpha, (W, H))
            map1, map2 = cv2.initUndistortRectifyMap(
                K, D, None, K_new, (W, H), cv2.CV_16SC2)

            def prep(frame, map1=map1, map2=map2):
                return cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)

            def post(xy, K_new=K_new, K=K, D=D):
                return _redistort_points(xy, K_new, K, D, cv2)
        else:
            def prep(frame):
                return frame

            def post(xy):
                return xy

        # ---------- probe: pick (rotation, clahe) per camera ----------
        combo = default_combo
        if len(rot_candidates) > 1 or len(clahe_candidates) > 1:
            probe_idx = np.linspace(0, max(n_total - 1, 0),
                                    num=min(args.probe_frames, max(n_total, 1)),
                                    dtype=int)
            probe_frames = []
            for idx in probe_idx:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ok, f = cap.read()
                if ok:
                    probe_frames.append(prep(f))

            hits = {}
            for use_clahe in clahe_candidates:
                for deg in rot_candidates:
                    n = sum(
                        run_detector(hands_static, f, deg, W, H, use_clahe) is not None
                        for f in probe_frames)
                    hits[(deg, use_clahe)] = n

            best = max(hits, key=lambda k: hits[k])
            base_n = hits.get(default_combo, 0)
            if hits[best] < PROBE_NOISE_FLOOR:
                print(f"{stem}: probe found almost nothing "
                      f"(best {hits[best]}/{len(probe_frames)}) -- "
                      f"detection is limited by the imagery, not settings")
                combo = default_combo
            elif hits[best] > max(base_n * PROBE_MARGIN, base_n + 1):
                combo = best
            else:
                combo = default_combo

            table = ", ".join(
                f"{d}{'+clahe' if c else ''}:{n}"
                for (d, c), n in sorted(hits.items(), key=lambda kv: -kv[1])[:4])
            print(f"{stem}: probe ({len(probe_frames)} frames) top -> {table} "
                  f"| using rotation {combo[0]}, clahe={combo[1]}")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        best_rot, use_clahe = combo
        others = [d for d in rot_candidates if d != best_rot]

        # FRESH tracking detector per video: sequential video mode, no state
        # carried over from the probe or from the previous camera.
        hands_track = mp.solutions.hands.Hands(
            static_image_mode=False, max_num_hands=1, model_complexity=1,
            min_detection_confidence=args.min_conf,
            min_tracking_confidence=args.min_conf)

        frames_xy, frames_conf = [], []
        n_det = n_rescued = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = prep(frame)

            got = run_detector(hands_track, frame, best_rot, W, H, use_clahe)
            if got is None and args.retry_rotations and others:
                for deg in others:
                    got = run_detector(hands_static, frame, deg, W, H, use_clahe)
                    if got is not None:
                        n_rescued += 1
                        break

            if got is not None:
                xy, score = got
                xy = post(xy)          # undistorted -> original pixel coords
                conf = np.full(21, score)
                n_det += 1
            else:
                xy = np.full((21, 2), np.nan)
                conf = np.zeros(21)
            frames_xy.append(xy)
            frames_conf.append(conf)

        cap.release()
        hands_track.close()

        out_path = os.path.join(args.out, f"{stem}.npz")
        np.savez(out_path, xy=np.array(frames_xy), conf=np.array(frames_conf),
                 width=W, height=H, rotation=best_rot, clahe=use_clahe)
        total = len(frames_xy)
        extra = f", {n_rescued} rescued by alt rotation" if n_rescued else ""
        print(f"{stem}: {total} frames, hand found in {n_det} "
              f"({100 * n_det / max(total, 1):.1f}%){extra} -> {out_path}")

    hands_static.close()


def cmd_write_h5(args):
    import pandas as pd

    os.makedirs(args.out, exist_ok=True)
    npzs = sorted(glob(os.path.join(args.detections_dir, "*.npz")))
    if not npzs:
        sys.exit(f"No .npz files found in {args.detections_dir}")

    for path in npzs:
        data = np.load(path)
        xy = data["xy"]          # (n, 21, 2)
        conf = data["conf"]      # (n, 21)
        n = xy.shape[0]

        columns = pd.MultiIndex.from_product(
            [[SCORER], BODYPARTS, ["x", "y", "likelihood"]],
            names=["scorer", "bodyparts", "coords"],
        )
        arr = np.empty((n, len(BODYPARTS) * 3))
        for k in range(len(BODYPARTS)):
            arr[:, 3 * k] = xy[:, k, 0]
            arr[:, 3 * k + 1] = xy[:, k, 1]
            arr[:, 3 * k + 2] = conf[:, k]

        df = pd.DataFrame(arr, columns=columns)
        stem = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(args.out, f"{stem}.h5")
        df.to_hdf(out_path, key="df_with_missing", format="table", mode="w")
        print(f"{stem}: wrote {out_path} ({n} frames)")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("detect", help="run MediaPipe on videos (modern env)")
    d.add_argument("videos_dir")
    d.add_argument("--out", default="detections_np")
    d.add_argument("--min-conf", type=float, default=0.3)
    d.add_argument("--calib", default=None,
                   help="calibration.toml (or multi_camera_calib.npz); enables "
                        "undistortion before detection, with keypoints mapped "
                        "back to original pixel coords")
    d.add_argument("--undistort-alpha", type=float, default=1.0,
                   help="0 = crop to valid pixels (hand appears larger), "
                        "1 = keep full field of view (default)")
    d.add_argument("--clahe", default="off", choices=["off", "on", "auto"],
                   help="CLAHE contrast enhancement; 'auto' probes both per camera")
    d.add_argument("--rotate", default="0",
                   choices=["0", "90", "180", "270", "auto"],
                   help="rotate frames before detection; 'auto' probes all four")
    d.add_argument("--probe-frames", type=int, default=30,
                   help="frames sampled per video when --rotate auto")
    d.add_argument("--retry-rotations", action="store_true",
                   help="on a failed frame, try the other rotations too (slower)")
    d.set_defaults(func=cmd_detect)

    w = sub.add_parser("write-h5", help="write DLC-format h5 (anipose venv)")
    w.add_argument("detections_dir")
    w.add_argument("--out", required=True,
                   help="session pose-2d folder, e.g. 2026-07-27_13/pose-2d")
    w.set_defaults(func=cmd_write_h5)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()