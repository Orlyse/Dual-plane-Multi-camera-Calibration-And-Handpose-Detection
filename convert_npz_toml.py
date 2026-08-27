"""
Convert multi_camera_calib.npz (from multicam_calibrate.py) into an
anipose-style calibration.toml.

Mapping:
  K{i}          -> [cam_N] matrix        (3x3 nested list)
  D{i}          -> [cam_N] distortions   (flattened list, standard 5-coeff)
  R{i} (3x3)    -> [cam_N] rotation      (Rodrigues 3-vector, world->cam)
  T{i}          -> [cam_N] translation   (world->cam, cam0 = world origin)
  cam0          -> rotation [0,0,0], translation [0,0,0]

Anipose notes baked in:
  * fisheye = false  (this pipeline uses OpenCV's standard distortion model)
  * `name` per camera MUST match the group your config.toml `cam_regex`
    extracts from your video filenames (e.g. videos named ...-camA.avi with
    cam_regex = 'cam([A-Z])$' need names A, B, C, D)
  * translation units in the toml define the units of triangulated output;
    the npz stores meters. Use --mm to emit millimeters instead.

Usage:
  python npz_to_anipose_toml.py multi_camera_calib.npz calibration.toml \
      --names A B C D --size 1920 1200 [--mm] [--error 0.3]
"""

import argparse
import cv2
import numpy as np
import time
from datetime import datetime
import os



def fmt_list(vals):
    return "[ " + ", ".join(repr(float(v)) for v in vals) + ",]"


def fmt_matrix(M):
    rows = ", ".join(fmt_list(row) for row in M)
    return "[ " + rows + ",]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("out_toml")
    ap.add_argument("--names", nargs="+", default=None,
                    help="camera names matching cam_regex groups (default A B C ...)")
    ap.add_argument("--size", nargs=2, type=int, default=[1920, 1200],
                    metavar=("W", "H"), help="image size as WIDTH HEIGHT")
    ap.add_argument("--mm", action="store_true",
                    help="write translations in millimeters (default: meters as stored)")
    ap.add_argument("--error", type=float, default=0.0,
                    help="reprojection error to record in metadata")
    args = ap.parse_args()

    data = np.load(args.npz)

    # count cameras by K keys
    n_cams = 0
    while f"K{n_cams}" in data:
        n_cams += 1
    if n_cams == 0:
        raise SystemExit("No K0 found in npz -- wrong file?")

    names = args.names or [chr(ord("A") + i) for i in range(n_cams)]
    if len(names) != n_cams:
        raise SystemExit(f"{n_cams} cameras in npz but {len(names)} names given")

    scale = 1000.0 if args.mm else 1.0

    blocks = []
    for c in range(n_cams):
        K = data[f"K{c}"]
        D = np.asarray(data[f"D{c}"]).flatten()
        if c == 0:
            rvec = np.zeros(3)
            tvec = np.zeros(3)
        else:
            R = data[f"R{c}"]
            rvec = cv2.Rodrigues(R)[0].flatten()
            tvec = np.asarray(data[f"T{c}"]).flatten() * scale

        block = (
            f"[cam_{c}]\n"
            f'name = "{names[c]}"\n'
            f"size = {fmt_list(args.size)}\n"
            f"matrix = {fmt_matrix(K)}\n"
            f"distortions = {fmt_list(D)}\n"
            f"rotation = {fmt_list(rvec)}\n"
            f"translation = {fmt_list(tvec)}\n"
            f"fisheye = false\n"
        )
        blocks.append(block)

    meta = (
        "[metadata]\n"
        "adjusted = false\n"
        f"error = {args.error}\n"
    )   

    # create output dir
    now = datetime.now()
    # outdir = f"anipose/{str(now.date())}_{str(now.hour)}/calibration/"
    outdir = args.out_toml

    assert len(outdir.split('/')) == 3, (f"Wrong path input for out, remove / at the end")
    os.makedirs(outdir, exist_ok=True)
    file_path = outdir + "/calibration.toml"

    with open(file_path, "w") as f:
        f.write("\n".join(blocks) + "\n" + meta)

    unit = "mm" if args.mm else "m"
    print(f"Wrote {file_path}: {n_cams} cameras, names {names}, "
          f"translations in {unit}")
    for c in range(1, n_cams):
        T = np.asarray(data[f"T{c}"]).flatten()
        print(f"  cam{c} baseline from cam0: {np.linalg.norm(T)*1000:.1f} mm")


if __name__ == "__main__":
    main()