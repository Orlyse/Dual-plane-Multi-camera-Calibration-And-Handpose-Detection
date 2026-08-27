# Multi-Camera Hand Pose Calibration Pipeline

A 4-camera calibration and triangulation pipeline built to support high-accuracy 3D hand pose
estimation. The pipeline combines a custom OpenCV/ChArUco
calibration stage with WiLoR 2D hand keypoint detection and Anipose 3D triangulation, achieving
triangulation error as low as **3.4px** and cross-camera reprojection error under **5px** on the
target 4-camera rig.

The pipeline is modular: with minor changes to camera indexing, it generalizes to rigs with more
than 4 cameras.

---

## Table of Contents

1. [Hardware Overview](#hardware-overview)
2. [Repository Structure](#repository-structure)
3. [Environment Setup](#environment-setup)
4. [Pipeline Overview](#pipeline-overview)
5. [Step-by-Step Usage](#step-by-step-usage)
6. [Output File Reference](#output-file-reference)
7. [Anipose Configuration Reference](#anipose-configuration-reference)
8. [Troubleshooting](#troubleshooting)
9. [References](#references)

---

## Hardware Overview

- **4x FLIR GigE camera** rig (BFS-PGE-23S3C-C PoE GigE Blackfly S, Color Camera), each connected via Ethernet to a
  **PoE switch** (NETGEAR Ultra60 MS510TXUP 8-Port Multi-Gigabit PoE++ Compliant Managed Switch with SFP+), which uplinks to the host machine.
- **Host machine**: laptop with a **Thunderbolt-to-10GbE adapter** (e.g. StarTech TB310G2) bridging
  the camera switch network to the machine's PCIe bus, since most laptops lack a native 10GbE port.
- **ChArUco calibration board**: see `board parameters` below — must be identical across every
  script that touches calibration or detection.

**Board parameters used throughout this repo** (must match physically and in every script):
```
Aruco dictionary : DICT_4X4_50
Squares X / Y    : 10 x 8
Checker size     : 22 mm  (0.022 m)
Marker size      : 16 mm  (0.016 m)
```
If you use a different physical board, update these constants in **every** script that defines
them (`intrinsic_calibrate.py`, `triangulate4.py`, `triangulation4_verification.py`, and `config.toml` in the anipose folder).

> ⚠️ **Known hardware failure mode:** Thunderbolt-to-Ethernet adapters can silently drop off the
> PCIe bus under sustained multi-camera streaming load due to aggressive PCIe power management
> (ASPM). This presents as all cameras dying simultaneously mid-capture with a
> `Spinnaker: Stream has been aborted. [-1012]` error. See
> [Troubleshooting → Cameras drop out mid-capture](#cameras-drop-out-mid-capture-thunderboltpcie).

---

## Repository Structure

```
main/
├── capture_video.py          # Records synchronized video from all 4 cameras
├── extract_video.py          # Extracts per-camera frame images from recorded video
├── intrinsic_calibrate.py    # Computes per-camera K (camera matrix) and distortion coeffs
├── triangulate4.py           # Computes extrinsics (R, T) of each camera relative to cam0
├── wilor_to_pose2d.py        # Runs WiLoR hand pose detection, writes Anipose-compatible pose-2d
│
├── data/
│   ├── intri_data/
│   │   ├── videos/<DATE_HOUR>/take0-camXX.mp4
│   │   ├── images/<DATE_HOUR>/take0-camXX/<NNNNNN>.jpg
│   │   └── params/<DATE_HOUR>/intrinsic_paramstake0-camXX.yaml
│   │
│   └── extri_data/
│       ├── videos/<DATE_HOUR>/take0-camXX.mp4
│       ├── images/<DATE_HOUR>/take0-camXX/<NNNNNN>.jpg
│       └── params/<DATE_HOUR>/
│           ├── multi_camera_calib.npz
│           ├── calibration.toml          # Anipose/aniposelib-format calibration
│           ├── per_frame_results.csv
│           ├── spacing_error_per_frame.png
│           ├── spacing_histogram.png
│           └── rig_3d.png
│
└── anipose/
    └── config.toml
    └── <DATE_HOUR>/
        ├── calibration
        |   ├── calibration.toml             # Calibration file generated from convert_npz_toml.py
        ├── videos-raw/take0-camXX.mp4       # Hand capture videos go here (input to Anipose)
        ├── images-raw/take0-camXX           # Folders generated from running extract_video.py using handpose command
        ├── pose-2d/
        │   ├── take0-camXX.h5               # WiLoR 2D keypoints, DeepLabCut-style format
        │   ├── detection_log.csv            # Per-frame, per-camera hand-detected yes/no
        │   └── keypoint_visibility_log.csv  # Per-frame, per-keypoint camera coverage count
        ├── pose2d-filtered/take0-camXX.h5   # ANipose filter output
        ├── pose-3d/                         # Anipose triangulation output (created by anipose CLI)
        ├── videos-labeled/                  # Anipose 2D-overlay videos (created by anipose CLI)
        ├── videos-3d/take0.mp4              # 3d video of only the hand from anipose label-3d
        └── videos-combined/take0.mp4
        
```

`<DATE_HOUR>` is the auto-generated folder name `YYYY-MM-DD_HH` used to keep separate capture
sessions from colliding.

> **Note:** `intrinsic_calibrate.py` output and `triangulate4.py` output are read back in by
> filename-sorted order (`sorted(os.listdir(...))`). Camera folders/files must sort into the same
> order as your physical camera indices (cam0, cam1, cam2, cam3) — using a consistent
> `take0-cam01`, `take0-cam02`, ... naming convention is what guarantees this.

---

## Environment Setup

This project uses **three separate conda environments**. They exist as separate environments
because their dependencies conflict — WiLoR needs a modern PyTorch/CUDA stack, Anipose's
tooling (aniposelib, DeepLabCut) is pinned to Python 3.7 and very old NumPy/TensorFlow versions,
and the camera-capture stack needs whatever Python version FLIR's Spinnaker SDK build supports.
**Do not try to merge these into one environment** — the pins conflict directly (e.g. `numpy==2.2.6`
vs `numpy==1.21.6`).
 
Exact, validated package lists for each environment are provided as
`requirements_pyspin.txt`, `requirements_wilormini.txt`, and `requirements_anipose.txt` in
this repo; use them for a reproducible setup rather than installing loose/latest versions.

### 1. `test_pyspin` — camera capture, image extraction, calibration
Used for: `capture_video.py`, `extract_video.py`, `intrinsic_calibrate.py`, `triangulate4.py`.
 
```bash
conda create -n test_pyspin python=3.10
conda activate test_pyspin
 
# Spinnaker SDK + PySpin must be installed FIRST, manually, from FLIR's SDK
# download for your camera model and OS (not available on PyPI as a normal
# package) — install the SDK, then pip install the wheel it provides.
# Only after that:
pip install -r requirements_pyspin.txt
```

This environment intentionally includes `mediapipe` alongside `opencv-contrib-python`, in case you
want to benchmark alternate 2D hand-pose detectors against WiLoR.

### 2. `wilor_mini` — WiLoR 2D hand keypoint detection
 
Used for: `wilor_to_pose2d.py`, wilor_mini_detector.py.
 
```bash
conda create -n wilor_mini python=3.10
conda activate wilor_mini
pip install -r requirements-wilor_mini.txt
```
 
Requires an NVIDIA GPU with a CUDA 12.4-compatible driver (the `nvidia-*` package pins assume
this). If you don't have a GPU available, WiLoR will fall back to CPU, but detection will be
substantially slower.
 
> Note: this environment uses plain `opencv-python`, not `opencv-contrib-python` — it never calls
> `cv2.aruco`, so the contrib build isn't needed here. Don't "fix" this by adding
> `opencv-contrib-python` on top of it; that reintroduces the same multi-OpenCV shadowing conflict
> described below for the `anipose` environment.
 
### 3. `anipose` — extrinsic bundle adjustment, 3D triangulation, Anipose CLI
 
Used for:  Used for the`anipose` CLI itself for the commands (`anipose filter`, `anipose label-2d/anipose label-2d-filter`, `anipose triangulate`, `anipose label-3d`, `anipose label-combined` etc). 
Additional information about anipose commnands can be found on this link: https://anipose.readthedocs.io/en/latest/tutorial.html (**Note:** Certain commands are not used as different methods were
used to generate their outputs; further description is provided below.)
Also includes DeepLabCut as an optional alternate 2D
detector, if you want to compare it against WiLoR.
 
```bash
conda create -n anipose python=3.7
conda activate anipose
pip install -r requirements-anipose.txt
```
 
**Why the OpenCV version is pinned:** `aniposelib` (as of `0.7.2`) still calls a legacy OpenCV
ArUco API (`cv2.aruco.estimatePoseCharucoBoard`, among others) that was removed in modern OpenCV.
This repo works around that incompatibility with a small monkey-patch inside `triangulate4.py`
(see comments in that file) and by using this repo's own `detect_corners()` function instead of
aniposelib's internal detector. `opencv-contrib-python==5.0.0.93` is the version this pipeline was
validated against alongside `numpy==1.21.6` on Python 3.7 — newer OpenCV releases may remove even
more of the legacy API aniposelib depends on internally, and installing any other `opencv-python`/
`opencv-python-headless` package in this same environment will shadow the working `aruco` module
(see Troubleshooting).

**Verify your environment after setup:**
```bash
python3 -c "import cv2; print(cv2.__version__); print(hasattr(cv2.aruco, 'CharucoDetector'))"
python3 -c "import numpy; print(numpy.__version__)"
anipose --help
```
The `hasattr` check should print `True`.

---

## Pipeline Overview

```
 [1] Intrinsic Calibration        [2] Extrinsic Calibration         [3] Hand Capture
 (per-camera lens model)    -->   (camera positions/rig geometry) --> (actual data collection)
        |                                    |                              |
        v                                    v                              v
   K, dist per camera          R, T per camera relative to cam0      raw video, 4 cameras
                                    (calibration.toml)                       |
                                                                              v
                                                                  [4] WiLoR 2D Detection
                                                                  (21 hand keypoints/frame,
                                                                   per camera, .h5 files)
                                                                              |
                                                                              v
                                                                  [5] Anipose Triangulation
                                                                  (combines calibration.toml
                                                                   + pose-2d .h5 -> 3D pose)
```

Each stage's output feeds the next. Steps 1 and 2 only need to be redone when the physical rig
(camera positions or lenses) changes — the same calibration can be reused across many hand
capture sessions.

---

## Step-by-Step Usage

### Step 1 — Intrinsic Calibration

Determines each camera's own lens characteristics (focal length, principal point, distortion) —
independent of the other cameras.

**1a. Capture calibration video** (environment: `test_pyspin`)
```bash
conda activate test_pyspin
python3 capture_video.py record intrinsic
```
Move the ChArUco board slowly through each camera's field of view individually — each camera
needs many (dozens+) clear, varied views of the board (different angles, positions, distances).
Cameras do **not** need to see the board simultaneously for this step.

Output: `data/intri_data/videos/<DATE_HOUR>/take0-camXX.mp4`

**1b. Extract frames**
```bash
python3 extract_video.py calibrate data/intri_data/videos/<DATE_HOUR>
```
Output: `data/intri_data/images/<DATE_HOUR>/take0-camXX/<NNNNNN>.jpg`

**1c. Run intrinsic calibration**
```bash
python3 intrinsic_calibrate.py data/intri_data/images/<DATE_HOUR>
```
This detects ChArUco corners in every image, fits camera intrinsics, then iteratively removes
high-reprojection-error views (default threshold: 0.5px, minimum view floor: see script) and
refits — improving calibration quality without overfitting to too few remaining views.

Output: `data/intri_data/params/<DATE_HOUR>/intrinsic_paramstake0-camXX.yaml` (contains `rms`,
`K`, `dist` per camera)

**Sanity check:** final RMS reprojection error per camera should typically land under ~0.5px
after outlier removal. If a camera's RMS stays high or its view count collapses far below the
others after filtering, that camera's raw calibration data was likely poor. Recapture calibration
video.

---

### Step 2 — Extrinsic Calibration (Triangulation)

Determines each camera's position and orientation relative to camera 0 (the rig's world origin).

**2a. Capture extrinsic video** (environment: `test_pyspin`)
```bash
python3 capture_video.py record extrinsic
```
This time, the board must be seen by **at least 2 (3 if possible) cameras simultaneously** in as many frames as
possible, ideally get direct co-visibility between *every* camera pair, not just adjacent ones,
since weak or missing pairwise coverage is the single biggest source of extrinsic error.

Output: `data/extri_data/videos/<DATE_HOUR>/take0-camXX.mp4`

**2b. Extract frames**
```bash
python3 extract_video.py anipose data/extri_data/videos/<DATE_HOUR>
```
Output: `data/extri_data/images/<DATE_HOUR>/take0-camXX/<NNNNNN>.jpg`

**2c. Run extrinsic calibration** (environment: `anipose`)
```bash
python3 triangulate4.py \
    data/extri_data/images/<DATE_HOUR>/ \
    data/intri_data/params/<INTRINSIC_DATE_HOUR>/ \
    --out_dir data/extri_data/params/<DATE_HOUR>
```
This script:
- Loads each camera's intrinsics from Step 1.
- Detects the ChArUco board in every extrinsic-capture image using this repo's own detector
  (not aniposelib's — see [Troubleshooting](#aniposelib-detects-nothing-empty-corners)).
- Validates the result via independent multi-view triangulation of ChArUco corner spacing
  (should triangulate to ~22mm between physically adjacent corners).

Output: `data/extri_data/params/<DATE_HOUR>/multi_camera_calib.npz`,
`calibration.toml`, `per_frame_results.csv`, `spacing_error_per_frame.png`,
`spacing_histogram.png`, `rig_3d.png`

**Sanity check:** open `spacing_histogram.png` — the distribution should peak tightly around
22mm. A histogram centered far from 22mm. Also check `rig_3d.png`
visually: camera positions/orientations should match your physical rig layout.

---

### Step 3 — Hand Capture

Capture the actual behavioral data.

```bash
conda activate pyspin
python3 capture_video.py record handpose
```
Position the hand roughly centered in the rig's shared field of view. Move naturally through the
hand poses/gestures you want captured.

Output: `anipose/<DATE_HOUR>/videos-raw/take0-camXX/<NNNNNN>.mp4`

Extract image frames from the handpose videos
```bash
conda activate pyspin
python3 extract_video.py handpose anipose/<DATE_HOUR>/videos-raw
```

---

### Step 4 — WiLoR 2D Hand Pose Detection

Runs WiLoR on every synchronized frame across all 4 cameras and writes DeepLabCut-style `.h5`
files that Anipose's triangulation step reads directly.

```bash
conda activate anipose
python3 wilor_to_pose2d.py \
    --img_root anipose/<DATE_HOUR>/images-raw \
    --out_dir anipose/<DATE_HOUR>/pose-2d
```

Output per camera: `anipose/<DATE_HOUR>/pose-2d/take0-camXX.h5`, plus two diagnostic logs:
- `detection_log.csv` — per-frame, per-camera, whether a hand was detected at all.
- `keypoint_visibility_log.csv` — per-frame, per-keypoint, how many cameras' predicted keypoint
  location fell inside WiLoR's own detected hand bounding box (a proxy for per-keypoint
  reliability, since WiLoR does not natively output per-keypoint confidence).

**Sanity check:** review the printed per-camera detection rate and the coverage histogram in the
console output. A camera with a chronically low detection rate relative to the others may indicate
a lighting, occlusion, or camera-placement issue worth addressing before relying on that camera's
data.

---

### Step 5 — Anipose 3D Triangulation
**Note:** Anipose requires a calibration file which can be obtained by running convert_npz_toml.py in the pyspin environment.
The last argument is the camera names that need to match what is in config.toml
```bash
conda activate pyspin
python3 convert_npz_toml.py data/extri_data/params/<DATE_HOUR>/multi_camera_calib.npz \
anipose/<DATE_HOUR>/calibration \
cam01 cam02 cam03 cam04
```

Set up an Anipose project `config.toml` (see
[Anipose Configuration Reference](#anipose-configuration-reference) below) in the `anipose/`
project root.

```bash
conda activate anipose
cd anipose/

anipose filter
anipose label-2d
anipose triangulate
anipose label-3d
anipose label-combined
anipose angles
```

Output: Similar to the outputs detailed in the anipose tutorial (https://anipose.readthedocs.io/en/latest/tutorial.html)

---

## Output File Reference

| File | Produced by | Contents |
|---|---|---|
| `intrinsic_paramstake0-camXX.yaml` | `intrinsic_calibrate.py` | `rms`, `K` (3x3 camera matrix), `dist` (distortion coefficients) |
| `multi_camera_calib.npz` | `triangulate4.py` | `K{c}`, `D{c}`, `R{c}`, `T{c}` per camera, cam0 = origin |
| `calibration.toml` | `triangulate4.py` (via aniposelib `CameraGroup.dump`) | Same extrinsic/intrinsic data in Anipose-native format |
| `per_frame_results.csv` | `triangulate4.py` | Per-frame triangulation validation: cameras used, spacing error (mm) |
| `take0-camXX.h5` | `wilor_to_pose2d.py` | DeepLabCut-format 2D keypoints (x, y, likelihood) x 21 joints x N frames |
| `detection_log.csv` | `wilor_to_pose2d.py` | Per-frame/camera hand-detected yes/no |
| `keypoint_visibility_log.csv` | `wilor_to_pose2d.py` | Per-frame/keypoint cross-camera visibility count |
| `pose-3d/*.csv` | `anipose triangulate` | Final 3D keypoint trajectories + reprojection error per point |

---

## Anipose Configuration Reference

Minimal `config.toml` for this project (place in the Anipose project root):

```toml
project = 'hand_pose_project'
model_folder = '.'
nesting = 0
video_extension = 'mp4'

[calibration]
board_type = "charuco"
board_size = [10, 8]
board_marker_bits = 4
board_marker_dict_number = 50
board_marker_length = 16     # mm
board_square_side_length = 22  # mm
fisheye = false

[triangulation]
triangulate = true
cam_regex = 'cam(\d+)'       # must match your calibration.toml camera names AND video filenames
ransac = true                # robust triangulation: discards outlier camera observations per-point
optim = true                 # enables bone-length/smoothness-constrained refinement
reproj_error_threshold = 3   # px; tune down toward 2 for tighter accuracy requirements
score_threshold = 0.6
scale_smooth = 4
scale_length = 4
constraints = [
    ["thumb_CMC", "thumb_MCP"], ["thumb_MCP", "thumb_DIP"], ["thumb_DIP", "thumb_tip"],
    ["index_MCP", "index_PIP"], ["index_PIP", "index_DIP"], ["index_DIP", "index_tip"],
    ["middle_MCP", "middle_PIP"], ["middle_PIP", "middle_DIP"], ["middle_DIP", "middle_tip"],
    ["ring_MCP", "ring_PIP"], ["ring_PIP", "ring_DIP"], ["ring_DIP", "ring_tip"],
    ["pinky_MCP", "pinky_PIP"], ["pinky_PIP", "pinky_DIP"], ["pinky_DIP", "pinky_tip"],
]

[filter]
enabled = true
medfilt = 9
offset_threshold = 25
score_threshold = 0.6
spline = true
```

> ⚠️ **Camera naming must match across three places**: (1) the `name=` field used when building
> `Camera()` objects in `triangulate4.py`, (2) the resulting `calibration.toml`, and (3) the
> `cam_regex` pattern's expectations against your actual video filenames (e.g.
> `take0-cam01.mp4`). A mismatch here causes Anipose to silently fail to associate calibration
> data with the correct camera's 2D pose data.

---

## Troubleshooting

### Cameras drop out mid-capture (Thunderbolt/PCIe)
Symptom: all 4 cameras die simultaneously mid-stream with
`Spinnaker: Stream has been aborted. [-1012]`, and `dmesg` shows `BadDLLP`, `AER:
Uncorrectable (Fatal)`, or `Unable to change power state from D0 to D3hot` around the same
timestamp.

This is a PCIe power-management issue on Thunderbolt-to-Ethernet adapters, **not** a bug in the
capture script. Fixes, in order of effectiveness:
1. Disable PCIe ASPM at the kernel level: add `pcie_aspm=off` to `GRUB_CMDLINE_LINUX_DEFAULT` in
   `/etc/default/grub`, then `sudo update-grub && sudo reboot`.
2. Force `power/control=on` for the Thunderbolt bridge and NIC devices via a udev rule (see
   `docs/thunderbolt-power-fix.md` if included, or search this repo's issue history) — this
   survives reboots, unlike a one-off `echo on | sudo tee .../power/control`.
3. Use a short, certified Thunderbolt 3/4 cable — signal integrity issues on uncertified or
   overly long cables can cause the same symptom independent of power management.
4. Check adapter temperature — sustained 4-camera GigE throughput can thermally stress
   USB/Thunderbolt-to-Ethernet adapters.

### `AttributeError: module 'cv2.aruco' has no attribute ...`
Multiple OpenCV variants (`opencv-python`, `opencv-contrib-python`, `opencv-python-headless`)
installed simultaneously in the same environment. Only `opencv-contrib-python` includes the
`aruco` module. Fix:
```bash
pip uninstall opencv-python opencv-contrib-python opencv-python-headless -y
pip install opencv-contrib-python   # pin a version per the anipose env note above, if needed
```

### aniposelib detects nothing (empty corners)
`aniposelib`'s own internal ChArUco detector uses a legacy OpenCV ArUco API
(`cv2.aruco.estimatePoseCharucoBoard`, old-style `cv2.aruco.interpolateCornersCharuco`, etc.) that
was removed in modern OpenCV. Symptoms include silently empty detections (`corners: array([])`
for every frame) or an `AttributeError` deep inside `aniposelib/boards.py`.

This repo works around it two ways, both already implemented in `triangulate4.py`:
- Board/corner **detection** is done with this repo's own `detect_corners()` (using
  `cv2.aruco.CharucoDetector`, the modern API) instead of calling `ani_board.detect_images()`.
  Results are reformatted into the row structure aniposelib's `calibrate_rows()` expects.
- Internal **pose estimation** calls are patched via a small monkey-patch
  (`cv2.aruco.estimatePoseCharucoBoard = _compat_estimatePoseCharucoBoard`) that reimplements the
  removed function using `solvePnP` against the board's own object points.

If you upgrade `aniposelib` to a version with native modern-OpenCV support in the future, these
workarounds can likely be removed.

### Units mismatch — triangulated spacing histogram centered far from 22mm
This repo's original triangulation code (`pose_from_corners`, `triangulate_multiview`) works in
**meters**. `aniposelib`'s `CharucoBoard` is constructed in **millimeters**
(`square_length=CHECKER_SIZE*1000`). If you pull `R`/`T` out of a `CameraGroup` after bundle
adjustment, remember to divide translations by 1000 before feeding them into any of this repo's
own meter-based functions.

### Charuco pose flips / a specific camera pair's baseline spikes unpredictably
Symptom: a per-pair baseline plot (distance between two rigidly-mounted cameras, which should be
constant) shows sharp, narrow dips or spikes rather than a flat line. This is the classic
IPPE planar-pose ambiguity — a near-fronto-parallel ChArUco view has two geometrically valid pose
solutions, and picking the wrong one corrupts that frame's contribution.

Fix: always compute board pose via `cv2.solvePnPGeneric(..., flags=cv2.SOLVEPNP_IPPE)` (which
returns *all* candidate solutions) rather than `cv2.solvePnP` (which silently returns only one),
and reject any frame where the best and second-best solutions have comparably low reprojection
error (see `pose_from_corners()`'s `ambiguity_ratio` parameter).

### One camera's images silently vanish during frame extraction
`extract_video.py` writes frames per camera into `image_out/<video_index>/`. If output images
overwrite each other (fewer images on disk than expected), confirm `os.makedirs()` is creating
the actual per-camera subdirectory, not just the parent output folder, and that frame numbers are
zero-padded (unpadded frame numbers sort lexicographically, not numerically, which can silently
scramble frame order across cameras).

---

## References

- [Kalibr](https://github.com/ethz-asl/kalibr) — multi-camera calibration toolbox (methodological reference)
- [EasyMocap](https://github.com/zju3dv/EasyMoCap) — multi-view human motion capture (methodological reference)
- [WiLoR](https://github.com/rolpotamias/WiLoR) — real-time 3D hand pose estimation model used for 2D keypoint detection
- [Anipose](https://anipose.readthedocs.io/) — markerless 3D pose estimation pipeline built on aniposelib
- [aniposelib](https://github.com/lambdaloop/aniposelib) — calibration/triangulation library underlying Anipose
- [OpenCV ArUco/ChArUco module documentation](https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html)
