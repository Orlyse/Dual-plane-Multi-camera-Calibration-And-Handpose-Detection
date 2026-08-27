"""
Diagnose an anipose pose-3d CSV, with plots.

No assumptions about anatomy: bone lengths are reported as measured
distributions (median, spread, coefficient of variation) and ranked by
consistency rather than checked against guessed ranges. A rigid hand gives
small CV regardless of what the true lengths are, so CV is the useful signal.

Outputs (into --out-dir):
  coverage.png        cameras per joint-frame, and per-joint measured fraction
  error.png           error distribution, split by ncams, per joint, over time
  error_vs_speed.png  scatter + binned medians (timing/blur vs bias)
  bones.png           per-bone length distributions and consistency ranking
  skeleton_3d.png     best-covered frame drawn in 3D
  summary.txt         the console output, saved

Usage:
  python triangulation_check_anipose.py path/to/pose-3d/take0.csv --fps 10 --fx 837
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib

# FINGERS = [1, 2, 3, 4, 5]
# JOINTS = ["base"] + [f"{p}{f}" for f in FINGERS
#                      for p in ("MCP", "PIP", "DIP", "tip")]

# # hand skeleton: palm link plus each finger chain
# BONES = []
# for _f in FINGERS:
#     BONES.append(("base", f"MCP{_f}"))
#     BONES.append((f"MCP{_f}", f"PIP{_f}"))
#     BONES.append((f"PIP{_f}", f"DIP{_f}"))
#     BONES.append((f"DIP{_f}", f"tip{_f}"))
FINGER_NAMES = ["thumb", "index", "middle", "ring", "pinky"]

def chain(f):
    """Joint names for one finger, proximal to distal."""
    if f == "thumb":
        return [f"{f}_CMC", f"{f}_MCP", f"{f}_DIP", f"{f}_tip"]
    return [f"{f}_MCP", f"{f}_PIP", f"{f}_DIP", f"{f}_tip"]

JOINTS = ["base"]
for _f in FINGER_NAMES:
    JOINTS += chain(_f)

BONES = []
for _f in FINGER_NAMES:
    c = chain(_f)
    BONES.append(("base", c[0]))
    BONES += list(zip(c[:-1], c[1:]))
def load(csv):
    df = pd.read_csv(csv)

    # labeling check
    joints = [j for j in JOINTS if f"{j}_x" in df.columns]
    missing = [j for j in JOINTS if f"{j}_x" not in df.columns]
    if missing:
        print(f"WARNING: {len(missing)} expected joints not in CSV: {missing}")

    joints = [j for j in JOINTS if f"{j}_x" in df.columns]
    n = len(df)
    X = np.stack([df[[f"{j}_x", f"{j}_y", f"{j}_z"]].values for j in joints],
                 axis=1)
    err = df[[f"{j}_error" for j in joints]].values.astype(float)
    ncams = df[[f"{j}_ncams" for j in joints]].values.astype(float)
    return df, joints, n, X, err, ncams


def nanmedian_safe(a, axis=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(a, axis=axis)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--fx", type=float, default=837.0,
                    help="focal length px, only for the px->mm annotation")
    ap.add_argument("--depth", type=float, default=0.8,
                    help="typical hand distance in m, for the px->mm annotation")
    ap.add_argument("--fps", type=float, default=20.0)
    ap.add_argument("--out-dir", default=None,
                    help="default: <csv without extension>_analysis/")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = args.out_dir or os.path.splitext(args.csv)[0] + "_analysis"
    os.makedirs(out_dir, exist_ok=True)

    df, joints, n, X, err, ncams = load(args.csv)
    measured = ncams >= 2
    err_m = np.where(measured, err, np.nan)
    lines = []

    def say(s=""):
        print(s)
        lines.append(s)

    say(f"{args.csv}: {n} frames, {len(joints)} joints")
    say(f"output -> {out_dir}/")

    # ---------------- coverage ----------------
    say("\n=== coverage ===")
    kmax = int(np.nanmax(ncams))
    counts = {k: int((ncams == k).sum()) for k in range(kmax + 1)}
    for k, c in counts.items():
        say(f"  {k} cameras: {c:6d} joint-frames ({100*c/ncams.size:5.1f}%)")
    say(f"  measured (>=2): {int(measured.sum())} / {ncams.size} "
        f"({100*measured.mean():.1f}%)")
    full = (ncams >= 2).all(axis=1).mean()
    say(f"  frames with all {len(joints)} joints measured: {100*full:.1f}%")

    per_joint_cov = measured.mean(axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    palette = ["#c44", "#c84", "#8a4", "#48a", "#24c"]
    axes[0].bar(list(counts.keys()), list(counts.values()),
                color=palette[:kmax + 1])
    axes[0].set_xlabel("cameras contributing")
    axes[0].set_ylabel("joint-frames")
    axes[0].set_title("Coverage: cameras per triangulated point")
    axes[0].grid(alpha=0.3, axis="y")

    order = np.argsort(per_joint_cov)
    axes[1].barh([joints[i] for i in order],
                 [100 * per_joint_cov[i] for i in order], color="#48a")
    axes[1].set_xlabel("% of frames measured (>=2 cams)")
    axes[1].set_title("Per-joint coverage (worst at bottom)")
    axes[1].grid(alpha=0.3, axis="x")
    axes[1].tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/coverage.png", dpi=150)
    plt.close(fig)

    # ---------------- error ----------------
    say("\n=== reprojection error (measured points) ===")
    flat = err_m[np.isfinite(err_m)]
    for k in range(2, kmax + 1):
        sel = err[(ncams == k) & np.isfinite(err)]
        if sel.size:
            say(f"  ncams={k}: median {np.median(sel):6.2f} px, "
                f"90th {np.percentile(sel, 90):6.2f} px  (n={sel.size})")
    med_all = float(np.median(flat))
    mm = med_all * args.depth * 1000.0 / args.fx
    say(f"  overall: median {med_all:.2f} px, "
        f"90th {np.percentile(flat, 90):.2f} px")
    say(f"           ~= {mm:.1f} mm at {args.depth:.1f} m (fx={args.fx:.0f})")

    per_joint_err = nanmedian_safe(err_m, axis=0)
    frame_err = nanmedian_safe(err_m, axis=1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    ax = axes[0, 0]
    ax.hist(flat, bins=80, range=(0, np.percentile(flat, 99)), color="#48a")
    ax.axvline(med_all, color="r", ls="--", label=f"median {med_all:.1f} px")
    ax.set_xlabel("reprojection error (px)")
    ax.set_ylabel("count")
    ax.set_title("Error distribution, all measured points")
    ax.legend()

    ax = axes[0, 1]
    data, labs = [], []
    for k in range(2, kmax + 1):
        sel = err[(ncams == k) & np.isfinite(err)]
        if sel.size:
            data.append(sel)
            labs.append(f"{k} cams\nn={sel.size}")
    if data:
        ax.boxplot(data, tick_labels=labs, showfliers=False)
    ax.set_ylabel("error (px)")
    ax.set_title("Error by camera count\n(more cameras can reveal more disagreement)")
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1, 0]
    o = np.argsort(per_joint_err)
    ax.barh([joints[i] for i in o], [per_joint_err[i] for i in o], color="#a54")
    ax.set_xlabel("median error (px)")
    ax.set_title("Per-joint error (hardest at bottom)")
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.3, axis="x")

    ax = axes[1, 1]
    im = ax.imshow(err_m.T, aspect="auto", cmap="magma",
                   vmax=np.percentile(flat, 95), interpolation="nearest")
    ax.set_yticks(range(len(joints)))
    ax.set_yticklabels(joints, fontsize=6)
    ax.set_xlabel("frame")
    ax.set_title("Error per joint over time (blank = not measured)")
    fig.colorbar(im, ax=ax, label="px")
    fig.tight_layout()
    fig.savefig(f"{out_dir}/error.png", dpi=150)
    plt.close(fig)

    # ---------------- error vs speed ----------------
    wrist = X[:, joints.index("base"), :]
    speed = np.full(n, np.nan)
    speed[1:] = np.linalg.norm(np.diff(wrist, axis=0), axis=1) * args.fps
    ok = np.isfinite(speed) & np.isfinite(frame_err)

    say("\n=== error vs hand speed ===")
    fig, ax = plt.subplots(figsize=(9, 5))
    if ok.sum() > 10:
        r = float(np.corrcoef(speed[ok], frame_err[ok])[0, 1])
        say(f"  correlation = {r:+.3f} (n={int(ok.sum())} frames)")
        ax.scatter(speed[ok], frame_err[ok], s=8, alpha=0.35, color="#48a",
                   label="frames")
        edges = np.nanpercentile(speed[ok], np.linspace(0, 100, 9))
        cx, cy = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = ok & (speed >= lo) & (speed <= hi)
            if m.sum() > 3:
                cx.append(np.nanmedian(speed[m]))
                cy.append(nanmedian_safe(frame_err[m]))
                say(f"  {lo:6.3f}-{hi:6.3f} m/s: median {cy[-1]:6.2f} px "
                    f"(n={int(m.sum())})")
        ax.plot(cx, cy, "o-", color="#c33", lw=2, label="binned median")
        ax.set_title(f"Error vs hand speed  (r = {r:+.3f})\n"
                     "rising = timing/blur    flat = detection or geometric bias")
        ax.legend()
    else:
        say("  not enough finite frames to correlate")
    ax.set_xlabel("wrist speed (m/s)")
    ax.set_ylabel("frame median error (px)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/error_vs_speed.png", dpi=150)
    plt.close(fig)

    # ---------------- bone lengths, no assumptions ----------------
    say("\n=== bone lengths as measured (ranked by consistency) ===")
    jidx = {j: i for i, j in enumerate(joints)}
    rows, dists, labels = [], [], []
    for a, b in BONES:
        if a not in jidx or b not in jidx:
            continue
        ia, ib = jidx[a], jidx[b]
        both = measured[:, ia] & measured[:, ib]
        L = np.linalg.norm(X[:, ia] - X[:, ib], axis=1) * 1000.0
        L = L[both & np.isfinite(L)]
        if L.size < 10:
            continue
        med, sd = float(np.median(L)), float(np.std(L))
        rows.append((f"{a}->{b}", med, sd, 100 * sd / med, L.size))
        dists.append(L)
        labels.append(f"{a}-{b}")
    rows.sort(key=lambda r: r[3])
    say(f"  {'bone':16s} {'median':>9s} {'std':>8s} {'CV':>7s} {'n':>6s}")
    for name, med, sd, cv, cnt in rows:
        say(f"  {name:16s} {med:7.1f}mm {sd:6.1f}mm {cv:6.1f}% {cnt:6d}")
    if rows:
        cvs = np.array([r[3] for r in rows])
        say(f"  median CV across bones: {np.median(cvs):.1f}%  "
            f"(lower = more rigid, more consistent reconstruction)")

    if rows:
        fig, axes = plt.subplots(2, 1, figsize=(13, 9))
        axes[0].boxplot(dists, tick_labels=labels, showfliers=False)
        axes[0].set_ylabel("length (mm)")
        axes[0].set_title("Measured bone-length distributions (no assumed values)")
        axes[0].tick_params(axis="x", rotation=90, labelsize=7)
        axes[0].grid(alpha=0.3, axis="y")

        axes[1].barh([r[0] for r in rows], [r[3] for r in rows], color="#4a8")
        axes[1].set_xlabel("coefficient of variation (%) = std / median")
        axes[1].set_title("Bone-length consistency (worst at bottom)")
        axes[1].tick_params(labelsize=7)
        axes[1].grid(alpha=0.3, axis="x")
        fig.tight_layout()
        fig.savefig(f"{out_dir}/bones.png", dpi=150)
        plt.close(fig)

    # ---------------- 3D skeleton of the best frame ----------------
    score = measured.sum(axis=1) * 1000.0 - np.nan_to_num(frame_err, nan=1e6)
    best = int(np.argmax(score))
    P = X[best]
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    for a, b in BONES:
        if a in jidx and b in jidx:
            pa, pb = P[jidx[a]], P[jidx[b]]
            if np.all(np.isfinite(pa)) and np.all(np.isfinite(pb)):
                ax.plot(*zip(pa, pb), color="#48a", lw=2)
    good = np.isfinite(P[:, 0])
    ax.scatter(P[good, 0], P[good, 1], P[good, 2], c="#c33", s=25)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title(f"Frame {best}: {int(measured[best].sum())}/{len(joints)} joints "
                 f"measured, median error {frame_err[best]:.1f} px")
    ctr = np.nanmean(P, axis=0)
    rad = float(np.nanmax(np.abs(P - ctr))) * 1.3
    ax.set_xlim(ctr[0] - rad, ctr[0] + rad)
    ax.set_ylim(ctr[1] - rad, ctr[1] + rad)
    ax.set_zlim(ctr[2] - rad, ctr[2] + rad)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/skeleton_3d.png", dpi=150)
    plt.close(fig)
    say(f"\nbest-covered frame: {best}")

    with open(f"{out_dir}/summary.txt", "w") as f:
        f.write("\n".join(lines) + "\n")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()