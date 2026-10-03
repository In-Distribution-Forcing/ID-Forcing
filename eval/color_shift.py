"""Color shift: how far a clip's colors move between its first and its last frame.

    python eval/color_shift.py outputs/self_forcing --out outputs/self_forcing_color.csv
    python eval/color_shift.py outputs/sf_240s --end 1917     # a 4 min clip scored as its first 2 min

For each video, both frames are converted to HSV and the Hue channel is binned into an L1-normalised
180-bin histogram (one bin per 8-bit OpenCV hue value). We report the L1 distance between the two
histograms (0 = same colors, 2 = disjoint) and their Pearson correlation, and over the whole set
ColorShift = 100 * (1 - mean L1 / 2), so higher means less color drift.

"First" and "last" are the first and last decoded frames. With --end N the clip is treated as if
it ended at frame N, so a long clip can be scored over its opening part without cutting files.
"""
import argparse
import csv
import glob
import os
from multiprocessing import Pool

import cv2
import numpy as np

HUE_BINS = 180
HUE_RANGE = (0, 180)


def hue_hist(frame_rgb):
    """L1-normalised 180-bin Hue histogram of an RGB uint8 frame; None if the frame is empty."""
    hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)       # H in [0, 179]
    hist, _ = np.histogram(hsv[..., 0].ravel(), bins=HUE_BINS, range=HUE_RANGE)
    total = hist.sum()
    if total == 0:
        return None
    return hist.astype(np.float64) / total


def first_last(path, end=None):
    """First and last decoded frame (RGB uint8), reading at most `end` frames, and the count read."""
    cap = cv2.VideoCapture(path)
    ok, bgr = cap.read()
    if not ok:
        cap.release()
        return None, None, 0
    first = last = bgr
    n = 1
    while end is None or n < end:
        ok, bgr = cap.read()
        if not ok:
            break
        last = bgr
        n += 1
    cap.release()
    return (cv2.cvtColor(first, cv2.COLOR_BGR2RGB), cv2.cvtColor(last, cv2.COLOR_BGR2RGB), n)


def score(job):
    path, end = job
    first, last, n = first_last(path, end)
    if first is None:
        return os.path.basename(path), None, None, 0
    h0, h1 = hue_hist(first), hue_hist(last)
    if h0 is None or h1 is None:
        return os.path.basename(path), None, None, n
    l1 = float(np.abs(h0 - h1).sum())
    pearson = float("nan") if h0.std() == 0 or h1.std() == 0 else float(np.corrcoef(h0, h1)[0, 1])
    return os.path.basename(path), l1, pearson, n


def main():
    ap = argparse.ArgumentParser(description="Color shift between the first and last frame")
    ap.add_argument("videos_dir")
    ap.add_argument("--glob", default="*.mp4")
    ap.add_argument("--end", type=int, default=0,
                    help="treat each clip as ending at this frame (0: use the whole clip)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--label", default=None)
    ap.add_argument("--out", default=None, help="per-video CSV")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.videos_dir, args.glob)))
    if not paths:
        raise SystemExit(f"no videos match {os.path.join(args.videos_dir, args.glob)}")
    with Pool(args.workers) as pool:
        rows = pool.map(score, [(p, args.end or None) for p in paths])

    l1s = [r[1] for r in rows if r[1] is not None]
    prs = [r[2] for r in rows if r[2] is not None and r[2] == r[2]]
    bad = [r[0] for r in rows if r[1] is None]
    if not l1s:
        raise SystemExit("no video could be scored")
    label = args.label or os.path.basename(os.path.normpath(args.videos_dir))
    print(f"{label}: n={len(l1s)}/{len(rows)}  L1={np.mean(l1s):.4f}  "
          f"pearson={np.mean(prs) if prs else float('nan'):.4f}  "
          f"ColorShift={100 * (1 - np.mean(l1s) / 2):.2f}")
    if bad:
        print(f"  {len(bad)} unscoreable, e.g. {bad[:3]}")
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["video", "l1_distance", "pearson_r", "frames_used"])
            for name, l1, pr, n in rows:
                w.writerow([name, "" if l1 is None else l1, "" if pr is None else pr, n])
        print(f"  -> {args.out}")


if __name__ == "__main__":
    main()
