"""Motion Drift: does a clip still move at its end the way it moved at its start?

    CUDA_VISIBLE_DEVICES=0 python eval/motion_drift.py outputs/self_forcing --out outputs/sf_motion.csv
    python eval/motion_drift.py --merge outputs/sf_motion.csv outputs/sf_motion.csv.part0 ...

VBench's dynamic-degree test is run separately on the first and on the last 5 s of every clip.
VBench's own DynamicDegree class (pip package `vbench`) is used without changes. Frames are
subsampled to 8 fps and RAFT (raft-things) estimates flow between consecutive frames. A pair
scores the mean of its top 5 % flow magnitudes. A segment counts as moving once round(4 * n / 16)
of its n frames' pairs exceed 6 * min(H, W) / 256 pixels (11.25 px at 480p).

Per video we write first_move, last_move, l1 = |first_move - last_move| and the mean pair score
of each end. Over the set we report

    Motion Drift = 100 - LOST, where LOST is the share (%) of clips that move at the start but
                   are static at the end. This is the number in our paper's tables.
    symmetric    = 100 - mean |d_start - d_end| (in %), which also counts clips that start
                   moving only at the end (gained)

together with the share of clips moving at each end.

With --end N each clip is treated as if it ended at frame N, so a long clip can be measured over
its opening part. --shard / --num_shards split the videos across processes; --merge joins the
per-shard CSVs and prints the summary.
"""
import argparse
import csv
import glob
import os

import cv2
import numpy as np

RAFT_DEFAULT = os.path.expanduser("~/.cache/vbench/raft_model/models/raft-things.pth")


def segment(path, seconds, which, end=None):
    """The first or last `seconds` of a clip, as RGB uint8 frames, and its frame rate."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 16.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if end:
        n = min(n, end)
    want = int(round(fps * seconds))
    start = 0 if which == "first" else max(0, n - want)
    if start:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    out = []
    while len(out) < want:
        ok, bgr = cap.read()
        if not ok:
            break
        out.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    return out, fps


def summarize(rows, label):
    def pct(x):
        return 100.0 * sum(x) / len(rows)
    first = [r["first_move"] in (True, "True") for r in rows]
    last = [r["last_move"] in (True, "True") for r in rows]
    lost = pct([a and not b for a, b in zip(first, last)])
    gained = pct([b and not a for a, b in zip(first, last)])
    l1 = pct([a != b for a, b in zip(first, last)])
    ff = np.mean([float(r["first_flow"]) for r in rows])
    lf = np.mean([float(r["last_flow"]) for r in rows])
    print(f"{label}: n={len(rows)}  Motion Drift {100 - lost:.2f}  (symmetric {100 - l1:.2f})  "
          f"moving first {pct(first):.2f} last {pct(last):.2f}  lost {lost:.2f} gained {gained:.2f}  "
          f"flow {ff:.1f} -> {lf:.1f}")


def write_csv(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["video", "first_move", "last_move", "l1",
                                          "first_flow", "last_flow"])
        w.writeheader()
        w.writerows(rows)


def merge(out, parts):
    rows = []
    for p in parts:
        rows += list(csv.DictReader(open(p)))
    if not rows:
        raise SystemExit("nothing to merge: every part is empty")
    rows.sort(key=lambda r: r["video"])
    write_csv(out, rows)
    summarize(rows, os.path.basename(out))
    print(f"  -> {out}")


def main():
    ap = argparse.ArgumentParser(description="Motion Drift: dynamic degree of the first vs last 5 s")
    ap.add_argument("videos_dir", nargs="?")
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--glob", default="*.mp4")
    ap.add_argument("--end", type=int, default=0, help="treat each clip as ending at this frame")
    ap.add_argument("--raft", default=RAFT_DEFAULT, help="RAFT raft-things checkpoint")
    ap.add_argument("--batch", type=int, default=1,
                    help="frame pairs per RAFT call. 1 is what our numbers use; a larger batch is "
                         "the same arithmetic, but cuDNN may pick another kernel and a clip on "
                         "the threshold can flip")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--label", default=None)
    ap.add_argument("--out", default=None, help="per-video CSV")
    ap.add_argument("--merge", nargs="+", metavar=("OUT", "PART"),
                    help="join per-shard CSVs into OUT and print the summary")
    a = ap.parse_args()

    if a.merge:
        if len(a.merge) < 2:
            ap.error("--merge needs OUT and at least one part")
        return merge(a.merge[0], a.merge[1:])
    if not a.videos_dir:
        ap.error("videos_dir is required")
    a.label = a.label or os.path.basename(os.path.normpath(a.videos_dir))
    a.out = a.out or os.path.normpath(a.videos_dir) + "_motion_drift.csv"

    import torch
    from easydict import EasyDict as edict
    from vbench.dynamic_degree import DynamicDegree
    from vbench.third_party.RAFT.core.utils_core.utils import InputPadder
    if not os.path.isfile(a.raft):
        raise SystemExit(f"RAFT checkpoint missing: {a.raft} (see README, 'Drift metrics')")
    device = torch.device("cuda")
    dyn = DynamicDegree(edict({"model": a.raft, "small": False, "mixed_precision": False,
                               "alternate_corr": False}), device)

    def score_of(flo):
        """VBench's get_score for one pair: the mean of the top 5 % flow magnitudes."""
        u, v = flo[0].cpu().numpy(), flo[1].cpu().numpy()
        rad = np.sqrt(u ** 2 + v ** 2).ravel()
        cut = int(rad.size * 0.05)
        return float(np.mean(np.sort(-rad)[:cut] * -1.0))

    def moving(frames, fps):
        interval = max(1, round(fps / 8))                  # VBench samples at 8 fps
        sub = frames[::interval]
        imgs = [torch.from_numpy(f.astype(np.uint8)).permute(2, 0, 1).float()[None].to(device)
                for f in sub]
        dyn.set_params(frame=imgs[0], count=len(imgs))
        scores = []
        with torch.no_grad():
            step = max(1, a.batch)
            for s in range(0, len(imgs) - 1, step):
                m = min(step, len(imgs) - 1 - s)
                a1 = torch.cat(imgs[s:s + m], 0)
                a2 = torch.cat(imgs[s + 1:s + 1 + m], 0)
                padder = InputPadder(a1.shape)
                p1, p2 = padder.pad(a1, a2)
                _, flow = dyn.model(p1, p2, iters=20, test_mode=True)
                scores += [score_of(flow[k]) for k in range(a1.shape[0])]
        return bool(dyn.check_move(scores)), float(np.mean(scores))

    files = sorted(glob.glob(os.path.join(a.videos_dir, a.glob)))
    files = [f for i, f in enumerate(files) if i % a.num_shards == a.shard]
    print(f"== {a.label}: {len(files)} videos, {a.seconds:g} s at each end", flush=True)
    if not files:                                      # more shards than videos
        write_csv(a.out, [])
        return
    rows = []
    for i, f in enumerate(files):
        fr_first, fps = segment(f, a.seconds, "first", a.end)
        fr_last, _ = segment(f, a.seconds, "last", a.end)
        if len(fr_first) < 2 or len(fr_last) < 2:
            print(f"   [{i}] too short, skipped: {os.path.basename(f)}", flush=True)
            continue
        m0, s0 = moving(fr_first, fps)
        m1, s1 = moving(fr_last, fps)
        rows.append(dict(video=os.path.basename(f), first_move=m0, last_move=m1,
                         l1=abs(int(m0) - int(m1)), first_flow=round(s0, 3), last_flow=round(s1, 3)))
        if (i + 1) % 16 == 0:
            print(f"   {i + 1}/{len(files)}", flush=True)
    if not rows:
        raise SystemExit("no video could be measured")
    write_csv(a.out, rows)
    summarize(rows, a.label)
    print(f"  -> {a.out}")


if __name__ == "__main__":
    main()
