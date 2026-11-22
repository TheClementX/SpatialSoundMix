#!/usr/bin/env python3
"""
Report how much the lengths of audio tracks that get mixed together differ.

Every audio file under ROOT is assigned to a "mix group" (the set of tracks that
are mixed together). By default a group is the folder that directly holds the
files, e.g.

    data/<hm3d_scene>/<instance>/speaker_0.wav
    data/<hm3d_scene>/<instance>/speaker_1.wav    -> one group per <instance>

Use --group-depth to group at a different level instead. Depth counts folders
below ROOT: --group-depth 1 puts everything under each <hm3d_scene> into one
group, --group-depth 2 groups by <hm3d_scene>/<instance>, and so on.

For each group with 2+ tracks it computes:
  * pairwise differences: |len(a) - len(b)| for every pair of tracks
  * spread: longest track - shortest track

It then reports the mean, variance, standard deviation, median and max of
those differences across the whole dataset (and per scene with --per-scene).

Files whose names match --exclude (e.g. the rendered mixture itself) are
skipped. Unreadable files are reported and skipped, not fatal.

Usage:
  # data/{train,val,test}/scene_X/instance_Y/{mix_audio,source_0,source_1}.wav
  for s in train val test; do
    python mix_length_stats.py data/$s --exclude "mix_audio*" --csv ${s}_mix_lengths.csv
  done

  python mix_length_stats.py /path/to/data
  python mix_length_stats.py /path/to/data --exclude "mix*" --exclude "*_reverb*"
  python mix_length_stats.py /path/to/data --group-depth 1 --csv per_mix.csv
  python mix_length_stats.py /path/to/data --list-groups 5     # check grouping first

Requires: pip install soundfile
"""
import argparse
import csv
import fnmatch
import itertools
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import soundfile as sf

AUDIO_EXTS = {".wav", ".flac", ".ogg", ".aiff", ".aif"}


def summarize(label, vals, unit="s"):
    if not vals:
        print(f"  {label}: (none)")
        return
    mean = statistics.fmean(vals)
    var = statistics.pvariance(vals, mu=mean)
    print(f"  {label}  (n={len(vals)})")
    print(f"    mean     {mean:10.3f} {unit}")
    print(f"    variance {var:10.3f} {unit}^2")
    print(f"    std dev  {var ** 0.5:10.3f} {unit}")
    print(f"    median   {statistics.median(vals):10.3f} {unit}")
    print(f"    max      {max(vals):10.3f} {unit}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path, help="Data folder (one subfolder per HM3D scene)")
    ap.add_argument("--group-depth", type=int, default=None,
                    help="Group tracks by the first N folders below ROOT "
                         "(default: by the folder directly containing each file)")
    ap.add_argument("--scene-depth", type=int, default=1,
                    help="Folder level below ROOT that names the HM3D scene (default 1)")
    ap.add_argument("--exclude", action="append", default=[], metavar="GLOB",
                    help="Skip files whose name matches GLOB (repeatable), e.g. 'mix*'")
    ap.add_argument("--per-scene", action="store_true",
                    help="Also print a mean/variance row for every scene")
    ap.add_argument("--csv", type=Path, help="Write one row per mix group to this CSV")
    ap.add_argument("--list-groups", type=int, default=0, metavar="N",
                    help="Print the first N groups and their files, then exit")
    args = ap.parse_args()

    root = args.root.resolve()
    if not root.is_dir():
        sys.exit(f"Not a folder: {root}")

    files = sorted(p for p in root.rglob("*")
                   if p.is_file() and p.suffix.lower() in AUDIO_EXTS
                   and not any(fnmatch.fnmatch(p.name, g) for g in args.exclude))
    if not files:
        sys.exit(f"No audio files found under {root}")

    groups = defaultdict(list)
    for p in files:
        parts = p.relative_to(root).parts
        if args.group_depth is None:
            key = parts[:-1]
        else:
            key = parts[:min(args.group_depth, len(parts) - 1)]
        groups["/".join(key) or "."].append(p)

    if args.list_groups:
        for key in sorted(groups)[:args.list_groups]:
            print(f"{key}/  ({len(groups[key])} files)")
            for p in groups[key]:
                print(f"    {p.name}")
        print(f"... {len(groups)} groups total")
        return 0

    pair_diffs, spreads, rows, bad = [], [], [], []
    by_scene = defaultdict(lambda: {"pairs": [], "spreads": []})
    n_single = 0
    for i, key in enumerate(sorted(groups), 1):
        lens = []
        for p in groups[key]:
            try:
                lens.append((p, sf.info(str(p)).duration))
            except Exception as e:  # unreadable / not really audio
                bad.append((p, str(e).splitlines()[0]))
        if len(lens) < 2:
            n_single += 1
            continue
        durs = [d for _, d in lens]
        pd = [abs(a - b) for a, b in itertools.combinations(durs, 2)]
        spread = max(durs) - min(durs)
        pair_diffs += pd
        spreads.append(spread)
        scene = "/".join(Path(key).parts[:args.scene_depth]) or "."
        by_scene[scene]["pairs"] += pd
        by_scene[scene]["spreads"].append(spread)
        rows.append((key, scene, len(durs), min(durs), max(durs), spread, statistics.fmean(pd)))
        if i % 500 == 0:
            print(f"  processed {i}/{len(groups)} groups", file=sys.stderr)

    print(f"\nRoot: {root}")
    print(f"Audio files: {len(files)}   groups: {len(groups)}   "
          f"groups with 2+ tracks: {len(spreads)}   scenes: {len(by_scene)}")
    if n_single:
        print(f"Skipped {n_single} groups with fewer than 2 readable tracks")
    if spreads:
        counts = defaultdict(int)
        for r in rows:
            counts[r[2]] += 1
        print("Tracks per mix: " + ", ".join(f"{k}: {v}" for k, v in sorted(counts.items())))

    print("\nDifference in length between tracks in the same mix")
    summarize("Pairwise |len(a) - len(b)|, every pair of tracks", pair_diffs)
    summarize("Spread per mix (longest - shortest)", spreads)

    if args.per_scene and len(by_scene) > 1:
        print("\nPer scene (mean / variance of pairwise difference, s and s^2)")
        w = max(len(s) for s in by_scene)
        print(f"  {'scene':<{w}}  {'mixes':>6}  {'mean':>8}  {'variance':>10}")
        for scene in sorted(by_scene):
            v = by_scene[scene]["pairs"]
            m = statistics.fmean(v)
            print(f"  {scene:<{w}}  {len(by_scene[scene]['spreads']):>6}  {m:>8.3f}  "
                  f"{statistics.pvariance(v, mu=m):>10.3f}")

    if bad:
        print(f"\n{len(bad)} unreadable files skipped:")
        for p, e in bad[:20]:
            print(f"  {p.relative_to(root)}: {e}")
        if len(bad) > 20:
            print(f"  ... and {len(bad) - 20} more")

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["group", "scene", "n_tracks", "shortest_s", "longest_s",
                        "spread_s", "mean_pairwise_diff_s"])
            for r in rows:
                w.writerow([r[0], r[1], r[2]] + [f"{x:.3f}" for x in r[3:]])
        print(f"\nPer-mix rows written to {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
