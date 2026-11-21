#!/usr/bin/env python3
"""
Build a copy of the EARS dataset with no clips longer than a cutoff (default 45 s).

Two modes:
  drop  (default)  Leave out every clip longer than the cutoff.
  split            Cut each long clip into equal pieces no longer than the cutoff,
                   so none of the audio is lost.

The original dataset is never modified. Kept clips are copied (or hard-linked
with --link) into the output folder with the same pXXX/ layout. A manifest.csv
listing every output clip and its duration is written alongside them.

Usage:
  python filter_ears.py /path/to/EARS /path/to/EARS_45s
  python filter_ears.py /path/to/EARS /path/to/EARS_45s --mode split
  python filter_ears.py /path/to/EARS /path/to/EARS_45s --max-seconds 30 --link
  python filter_ears.py /path/to/EARS /path/to/EARS_45s --dry-run

Requires: pip install soundfile
"""
import argparse
import csv
import math
import os
import shutil
import sys
from pathlib import Path

import soundfile as sf


def place(src: Path, dst: Path, link: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if link:
        try:
            os.link(src, dst)
            return
        except OSError:
            pass  # different filesystem etc. -> fall back to copying
    shutil.copy2(src, dst)


def split_clip(src: Path, dst_dir: Path, max_s: float, dry_run: bool):
    """Split into the fewest equal-length pieces that are each <= max_s."""
    info = sf.info(str(src))
    n_pieces = math.ceil(info.duration / max_s)
    frames_per = math.ceil(info.frames / n_pieces)
    out = []
    for k in range(n_pieces):
        start = k * frames_per
        stop = min(start + frames_per, info.frames)
        dst = dst_dir / f"{src.stem}_part{k + 1:02d}{src.suffix}"
        if not dry_run and not dst.exists():
            data, sr = sf.read(str(src), start=start, stop=stop, always_2d=True)
            dst.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(dst), data, sr, subtype=info.subtype)
        out.append((dst, (stop - start) / info.samplerate))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path, help="EARS root folder (contains p001/ ... p107/)")
    ap.add_argument("dst", type=Path, help="Output folder (created if missing)")
    ap.add_argument("--max-seconds", type=float, default=45.0, help="Cutoff in seconds (default 45)")
    ap.add_argument("--mode", choices=["drop", "split"], default="drop",
                    help="drop long clips (default) or split them into pieces")
    ap.add_argument("--link", action="store_true",
                    help="Hard-link kept clips instead of copying (saves disk space)")
    ap.add_argument("--dry-run", action="store_true", help="Report what would happen; write nothing")
    args = ap.parse_args()

    src, dst = args.src.resolve(), args.dst.resolve()
    if not src.is_dir():
        sys.exit(f"Not a folder: {src}")
    if dst == src or src in dst.parents:
        sys.exit("Output folder must be outside the source folder.")

    wavs = sorted(src.rglob("*.wav"))
    if not wavs:
        sys.exit(f"No .wav files found under {src}")

    rows = []  # (relative path, seconds, source relative path)
    n_kept = n_long = 0
    kept_s = long_s = 0.0

    for i, f in enumerate(wavs, 1):
        rel = f.relative_to(src)
        dur = sf.info(str(f)).duration
        if dur <= args.max_seconds:
            n_kept += 1
            kept_s += dur
            if not args.dry_run:
                place(f, dst / rel, args.link)
            rows.append((rel.as_posix(), dur, rel.as_posix()))
        else:
            n_long += 1
            long_s += dur
            if args.mode == "split":
                for piece, pdur in split_clip(f, (dst / rel).parent, args.max_seconds, args.dry_run):
                    rows.append((piece.relative_to(dst).as_posix(), pdur, rel.as_posix()))
        if i % 1000 == 0:
            print(f"  scanned {i}/{len(wavs)}", file=sys.stderr)

    if not args.dry_run:
        dst.mkdir(parents=True, exist_ok=True)
        with open(dst / "manifest.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["path", "duration_s", "source"])
            for r in rows:
                w.writerow([r[0], f"{r[1]:.3f}", r[2]])

    total_s = kept_s + long_s
    print(f"\nScanned {len(wavs)} clips, {total_s / 3600:.2f} h")
    print(f"  <= {args.max_seconds:g}s: {n_kept} clips, {kept_s / 3600:.2f} h")
    print(f"  >  {args.max_seconds:g}s: {n_long} clips, {long_s / 3600:.2f} h "
          f"({'split into pieces' if args.mode == 'split' else 'left out'})")
    durs = [r[1] for r in rows]
    if durs:
        durs.sort()
        print(f"Output: {len(durs)} clips, {sum(durs) / 3600:.2f} h, "
              f"min {durs[0]:.1f}s / median {durs[len(durs) // 2]:.1f}s / max {durs[-1]:.1f}s")
    print("(dry run, nothing written)" if args.dry_run else f"Written to {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
