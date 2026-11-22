#!/usr/bin/env python3
"""
Build a copy of the EARS dataset keeping only clips between a minimum and a
maximum length (default: no minimum, 45 s maximum), and log the distribution of
clip lengths. Clips shorter than --min-seconds are always left out.

Two cleaning modes:
  drop  (default)  Leave out every clip longer than the cutoff.
  split            Cut each long clip into equal pieces no longer than the cutoff,
                   so none of the audio is lost.

The original dataset is never modified. Kept clips are copied (or hard-linked
with --link) into the output folder with the same pXXX/ layout. A manifest.csv
listing every output clip and its duration is written alongside them.

Every cleaning run logs the length distribution of the source AND of the result.
With --dry-run the result's distribution is a projection; nothing is written.

Usage:
  # Clean
  python filter_ears.py /path/to/EARS /path/to/EARS_45s
  python filter_ears.py /path/to/EARS /path/to/EARS_45s --mode split
  python filter_ears.py /path/to/EARS /path/to/EARS_45s --max-seconds 30 --link
  python filter_ears.py /path/to/EARS /path/to/EARS_5-45s --min-seconds 5

  # Preview what the cleaned lengths would look like, without writing anything
  python filter_ears.py /path/to/EARS /path/to/EARS_45s --dry-run

  # Just report the length distribution of any folder (original or cleaned)
  python filter_ears.py --stats /path/to/EARS
  python filter_ears.py --stats /path/to/EARS_45s --bin-seconds 2 --log stats.txt

Requires: pip install soundfile
"""
import argparse
import csv
import logging
import math
import os
import shutil
import sys
from pathlib import Path

import soundfile as sf

log = logging.getLogger("filter_ears")


# --------------------------------------------------------------------------- #
# Distribution logging
# --------------------------------------------------------------------------- #
def percentile(sorted_vals, q):
    """Linear-interpolated percentile of an already-sorted list (q in 0..100)."""
    if not sorted_vals:
        return float("nan")
    pos = (len(sorted_vals) - 1) * q / 100
    lo, hi = math.floor(pos), math.ceil(pos)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def log_distribution(title, durations, bin_s=5.0, width=40, csv_path=None):
    """Log summary stats and a text histogram of clip durations (seconds)."""
    d = sorted(durations)
    log.info("")
    log.info("=== %s ===", title)
    if not d:
        log.info("  (no clips)")
        return
    total = sum(d)
    log.info("  clips: %d   total: %.2f h   mean: %.1f s", len(d), total / 3600, total / len(d))
    log.info("  min %.1f | p5 %.1f | p25 %.1f | median %.1f | p75 %.1f | p95 %.1f | max %.1f  (s)",
             d[0], percentile(d, 5), percentile(d, 25), percentile(d, 50),
             percentile(d, 75), percentile(d, 95), d[-1])

    n_bins = int(d[-1] // bin_s) + 1
    counts = [0] * n_bins
    secs = [0.0] * n_bins
    for x in d:
        b = min(int(x // bin_s), n_bins - 1)
        counts[b] += 1
        secs[b] += x
    peak = max(counts)
    label_w = len(f"{n_bins * bin_s:g}") * 2 + 4
    log.info("  %-*s %7s %6s %7s", label_w, "length (s)", "clips", "%clip", "%audio")
    for b, c in enumerate(counts):
        if c == 0 and b not in (0, n_bins - 1):
            # collapse empty bins in long tails to keep the log readable
            if counts[b - 1] == 0:
                continue
            log.info("  %-*s", label_w, "   ...")
            continue
        lo, hi = b * bin_s, (b + 1) * bin_s
        bar = "#" * max(1 if c else 0, round(width * c / peak))
        log.info("  %-*s %7d %5.1f%% %6.1f%%  %s", label_w, f"{lo:g}-{hi:g}",
                 c, 100 * c / len(d), 100 * secs[b] / total, bar)

    if csv_path:
        with open(csv_path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["bin_start_s", "bin_end_s", "clips", "pct_clips", "seconds", "pct_audio"])
            for b, c in enumerate(counts):
                w.writerow([b * bin_s, (b + 1) * bin_s, c, f"{100 * c / len(d):.3f}",
                            f"{secs[b]:.3f}", f"{100 * secs[b] / total:.3f}"])
        log.info("  histogram written to %s", csv_path)


def scan_durations(root: Path):
    """Return [(path, seconds)] for every .wav under root."""
    wavs = sorted(root.rglob("*.wav"))
    out = []
    for i, f in enumerate(wavs, 1):
        out.append((f, sf.info(str(f)).duration))
        if i % 1000 == 0:
            print(f"  scanned {i}/{len(wavs)}", file=sys.stderr)
    return out


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #
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


def split_clip(src: Path, dst_dir: Path, max_s: float, min_s: float, dry_run: bool):
    """Split into the fewest equal-length pieces that are each <= max_s.

    Pieces come out at least max_s/2 long, so they only fall below min_s when
    min_s is more than half of max_s; any such piece is skipped, not written.
    Returns (kept [(path, seconds)], number of pieces skipped as too short).
    """
    info = sf.info(str(src))
    n_pieces = math.ceil(info.duration / max_s)
    frames_per = math.ceil(info.frames / n_pieces)
    out, skipped = [], 0
    for k in range(n_pieces):
        start = k * frames_per
        stop = min(start + frames_per, info.frames)
        secs = (stop - start) / info.samplerate
        if secs < min_s:
            skipped += 1
            continue
        dst = dst_dir / f"{src.stem}_part{k + 1:02d}{src.suffix}"
        if not dry_run and not dst.exists():
            data, sr = sf.read(str(src), start=start, stop=stop, always_2d=True)
            dst.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(dst), data, sr, subtype=info.subtype)
        out.append((dst, secs))
    return out, skipped


def clean(args) -> int:
    src, dst = args.src.resolve(), args.dst.resolve()
    if dst == src or src in dst.parents:
        sys.exit("Output folder must be outside the source folder.")

    scanned = scan_durations(src)
    if not scanned:
        sys.exit(f"No .wav files found under {src}")

    rows = []  # (relative path, seconds, source relative path)
    n_long = n_short = n_short_pieces = 0
    long_s = short_s = 0.0
    for f, dur in scanned:
        rel = f.relative_to(src)
        if dur < args.min_seconds:
            n_short += 1
            short_s += dur
        elif dur <= args.max_seconds:
            if not args.dry_run:
                place(f, dst / rel, args.link)
            rows.append((rel.as_posix(), dur, rel.as_posix()))
        else:
            n_long += 1
            long_s += dur
            if args.mode == "split":
                pieces, skipped = split_clip(f, (dst / rel).parent, args.max_seconds,
                                             args.min_seconds, args.dry_run)
                n_short_pieces += skipped
                for piece, pdur in pieces:
                    rows.append((piece.relative_to(dst).as_posix(), pdur, rel.as_posix()))

    if not args.dry_run:
        dst.mkdir(parents=True, exist_ok=True)
        with open(dst / "manifest.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["path", "duration_s", "source"])
            for r in rows:
                w.writerow([r[0], f"{r[1]:.3f}", r[2]])

    log_distribution(f"Original: {src}", [d for _, d in scanned], args.bin_seconds)
    verb = "split into pieces" if args.mode == "split" else "left out"
    log.info("")
    log.info("Clips under %gs: %d (%.2f h) -> left out", args.min_seconds, n_short, short_s / 3600)
    log.info("Clips over %gs: %d (%.2f h) -> %s", args.max_seconds, n_long, long_s / 3600, verb)
    if n_short_pieces:
        log.info("  %d split pieces came out under %gs and were left out", n_short_pieces, args.min_seconds)
    title = ("Projected after cleaning (dry run, nothing written)" if args.dry_run
             else f"After cleaning: {dst}")
    log_distribution(title, [r[1] for r in rows], args.bin_seconds,
                     csv_path=None if args.dry_run else dst / "length_histogram.csv")
    return 0


def stats(args) -> int:
    root = args.stats.resolve()
    scanned = scan_durations(root)
    if not scanned:
        sys.exit(f"No .wav files found under {root}")
    log_distribution(f"Clip lengths: {root}", [d for _, d in scanned], args.bin_seconds,
                     csv_path=args.hist_csv)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path, nargs="?", help="EARS root folder (contains p001/ ... p107/)")
    ap.add_argument("dst", type=Path, nargs="?", help="Output folder (created if missing)")
    ap.add_argument("--stats", type=Path, metavar="DIR",
                    help="Only report the clip-length distribution of DIR (original or cleaned)")
    ap.add_argument("--max-seconds", type=float, default=45.0, help="Longest clip to keep, in seconds (default 45)")
    ap.add_argument("--min-seconds", type=float, default=0.0,
                    help="Shortest clip to keep, in seconds (default 0 = no minimum)")
    ap.add_argument("--mode", choices=["drop", "split"], default="drop",
                    help="drop long clips (default) or split them into pieces")
    ap.add_argument("--link", action="store_true",
                    help="Hard-link kept clips instead of copying (saves disk space)")
    ap.add_argument("--dry-run", action="store_true", help="Report what would happen; write nothing")
    ap.add_argument("--bin-seconds", type=float, default=5.0, help="Histogram bin width (default 5)")
    ap.add_argument("--hist-csv", type=Path, help="With --stats: also save the histogram as CSV")
    ap.add_argument("--log", type=Path, help="Also write the log to this file")
    args = ap.parse_args()

    handlers = [logging.StreamHandler(sys.stdout)]
    if args.log:
        handlers.append(logging.FileHandler(args.log, mode="w"))
    logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=handlers)

    if args.stats:
        if not args.stats.is_dir():
            sys.exit(f"Not a folder: {args.stats}")
        return stats(args)
    if not (args.src and args.dst):
        ap.error("give SRC and DST to clean, or --stats DIR to only report lengths")
    if not 0 <= args.min_seconds < args.max_seconds:
        ap.error("--min-seconds must be at least 0 and less than --max-seconds")
    if not args.src.is_dir():
        sys.exit(f"Not a folder: {args.src}")
    return clean(args)


if __name__ == "__main__":
    raise SystemExit(main())