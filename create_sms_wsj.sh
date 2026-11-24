#!/usr/bin/env bash
# =============================================================================
# create_sms_wsj.sh
#
# Builds the raw SMS-WSJ database (https://github.com/fgnt/sms_wsj) from local
# copies of WSJ0 (LDC93S6A/B) and WSJ1 (LDC94S13A/B).
#
# What it does, start to finish:
#   1. Sets up a self-contained Python venv with the official sms_wsj package
#      (pinned commit, small NumPy-2 fix applied) and an MPI runtime from pip.
#   2. Builds sph2pipe and runs the part of Kaldi's WSJ data preparation that
#      sms_wsj needs (normalised transcripts + speaker info). No Kaldi
#      compilation is required. Pass --kaldi-root to use an existing Kaldi
#      where egs/wsj/s5 stage 0 has already been run instead.
#   3. Converts WSJ .wv1 (sphere/shorten) to 8 kHz zero-mean float wav.
#   4. Downloads the official SMS-WSJ RIRs from Zenodo (record 3517889).
#   5. Simulates and writes every signal of SMS-WSJ to disk:
#        observation/  speech_source/  early/  tail/  noise/
#      for train_si284, cv_dev93 and test_eval92 (35,875 examples).
#   6. Writes sms_wsj.json (paths + all metadata: offsets, SNR, transcripts,
#      speaker ids, RIR ids ...) next to the wav files.
#
# The script is resumable: re-run the same command after an interruption and
# completed stages are skipped; a stage that was cut off is redone.
#
# Final layout of --out:
#   wsj_8k_zeromean/          WSJ0+1 as 8 kHz zero-mean wav (+ transcripts)
#   rirs/                     multichannel RIRs + scenarios.json
#   observation/<dataset>/<id>.wav          6-ch mixture
#   speech_source/<dataset>/<id>_{0,1}.wav  1-ch source per speaker (padded/aligned)
#   early/<dataset>/<id>_{0,1}.wav          6-ch early-reverberant image per speaker
#   tail/<dataset>/<id>_{0,1}.wav           6-ch late-reverberant image per speaker
#   noise/<dataset>/<id>.wav                6-ch noise
#   sms_wsj.json, wsj_8k_zeromean.json, intermediate_sms_wsj.json
#   .setup/                   venv, sources, logs, stage markers
#
#   Note: observation = early + tail + noise; speech image = early + tail.
#
# Disk: ~442 GiB in total (+ ~47 GiB of RIR archives unless --stream-rirs or
#       --delete-rir-archives). Time: ~4 h on 32 cores for the simulation part.
# =============================================================================

set -Eeuo pipefail

# -----------------------------------------------------------------------------
# Pinned versions (override via environment if you know what you are doing)
# -----------------------------------------------------------------------------
SMS_WSJ_REPO="${SMS_WSJ_REPO:-https://github.com/fgnt/sms_wsj.git}"
SMS_WSJ_COMMIT="${SMS_WSJ_COMMIT:-f6b94bb987d620fc8bbe005a92bd042e09a68d9b}"
# Kaldi commit the sms_wsj authors tested against; only 4 perl helpers are used.
KALDI_COMMIT="${KALDI_COMMIT:-7637de77e0a77bf280bef9bf484e4f37c4eb9475}"
SPH2PIPE_URL="${SPH2PIPE_URL:-https://github.com/burrmill/sph2pipe/archive/2.5.tar.gz}"
ZENODO_BASE="${ZENODO_BASE:-https://zenodo.org/records/3517889/files}"
RIR_PARTS=(aa ab ac ad ae)
declare -A RIR_MD5=(
  [aa]=1174168b15023db578cfb38e02d80db7
  [ab]=9f8f8227b6b543e9a196d169b491b7f7
  [ac]=e8a89bbb08865bb3e2f20092748bf485
  [ad]=28a323d35393ac04da721844a2a88619
  [ae]=4064ec018384b18cca4f31f616c639ee
)
SPKRINFO_URLS=(
  "https://catalog.ldc.upenn.edu/docs/LDC93S6A/wsj0-train-spkrinfo.txt"
  "https://sourceforge.net/projects/kaldi/files/wsj0-train-spkrinfo.txt"
)
NUM_EXAMPLES=35875   # train_si284 + cv_dev93 + test_eval92

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
usage() {
  cat <<EOF
Usage:
  $(basename "$0") --wsj0 DIR --wsj1 DIR --out DIR [options]
  $(basename "$0") --wsj DIR --out DIR [options]     # WSJ0 and WSJ1 both under DIR

Required:
  --wsj0 DIR             Root of your WSJ0 copy (contains CD dirs like 11-13.1)
  --wsj1 DIR             Root of your WSJ1 copy (contains CD dirs like 13-34.1)
  --wsj DIR              Shortcut when both live under a single directory
  --out DIR              Where SMS-WSJ is written (~442 GiB free space needed)

Options:
  -j, --jobs N           Parallel processes (default: all cores, capped at
                         ~2 GB RAM per process). 1 = no MPI.
  --rir-dir DIR          Use RIRs you already downloaded and extracted
                         (directory containing scenarios.json). Skips download.
  --stream-rirs          Pipe the RIR download straight into tar: saves ~47 GiB
                         of temporary space, but no resume and no md5 check.
  --delete-rir-archives  Delete the downloaded RIR .tar.gz parts after extraction.
  --spkrinfo FILE        Local copy of wsj0-train-spkrinfo.txt, if downloading it
                         from LDC/sourceforge does not work on your network.
  --kaldi-root DIR       Use an existing Kaldi checkout in which the WSJ recipe
                         (egs/wsj/s5, stage 0) was already run. By default a
                         minimal stand-in is built under OUT/.setup.
  --python PY            Python (>=3.8) used to create the venv (default: python3)
  --no-space-check       Do not abort when free disk space looks too small.
  -h, --help             Show this help.

System requirements: bash 4+, python3 with venv, gcc, make, perl, sox, tar,
md5sum, and wget or curl. Internet access to github.com, pypi.org, zenodo.org
and catalog.ldc.upenn.edu (or sourceforge.net).
EOF
}

WSJ0=""; WSJ1=""; OUT=""; JOBS=""; RIR_DIR_USER=""; STREAM_RIRS=0
DELETE_RIR_ARCHIVES=0; KALDI_ROOT_USER=""; PYTHON_BIN="python3"; SPACE_CHECK=1
SPKRINFO_FILE=""

die()  { echo -e "\n[ERROR] $*" >&2; exit 1; }
info() { echo -e "[$(date +'%F %T')] $*"; }
warn() { echo -e "[WARNING] $*" >&2; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --wsj0) WSJ0="${2:?}"; shift 2 ;;
    --wsj1) WSJ1="${2:?}"; shift 2 ;;
    --wsj)  WSJ0="${2:?}"; WSJ1="${2:?}"; shift 2 ;;
    --out)  OUT="${2:?}"; shift 2 ;;
    -j|--jobs) JOBS="${2:?}"; shift 2 ;;
    --rir-dir) RIR_DIR_USER="${2:?}"; shift 2 ;;
    --stream-rirs) STREAM_RIRS=1; shift ;;
    --delete-rir-archives) DELETE_RIR_ARCHIVES=1; shift ;;
    --kaldi-root) KALDI_ROOT_USER="${2:?}"; shift 2 ;;
    --spkrinfo) SPKRINFO_FILE="${2:?}"; shift 2 ;;
    --python) PYTHON_BIN="${2:?}"; shift 2 ;;
    --no-space-check) SPACE_CHECK=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; die "Unknown argument: $1" ;;
  esac
done

[[ -n "$WSJ0" && -n "$WSJ1" && -n "$OUT" ]] || { usage >&2; die "--wsj0/--wsj1 (or --wsj) and --out are required."; }
[[ -d "$WSJ0" ]] || die "WSJ0 directory not found: $WSJ0"
[[ -d "$WSJ1" ]] || die "WSJ1 directory not found: $WSJ1"
if [[ -n "$SPKRINFO_FILE" ]]; then
  [[ -f "$SPKRINFO_FILE" ]] || die "--spkrinfo file not found: $SPKRINFO_FILE"
  SPKRINFO_FILE="$(cd "$(dirname "$SPKRINFO_FILE")" && pwd -P)/$(basename "$SPKRINFO_FILE")"
fi

WSJ0="$(cd "$WSJ0" && pwd -P)"
WSJ1="$(cd "$WSJ1" && pwd -P)"
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd -P)"

# sacred (the CLI used by sms_wsj) is not happy with whitespace in values.
for p in "$WSJ0" "$WSJ1" "$OUT" "$RIR_DIR_USER" "$KALDI_ROOT_USER"; do
  if [[ "$p" =~ [[:space:]] ]]; then die "Paths must not contain whitespace: '$p'"; fi
done
case "$OUT/" in "$WSJ0"/*|"$WSJ1"/*) die "--out must not be inside the WSJ directories." ;; esac

SETUP="$OUT/.setup"
VENV="$SETUP/venv"
SRC="$SETUP/src"
LOGS="$SETUP/logs"
MARKS="$SETUP/stages"
ARCHIVES="$SETUP/rir_archives"
WSJ8K="$OUT/wsj_8k_zeromean"
RIR_DIR="${RIR_DIR_USER:-$OUT/rirs}"
WSJ_JSON="$OUT/wsj_8k_zeromean.json"
INTERMED_JSON="$OUT/intermediate_sms_wsj.json"
FINAL_JSON="$OUT/sms_wsj.json"
SIGNAL_DIRS=(observation speech_source early tail noise)
mkdir -p "$SETUP" "$SRC" "$LOGS" "$MARKS"

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
# shellcheck disable=SC2154
trap 'rc=$?; if [[ $BASHPID == "$$" ]]; then echo -e "\n[ERROR] Failed at line $LINENO (exit $rc). Logs: $LOGS\nRe-run the same command to resume." >&2; fi' ERR

have() { command -v "$1" >/dev/null 2>&1; }

if have wget; then DL=wget; elif have curl; then DL=curl; else DL=""; fi

# fetch URL DEST   (resumable)
fetch() {
  local url="$1" dest="$2"
  if [[ $DL == wget ]]; then
    wget -c --tries=10 --retry-connrefused --waitretry=10 --progress=dot:giga -O "$dest" "$url"
  else
    curl -fL --retry 10 --retry-delay 10 -C - -o "$dest" "$url"
  fi
}

# fetch_stdout URL...   (concatenated to stdout)
fetch_stdout() {
  if [[ $DL == wget ]]; then wget -q --tries=10 -O- "$@"; else curl -fsSL --retry 10 "$@"; fi
}

is_done()   { [[ -f "$MARKS/$1.done" ]]; }
was_started() { [[ -f "$MARKS/$1.started" ]]; }
mark_started() { touch "$MARKS/$1.started"; }
mark_done() { touch "$MARKS/$1.done"; rm -f "$MARKS/$1.started"; }

# run_logged STAGE cmd...  -> stdout+stderr to terminal and log
run_logged() {
  local stage="$1"; shift
  info "Running: $*" | tee -a "$LOGS/$stage.log"
  "$@" 2>&1 | tee -a "$LOGS/$stage.log"
}

PY="$VENV/bin/python"
USE_MPI=0
# run_py STAGE module [sacred args...]   (single process)
run_py() {
  local stage="$1" module="$2"; shift 2
  (cd "$SETUP" && run_logged "$stage" "$PY" -m "$module" with "$@")
}
# run_py_mpi STAGE module [sacred args...]   (parallel if available)
run_py_mpi() {
  local stage="$1" module="$2"; shift 2
  if [[ $USE_MPI == 1 ]]; then
    (cd "$SETUP" && run_logged "$stage" "$VENV/bin/mpiexec" -np "$JOBS" "$PY" -m "$module" with "$@")
  else
    run_py "$stage" "$module" "$@"
  fi
}

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# -----------------------------------------------------------------------------
# Stage 0: sanity checks
# -----------------------------------------------------------------------------
info "SMS-WSJ will be created in: $OUT"

missing=()
for t in gcc make perl sox tar md5sum find sort awk sed; do have "$t" || missing+=("$t"); done
have "$PYTHON_BIN" || missing+=("$PYTHON_BIN")
[[ -n $DL ]] || missing+=("wget|curl")
((${#missing[@]} == 0)) || die "Missing required programs: ${missing[*]}\n  e.g. Debian/Ubuntu: sudo apt install build-essential perl sox wget python3-venv"
((BASH_VERSINFO[0] >= 4)) || die "bash >= 4 is required."
"$PYTHON_BIN" -c 'import sys; sys.exit(sys.version_info < (3, 8))' \
  || die "$PYTHON_BIN is older than 3.8; pass a newer one with --python."

# Locate the WSJ CD directories (names like 11-13.1, 13-34.1)
declare -A CD
CDS=()
while IFS= read -r d; do
  b="$(basename "$d")"
  if [[ -n "${CD[$b]:-}" ]]; then
    [[ "${CD[$b]}" == "$d" ]] || warn "Duplicate WSJ disk $b: using ${CD[$b]}, ignoring $d"
    continue
  fi
  CD[$b]="$d"; CDS+=("$d")
done < <(find -L "$WSJ0" "$WSJ1" -maxdepth 4 -type d -regextype posix-extended \
           -regex '.*/[0-9]{2}-[0-9]{1,2}\.[0-9]' 2>/dev/null | sort)

for need in 11-2.1 11-13.1 13-16.1 13-32.1 13-33.1 13-34.1; do
  [[ -n "${CD[$need]:-}" ]] || die "Could not find WSJ disk directory '$need' under $WSJ0 or $WSJ1.\n  Expected the LDC layout with directories like 11-1.1 ... 11-13.1 (WSJ0) and 13-1.1 ... 13-34.1 (WSJ1).\n  Note: SMS-WSJ needs BOTH WSJ0 and WSJ1."
done
for f in "${CD[11-13.1]}/wsj0/doc/indices/train/tr_s_wv1.ndx" \
         "${CD[11-13.1]}/wsj0/doc/indices/test/nvp/si_et_20.ndx" \
         "${CD[13-34.1]}/wsj1/doc/indices/si_tr_s.ndx" \
         "${CD[13-34.1]}/wsj1/doc/indices/h1_p0.ndx" \
         "${CD[13-32.1]}/wsj1/doc/indices/wsj1/eval/h1_p0.ndx"; do
  [[ -f "$f" ]] || die "Expected WSJ index file not found: $f\n  (file names must be lower-case as on the LDC disks)"
done
info "Found ${#CDS[@]} WSJ disk directories."

# Parallel jobs: default all cores, at most ~2 GB RAM each (sms_wsj README)
if [[ -z "$JOBS" ]]; then
  JOBS="$(nproc --all 2>/dev/null || echo 1)"
  mem_gb=$(awk '/MemTotal/ {printf "%d", $2/1024/1024}' /proc/meminfo 2>/dev/null || echo 0)
  if (( mem_gb > 0 && JOBS > mem_gb / 2 )); then
    JOBS=$(( mem_gb / 2 > 0 ? mem_gb / 2 : 1 ))
    info "Capping jobs to $JOBS (≈2 GB RAM per process, ${mem_gb} GB detected)."
  fi
fi
[[ "$JOBS" =~ ^[0-9]+$ && "$JOBS" -ge 1 ]] || die "--jobs must be a positive integer."

# Disk space (GiB still needed for stages not yet done)
if [[ $SPACE_CHECK == 1 ]]; then
  need=0
  is_done wav8k   || need=$((need + 30))
  if [[ -z "$RIR_DIR_USER" ]] && ! is_done rirs; then
    need=$((need + 53)); [[ $STREAM_RIRS == 1 ]] || need=$((need + 48))
  fi
  is_done write_files || need=$((need + 361))
  avail=$(df -Pk "$OUT" | awk 'NR==2 {printf "%d", $4/1024/1024}')
  if (( avail < need )); then
    die "About ${need} GiB free space needed in $OUT, only ${avail} GiB available.\n  Free up space, use --stream-rirs, or pass --no-space-check to try anyway."
  fi
  info "Disk: ${avail} GiB available, ~${need} GiB needed."
fi

# -----------------------------------------------------------------------------
# Stage 1: Python environment
# -----------------------------------------------------------------------------
if ! is_done python_env; then
  info "Stage 1/7: Creating Python environment in $VENV"
  rm -rf "$VENV"
  "$PYTHON_BIN" -m venv "$VENV" \
    || die "Could not create a venv. On Debian/Ubuntu: sudo apt install python3-venv"

  # sms_wsj source at a pinned commit
  rm -rf "$SRC/sms_wsj"
  if have git; then
    git clone -q "$SMS_WSJ_REPO" "$SRC/sms_wsj"
    git -C "$SRC/sms_wsj" checkout -q "$SMS_WSJ_COMMIT"
  else
    mkdir -p "$SRC/sms_wsj"
    fetch_stdout "https://github.com/fgnt/sms_wsj/archive/$SMS_WSJ_COMMIT.tar.gz" \
      | tar -xz -C "$SRC/sms_wsj" --strip-components=1
  fi
  # NumPy >= 1.24 removed the np.object alias used in write_files.py
  sed -i 's/\bnp\.object\b/object/g' "$SRC/sms_wsj/sms_wsj/database/write_files.py"

  {
    "$PY" -m pip install -q --upgrade pip wheel "setuptools<81"   # sacred needs pkg_resources
    # sh<2: sms_wsj relies on sh 1.x command piping semantics.
    # tqdm: dlp_mpi crashes without it when running under MPI.
    "$PY" -m pip install -q numpy scipy soundfile tqdm "sh<2" sacred lazy_dataset dlp_mpi
    "$PY" -m pip install -q --no-build-isolation --no-deps "$SRC/sms_wsj"
  } 2>&1 | tee -a "$LOGS/python_env.log"
  mark_done python_env
fi

# MPI (from pip: mpi4py + MPICH wheels, so no system MPI is needed)
mpi_works() {
  [[ -x "$VENV/bin/mpiexec" ]] && timeout 120 "$VENV/bin/mpiexec" -np 2 "$PY" -c \
    'from mpi4py import MPI; import sys; sys.exit(MPI.COMM_WORLD.size != 2)' >/dev/null 2>&1
}
if (( JOBS > 1 )); then
  if ! mpi_works; then
    info "Installing MPI runtime into the venv (mpi4py + mpich wheels)"
    "$PY" -m pip install -q mpi4py mpich 2>&1 | tee -a "$LOGS/python_env.log" || true
  fi
  if mpi_works; then
    USE_MPI=1
    info "Using MPI with $JOBS processes (1 coordinator + $((JOBS - 1)) workers)."
  else
    warn "MPI is not usable here; running single-process (this will be MUCH slower)."
  fi
fi
# dlp_mpi only falls back to serial mode on ImportError; a broken mpi4py
# (wheel without an MPI library) raises RuntimeError instead, so remove it.
if [[ $USE_MPI == 0 ]] && ! "$PY" -c 'from mpi4py import MPI' >/dev/null 2>&1; then
  "$PY" -m pip uninstall -y -q mpi4py >/dev/null 2>&1 || true
fi

# NumPy >= 1.24 compatibility patches for the INSTALLED sms_wsj package.
# Idempotent, so it also fixes venvs created by earlier versions of this script.
#  - np.object alias was removed.
#  - np.array([...]) on the two speakers' utterances (different lengths) used to
#    give an object array; it now raises "inhomogeneous shape". Build the
#    object array explicitly (downstream code only iterates over it).
"$PY" -I - <<'PYEOF' 2>&1 | tee -a "$LOGS/python_env.log"
import importlib.util, pathlib, re

def ragged(expr):
    return f"__import__('sms_wsj.database.write_files').database.write_files._sms_ragged({expr})"

HELPER = '''

def _sms_ragged(items):
    """np.array(items), but an object array if the items have different shapes."""
    items = list(items)
    try:
        return np.array(items)
    except ValueError:
        out = np.empty(len(items), dtype=object)
        for i, x in enumerate(items):
            out[i] = x
        return out
'''

spec = importlib.util.find_spec("sms_wsj")
root = pathlib.Path(spec.submodule_search_locations[0]) / "database"

wf = root / "write_files.py"
s = wf.read_text()
orig = s
s = re.sub(r"\bnp\.object\b", "object", s)
if "_sms_ragged" not in s:
    s = s.replace("\n\ndef audio_read(", HELPER + "\n\ndef audio_read(", 1)
    s = s.replace("example['audio_data'][audio_key] = np.array(audio_data)",
                  "example['audio_data'][audio_key] = _sms_ragged(audio_data)")
assert "_sms_ragged(audio_data)" in s, "could not patch sms_wsj write_files.py"
if s != orig:
    wf.write_text(s)
    print(f"patched {wf}")

db = root / "database.py"
s = db.read_text()
old = "return np.array([cls._rec_audio_read(f) for f in file])"
if old in s:
    db.write_text(s.replace(old, "return " + ragged("[cls._rec_audio_read(f) for f in file]")))
    print(f"patched {db}")
PYEOF
[[ ${PIPESTATUS[0]} == 0 ]] || die "Patching sms_wsj for NumPy compatibility failed (see $LOGS/python_env.log)."

# Make sure all the modules import before spending hours on anything.
"$PY" -W ignore -c 'import sms_wsj, sacred, lazy_dataset, dlp_mpi, soundfile, scipy.signal, sh' \
  || die "Python environment is broken; delete $VENV and $MARKS/python_env.done and re-run."

# -----------------------------------------------------------------------------
# Stage 2: Kaldi bits (sph2pipe + WSJ transcript normalisation + speaker info)
# -----------------------------------------------------------------------------
if [[ -n "$KALDI_ROOT_USER" ]]; then
  KROOT="$(cd "$KALDI_ROOT_USER" && pwd -P)"
  [[ -x "$KROOT/tools/sph2pipe_v2.5/sph2pipe" ]] || die "Not found: $KROOT/tools/sph2pipe_v2.5/sph2pipe"
  [[ -f "$KROOT/egs/wsj/s5/local/normalize_transcript.pl" ]] || die "Not found: $KROOT/egs/wsj/s5/local/normalize_transcript.pl"
  [[ -f "$KROOT/egs/wsj/s5/data/local/data/test_eval92.txt" ]] \
    || die "Kaldi WSJ data prep not found in $KROOT/egs/wsj/s5/data/local/data.\n  Run stage 0 of egs/wsj/s5/run.sh, or drop --kaldi-root to let this script do it."
  compgen -G "$KROOT/egs/wsj/s5/data/local/data/*spkrinfo.txt" >/dev/null \
    || die "No *spkrinfo.txt in $KROOT/egs/wsj/s5/data/local/data (Kaldi data prep incomplete)."
  info "Stage 2/7: Using existing Kaldi at $KROOT"
else
  KROOT="$SETUP/kaldi"
  if ! is_done kaldi_prep; then
    info "Stage 2/7: Building sph2pipe and preparing WSJ transcripts (Kaldi recipe subset)"
    S5="$KROOT/egs/wsj/s5"
    LOCAL="$S5/local"
    DATA="$S5/data/local/data"
    rm -rf "$KROOT"; mkdir -p "$KROOT/tools" "$LOCAL" "$DATA"

    # sph2pipe at the location sms_wsj expects: $KALDI_ROOT/tools/sph2pipe_v2.5/sph2pipe
    if have sph2pipe; then
      mkdir -p "$KROOT/tools/sph2pipe_v2.5"
      ln -s "$(command -v sph2pipe)" "$KROOT/tools/sph2pipe_v2.5/sph2pipe"
    else
      mkdir -p "$KROOT/tools/sph2pipe_v2.5"
      fetch_stdout "$SPH2PIPE_URL" | tar -xz -C "$KROOT/tools/sph2pipe_v2.5" --strip-components=1
      # sph2pipe is K&R-style C: GCC >= 14/15 (C23 default) rejects it unless
      # compiled in the old dialect.
      make -s -C "$KROOT/tools/sph2pipe_v2.5" CCFLAGS="-std=gnu89" 2>&1 | tee -a "$LOGS/kaldi_prep.log"
    fi
    [[ -x "$KROOT/tools/sph2pipe_v2.5/sph2pipe" ]] || die "Building sph2pipe failed."

    for f in ndx2flist.pl flist2scp.pl find_transcripts.pl normalize_transcript.pl; do
      fetch_stdout "https://raw.githubusercontent.com/kaldi-asr/kaldi/$KALDI_COMMIT/egs/wsj/s5/local/$f" > "$LOCAL/$f"
      chmod +x "$LOCAL/$f"
    done

    # --- Equivalent of the relevant part of egs/wsj/s5/local/wsj_data_prep.sh ---
    (
      cd "$DATA"
      export LC_ALL=C
      n2f() { perl "$LOCAL/ndx2flist.pl" "${CDS[@]}"; }

      n2f < "${CD[11-13.1]}/wsj0/doc/indices/train/tr_s_wv1.ndx" | sort \
        | grep -v -i 11-2.1/wsj0/si_tr_s/401 > train_si84.flist
      cat "${CD[13-34.1]}/wsj1/doc/indices/si_tr_s.ndx" \
          "${CD[11-13.1]}/wsj0/doc/indices/train/tr_s_wv1.ndx" | n2f | sort \
        | grep -v -i 11-2.1/wsj0/si_tr_s/401 > train_si284.flist
      n2f < "${CD[11-13.1]}/wsj0/doc/indices/test/nvp/si_et_20.ndx" \
        | awk '{printf("%s.wv1\n", $1)}' | sort > test_eval92.flist
      n2f < "${CD[11-13.1]}/wsj0/doc/indices/test/nvp/si_et_05.ndx" \
        | awk '{printf("%s.wv1\n", $1)}' | sort > test_eval92_5k.flist
      sed s/13_32_1/13_33_1/ "${CD[13-32.1]}/wsj1/doc/indices/wsj1/eval/h1_p0.ndx" | n2f | sort > test_eval93.flist
      sed s/13_32_1/13_33_1/ "${CD[13-32.1]}/wsj1/doc/indices/wsj1/eval/h2_p0.ndx" | n2f | sort > test_eval93_5k.flist
      n2f < "${CD[13-34.1]}/wsj1/doc/indices/h1_p0.ndx" | sort > test_dev93.flist
      n2f < "${CD[13-34.1]}/wsj1/doc/indices/h2_p0.ndx" | sort > test_dev93_5k.flist
      { find "${CD[13-16.1]}"/???1/??_??_20 -print 2>/dev/null || true; } | grep -i ".wv1" | sort > dev_dt_20.flist || true
      { find "${CD[13-16.1]}"/???1/??_??_05 -print 2>/dev/null || true; } | grep -i ".wv1" | sort > dev_dt_05.flist || true

      for x in "${CDS[@]}"; do find -L "$x" -iname '*.dot'; done > dot_files.flist

      for x in train_si84 train_si284 test_eval92 test_eval93 test_dev93 \
               test_eval92_5k test_eval93_5k test_dev93_5k dev_dt_05 dev_dt_20; do
        perl "$LOCAL/flist2scp.pl" "$x.flist" | sort > "${x}_sph.scp"
        awk '{print $1}' "${x}_sph.scp" | perl "$LOCAL/find_transcripts.pl" dot_files.flist > "$x.trans1"
        perl "$LOCAL/normalize_transcript.pl" "<NOISE>" < "$x.trans1" | sort > "$x.txt"
      done

      nl=$(wc -l < train_si284.flist)
      [[ "$nl" -eq 37416 ]] || echo "[WARNING] expected 37416 lines in train_si284.flist, got $nl" >&2

      # Speaker/gender info for the WSJ0 training speakers (not on the disks).
      if [[ -n "$SPKRINFO_FILE" ]]; then
        cp "$SPKRINFO_FILE" wsj0-train-spkrinfo.txt
        [[ $(wc -l < wsj0-train-spkrinfo.txt) -eq 134 ]] \
          || echo "[WARNING] expected 134 lines in $SPKRINFO_FILE" >&2
      else
        ok=0
        for u in "${SPKRINFO_URLS[@]}"; do
          rm -f wsj0-train-spkrinfo.txt
          if fetch "$u" wsj0-train-spkrinfo.txt >/dev/null 2>&1 \
             && [[ $(wc -l < wsj0-train-spkrinfo.txt) -eq 134 ]]; then ok=1; break; fi
        done
        if [[ $ok != 1 ]]; then
          echo "[ERROR] Could not download wsj0-train-spkrinfo.txt from:" >&2
          printf '          %s\n' "${SPKRINFO_URLS[@]}" >&2
          echo "        Download it manually and re-run with --spkrinfo /path/to/wsj0-train-spkrinfo.txt" >&2
          exit 1
        fi
      fi
    ) 2>&1 | tee -a "$LOGS/kaldi_prep.log"
    mark_done kaldi_prep
  else
    info "Stage 2/7: Kaldi bits already prepared."
  fi
fi
export KALDI_ROOT="$KROOT"

# -----------------------------------------------------------------------------
# Stage 3: WSJ sphere -> 8 kHz zero-mean wav
# -----------------------------------------------------------------------------
if ! is_done wav8k; then
  info "Stage 3/7: Converting WSJ to 8 kHz wav -> $WSJ8K"
  if [[ -e "$WSJ8K" ]]; then
    was_started wav8k || die "$WSJ8K already exists and was not created by this script. Move it away first."
    info "Removing partial output from an interrupted run: $WSJ8K"
    rm -rf "$WSJ8K"
  fi
  mark_started wav8k
  run_py_mpi wav8k sms_wsj.database.wsj.write_wav \
    "dst_dir=$WSJ8K" "wsj0_root=$WSJ0" "wsj1_root=$WSJ1" sample_rate=8000
  mark_done wav8k
else
  info "Stage 3/7: 8 kHz WSJ already done."
fi

# -----------------------------------------------------------------------------
# Stage 4: WSJ json
# -----------------------------------------------------------------------------
if ! is_done wsj_json; then
  info "Stage 4/7: Creating $WSJ_JSON"
  rm -f "$WSJ_JSON"
  run_py wsj_json sms_wsj.database.wsj.create_json \
    "json_path=$WSJ_JSON" "database_dir=$WSJ8K" as_wav=True
  mark_done wsj_json
else
  info "Stage 4/7: WSJ json already done."
fi

# -----------------------------------------------------------------------------
# Stage 5: RIRs from Zenodo
# -----------------------------------------------------------------------------
if [[ -n "$RIR_DIR_USER" ]]; then
  RIR_DIR="$(cd "$RIR_DIR_USER" && pwd -P)"
  [[ -f "$RIR_DIR/scenarios.json" ]] || die "No scenarios.json in --rir-dir $RIR_DIR"
  info "Stage 5/7: Using existing RIRs in $RIR_DIR"
elif ! is_done rirs; then
  info "Stage 5/7: Downloading RIRs (~51 GB) from Zenodo -> $RIR_DIR"
  if [[ -e "$RIR_DIR" ]]; then
    was_started rirs || die "$RIR_DIR already exists and was not created by this script.\n  If these are the extracted SMS-WSJ RIRs, pass --rir-dir $RIR_DIR"
    rm -rf "$RIR_DIR"
  fi
  mark_started rirs
  urls=(); for p in "${RIR_PARTS[@]}"; do urls+=("$ZENODO_BASE/sms_wsj.tar.gz.part$p?download=1"); done

  mkdir -p "$RIR_DIR"
  if [[ $STREAM_RIRS == 1 ]]; then
    fetch_stdout "${urls[@]}" | tar -xz -C "$RIR_DIR" --checkpoint=100000 --checkpoint-action=dot
    echo
  else
    mkdir -p "$ARCHIVES"
    for i in "${!RIR_PARTS[@]}"; do
      p="${RIR_PARTS[$i]}"; f="$ARCHIVES/sms_wsj.tar.gz.part$p"
      if [[ -f "$f.ok" ]]; then continue; fi
      info "  downloading part $p ($((i + 1))/${#RIR_PARTS[@]})"
      fetch "${urls[$i]}" "$f" 2>&1 | tee -a "$LOGS/rirs.log"
      info "  verifying md5 of part $p"
      echo "${RIR_MD5[$p]}  $f" | md5sum -c --quiet - \
        || { rm -f "$f"; die "md5 mismatch for $f (deleted). Re-run to download it again."; }
      touch "$f.ok"
    done
    info "  extracting RIRs (this takes a while)"
    cat "${RIR_PARTS[@]/#/$ARCHIVES/sms_wsj.tar.gz.part}" \
      | tar -xz -C "$RIR_DIR" --checkpoint=100000 --checkpoint-action=dot
    echo
  fi
  [[ -f "$RIR_DIR/scenarios.json" ]] || die "RIR extraction finished but $RIR_DIR/scenarios.json is missing."
  if [[ $DELETE_RIR_ARCHIVES == 1 ]]; then rm -rf "$ARCHIVES"; fi
  mark_done rirs
else
  info "Stage 5/7: RIRs already done."
fi

# -----------------------------------------------------------------------------
# Stage 6: intermediate json (pairs utterances with RIRs, offsets, ...)
# -----------------------------------------------------------------------------
if ! is_done intermediate_json; then
  info "Stage 6/7: Creating $INTERMED_JSON"
  rm -f "$INTERMED_JSON"
  run_py intermediate_json sms_wsj.database.create_intermediate_json \
    "json_path=$INTERMED_JSON" "rir_dir=$RIR_DIR" "wsj_json_path=$WSJ_JSON" debug=False
  mark_done intermediate_json
else
  info "Stage 6/7: Intermediate json already done."
fi

# -----------------------------------------------------------------------------
# Stage 7: simulate and write all signals, then the final json
# -----------------------------------------------------------------------------
if ! is_done write_files; then
  info "Stage 7/7: Simulating and writing SMS-WSJ wav files (the long part)"
  existing=()
  for d in "${SIGNAL_DIRS[@]}"; do if [[ -e "$OUT/$d" ]]; then existing+=("$OUT/$d"); fi; done
  if ((${#existing[@]})); then
    was_started write_files || die "Output dirs already exist and were not created by this script: ${existing[*]}"
    info "Removing partial output from an interrupted run."
    rm -rf "${existing[@]}"
  fi
  mark_started write_files
  run_py_mpi write_files sms_wsj.database.write_files \
    "dst_dir=$OUT" "json_path=$INTERMED_JSON" write_all=True debug=False
  mark_done write_files
else
  info "Stage 7/7: wav files already written."
fi

if ! is_done final_json; then
  info "Writing $FINAL_JSON"
  rm -f "$FINAL_JSON"
  run_py final_json sms_wsj.database.create_json_for_written_files \
    "db_dir=$OUT" "intermed_json_path=$INTERMED_JSON" write_all=True "json_path=$FINAL_JSON" debug=False
  mark_done final_json
fi

# -----------------------------------------------------------------------------
# Summary / verification
# -----------------------------------------------------------------------------
declare -A EXPECTED=(
  [observation]=$NUM_EXAMPLES [noise]=$NUM_EXAMPLES
  [speech_source]=$((2 * NUM_EXAMPLES)) [early]=$((2 * NUM_EXAMPLES)) [tail]=$((2 * NUM_EXAMPLES))
)
echo
info "Done. Contents of $OUT:"
all_ok=1
for d in "${SIGNAL_DIRS[@]}"; do
  n=$(find "$OUT/$d" -name '*.wav' | wc -l)
  status="ok"; [[ "$n" -eq "${EXPECTED[$d]}" ]] || { status="EXPECTED ${EXPECTED[$d]}"; all_ok=0; }
  printf '  %-15s %7d wav  %s\n' "$d/" "$n" "$status"
  for ds in train_si284 cv_dev93 test_eval92; do
    printf '      %-12s %7d\n' "$ds" "$(find "$OUT/$d/$ds" -name '*.wav' | wc -l)"
  done
done
printf '  %-15s %s\n' "rirs/" "$RIR_DIR"
printf '  %-15s %s\n' "wsj_8k_zeromean/" "$WSJ8K"
printf '  %-15s %s\n' "sms_wsj.json" "$FINAL_JSON"
[[ $all_ok == 1 ]] || warn "Some file counts differ from the expected values; check $LOGS."
echo
echo "All signals are 8 kHz, 32-bit float wav. Multichannel files have 6 channels."
echo "sms_wsj.json maps every example id to its files plus metadata (offsets, SNR,"
echo "speaker ids, transcriptions, RIR ids), handy for writing your own loaders."
[[ $STREAM_RIRS == 1 || $DELETE_RIR_ARCHIVES == 1 || -n "$RIR_DIR_USER" || ! -d "$ARCHIVES" ]] \
  || echo "You can delete the RIR archives to free ~47 GiB: rm -rf $ARCHIVES"
