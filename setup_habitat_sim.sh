#!/usr/bin/env bash
# setup_habitat_sim.sh
#
# Downloads habitat-sim v0.3.4, fixes the audio source position bug, builds it with
# audio support into a fresh conda environment, applies the numba CPU detection fix
# and verifies the result.
#
#   ./setup_habitat_sim.sh                    # env "habitat-sim", source in ./habitat-sim-v0.3.4
#   ./setup_habitat_sim.sh --recreate         # delete and rebuild an existing env
#
# Options:
#   --env NAME     conda environment to create (default: habitat-sim)
#   --dir PATH     where the source is cloned (default: ./habitat-sim-v0.3.4);
#                  an existing clone there is reused
#   --jobs N       parallel compile jobs (default: from CPU count and memory)
#   --recreate     delete the environment first if it already exists
#   --gui          also build the GUI viewer (default: headless)
#
# The fixes:
#   1. AudioSensor::runSimulation passes lastSourcePos_[1], [2], [3] to the ray tracer
#      instead of [0], [1], [2], so sources are simulated at (y, z, ~0).
#   2. numba's CPU feature detection crashes on some CPUs ("free(): invalid pointer"
#      when importing habitat_sim). NUMBA_CPU_NAME=generic is saved in the environment.

set -euo pipefail

ENV_NAME="habitat-sim"
SRC_DIR="$PWD/habitat-sim-v0.3.4"
TAG="v0.3.4"
REPO="https://github.com/facebookresearch/habitat-sim.git"
PYTHON_VERSION="3.12"
RECREATE=0
GUI=OFF
JOBS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --env) ENV_NAME="$2"; shift 2 ;;
        --dir) SRC_DIR="$(realpath -m "$2")"; shift 2 ;;
        --jobs) JOBS="$2"; shift 2 ;;
        --recreate) RECREATE=1; shift ;;
        --gui) GUI=ON; shift ;;
        -h|--help) sed -n '2,25p' "$0"; exit 0 ;;
        *) echo "Unknown option: $1 (see --help)" >&2; exit 1 ;;
    esac
done

step() { echo; echo "=== $* ==="; }
die() { echo; echo "ERROR: $*" >&2; exit 1; }

# ---------------------------------------------------------------- preflight
step "Checking prerequisites"
if ! command -v conda >/dev/null 2>&1 && [[ -n "${CONDA_EXE:-}" ]]; then
    export PATH="$(dirname "$CONDA_EXE"):$PATH"
fi
command -v conda >/dev/null 2>&1 || die "conda not found. Run this from a shell where conda works."
command -v git >/dev/null 2>&1 || die "git not found (sudo apt-get install git)."
command -v g++ >/dev/null 2>&1 || die "g++ not found (sudo apt-get install build-essential)."

missing=()
for header in /usr/include/EGL/egl.h /usr/include/GL/gl.h /usr/include/jpeglib.h /usr/include/glm/glm.hpp; do
    [[ -f "$header" ]] || missing+=("$header")
done
if [[ "$GUI" == ON ]]; then
    [[ -f /usr/include/X11/Xlib.h ]] || missing+=(/usr/include/X11/Xlib.h)
fi
if [[ ${#missing[@]} -gt 0 ]]; then
    echo "Missing system headers: ${missing[*]}"
    echo "Install them with:"
    echo "  sudo apt-get install -y libjpeg-dev libglm-dev libegl1-mesa-dev libgl1-mesa-dev mesa-utils xorg-dev freeglut3-dev"
    exit 1
fi

if [[ -z "$JOBS" ]]; then
    # each compile job can use ~2 GB, so limit by memory as well as cores
    mem_gb=$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo)
    JOBS=$(( mem_gb / 2 ))
    (( JOBS > $(nproc) )) && JOBS=$(nproc)
    (( JOBS < 1 )) && JOBS=1
fi

CONDA_BASE="$(conda info --base)"
# conda's activation scripts read unset variables, so relax -u around them
set +u
source "$CONDA_BASE/etc/profile.d/conda.sh"
set -u

env_exists() { conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; }
if env_exists; then
    [[ $RECREATE -eq 1 ]] || die "conda environment '$ENV_NAME' already exists. Rerun with --recreate to delete and rebuild it, or pick another name with --env."
fi
echo "env: $ENV_NAME | source: $SRC_DIR | jobs: $JOBS | GUI viewer: $GUI"

# ---------------------------------------------------------------- source
step "Getting habitat-sim $TAG"
if [[ -d "$SRC_DIR/.git" ]]; then
    echo "Reusing existing clone at $SRC_DIR"
    current=$(git -C "$SRC_DIR" describe --tags --exact-match 2>/dev/null || git -C "$SRC_DIR" rev-parse --short HEAD)
    [[ "$current" == "$TAG" ]] || echo "WARNING: clone is at $current, not $TAG. Your code needs $TAG or later."
else
    [[ -e "$SRC_DIR" ]] && die "$SRC_DIR exists but is not a git clone. Remove it or pass --dir."
    git clone --branch "$TAG" --depth 1 "$REPO" "$SRC_DIR"
fi
git -C "$SRC_DIR" submodule update --init --recursive --jobs 4
[[ -f "$SRC_DIR/src/deps/rlr-audio-propagation/RLRAudioPropagationPkg/libs/linux/x64/libRLRAudioPropagation.so" ]] \
    || die "the audio library submodule (src/deps/rlr-audio-propagation) is missing."

# ---------------------------------------------------------------- fix 1
step "Fixing the audio source position bug"
AUDIO_CPP="$SRC_DIR/src/esp/sensor/AudioSensor.cpp"
sed -i 's/lastSourcePos_\[1\], lastSourcePos_\[2\], lastSourcePos_\[3\]/lastSourcePos_[0], lastSourcePos_[1], lastSourcePos_[2]/' "$AUDIO_CPP"
grep -q 'lastSourcePos_\[3\]' "$AUDIO_CPP" && die "patch did not apply to $AUDIO_CPP"
grep -q 'lastSourcePos_\[0\], lastSourcePos_\[1\], lastSourcePos_\[2\]' "$AUDIO_CPP" \
    || die "could not find the AddSource call in $AUDIO_CPP; the file differs from $TAG"
grep -n -A1 'AddSource(' "$AUDIO_CPP"

# ---------------------------------------------------------------- environment
step "Creating conda environment $ENV_NAME"
if env_exists; then
    conda env remove -n "$ENV_NAME" -y
fi
conda create -n "$ENV_NAME" "python=$PYTHON_VERSION" -y
# fix 2: saved in the environment, so it is set whenever the environment is active
conda env config vars set NUMBA_CPU_NAME=generic -n "$ENV_NAME"
set +u
conda activate "$ENV_NAME"
set -u
export NUMBA_CPU_NAME=generic
[[ "$(command -v python)" == "$CONDA_PREFIX/bin/python" ]] || die "could not activate $ENV_NAME"

step "Installing build tools and Python dependencies"
# CMake 4 rejects some of the bundled dependencies, so stay below it
python -m pip install --upgrade pip
python -m pip install "cmake>=3.22,<4" ninja "scikit-build-core>=0.10" "pybind11>=2.10" \
    -r "$SRC_DIR/requirements.txt" soundfile

# ---------------------------------------------------------------- build
step "Building habitat-sim (this takes a while)"
LOG="$SRC_DIR/build.log"
rm -rf "$SRC_DIR/build"
echo "full log: $LOG"
if ! ( cd "$SRC_DIR" && \
       HABITAT_WITH_AUDIO=ON HABITAT_WITH_BULLET=ON HABITAT_BUILD_GUI_VIEWERS="$GUI" \
       CMAKE_BUILD_PARALLEL_LEVEL="$JOBS" \
       python -m pip install . --no-build-isolation -v > "$LOG" 2>&1 ); then
    tail -n 40 "$LOG"
    grep -q -i "killed signal\|out of memory" "$LOG" && echo "The compiler ran out of memory: rerun with a smaller --jobs."
    die "build failed, see $LOG"
fi
echo "build finished"

# ---------------------------------------------------------------- verify
step "Verifying"
TEST_SCENE="$SRC_DIR/data/test_assets/objects/nested_box.glb"
VERIFY_DIR="$(mktemp -d)"
trap 'rm -rf "$VERIFY_DIR"' EXIT
# run outside the source folder so the installed package is the one imported
( cd "$VERIFY_DIR" && python - "$TEST_SCENE" "$CONDA_PREFIX" <<'EOF'
import os, sys
scene, prefix = sys.argv[1], sys.argv[2]

import numba
assert numba.njit(lambda x: x + 1)(1) == 2
print(f"numba OK (NUMBA_CPU_NAME={os.environ.get('NUMBA_CPU_NAME')})")

import numpy as np
import habitat_sim
assert habitat_sim.__file__.startswith(prefix), f"imported the wrong habitat_sim: {habitat_sim.__file__}"
assert habitat_sim.audio_enabled, "habitat_sim was built without audio"
print(f"habitat_sim OK, audio enabled: {habitat_sim.__file__}")

#a source 0.3 m above the listener vs the same source 200 m along x: with the fix the
#far one is much quieter, with the bug x is ignored and both give the same RIR
backend = habitat_sim.SimulatorConfiguration()
backend.scene_id = scene
backend.enable_physics = True
#any EGL device: habitat-sim always opens a renderer once sensors exist,
#even though audio does not use it
backend.gpu_device_id = -1
spec = habitat_sim.AudioSensorSpec()
spec.uuid = "audio_sensor"
acoustics = habitat_sim.RLRAudioPropagationConfiguration()
acoustics.enableMaterials = False  #the test asset has no semantic mesh
spec.acousticsConfig = acoustics
spec.channelLayout.channelType = habitat_sim.RLRAudioPropagationChannelLayoutType.Binaural
spec.position = [0.0, 1.5, 0.0]
agent_cfg = habitat_sim.AgentConfiguration()
agent_cfg.sensor_specifications = [spec]
sim = habitat_sim.Simulator(habitat_sim.Configuration(backend, [agent_cfg]))
sensor = sim.get_agent(0)._sensors["audio_sensor"]
listener = np.array(sensor.node.absolute_translation, dtype=np.float32)

def energy(offset):
    sensor.setAudioSourceTransform(listener + np.array(offset, dtype=np.float32))
    rir = np.asarray(sim.get_sensor_observations()["audio_sensor"], dtype=np.float64)
    return float(np.sum(rir ** 2))

near, far = energy([0.0, 0.3, 0.0]), energy([200.0, 0.3, 0.0])
sim.close()
print(f"RIR energy: source 0.3 m away {near:.3g}, source 200 m away {far:.3g}")
assert near > 0, "the audio simulation returned a silent RIR"
assert far < 0.1 * near, "moving the source does not change its RIR: the position bug is still present"
print("audio source positions OK")
EOF
) || die "verification failed (see above)"

echo
echo "Done. habitat-sim $TAG with audio is installed in the conda environment '$ENV_NAME'."
echo "Use it with:  conda activate $ENV_NAME"
