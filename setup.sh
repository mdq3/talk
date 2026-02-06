#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

VARIANT="${1:-base}"
VENV_DIR="$SCRIPT_DIR/venv"
MODELS_DIR="$SCRIPT_DIR/models"
HAILORT_WHEEL="/usr/local/hailo/resources/packages/hailort-*-linux_aarch64.whl"
BASE_HEF="https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources/whisper"
BASE_ASSETS="https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources/npy%20files/whisper/decoder_assets"
MEL_FILTERS_SRC="https://github.com/openai/whisper/raw/main/whisper/assets/mel_filters.npz"

# --- Helpers ---

info()  { echo "==> $*"; }
error() { echo "ERROR: $*" >&2; exit 1; }

# --- Preflight checks ---

info "Checking prerequisites..."

command -v python3 >/dev/null || error "python3 not found. Install with: sudo apt install python3"
command -v ffmpeg  >/dev/null || error "ffmpeg not found. Install with: sudo apt install ffmpeg"
command -v wget    >/dev/null || error "wget not found. Install with: sudo apt install wget"

if ! dpkg -s libportaudio2 &>/dev/null; then
    error "libportaudio2 not found. Install with: sudo apt install libportaudio2"
fi

if ! compgen -G $HAILORT_WHEEL >/dev/null 2>&1; then
    error "HailoRT wheel not found at $HAILORT_WHEEL. Is the hailort package installed?"
fi

if ! hailortcli fw-control identify &>/dev/null; then
    error "Hailo device not detected. Check your AI HAT+ connection."
fi

case "$VARIANT" in
    base|tiny|tiny.en) ;;
    *) error "Unknown variant '$VARIANT'. Choose: base, tiny, tiny.en" ;;
esac

info "Variant: $VARIANT"

# --- Virtual environment ---

if [ ! -d "$VENV_DIR" ]; then
    info "Creating virtual environment..."
    python3 -m venv --system-site-packages "$VENV_DIR"
else
    info "Virtual environment already exists."
fi

info "Installing HailoRT Python bindings..."
"$VENV_DIR/bin/pip" install --quiet $HAILORT_WHEEL 2>/dev/null || true

info "Installing Python dependencies..."
"$VENV_DIR/bin/pip" install --quiet -r requirements.txt

# --- Download models ---

# HEF files
HEF_DIR="$MODELS_DIR/hefs/h10h/$VARIANT"
mkdir -p "$HEF_DIR"

declare -A ENCODER_HEFS=(
    ["base"]="base-whisper-encoder-10s.hef"
    ["tiny"]="tiny-whisper-encoder-10s.hef"
    ["tiny.en"]="tiny_en-whisper-encoder-10s.hef"
)
declare -A DECODER_HEFS=(
    ["base"]="base-whisper-decoder-10s-out-seq-64.hef"
    ["tiny"]="tiny-whisper-decoder-fixed-sequence.hef"
    ["tiny.en"]="tiny_en-whisper-decoder-fixed-sequence.hef"
)

ENCODER="${ENCODER_HEFS[$VARIANT]}"
DECODER="${DECODER_HEFS[$VARIANT]}"

if [ -f "$HEF_DIR/$ENCODER" ] && [ -f "$HEF_DIR/$DECODER" ]; then
    info "HEF model files already downloaded."
else
    info "Downloading HEF model files..."
    wget -q --show-progress -P "$HEF_DIR" "$BASE_HEF/h10h/$ENCODER"
    wget -q --show-progress -P "$HEF_DIR" "$BASE_HEF/h10h/$DECODER"
fi

# Decoder tokenization assets
# tiny.en uses "tiny.en" in filenames
ASSET_VARIANT="$VARIANT"
ASSET_DIR="$MODELS_DIR/decoder_assets/$VARIANT/decoder_tokenization"
mkdir -p "$ASSET_DIR"

WEIGHT_FILE="token_embedding_weight_${ASSET_VARIANT}.npy"
ADD_FILE="onnx_add_input_${ASSET_VARIANT}.npy"

if [ -f "$ASSET_DIR/$WEIGHT_FILE" ] && [ -f "$ASSET_DIR/$ADD_FILE" ]; then
    info "Decoder assets already downloaded."
else
    info "Downloading decoder assets..."
    wget -q --show-progress -P "$ASSET_DIR" "$BASE_ASSETS/$VARIANT/decoder_tokenization/$WEIGHT_FILE"
    wget -q --show-progress -P "$ASSET_DIR" "$BASE_ASSETS/$VARIANT/decoder_tokenization/$ADD_FILE"
fi

# Mel filter bank
if [ -f "$MODELS_DIR/mel_filters.npz" ]; then
    info "Mel filters already present."
else
    info "Downloading mel filters..."
    wget -q --show-progress -O "$MODELS_DIR/mel_filters.npz" "$MEL_FILTERS_SRC"
fi

# --- Done ---

echo ""
echo "Setup complete! To run:"
echo ""
echo "  source venv/bin/activate"
echo "  python talk.py"
echo ""
