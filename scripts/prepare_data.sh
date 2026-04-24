#!/bin/bash
# scripts/prepare_data.sh
# Normalize working directory to project root
cd "$(dirname "$0")/.."
set -e

RAW_DATA_PATH="/data/efs/jiajy16/mani_skill_related/settable_pick_bowl/open_pick_rgb_processed/0_10_test"
DATASET_NAME="fetch_robot_data_test"
BASE_DIR="/data/efs/duwy18/code/robometer/my_processed_data_test"
CACHE_ROOT="$BASE_DIR"

echo "📍 Working directory: $(pwd)"
echo "Starting data processing pipeline..."
source .venv/bin/activate

echo ">>> [Step 1/2] Converting raw data to HF format..."
uv run python dataset_upload/generate_hf_dataset.py \
    --dataset.dataset_path="$RAW_DATA_PATH" \
    --dataset.dataset_name="$DATASET_NAME" \
    --output.output_dir="$BASE_DIR/standard" \
    --output.use_video=true \
    --output.max_frames=32 \
    --output.num_workers=8

echo ">>> [Step 2/2] Generating training cache (NPZ frames & index mappings)..."
export ROBOMETER_PROCESSED_DATASETS_PATH="$CACHE_ROOT"

uv run python -m robometer.data.scripts.preprocess_datasets \
    --cache_dir "$CACHE_ROOT" \
    --train_datasets "['$BASE_DIR/standard/$DATASET_NAME']" \
    --train_subsets "[['']]" \
    --num_proc 8 \
    --force_reprocess True

echo ">>> Setting up symlinks for cleaner dataset names..."
cd "$CACHE_ROOT"
LONG_NAME=$(ls -d _data_*${DATASET_NAME}* 2>/dev/null | head -n 1)
if [ -n "$LONG_NAME" ]; then
    rm -f "$DATASET_NAME"
    ln -s "$LONG_NAME" "$DATASET_NAME"
    echo "✅ Success: Symlinked $DATASET_NAME -> $LONG_NAME"
else
    echo "❌ Error: Could not find generated cache directory!"
    exit 1
fi
echo "✅ Data preparation complete!"
