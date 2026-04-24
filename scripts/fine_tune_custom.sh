#!/bin/bash
# scripts/fine_tune_custom.sh
# Normalize working directory to project root
cd "$(dirname "$0")/.."

export ROBOMETER_PROCESSED_DATASETS_PATH="/data/efs/duwy18/code/robometer/my_processed_data"
MODEL_CHECKPOINT="robometer/Robometer-4B"
DATASET_NAME="fetch_robot_data"
MAX_STEPS=2000
LR=5e-5

echo "📍 Working directory: $(pwd)"
echo "🚀 Launching Robometer fine-tuning..."
source .venv/bin/activate

# Clean up old logs to prevent interference
rm -rf ./logs/rbm

uv run accelerate launch --config_file robometer/configs/distributed/fsdp.yaml train.py \
  training.load_from_checkpoint="$MODEL_CHECKPOINT" \
  training.resume_from_checkpoint=null \
  model.use_peft=true \
  "data.train_datasets=[$DATASET_NAME]" \
  "data.eval_datasets=[$DATASET_NAME]" \
  training.max_steps=$MAX_STEPS \
  training.per_device_train_batch_size=1 \
  training.learning_rate=$LR \
  training.overwrite_output_dir=True \
  "logging.log_to=[]" \
  data.min_frames_per_trajectory=1
