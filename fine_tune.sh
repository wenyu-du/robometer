export ROBOMETER_PROCESSED_DATASETS_PATH=/data/efs/duwy18/code/robometer/my_processed_data

uv run python -m robometer.data.scripts.preprocess_datasets \
    --config robometer/configs/preprocess.yaml \
    "data.train_datasets=[fetch_robot_data]" \
    "data.eval_datasets=[fetch_robot_data]"

uv run accelerate launch --config_file robometer/configs/distributed/fsdp.yaml train.py \
  training.resume_from_checkpoint=robometer/Robometer-4B \
  model.use_peft=true \
  data.train_datasets=[fetch_robot_data] \
  data.eval_datasets=[fetch_robot_data] \
  data.max_frames=8 \
  training.max_steps=2000 \
  training.per_device_train_batch_size=1 \
  training.learning_rate=5e-5 \
  logging.log_to=[]  \
  training.overwrite_output_dir=True
