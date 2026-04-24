# 1. 建立软连接，模拟标准目录结构 (如果还没建的话)
cd /data/efs/duwy18/code/robometer/my_processed_data/fetch_robot_data
mkdir -p processed_dataset
ln -s ../cache processed_dataset/cache
ln -s ../data-00000-of-00001.arrow processed_dataset/
ln -s ../dataset_info.json processed_dataset/
ln -s ../state.json processed_dataset/

# 2. 启动训练
cd /data/private/code/robometer
# 1. 彻底清理旧日志，防止干扰
rm -rf ./logs/rbm

# 2. 设置路径
export ROBOMETER_PROCESSED_DATASETS_PATH=/data/efs/duwy18/code/robometer/my_processed_data

# 3. 启动训练：注意将 resume_from_checkpoint 改为 load_from_checkpoint
uv run accelerate launch --config_file robometer/configs/distributed/fsdp.yaml train.py \
  training.load_from_checkpoint=robometer/Robometer-4B \
  training.resume_from_checkpoint=null \
  model.use_peft=true \
  "data.train_datasets=[fetch_robot_data]" \
  "data.eval_datasets=[fetch_robot_data]" \
  training.max_steps=2000 \
  training.per_device_train_batch_size=1 \
  training.learning_rate=5e-5 \
  training.overwrite_output_dir=True \
  "logging.log_to=[]" \
  data.min_frames_per_trajectory=1