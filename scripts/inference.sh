python scripts/example_inference_local.py \
  --model-path /data/efs/duwy18/code/robometer/logs/rbm/ckpt-latest-avg-5metrics=-inf_step=2000 \
  --video /data/efs/duwy18/code/robometer/my_processed_data_0_999/output/fetch_robot_data_0_999/batch_0000/trajectory_0999.mp4 \
  --task "OpenPickBowl"  \
  --fps 20.0 
