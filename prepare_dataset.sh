
#!/bin/bash

# 1. 彻底禁用 GPU 冲突 (防止 JAX/Torch/TF 报错)
# export CUDA_VISIBLE_DEVICES=-1
# export JAX_PLATFORM_NAME=cpu
# export JAX_PLATFORMS=cpu
# export TF_CPP_MIN_LOG_LEVEL=3
# export XLA_PYTHON_CLIENT_PREALLOCATE=false

# 定义路径变量
DATASET_NAME="fetch_robot_data"
RAW_DATA_PATH="/data/efs/jiajy16/mani_skill_related/settable_pick_bowl/open_pick_rgb_processed/0_10_test"
OUTPUT_ROOT="my_processed_data"

# 2. 运行数据转换脚本
echo "正在开始数据转换..."
uv run python dataset_upload/generate_hf_dataset.py \
    --dataset.dataset_name=$DATASET_NAME \
    --dataset.dataset_path=$RAW_DATA_PATH \
    --output.output_dir=$OUTPUT_ROOT \
    --output.use_video=true \
    --output.fps=10 \
    --output.shortest_edge_size=128  \
    --output.num_workers=1

# 3. 自动整理目录结构 (满足 Robometer 训练要求的子文件夹格式)
TARGET_DIR="$OUTPUT_ROOT/$DATASET_NAME"
TEMP_SUBDIR="$TARGET_DIR/processed_dataset"

if [ -d "$TARGET_DIR" ]; then
    echo "转换完成，正在调整目录结构以适配 Robometer 训练..."
    
    # 如果子目录不存在，则创建
    mkdir -p "$TEMP_SUBDIR"
    
    # 将除了新创建的子目录以外的所有内容移动进去
    # 使用 find 确保不会移动子目录本身，且支持隐藏文件
    find "$TARGET_DIR" -maxdepth 1 -not -name "." -not -name "processed_dataset" -exec mv {} "$TEMP_SUBDIR/" \;
    
    echo "目录结构调整成功！当前数据路径: $TEMP_SUBDIR"
else
    echo "错误: 转换脚本似乎未生成输出目录 $TARGET_DIR"
    exit 1
fi

echo "现在你可以运行微调命令了。"
