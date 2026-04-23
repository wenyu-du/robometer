import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict

def load_fetch_robot_dataset(base_path: str) -> Dict[str, List[Dict]]:
    task_data = {}
    base_path = Path(base_path)
    
    # 查找所有 hdf5 文件
    data_files = list(base_path.glob("*.hdf5"))
    print(f"Found {len(data_files)} files in {base_path}")
    
    for file_path in tqdm(data_files, desc="Processing Fetch HDF5"):
        with h5py.File(file_path, 'r') as f:
            # 1. 提取图像 (建议用头部视角，如果是 128x128)
            # shape 是 (111, 128, 128, 3)
            frames = f['obs/fetch_head_rgb'][:] 
            
            # 2. 提取动作
            actions = f['action'][:]
            
            # 3. 提取任务名 (如果 HDF5 没存，就用文件名去掉后缀)
            # 优先从属性里找，找不到就用文件名
            # task_name = f.attrs.get('task', file_path.stem).replace('_', ' ')

            # 写死任务名称
            task_name = 'OpenPickBowl'
            
            # 4. 确定是否成功
            # 根据你的输出有 success (111,)，我们看最后一帧是否为 1
            is_success = f['success'][-1] > 0
            optimal_status = 'success' if is_success else 'failed'

            trajectory = {
                'frames': frames,        # 图像数组
                'actions': actions,      # 动作数组
                'is_robot': True,
                'task': task_name,
                'optimal': optimal_status
            }
            
            if task_name not in task_data:
                task_data[task_name] = []
            task_data[task_name].append(trajectory)
            
    return task_data
