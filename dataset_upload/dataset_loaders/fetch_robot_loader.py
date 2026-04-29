import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict
from dataset_upload.helpers import generate_unique_id

def load_fetch_robot_dataset(base_path: str) -> Dict[str, List[Dict]]:
    task_data = {}
    
    # 支持多路径解析
    paths = base_path.replace(',', ':').split(':')
    data_files = []
    for p in paths:
        p_path = Path(p.strip())
        if p_path.exists():
            found = list(p_path.glob("*.hdf5"))
            print(f"Found {len(found)} files in {p_path}")
            data_files.extend(found)
        else:
            print(f"Warning: Path {p_path} does not exist")
    
    print(f"Total files to process: {len(data_files)}")
    
    # 定义要提取的视角名称
    camera_views = ['fetch_head_rgb', 'fetch_hand_rgb']
    
    for file_path in tqdm(data_files, desc="Loading Fetch Data"):
        with h5py.File(file_path, 'r') as f:
            # 提取共享元数据
            actions = f['action'][:]
            task_name = 'OpenPickBowl'
            
            rewards = f['reward'][:]
            max_reward = np.max(rewards)
            partial_success = float(max_reward) / 3.0
            
            is_success = f['success'][-1] > 0
            quality_label = 'successful' if is_success else 'failure'

            # 为每一个视角创建一条独立的轨迹
            for view_key in camera_views:
                full_view_path = f'obs/{view_key}'
                if full_view_path not in f:
                    continue
                
                # 直接读取图像数组到内存
                # 在 2TB 内存的机器上，这样做最快且最稳定
                frames = f[full_view_path][:] 

                trajectory = {
                    'id': generate_unique_id(),
                    'frames': frames,        
                    'actions': actions,      
                    'is_robot': True,
                    'task': task_name,
                    'quality_label': quality_label, 
                    'partial_success': partial_success, 
                    'view': view_key         
                }
                
                if task_name not in task_data:
                    task_data[task_name] = []
                task_data[task_name].append(trajectory)
            
    return task_data
