import h5py
import os
from pathlib import Path
from tqdm import tqdm

def load_my_hdf5_dataset(base_path: str):
    task_data = {}
    base_path = Path(base_path)
    
    # 获取文件夹下所有 hdf5 文件
    data_files = list(base_path.glob(".hdf5"))
    
    for file_path in tqdm(data_files, desc="Processing HDF5s"):
        with h5py.File(file_path, 'r') as f:
            # --- 关键：根据你的 HDF5 结构提取数据 ---
            # 假设你的 HDF5 里有 'observations/images' 和 'task_description'
            images = f['observations/images'][:] 
            task_name = f.attrs.get('task_description', 'Robot task') 
            
            trajectory = {
                'frames': images,       # 直接传入 numpy 数组，转换器会自动处理
                'actions': f['actions'][:], 
                'is_robot': True,
                'task': task_name,
                'optimal': 'success'    # <--- 因为全是成功的，直接写死
            }
            
            if task_name not in task_data:
                task_data[task_name] = []
            task_data[task_name].append(trajectory)
            
    return task_data