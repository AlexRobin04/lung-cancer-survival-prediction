import h5py
import torch
import os
from tqdm import tqdm

def convert_all_h5_to_pt(folder_path):
    # 获取所有 .h5 文件
    h5_files = [f for f in os.listdir(folder_path) if f.endswith('.h5')]

    for h5_file in tqdm(h5_files, desc="转换中"):
        h5_path = os.path.join(folder_path, h5_file)
        pt_filename = os.path.splitext(h5_file)[0] + '.pt'
        pt_path = os.path.join(folder_path, pt_filename)

        try:
            with h5py.File(h5_path, 'r') as f:
                if 'features' not in f:
                    print(f"跳过 {h5_file}：未找到 'features'")
                    continue
                features = f['features'][:]
                features_tensor = torch.tensor(features)
                torch.save(features_tensor, pt_path)
        except Exception as e:
            print(f"处理 {h5_file} 时出错：{e}")
    print("全部处理完成。")


if __name__ == "__main__":
    convert_all_h5_to_pt('/media/ubuntu/015eb652-0471-4534-8f64-2a82c75fed771/yuehailin/TCGA_feature/CONCH/ESCA/20')
    convert_all_h5_to_pt('/media/ubuntu/015eb652-0471-4534-8f64-2a82c75fed771/yuehailin/TCGA_feature/CONCH/ESCA/10')

    
    
    


