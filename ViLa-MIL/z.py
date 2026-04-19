import pandas as pd
import os

# 配置路径
csv_path = '/home/ubuntu/project/ViLa-MIL/datasets_csv/LGG.csv'            # CSV 文件路径
folder_path = '/media/ubuntu/015eb652-0471-4534-8f64-2a82c75fed771/yuehailin/TCGA_wsi/TCGA_LGG_generation/CONCH/20/20x_512px_0px_overlap/features_conch_v1'       # 存放 .h5 文件的文件夹路径

# 读取 CSV 文件中的 slide_id 列，并移除 .svs 后缀
df = pd.read_csv(csv_path)
slide_ids_raw = df['slide_id'].astype(str)
slide_ids = [sid.replace('.svs', '') for sid in slide_ids_raw]

# 检查对应 .h5 文件是否存在
existing_files = []
missing_files = []

for slide_id in slide_ids:
    h5_filename = f"{slide_id}.h5"
    h5_path = os.path.join(folder_path, h5_filename)
    if os.path.exists(h5_path):
        existing_files.append(slide_id)
    else:
        missing_files.append(slide_id)

# 输出结果
print(f"找到的 .h5 文件数量: {len(existing_files)}")
print(f"缺失的 .h5 文件数量: {len(missing_files)}")


# 可选：保存缺失列表到 CSV
pd.DataFrame(missing_files, columns=['missing_slide_id']).to_csv('LGG_missing_slide_ids_20.csv', index=False)
