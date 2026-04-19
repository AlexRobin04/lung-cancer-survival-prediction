import pandas as pd
import os

# 替换成你的文件路径
csv_file = '/home/ubuntu/project/ViLa-MIL/datasets_csv/LGG.csv'
folder_path = '/media/ubuntu/015eb652-0471-4534-8f64-2a82c75fed771/yuehailin/TCGA_wsi/TCGA_LGG_generation/CONCH/10/10x_512px_0px_overlap/features_conch_v1'

# 读取CSV并去掉后缀
df = pd.read_csv(csv_file)
df['base_id'] = df['slide_id'].str[:-4]  # 去掉后4个字符，例如 .svs

# 检查是否存在对应的 .h5 文件
def check_h5_exists(slide_id):
    h5_file = f"{slide_id}.h5"
    return os.path.exists(os.path.join(folder_path, h5_file))

# 添加一列标记是否存在对应的h5文件
df['h5_exists'] = df['base_id'].apply(check_h5_exists)

# 打印结果统计
print("存在的 .h5 文件数量：", df['h5_exists'].sum())
print("不存在的 .h5 文件数量：", (~df['h5_exists']).sum())

# 如果需要可以保存到新CSV
df.to_csv('/home/ubuntu/project/ViLa-MIL/dataset_csv/LGG_checked.csv', index=False)
