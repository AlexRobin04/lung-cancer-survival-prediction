import pandas as pd

def add_svs_suffix_to_missing_slide_id(csv_path, output_path=None):
    # 读取CSV文件
    df = pd.read_csv(csv_path)

    # 检查是否有 'missing_slide_id' 列
    if 'missing_slide_id' not in df.columns:
        raise ValueError("CSV中没有名为 'missing_slide_id' 的列。")

    # 在原列后添加一个加了 .svs 后缀的新列
    df['missing_slide_id'] = df['missing_slide_id'].astype(str) + '.svs'

    # 保存结果（可选）
    if output_path:
        df.to_csv(output_path, index=False)
    
    return df

# 示例调用
df = add_svs_suffix_to_missing_slide_id("/home/ubuntu/project/ViLa-MIL/LGG_missing_slide_ids.csv", "/home/ubuntu/project/ViLa-MIL/LGG_missing_slide_ids1.csv")
