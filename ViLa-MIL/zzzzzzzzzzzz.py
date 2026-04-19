import pandas as pd

def filter_txt_by_csv(csv_path, txt_path, output_path=None, txt_sep='\t'):
    # 1. 读取 CSV 中 missing_slide_id 列
    csv_df = pd.read_csv(csv_path)
    if 'missing_slide_id' not in csv_df.columns:
        raise ValueError("CSV 文件中未找到 'missing_slide_id' 列")
    missing_ids = set(csv_df['missing_slide_id'].dropna().astype(str))

    # 2. 读取 TXT 中的 filename 列
    txt_df = pd.read_csv(txt_path, sep=txt_sep)
    if 'filename' not in txt_df.columns:
        raise ValueError("TXT 文件中未找到 'filename' 列")

    # 3. 筛选：仅保留 filename 在 missing_slide_id 中的行
    filtered_df = txt_df[txt_df['filename'].astype(str).isin(missing_ids)]

    # 4. 写入结果
    if output_path is None:
        output_path = txt_path  # 覆盖原文件
    filtered_df.to_csv(output_path, sep=txt_sep, index=False)

    print(f"已保留 {len(filtered_df)} 行，写入到 {output_path}")

# 示例调用
filter_txt_by_csv("/home/ubuntu/project/ViLa-MIL/LGG_missing_slide_ids1.csv", "/home/ubuntu/project/ViLa-MIL/datasets_csv/gdc_manifest_gbm.txt")
