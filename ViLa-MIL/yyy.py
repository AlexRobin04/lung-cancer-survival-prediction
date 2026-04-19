svs_count = 0

with open('/media/ubuntu/015eb652-0471-4534-8f64-2a82c75fed771/yuehailin/TCGA_wsi_txt/gdc_manifest_brca.txt', 'r', encoding='utf-8') as file:
    for line in file:
        columns = line.strip().split()  # 默认以空白字符分割
        if len(columns) >= 2:
            second_column = columns[1]
            print(second_column)
            if second_column.endswith('.svs'):
                svs_count += 1

print(f"\n共有 {svs_count} 个以 .svs 结尾的项。")
