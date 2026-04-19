import shutil
from pathlib import Path

def move_all_svs_files(src_dir, dst_dir):
    src_path = Path(src_dir)
    dst_path = Path(dst_dir)
    dst_path.mkdir(parents=True, exist_ok=True)  # 创建目标文件夹

    # 遍历所有子目录下的.svs文件
    for svs_file in src_path.rglob("*.svs"):
        target = dst_path / svs_file.name
        print(f"Moving: {svs_file} -> {target}")
        shutil.move(str(svs_file), str(target))

# 示例用法：将当前目录所有.svs文件移动到 ./all_svs 目录
move_all_svs_files("/media/ubuntu/015eb652-0471-4534-8f64-2a82c75fed771/yuehailin/private/beijing/Double", "/media/ubuntu/015eb652-0471-4534-8f64-2a82c75fed771/yuehailin/private/beijing/wsi")
