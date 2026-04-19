import subprocess
import os

# 每个脚本和其对应使用的GPU编号（手动指定）
tasks = [
    # ('main_BLCA.py', 3),
    # ('main_BLCA1.py', 0),
    # ('main_BRCA.py', 3),
    # ('main_BRCA1.py', 0),
    # ('main_COAD.py', 3),
    # ('main_COAD1.py', 0),

    # ('main_ESCA.py', 3),
    # ('main_ESCA1.py', 1),
    # ('main_HNSC.py', 3),
    # ('main_HNSC1.py', 1),
    # ('main_KIRC.py', 3)
    # ('main_KIRC1.py', 1),


    ('main_LGG.py', 3),
    # ('main_LGG1.py', 2),
    ('main_LIHC.py', 3),
    # ('main_LIHC1.py', 2),
    ('main_LUAD.py', 3),
    # ('main_LUAD1.py', 2),


    ('main_LUSC.py', 3),
    # ('main_LUSC1.py', 3),
    ('main_STAD.py', 3),
    # ('main_STAD1.py', 3),
    ('main_UCEC.py', 3)
    # ('main_UCEC1.py', 3)
   
]

for script, gpu_id in tasks:
    log_file = f'{os.path.splitext(script)[0]}.log'  # e.g., a.log
    cmd = f'nohup bash -c "CUDA_VISIBLE_DEVICES={gpu_id} python {script}" > {log_file} 2>&1 &'
    subprocess.call(cmd, shell=True)

print("所有脚本已分配GPU并用 nohup 启动。")
