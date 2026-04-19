import torch
import time

# 设定运行时间（秒）
target_runtime = 40 * 60  # 20分钟

import os
# 仅设置一块可见
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda:1')

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'


# 创建占用大约20GB的张量
# 20GB / 4B = 元素数量
num_elements = 6 * 1024**3 // 4
big_tensor = torch.randn(num_elements, dtype=torch.float32, device=device)

print(f"Allocated tensor with {big_tensor.numel()} elements, about {big_tensor.element_size() * big_tensor.numel() / 1024**3:.2f} GB.")

# 一个简单的循环运算，避免被认为是空闲
start_time = time.time()
iteration = 0

print("Starting computation...")

while time.time() - start_time < target_runtime:
    big_tensor = big_tensor * 1.000001 + 0.000001  # 做简单的无害操作
    iteration += 1
    if iteration % 100 == 0:
        print(f"Iteration {iteration}, elapsed time {int(time.time() - start_time)}s")
    torch.cuda.synchronize()  # 保证计算真的执行，而不是延迟

print("Finished 20 minutes of execution.")
