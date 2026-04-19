import argparse
import time
import signal
import sys
import torch
import math

def human(n):
    return f"{n / (1024**3):.2f} GB"

def allocate_mem(gb: float, device: int, chunk_mb: int = 256):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，请检查 CUDA / 驱动 / PyTorch 安装。")

    torch.cuda.set_device(device)
    total = torch.cuda.get_device_properties(device).total_memory
    free_before = torch.cuda.mem_get_info(device)[0]  # CUDA 可用的空闲显存（近似）

    bytes_needed = int(gb * (1024**3))
    chunk_bytes = int(chunk_mb * 1024**2)

    print(f"[INFO] 目标占用: {human(bytes_needed)}  | 每块: {chunk_mb} MB")
    print(f"[INFO] 设备 {device} 总显存: {human(total)} | 预计可分配前空闲: {human(free_before)}")

    if free_before < bytes_needed:
        print(f"[WARN] 空闲显存({human(free_before)}) 低于目标({human(bytes_needed)}). 将尝试分配，但可能 OOM。")

    tensors = []
    full_chunks = bytes_needed // chunk_bytes
    remainder = bytes_needed % chunk_bytes

    try:
        for _ in range(full_chunks):
            tensors.append(torch.empty(chunk_bytes, dtype=torch.uint8, device=device))
        if remainder > 0:
            tensors.append(torch.empty(remainder, dtype=torch.uint8, device=device))
        torch.cuda.synchronize()
    except RuntimeError as e:
        print("[ERROR] 分配过程中 OOM 或其它错误：", e)
        print("       可尝试：1) 调小 --gb 或 --chunk-mb；2) 关闭其它占用显存的程序。")
        sys.exit(1)

    used_after = torch.cuda.max_memory_allocated(device)
    free_after = torch.cuda.mem_get_info(device)[0]
    print(f"[OK] 已分配 ~{human(used_after)}（PyTorch 报告的峰值），当前空闲约: {human(free_after)}")
    print("[HOLD] 进程将保持占用。按 Ctrl+C 结束进程才会释放。")

    return tensors  # 把引用保留在作用域里，防止被 GC

# === 新增：轻量计算，零额外显存（安全对齐到4字节再view成float32） ===
@torch.inference_mode()
def light_compute_forever(tensors, device, tick_sleep: float = 0.0, iters_per_tick: int = 1):
    torch.cuda.set_device(device)
    print("[RUN] light 计算模式（零额外显存）...")
    while True:
        for t in tensors:
            # 关键点：对齐到4字节，避免 .view(torch.float32) 报错
            n = (t.numel() // 4) * 4
            if n == 0:
                continue
            buf = t[:n]  # 按4字节对齐后的缓冲区
            # 用 as_strided 按 float32 的步长映射到原内存，不会复制数据
            f32 = torch.as_strided(buf, size=(n // 4,), stride=(4,)).view(torch.float32)
            for _ in range(iters_per_tick):
                f32.mul_(1.0001)
                f32.add_(0.0003)
                f32.tanh_()
        torch.cuda.synchronize()
        if tick_sleep > 0:
            time.sleep(tick_sleep)

# === 新增：重计算，用小矩阵做 matmul，更容易拉高 GPU 利用率 ===
@torch.inference_mode()
def heavy_compute_forever(device: int,
    extra_compute_mb: int = 128,
    tick_sleep: float = 0.0,
    iters_per_tick: int = 1,
    duty_cycle: float = 0.5,     # 新增：占空比，0~1，0.5≈约一半负载
    work_scale: float = 1.0):
    torch.cuda.set_device(device)
    base_bytes = extra_compute_mb * (1024**2)
    n = max(512, int(((base_bytes / 8.0) ** 0.5)))
    if work_scale > 0 and work_scale != 1.0:
        n = max(256, int(n * math.sqrt(max(work_scale, 1e-3))))  # 缩边长 → 降算量
    print(f"[RUN] heavy(节流) 模式：工作区 ~{extra_compute_mb} MB，矩阵 n={n}，duty={duty_cycle}, scale={work_scale}")

    A = torch.randn((n, n), device=device, dtype=torch.float32)
    B = torch.randn((n, n), device=device, dtype=torch.float32)
    C = torch.empty_like(A)

    duty_cycle = float(min(max(duty_cycle, 1e-3), 1.0))  # clamp 到 (0,1]
    while True:
        t0 = time.perf_counter()
        for _ in range(iters_per_tick):
            C = A @ B
            C.tanh_()
            A.add_(1e-4).mul_(0.9999)
            B.add_(1e-4).mul_(1.0001)
        torch.cuda.synchronize()
        t_compute = time.perf_counter() - t0

        # 按占空比估算需要“休息”的时间，使 avg_util ≈ duty_cycle
        # 计算时间 : 总时间 = duty_cycle  ⇒  t_compute : (t_compute + t_idle) = duty_cycle
        # 解得 t_idle = t_compute * (1 - duty_cycle) / duty_cycle
        t_idle = t_compute * (1.0 - duty_cycle) / duty_cycle
        if tick_sleep > 0:
            t_idle += tick_sleep

        if t_idle > 0:
            time.sleep(t_idle)

def main():
    parser = argparse.ArgumentParser(description="占用指定 GPU 显存并保持不释放（可选持续运算）")
    parser.add_argument("--gb", type=float, default=7.0, help="需要占用的显存大小（GB），默认 8")
    parser.add_argument("--device", type=int, default=2, help="GPU 设备编号，默认 0")
    parser.add_argument("--chunk-mb", type=int, default=256, help="按块分配大小（MB），默认 256")
    # === 新增参数 ===
    parser.add_argument("--compute", choices=["none", "light", "heavy"], default="heavy",
                        help="运算强度：none/light/heavy（默认 light）")
    parser.add_argument("--extra-compute-mb", type=int, default=128,
                        help="heavy 模式的额外显存（MB），默认 128")
    parser.add_argument("--tick-sleep", type=float, default=0.0,
                        help="每轮计算后的 sleep 秒数，增大可降功耗，默认 0")
    parser.add_argument("--iters-per-tick", type=int, default=1,
                        help="每轮里重复计算的次数，增大可提高利用率，默认 1")
    args = parser.parse_args()

    def handler(signum, frame):
        print("\n[EXIT] 接收到信号，退出进程（显存将被释放）。")
        sys.exit(0)

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, handler)
        except Exception:
            pass

    tensors = allocate_mem(args.gb, args.device, args.chunk_mb)

    try:
        if args.compute == "none":
            print("[IDLE] 不做运算，仅占显存。")
            while True:
                time.sleep(3600)
        elif args.compute == "light":
            light_compute_forever(
                tensors=tensors,
                device=args.device,
                tick_sleep=args.tick_sleep,
                iters_per_tick=args.iters_per_tick,
            )
        elif args.compute == "heavy":
            heavy_compute_forever(
                device=args.device,
                extra_compute_mb=args.extra_compute_mb,
                tick_sleep=args.tick_sleep,
                iters_per_tick=args.iters_per_tick,
            )
    except KeyboardInterrupt:
        print("\n[EXIT] 用户中断，退出进程。")

if __name__ == "__main__":
    main()
