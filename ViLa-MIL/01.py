# -*- coding: utf-8 -*-
"""
KM 曲线：使用 lusc (1).csv（列：statue, time, other, ours）。
默认绘制 Ours vs Others；可通过「baseline」单选切换哪一条作为参照曲线样式。

说明：数据中部分样本 other=1 且 ours=1（重叠），两条 KM 为各自标签下的子队列生存曲线；
     log-rank 在互斥子集（仅 ours / 仅 others）上计算，并在控制台打印样本量。
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

try:
    from matplotlib.widgets import RadioButtons
except ImportError:
    RadioButtons = None

# 中文字体（无则回退）
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 12

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "lusc (1).csv"


def load_data(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    if not {"time", "statue", "ours", "other"}.issubset(df.columns):
        raise ValueError(f"CSV 需包含列: time, statue, ours, other；当前为 {list(df.columns)}")
    time = df["time"].to_numpy(dtype=float)
    event = df["statue"].to_numpy(dtype=int)
    ours = (df["ours"].to_numpy(dtype=int) == 1)
    other = (df["other"].to_numpy(dtype=int) == 1)
    return time, event, ours, other


def fit_km(
    time: np.ndarray,
    event: np.ndarray,
    mask: np.ndarray,
    label: str,
) -> KaplanMeierFitter:
    kmf = KaplanMeierFitter()
    kmf.fit(time[mask], event_observed=event[mask], label=label)
    return kmf


def draw_km(
    ax,
    km_ours: KaplanMeierFitter,
    km_other: KaplanMeierFitter,
    baseline: str,
) -> None:
    ax.clear()
    # baseline：虚线 + 深色；对比：实线 + 醒目色
    if baseline.lower() == "others":
        km_other.plot_survival_function(
            ax=ax,
            color="#37474F",
            linewidth=2.8,
            ci_show=True,
            linestyle="--",
            label="Others (baseline)",
        )
        km_ours.plot_survival_function(
            ax=ax,
            color="#E53935",
            linewidth=2.8,
            ci_show=True,
            linestyle="-",
            label="Ours",
        )
    else:
        km_ours.plot_survival_function(
            ax=ax,
            color="#37474F",
            linewidth=2.8,
            ci_show=True,
            linestyle="--",
            label="Ours (baseline)",
        )
        km_other.plot_survival_function(
            ax=ax,
            color="#1E88E5",
            linewidth=2.8,
            ci_show=True,
            linestyle="-",
            label="Others",
        )

    ax.set_xlabel("Time (months)")
    ax.set_ylabel("Survival probability")
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", framealpha=0.9)


def print_stats(
    time: np.ndarray,
    event: np.ndarray,
    ours: np.ndarray,
    other: np.ndarray,
) -> None:
    n_ours = int(ours.sum())
    n_other = int(other.sum())
    both = int((ours & other).sum())
    ex_ours = ours & ~other
    ex_other = other & ~ours
    n_ex_o = int(ex_ours.sum())
    n_ex_t = int(ex_other.sum())

    print("=== 样本量 ===")
    print(f"  ours==1: {n_ours}  |  other==1: {n_other}  | 同时=1: {both}")
    print(f"  互斥: 仅 ours ({n_ex_o}) vs 仅 others ({n_ex_t}) — 用于 log-rank")

    if n_ex_o > 0 and n_ex_t > 0:
        lr = logrank_test(
            time[ex_ours],
            time[ex_other],
            event_observed_A=event[ex_ours],
            event_observed_B=event[ex_other],
        )
        print(f"  Log-rank p（互斥子集）: {float(lr.p_value):.6g}")
    else:
        print("  Log-rank（互斥子集）: 样本不足，跳过")


def run_interactive(
    km_ours: KaplanMeierFitter,
    km_other: KaplanMeierFitter,
) -> None:
    if RadioButtons is None:
        print("matplotlib.widgets 不可用，跳过交互。", file=sys.stderr)
        return

    fig, ax = plt.subplots(figsize=(10, 7))
    plt.subplots_adjust(left=0.22, bottom=0.12)

    baseline = ["others"]

    def redraw(label: str) -> None:
        baseline[0] = label.lower()
        draw_km(ax, km_ours, km_other, baseline[0])
        fig.canvas.draw_idle()

    draw_km(ax, km_ours, km_other, "others")
    rax = plt.axes([0.02, 0.45, 0.16, 0.12])
    radio = RadioButtons(rax, ("Others baseline", "Ours baseline"), active=0)

    def on_click(label: str) -> None:
        if "Others" in label:
            redraw("others")
        else:
            redraw("ours")

    radio.on_clicked(on_click)
    plt.suptitle("KM：Ours vs Others（切换 baseline）", fontsize=14)
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="KM 曲线：lusc (1).csv")
    parser.add_argument(
        "--csv",
        type=Path,
        default=CSV_PATH,
        help="CSV 路径",
    )
    parser.add_argument(
        "--baseline",
        choices=("others", "ours"),
        default="others",
        help="非交互模式下保存图时的 baseline（默认 others）",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="弹出窗口，用单选按钮切换 baseline（需图形界面）",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="输出 PNG 路径（默认写入脚本目录）",
    )
    args = parser.parse_args()

    time, event, ours, other = load_data(args.csv)
    print_stats(time, event, ours, other)

    km_ours = fit_km(time, event, ours, "Ours")
    km_other = fit_km(time, event, other, "Others")

    if args.interactive:
        if os.environ.get("DISPLAY") or sys.platform == "darwin":
            run_interactive(km_ours, km_other)
            return
        print("未检测到 DISPLAY，已改为保存 PNG 文件。", file=sys.stderr)

    # 保存两张图（两种 baseline），便于无 GUI 环境直接查看
    out_dir = args.out.parent if args.out else BASE_DIR
    stem = (args.out.stem if args.out else "KM_lusc")
    if args.out:
        fig, ax = plt.subplots(figsize=(10, 7))
        draw_km(ax, km_ours, km_other, args.baseline)
        plt.tight_layout()
        plt.savefig(args.out, dpi=300, bbox_inches="tight")
        print(f"已保存: {args.out.resolve()}")
        plt.close(fig)
        return

    for bl in ("others", "ours"):
        fig, ax = plt.subplots(figsize=(10, 7))
        draw_km(ax, km_ours, km_other, bl)
        plt.tight_layout()
        path = out_dir / f"{stem}_baseline_{bl}.png"
        plt.savefig(path, dpi=300, bbox_inches="tight")
        print(f"已保存: {path.resolve()}")
        plt.close(fig)


if __name__ == "__main__":
    main()
