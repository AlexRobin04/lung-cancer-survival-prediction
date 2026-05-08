import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import font_manager as fm


def _pick_font():
    candidates = [
        "PingFang SC",
        "Hiragino Sans GB",
        "Heiti SC",
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            return name
    return "DejaVu Sans"


def create_academic_flowchart(output_png: str, output_svg: str):
    font_name = _pick_font()
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [font_name]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.weight"] = "bold"

    fig, ax = plt.subplots(figsize=(16, 9))
    fig.patch.set_facecolor("#F4F6FA")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60)
    ax.axis("off")

    def card(x, y, w, h, text, fc="#FFFFFF", ec="#D7DEE9", lw=1.2, fs=10, tc="#111111", bold=True):
        # soft shadow
        shadow = patches.FancyBboxPatch(
            (x + 0.5, y - 0.5), w, h,
            boxstyle="round,pad=0.25,rounding_size=0.7",
            ec="none", fc="#DDE3EE", alpha=0.35
        )
        ax.add_patch(shadow)
        box = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.25,rounding_size=0.7",
            ec=ec, fc=fc, lw=lw
        )
        ax.add_patch(box)
        ax.text(
            x + w / 2, y + h / 2, text,
            ha="center", va="center", fontsize=fs, color=tc,
            weight="bold" if bold else "semibold"
        )

    def diamond(cx, cy, dx, dy, text, ec="#2F79A9", fc="#F8FCFF", fs=10, tc="#111111"):
        poly = plt.Polygon(
            [[cx - dx, cy], [cx, cy + dy], [cx + dx, cy], [cx, cy - dy]],
            fc=fc, ec=ec, lw=1.4
        )
        ax.add_patch(poly)
        ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, color=tc, weight="bold")

    def arrow(x0, y0, x1, y1, lw=1.5, cs="arc3,rad=0.0", text=None, tx=None, ty=None):
        ax.annotate(
            "", xy=(x1, y1), xytext=(x0, y0),
            arrowprops=dict(arrowstyle="-|>", lw=lw, color="#445566", shrinkA=2, shrinkB=2, connectionstyle=cs)
        )
        if text:
            ax.text(tx, ty, text, fontsize=8.8, color="#334455", weight="bold")

    # ========================
    # 阶段一：基础融合
    # ========================
    rect_stage1 = patches.FancyBboxPatch(
        (3, 35.5),
        94,
        22,
        boxstyle="round,pad=0.5,rounding_size=1.0",
        ec="#BCC7D8",
        fc="#F8FBFF",
        lw=1.2,
    )
    ax.add_patch(rect_stage1)
    header1 = patches.FancyBboxPatch((36, 54.8), 28, 3.2, boxstyle="round,pad=0.25,rounding_size=0.8", ec="none", fc="#EAF2FF")
    ax.add_patch(header1)
    ax.text(
        50,
        56.4,
        "阶段一：概率层基础融合",
        ha="center",
        va="center",
        weight="bold",
        fontsize=13,
        color="#1E3350",
    )

    card(7, 44.8, 12, 6.4, "输入特征\nX", fc="#2E4A68", ec="#2E4A68", tc="white", fs=10.8, bold=True)
    card(24, 43.8, 17, 8.4, "五路基线\nMIL前向", fs=10.6)
    card(46, 43.8, 16, 8.4, "分支概率映射\nSoftmax", fs=10.3)
    diamond(73, 48.0, 7.3, 5.0, "avg_prob\n融合", fs=10.0)
    bubble = patches.Circle((89, 48), 3.4, fc="#FFFFFF", ec="#AAB6C5", lw=1.3)
    ax.add_patch(bubble)
    ax.text(89, 48, "r_base", ha="center", va="center", fontsize=10.2, color="#111111", weight="bold")

    arrow(19, 48, 24, 48)
    arrow(41, 48, 46, 48)
    arrow(62, 48, 65.8, 48)
    arrow(80.2, 48, 85.4, 48)

    # ========================
    # 阶段二：稳健增强
    # ========================
    rect_stage2 = patches.FancyBboxPatch(
        (3, 2.5),
        94,
        30,
        boxstyle="round,pad=0.5,rounding_size=1.0",
        ec="#C8D0DD",
        fc="#FFFFFF",
        lw=1.2,
    )
    ax.add_patch(rect_stage2)
    header2 = patches.FancyBboxPatch((35, 29.0), 30, 3.2, boxstyle="round,pad=0.25,rounding_size=0.8", ec="none", fc="#FFF1E7")
    ax.add_patch(header2)
    ax.text(
        50,
        30.5,
        "阶段二：稳健自适应增强",
        ha="center",
        va="center",
        weight="bold",
        fontsize=13,
        color="#5A341F",
    )

    # 下半区按“右 -> 左”排版：网格化位置，避免箭头与文字重叠
    card(79, 17.5, 16, 8.0, "历史排序评估\n(best / second)", fs=9.8)
    diamond(66.5, 21.5, 6.8, 5.0, "分支选择", fs=10.0)
    card(47, 22.8, 14, 5.0, "复用最强分支\np_best", fs=9.2)
    card(47, 15.8, 14, 5.0, "使用融合结果\np_fused", fs=9.2)
    card(29, 17.5, 15, 8.0, "轻量校准\nr_λ = r_best + λ·sign(Δ)", fs=8.9)
    diamond(19.5, 21.5, 5.8, 4.8, "稳健门控", fs=10.0)
    card(2.5, 5.5, 15, 6.0, "最终风险分数", fc="#2E4A68", ec="#2E4A68", tc="white", fs=10.8, bold=True)

    note = patches.FancyBboxPatch((30, 4.2), 19, 10.8, boxstyle="round,pad=0.25,rounding_size=0.7", fc="#FFFDF8", ec="#E1D8C8", lw=1.1)
    ax.add_patch(note)
    ax.text(
        39.5, 9.8,
        "门控约束\n• λ 幅度上限\n• 时间切分验证\n• 最小提升阈值\n• 全量退化回退 λ=0",
        ha="center", va="center", fontsize=8.4, color="#4D463D", weight="bold"
    )

    # 右 -> 左主流程连线
    arrow(79, 21.5, 73.2, 21.5)
    arrow(59.7, 21.5, 61.0, 24.9, text="可用", tx=60.8, ty=24.2)
    arrow(59.7, 21.5, 61.0, 18.3, text="不可用", tx=60.4, ty=18.6)
    arrow(47.0, 24.9, 44.1, 21.5)
    arrow(47.0, 18.3, 44.1, 21.5)
    arrow(29.0, 21.5, 25.3, 21.5)
    arrow(19.5, 16.7, 19.5, 8.6, text="通过", tx=20.8, ty=13.4)
    arrow(19.5, 8.6, 17.5, 8.6, lw=1.2, cs="arc3,rad=0.0")

    # 跨阶段桥接：由 r_base 向下接入“历史排序评估”（右侧入口），避免压在框边
    arrow(89.0, 44.6, 89.0, 30.0, lw=1.5, cs="arc3,rad=0.0")
    arrow(89.0, 30.0, 87.0, 30.0, lw=1.5, cs="arc3,rad=0.0")
    arrow(87.0, 30.0, 87.0, 25.5, lw=1.5, cs="arc3,rad=0.0")

    plt.tight_layout()
    plt.savefig(output_png, dpi=320, bbox_inches="tight", facecolor="white")
    plt.savefig(output_svg, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    create_academic_flowchart(
        output_png="EnsembleDecision_flowchart_paper.png",
        output_svg="EnsembleDecision_flowchart_paper.svg",
    )
