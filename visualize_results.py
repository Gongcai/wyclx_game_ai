"""把 runs/ 下的评测数据可视化为 2D/3D 图表（PNG 输出到 runs/charts/）。

数据源：
- runs/competition-*.json     各策略比赛吞吐（每千步/死亡/首9/决策/30分钟估算）
- runs/eval-paired-*.json     配对评测（逐 seed 每模型 n9/moves）
- runs/death-analysis-*.jsonl 死亡分析（死因/死亡前列高）

用 dataviz 技能验证过的分类调色板；每图单轴；分类色固定顺序。
"""

import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# 验证过的分类调色板（dataviz 参考调色板，固定顺序）
CAT = ["#2a78d6", "#199e70", "#d95926", "#4a3aa7", "#eda100", "#d55181", "#184f95", "#e34948"]
INK = "#1a1a19"
MUTED = "#898781"
GRID = "#e1e0d9"

# CJK 字体（Arch 常见），找不到就退回英文
import matplotlib.font_manager as fm
CJK_CANDIDATES = ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "Source Han Sans SC", "Microsoft YaHei"]
available = {f.name for f in fm.fontManager.ttflist}
cjk = next((c for c in CJK_CANDIDATES if c in available), None)
if cjk:
    plt.rcParams["font.sans-serif"] = [cjk]
    plt.rcParams["axes.unicode_minus"] = False

OUT = "runs/charts"
os.makedirs(OUT, exist_ok=True)
figs = []


def style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for label in ax.get_xticklabels():
        label.set_rotation(30 if len(ax.get_xticklabels()) > 6 else 0)
        label.set_ha("right")


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  {path}")
    return path


def load_competition():
    rows = []
    for f in sorted(glob.glob("runs/competition-*.json")):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        rows.append({
            "name": os.path.basename(f).replace("competition-", "").replace(".json", ""),
            "per1000": d.get("n9_per_1000", 0), "deaths": d.get("deaths", 0),
            "first9": d.get("first9_avg_moves", 0), "gap": d.get("post9_avg_gap", 0),
            "decision_ms": d.get("avg_decision_seconds", 0) * 1000,
            "est30m": d.get("estimate_30m", 0), "n9": d.get("n9", 0),
            "moves": d.get("moves", 0),
        })
    return rows


def load_paired():
    rows = []
    for f in sorted(glob.glob("runs/eval-paired-*.json")):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        for r in d.get("rows", []):
            rows.append({
                "eval": os.path.basename(f).replace("eval-paired-", "").replace(".json", ""),
                "model": r["model"], "seed": r["seed"], "n9": r["n9"],
                "moves": r["moves"], "dead": r["dead"],
            })
    return rows


def load_deaths():
    rows = []
    for f in sorted(glob.glob("runs/death-analysis-*.jsonl")):
        try:
            for line in open(f):
                rows.append(json.loads(line))
        except Exception:
            continue
    return rows


def chart_competition_bars(rows):
    """每千步吞吐 + 30分钟估算（两个独立条形图，单轴原则）"""
    rows = sorted(rows, key=lambda r: -r["per1000"])
    names = [r["name"] for r in rows]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(names, [r["per1000"] for r in rows], color=CAT[0], width=0.62)
    for b, r in zip(bars, rows):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.4,
                f"{r['per1000']:.1f}", ha="center", fontsize=8, color=INK)
    ax.set_ylabel("每千步合成 9 数")
    ax.set_title("各策略比赛吞吐（每千步 9 数）")
    style(ax)
    save(fig, "01_competition_per1000.png")

    rows2 = sorted(rows, key=lambda r: -r["est30m"])
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar([r["name"] for r in rows2], [r["est30m"] for r in rows2],
                  color=CAT[1], width=0.62)
    for b, r in zip(bars, rows2):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1,
                f"{r['est30m']:.0f}", ha="center", fontsize=8, color=INK)
    ax.axhline(150, color=CAT[3], linewidth=1.2, linestyle="--")
    ax.text(0.01, 152, "目标 150", fontsize=8, color=CAT[3])
    ax.set_ylabel("30 分钟估算合成 9 数")
    ax.set_title("各策略 30 分钟估算（含决策延迟）")
    style(ax)
    save(fig, "02_competition_est30m.png")


def chart_competition_scatter3d(rows):
    """3D：每千步 × 平均存活 × 30分钟估算，气泡=决策时间"""
    fig = plt.figure(figsize=(9, 6.5))
    ax = fig.add_subplot(111, projection="3d")
    for i, r in enumerate(rows):
        steps_per_game = r["moves"] / max(1, r["deaths"] + 1) if r["moves"] else 0
        size = max(40, r["decision_ms"] * 0.6)
        ax.scatter(r["per1000"], steps_per_game, r["est30m"],
                   color=CAT[i % len(CAT)], s=size, alpha=0.85,
                   label=f"{r['name']} ({r['decision_ms']:.0f}ms)")
    ax.set_xlabel("每千步 9 数", fontsize=9)
    ax.set_ylabel("平均存活步数", fontsize=9)
    ax.set_zlabel("30分钟估算", fontsize=9)
    ax.set_title("策略空间：吞吐 × 存活 × 30分钟（气泡=决策时间）")
    ax.legend(fontsize=7, loc="upper left")
    save(fig, "03_competition_3d.png")


def chart_paired_scatter(rows):
    """配对评测散点：同一 eval 的逐 seed n9 对比（每图一对模型）"""
    evals = sorted({r["eval"] for r in rows})
    shown = 0
    for ev in evals:
        sub = [r for r in rows if r["eval"] == ev]
        models = sorted({r["model"] for r in sub})
        if len(models) != 2:
            continue
        a, b = models
        da = {r["seed"]: r["n9"] for r in sub if r["model"] == a}
        db = {r["seed"]: r["n9"] for r in sub if r["model"] == b}
        seeds = sorted(set(da) & set(db))
        if len(seeds) < 4:
            continue
        xs = [da[s] for s in seeds]
        ys = [db[s] for s in seeds]
        fig, ax = plt.subplots(figsize=(6.5, 6))
        ax.scatter(xs, ys, s=22, color=CAT[0], alpha=0.8, edgecolors="white", linewidths=0.4)
        lim = max(max(xs), max(ys), 1)
        ax.plot([0, lim], [0, lim], color=MUTED, linewidth=1, linestyle="--")
        wins = sum(y > x for x, y in zip(xs, ys))
        ax.text(0.03, 0.94, f"{a}: {sum(xs) / len(xs):.2f}  {b}: {sum(ys) / len(ys):.2f}\n"
                f"{b} 胜 {wins}/{len(seeds)}", transform=ax.transAxes,
                fontsize=9, color=INK, va="top")
        ax.set_xlabel(f"{a} 每局 9 数", fontsize=9)
        ax.set_ylabel(f"{b} 每局 9 数", fontsize=9)
        ax.set_title(f"配对评测 {ev}")
        ax.set_xlim(0, lim + 1)
        ax.set_ylim(0, lim + 1)
        style(ax)
        save(fig, f"04_paired_{ev}.png")
        shown += 1


def chart_game_lengths(rows):
    """单局长度分布直方图（对比每个评测里的模型）"""
    for ev in sorted({r["eval"] for r in rows})[:2]:
        sub = [r for r in rows if r["eval"] == ev]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for i, model in enumerate(sorted({r["model"] for r in sub})):
            moves = [r["moves"] for r in sub if r["model"] == model]
            ax.hist(moves, bins=16, alpha=0.55, color=CAT[i % len(CAT)],
                    label=f"{model} (n={len(moves)})", edgecolor="white", linewidth=0.4)
        ax.set_xlabel("单局步数", fontsize=9)
        ax.set_ylabel("局数", fontsize=9)
        ax.set_title(f"单局长度分布 {ev}")
        ax.legend(fontsize=8)
        style(ax)
        save(fig, f"05_length_{ev}.png")


def chart_deaths(rows):
    """死亡原因分布 + 死亡前列高直方图"""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    causes = {}
    for r in rows:
        causes[r.get("cause", "unknown")] = causes.get(r.get("cause", "unknown"), 0) + 1
    items = sorted(causes.items(), key=lambda kv: -kv[1])
    bars = ax.bar([k for k, _ in items], [v for _, v in items],
                  color=[CAT[0], CAT[1]], width=0.55)
    for b, (k, v) in zip(bars, items):
        ax.text(b.get_x() + b.get_width() / 2, v + 1, str(v), ha="center",
                fontsize=9, color=INK)
    ax.set_ylabel("死亡次数")
    ax.set_title("死亡原因分布（全部死亡分析汇总）")
    style(ax)
    save(fig, "06_death_causes.png")

    # 死亡前列高分布（从 death-analysis-*.jsonl 的 tail 最后一步）
    pre_heights = []
    for r in rows:
        tail = r.get("tail")
        if tail:
            stacks = tail[-1].get("pre_stacks")
            if stacks:
                pre_heights.extend(len(s) for s in stacks)
    if pre_heights:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.hist(pre_heights, bins=range(0, 10), color=CAT[2], alpha=0.85,
                edgecolor="white", linewidth=0.5)
        ax.set_xlabel("死亡前各列高度", fontsize=9)
        ax.set_ylabel("列数")
        ax.set_title("死亡前盘面列高分布")
        style(ax)
        save(fig, "07_death_heights.png")


def main():
    print("生成图表到 runs/charts/ ...")
    comp = load_competition()
    paired = load_paired()
    deaths = load_deaths()
    print(f"  数据: competition={len(comp)} 配对行={len(paired)} 死亡={len(deaths)}")
    if comp:
        chart_competition_bars(comp)
        chart_competition_scatter3d(comp)
    if paired:
        chart_paired_scatter(paired)
        chart_game_lengths(paired)
    if deaths:
        chart_deaths(deaths)
    print("完成")


if __name__ == "__main__":
    main()
