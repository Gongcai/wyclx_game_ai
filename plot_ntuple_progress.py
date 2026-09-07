"""把 N-Tuple 本轮（ext-* 续训）的训练与评测数据可视化到 runs/charts/。

数据源：
- runs/ntuple-*/metrics.jsonl            训练期 1-ply 每千步 9 数
- runs/eval-paired-ntuple-epsvsl2ext-nocap.json   无步数上限配对评测（128 局）
- runs/eval-paired-ntuple-epsvsl2ext.json         300 步上限配对评测（120 局，部分）

风格沿用 visualize_results.py（同一套调色板、去边框、浅网格）。
"""

import glob
import json
import os
import random
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

CAT = ["#2a78d6", "#199e70", "#d95926", "#4a3aa7", "#eda100", "#d55181", "#184f95", "#e34948"]
INK = "#1a1a19"
MUTED = "#898781"
GRID = "#e1e0d9"

CJK_CANDIDATES = ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "Source Han Sans SC", "Microsoft YaHei"]
available = {f.name for f in fm.fontManager.ttflist}
cjk = next((c for c in CJK_CANDIDATES if c in available), None)
if cjk:
    plt.rcParams["font.sans-serif"] = [cjk]
plt.rcParams["axes.unicode_minus"] = False

OUT = "runs/charts"
os.makedirs(OUT, exist_ok=True)

RUNS = [
    ("ntuple-ext-eps", "eps (ε下限 0.10)", CAT[0]),
    ("ntuple-ext-base", "base (ε 0.01)", CAT[2]),
    ("ntuple-ext-lam80", "lam80 (λ 0.8)", CAT[1]),
    ("ntuple-l2-ext", "l2-ext（旧纪录 27.33）", CAT[3]),
]


def style(ax, logy=False):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    if logy:
        ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  {path}")


def load_series(run_id):
    path = f"runs/{run_id}/metrics.jsonl"
    if not os.path.exists(path):
        return []
    out = []
    for line in open(path):
        r = json.loads(line)
        out.append((r["step"], r["ntuple"]["n9_per_1000_moves"]))
    return out


def rolling(vals, k=10):
    return [statistics.mean(vals[max(0, i - k + 1):i + 1]) for i in range(len(vals))]


# ── 图 1：训练曲线（1-ply 每千步） ────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5.5))
for run_id, label, color in RUNS:
    s = load_series(run_id)
    if not s:
        continue
    steps = [x / 1e6 for x, _ in s]
    vals = [v for _, v in s]
    ax.plot(steps, vals, color=color, alpha=0.18, linewidth=0.8)
    ax.plot(steps, rolling(vals), color=color, linewidth=2.2, label=label)
ax.axhline(27.33, color=MUTED, linestyle="--", linewidth=1.2)
ax.text(1.15, 28.2, "旧纪录 27.33 (l2-ext)", color=MUTED, fontsize=9)
ax.set_xlabel("训练步数（百万）")
ax.set_ylabel("1-ply 每千步 9 数（10 次 eval 滑动均值）")
ax.set_title("N-Tuple 续训：抬高 ε 下限是唯一有效杠杆", color=INK, fontsize=13, pad=12)
ax.legend(frameon=False, fontsize=9, loc="upper left")
style(ax)
save(fig, "nt-01-training-curves.png")

# ── 图 2：分段均值（每 500k） ────────────────────────────────────
edges = list(range(5_000_000, 8_500_000, 500_000))
fig, ax = plt.subplots(figsize=(10, 5))
width = 0.26
for i, (run_id, label, color) in enumerate(RUNS[:3]):
    s = load_series(run_id)
    means, present = [], []
    for a in edges:
        seg = [v for st, v in s if a <= st < a + 500_000]
        if seg:
            means.append(statistics.mean(seg))
            present.append(a)
    xs = [p / 1e6 + (i - 1) * width for p in present]
    ax.bar(xs, means, width=width, color=color, label=label)
    for x, m in zip(xs, means):
        ax.text(x, m + 0.6, f"{m:.1f}", ha="center", fontsize=8, color=MUTED)
ax.set_xticks([e / 1e6 + 0.25 for e in edges])
ax.set_xticklabels([f"{e/1e6:.1f}-{(e+500_000)/1e6:.1f}M" for e in edges])
ax.set_xlabel("训练步数区间")
ax.set_ylabel("1-ply 每千步 9 数（区间均值）")
ax.set_title("分段均值：eps 持续上升，base 缓慢，lam80 死平", color=INK, fontsize=13, pad=12)
ax.legend(frameon=False, fontsize=9)
style(ax)
save(fig, "nt-02-segmented-means.png")

# ── 图 3/4：无上限评测的分布 ─────────────────────────────────────
d = json.load(open("runs/eval-paired-ntuple-epsvsl2ext-nocap.json"))
per = {}
for r in d["rows"]:
    per.setdefault(r["model"], []).append(r)
order = ["eps-final", "eps-best", "l2ext", "puct-r2"]
colors = {m: CAT[i] for i, m in enumerate(order)}
pretty = {"eps-final": "eps-final (8M)", "eps-best": "eps-best (7.74M)",
          "l2ext": "l2-ext（旧候选）", "puct-r2": "PUCT r2（现部署）"}

for key, title, ylabel, fname in [
    ("n9", "每局 9 数分布（无步数上限，32 seed）", "每局合成 9 的数量（对数轴）", "nt-03-n9-distribution.png"),
    ("moves", "存活步数分布（无步数上限，32 seed）", "存活步数（对数轴）", "nt-04-survival-distribution.png"),
]:
    fig, ax = plt.subplots(figsize=(9, 5.2))
    data = [[r[key] for r in per[m]] for m in order]
    bp = ax.boxplot(data, orientation="vertical", patch_artist=True, widths=0.5, showfliers=False)
    for patch, m in zip(bp["boxes"], order):
        patch.set_facecolor(colors[m])
        patch.set_alpha(0.55)
        patch.set_edgecolor(colors[m])
    for med in bp["medians"]:
        med.set_color(INK)
    rnd = random.Random(0)
    for i, (m, vals) in enumerate(zip(order, data)):
        xs = [i + 1 + rnd.uniform(-0.17, 0.17) for _ in vals]
        ax.scatter(xs, vals, s=14, color=colors[m], alpha=0.75, edgecolor="white", linewidth=0.4, zorder=3)
        ax.text(i + 1, max(vals) * 1.12, f"均值 {statistics.mean(vals):.1f}",
                ha="center", fontsize=9, color=INK)
    ax.set_yscale("log")
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels([pretty[m] for m in order])
    ax.set_ylabel(ylabel)
    ax.set_title(title, color=INK, fontsize=13, pad=12)
    style(ax, logy=True)
    save(fig, fname)

# ── 图 5：配对差森林图 ───────────────────────────────────────────
byseed = {}
for r in d["rows"]:
    byseed.setdefault(r["seed"], {})[r["model"]] = r


def paired(a, b, key="n9"):
    ds = [byseed[s][a][key] - byseed[s][b][key] for s in sorted(byseed)
          if a in byseed[s] and b in byseed[s]]
    n = len(ds)
    mean = statistics.mean(ds)
    rnd = random.Random(0)
    boots = sorted(statistics.mean([ds[rnd.randrange(n)] for _ in range(n)]) for _ in range(10000))
    return mean, boots[250], boots[9750]


pairs = [("eps-final", "l2ext"), ("eps-best", "l2ext"), ("eps-final", "puct-r2"),
         ("eps-best", "puct-r2"), ("l2ext", "puct-r2"), ("eps-final", "eps-best")]
fig, ax = plt.subplots(figsize=(9.5, 4.8))
for i, (a, b) in enumerate(pairs):
    mean, lo, hi = paired(a, b)
    y = len(pairs) - i
    sig = lo > 0 or hi < 0
    color = CAT[0] if sig else MUTED
    ax.plot([lo, hi], [y, y], color=color, linewidth=2.2, solid_capstyle="round")
    ax.scatter([mean], [y], color=color, s=52, zorder=3)
    ax.text(hi + 2.0, y, f"{mean:+.1f}  [{lo:+.1f}, {hi:+.1f}]{'  *' if sig else '  n.s.'}",
            va="center", fontsize=9, color=INK if sig else MUTED)
ax.axvline(0, color=MUTED, linewidth=1, linestyle="--")
ax.set_yticks(range(1, len(pairs) + 1))
ax.set_yticklabels([f"{a} − {b}" for a, b in reversed(pairs)])
ax.set_xlabel("配对每局 9 数之差（挑战者 − 基线），bootstrap 95% CI")
ax.set_title("配对差：主结论显著，best vs final 无法区分", color=INK, fontsize=13, pad=12)
ax.set_xlim(-30, 115)
style(ax)
save(fig, "nt-05-paired-forest.png")

# ── 图 6：300 步上限的截断影响 ───────────────────────────────────
cap = json.load(open("runs/eval-paired-ntuple-epsvsl2ext.json"))
capper = {}
for r in cap["rows"]:
    capper.setdefault(r["model"], []).append(r)

fig, ax = plt.subplots(figsize=(9, 5))
width = 0.35
xs = range(len(order))
for i, (src, label, color) in enumerate([(capper, "300 步上限", CAT[2]), (per, "无上限", CAT[0])]):
    means = [statistics.mean([r["n9"] for r in src[m]]) for m in order]
    pos = [x + (i - 0.5) * width for x in xs]
    ax.bar(pos, means, width=width, color=color, label=label)
    for x, m in zip(pos, means):
        ax.text(x, m + 1.2, f"{m:.1f}", ha="center", fontsize=9, color=INK)
ax.set_xticks(list(xs))
ax.set_xticklabels([pretty[m] for m in order])
ax.set_ylabel("每局合成 9 的数量")
ax.set_title("300 步上限把真实差距压掉了数倍", color=INK, fontsize=13, pad=12)
ax.legend(frameon=False, fontsize=9)
style(ax)
save(fig, "nt-06-cap-effect.png")

print("完成")
