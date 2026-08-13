#!/bin/bash
# ============================================================
# 8 卡平台：长局 bootstrap 一轮作业脚本（SLURM / 模板提交通用）
#
# 用法一（SLURM）:
#   sbatch --gres=dcu:8 --partition=<队列> --time=08:00:00 \
#          deploy/slurm_bootstrap.sh --round 1
# 用法二（模板提交）: 在平台的作业模板里填下面这一行命令行
#   bash deploy/slurm_bootstrap.sh --round 1
#
# 模板表单通常还需要：队列/分区、GPU 数(8)、节点数(1)、时长(≥8h)、
# 工作目录(仓库根目录)、Python 环境(venv 路径)。
# ============================================================

set -euo pipefail

# ---------- 用户需按平台调整 ----------
VENV="${VENV:-$PWD/.venv}"                 # Python 环境（海光平台装 DTK 版 torch）
CONDA_ENV="${CONDA_ENV:-dcu}"              # 若用 conda 环境（海光推荐），设非空即激活它
DTK_MODULE="${DTK_MODULE:-compiler/dtk/25.04}"   # 与 torch das 版本匹配的 DTK 模块
GPUS="${GPUS:-8}"                          # 本作业使用的 GPU 数
ROUND="${ROUND:-1}"                        # bootstrap 轮次（决定 max-moves 与产物名）
BASE_MODEL="${BASE_MODEL:-runs/demos/high-halving8-gumbel48-pv.pt}"
DIST="${DIST:-high}"
# 海光 DCU 平台：加载 DTK 模块 + 环境（SLURM --gres=dcu:8 会自动设置可见设备）
module load "$DTK_MODULE"
source /opt/hygon/env.sh
# Python 解释器：conda 环境（用路径直接调用，不依赖 conda activate）→ venv → PATH
if [ -n "${CONDA_ENV:-}" ]; then
    module load anaconda3/2023.09
    PY="$(dirname "$(dirname "$(which conda)")")/envs/$CONDA_ENV/bin/python"
    [ -x "$PY" ] || PY=python
else
    PY="$VENV/bin/python"
    [ -x "$PY" ] || PY=python
fi
# ---------------------------------------

# 每轮拉长局上限：r1=2000, r2=4000, r3=8000 ...
MAX_MOVES=$((2000 * ROUND))
EPISODES_PER_GPU=50
SELFPLAY_OUT="runs/demos/long-bootstrap-r${ROUND}.pt"
MODEL_OUT="runs/demos/high-h256-bootstrap-r${ROUND}-pv.pt"
PAIRED_OUT="runs/eval-paired-h256-r${ROUND}.json"
SEED_BASE=$((600000 + ROUND * 1000000))

cd "$(dirname "$0")/.."                    # 进入仓库根目录
mkdir -p runs/demos
echo "===== 轮次 $ROUND: max_moves=$MAX_MOVES, gpus=$GPUS, seed=$SEED_BASE ====="

# 0) 平台冒烟（失败即退出，避免空跑 4 小时）
"$PY" deploy/smoke_dcu.py --model "$BASE_MODEL"
echo "冒烟通过，开始长局自对弈..."

# 1) 多卡长局自我对弈（8 卡并行，约 4 小时）
"$PY" launch_selfplay.py \
    --model "$BASE_MODEL" \
    --gpus "$GPUS" \
    --episodes-per-gpu "$EPISODES_PER_GPU" \
    --max-moves "$MAX_MOVES" \
    --dist "$DIST" \
    --seed "$SEED_BASE" \
    --out "$SELFPLAY_OUT"

# 2) 更大网络从零训练（hidden 256，混入旧长局示范）
"$PY" train_policy_value.py \
    --demos "$SELFPLAY_OUT" \
           runs/demos/high-long-64.pt \
           runs/demos/high-long-64-s1000.pt \
           runs/demos/high-halving8-selfplay48.pt \
    --hidden 256 \
    --mature-policy-weight 2.0 \
    --high-tile-policy-weight 1.0 \
    --elite-score-threshold 10 \
    --elite-policy-weight 2.0 \
    --death-horizon 32 \
    --death-w 0.5 \
    --out "$MODEL_OUT"

# 3) 配对评测 vs 当前推荐（CI 为正才考虑替换）
"$PY" eval_paired_puct.py \
    --models gumbel48="$BASE_MODEL" \
             "h256-r${ROUND}=$MODEL_OUT" \
    --tree-reuse-models gumbel48 "h256-r${ROUND}" \
    --episodes 128 --seed "$SEED_BASE" --max-moves 500 \
    --simulations 128 --depth 16 --chance-samples 8 \
    --chance-widening 0.5 --root-min-visits 2 --death-penalty 0.5 \
    --json-out "$PAIRED_OUT" --max-new 64

echo "===== 轮次 $ROUND 完成。查看 $PAIRED_OUT 的配对差与 95% CI ====="
