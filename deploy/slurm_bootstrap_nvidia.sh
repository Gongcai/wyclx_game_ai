#!/bin/bash
# ============================================================
# NVIDIA 平台：长局 bootstrap 一轮作业脚本（SLURM / 模板提交）
#
# 用法一（SLURM）:
#   sbatch --gres=gpu:8 --partition=<队列> --time=04:00:00 \
#          deploy/slurm_bootstrap_nvidia.sh --round 1
# 用法二（模板提交）: 命令行填
#   bash deploy/slurm_bootstrap_nvidia.sh --round 1
#
# 前置：标准 venv（pip 装 torch+cuda 版即可，代码为 CUDA 原生，零适配）
# ============================================================

set -euo pipefail

# ---------- 按平台调整 ----------
VENV="${VENV:-$PWD/.venv}"                 # venv 路径
GPUS="${GPUS:-8}"                          # GPU 数（--gres=gpu:N 需一致）
ROUND="${ROUND:-1}"                        # bootstrap 轮次
BASE_MODEL="${BASE_MODEL:-runs/demos/high-halving8-gumbel48-pv.pt}"
DIST="${DIST:-high}"
# ---------------------------------------

PY="$VENV/bin/python"
[ -x "$PY" ] || PY=python3

MAX_MOVES=$((2000 * ROUND))
EPISODES_PER_GPU=50
SELFPLAY_OUT="runs/demos/long-bootstrap-r${ROUND}.pt"
MODEL_OUT="runs/demos/high-h256-bootstrap-r${ROUND}-pv.pt"
PAIRED_OUT="runs/eval-paired-h256-r${ROUND}.json"
SEED_BASE=$((600000 + ROUND * 1000000))

cd "$(dirname "$0")/.."
mkdir -p runs/demos
echo "===== 轮次 $ROUND: max_moves=$MAX_MOVES, gpus=$GPUS ====="

# 0) 冒烟（验证 torch/cuda/模型）
"$PY" deploy/smoke_dcu.py --model "$BASE_MODEL"
echo "冒烟通过，开始长局自对弈..."

# 1) 多卡长局自我对弈
"$PY" launch_selfplay.py \
    --model "$BASE_MODEL" \
    --gpus "$GPUS" \
    --episodes-per-gpu "$EPISODES_PER_GPU" \
    --max-moves "$MAX_MOVES" \
    --dist "$DIST" \
    --seed "$SEED_BASE" \
    --out "$SELFPLAY_OUT"

# 2) 更大网络从零训练
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

# 3) 配对评测
"$PY" eval_paired_puct.py \
    --models gumbel48="$BASE_MODEL" \
             "h256-r${ROUND}=$MODEL_OUT" \
    --tree-reuse-models gumbel48 "h256-r${ROUND}" \
    --episodes 128 --seed "$SEED_BASE" --max-moves 500 \
    --simulations 128 --depth 16 --chance-samples 8 \
    --chance-widening 0.5 --root-min-visits 2 --death-penalty 0.5 \
    --json-out "$PAIRED_OUT" --max-new 64

echo "===== 轮次 $ROUND 完成。查看 $PAIRED_OUT 的配对差与 95% CI ====="
