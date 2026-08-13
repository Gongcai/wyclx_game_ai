"""8 卡平台多 GPU 长局自我对弈启动器：分片 generate_puct_demos 到多卡并行，合并产物。

用法（8 卡平台）：
    .venv/bin/python launch_selfplay.py \
        --model runs/demos/high-halving8-gumbel48-pv.pt \
        --gpus 8 --episodes-per-gpu 50 --max-moves 2000 \
        --out runs/demos/long-bootstrap-r1.pt

每个 GPU 一个 worker（--device cuda:i，seed 错开），完成后合并 episodes 到 --out。
"""

import argparse
import os
import subprocess
import sys

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--gpus", type=int, default=8)
    ap.add_argument("--episodes-per-gpu", type=int, default=50)
    ap.add_argument("--max-moves", type=int, default=2000, help="长局上限（bootstrap 逐轮拉长）")
    ap.add_argument("--dist", default="high")
    ap.add_argument("--simulations", type=int, default=128)
    ap.add_argument("--depth", type=int, default=16)
    ap.add_argument("--c-puct", type=float, default=1.5)
    ap.add_argument("--death-penalty", type=float, default=0.5)
    ap.add_argument("--chance-samples", type=int, default=8)
    ap.add_argument("--chance-widening", type=float, default=0.5)
    ap.add_argument("--root-min-visits", type=int, default=2)
    ap.add_argument("--seed", type=int, default=600000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep-tmp", action="store_true")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    tmp_files = []
    procs = []
    for i in range(args.gpus):
        tmp = f"{args.out}.tmp{i}.pt"
        tmp_files.append(tmp)
        cmd = [
            sys.executable, "generate_puct_demos.py",
            "--model", args.model, "--dist", args.dist,
            "--episodes", str(args.episodes_per_gpu),
            "--max-moves", str(args.max_moves),
            "--simulations", str(args.simulations),
            "--depth", str(args.depth),
            "--c-puct", str(args.c_puct),
            "--death-penalty", str(args.death_penalty),
            "--chance-samples", str(args.chance_samples),
            "--chance-widening", str(args.chance_widening),
            "--root-min-visits", str(args.root_min_visits),
            "--seed", str(args.seed + i * 100000),
            "--device", f"cuda:{i}",
            "--out", tmp,
        ]
        print(f"[GPU{i}] 启动: {' '.join(cmd)}", flush=True)
        procs.append(subprocess.Popen(cmd))

    failed = False
    for i, proc in enumerate(procs):
        rc = proc.wait()
        print(f"[GPU{i}] 完成 rc={rc}", flush=True)
        if rc != 0:
            failed = True
    if failed:
        print("有 worker 失败，中止合并", flush=True)
        sys.exit(1)

    episodes = []
    merged_meta = None
    for tmp in tmp_files:
        data = torch.load(tmp, weights_only=True, map_location="cpu")
        if merged_meta is None:
            merged_meta = data
        episodes.extend(data["episodes"])
    merged_meta["episodes"] = episodes
    merged_meta["gpus"] = args.gpus
    torch.save(merged_meta, args.out)
    if not args.keep_tmp:
        for tmp in tmp_files:
            os.remove(tmp)

    total_n9 = sum(len(e["n9_steps"]) for e in episodes)
    steps = sum(len(e["actions"]) for e in episodes)
    longest = max(len(e["actions"]) for e in episodes)
    print(
        f"已合并 {args.out}: {len(episodes)} 局, {steps} 步, {total_n9} 个9, "
        f"最长单局 {longest} 步",
        flush=True,
    )


if __name__ == "__main__":
    main()
