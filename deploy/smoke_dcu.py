"""8 卡平台（海光 DCU / 任意多 GPU）部署冒烟测试。

    .venv/bin/python deploy/smoke_dcu.py [--model runs/demos/high-halving8-gumbel48-pv.pt]

检查：torch 可用性、设备数量与名称、test_game.py、PV 模型加载与前向。
海光 DCU 需 DTK/ROCm 版 torch（torch_dcu），此时 torch.cuda 被映射为 DCU。
"""

import argparse
import os
import subprocess
import sys

# 让脚本可从 deploy/ 子目录运行也能 import agents
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    ok = True

    print(f"torch 版本: {torch.__version__}")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"设备数: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  cuda:{i}: {torch.cuda.get_device_name(i)}")
    else:
        ok = False
        print("!! 无可用 CUDA/DCU 设备（海光平台需 DTK 版 torch）")

    # test_game.py
    result = subprocess.run(
        [sys.executable, "test_game.py"], capture_output=True, text=True
    )
    print(f"test_game.py: rc={result.returncode} | {result.stdout.strip()}")
    if result.returncode != 0:
        ok = False

    # 模型加载 + 前向
    if args.model:
        try:
            from agents.policy_value import PolicyValueNet, hidden_from_checkpoint
            from agents.dqn import encode
            from game import Game

            checkpoint = torch.load(args.model, weights_only=True, map_location="cpu")
            net = PolicyValueNet(
                hidden=hidden_from_checkpoint(checkpoint),
                value_outputs=checkpoint.get("value_outputs", 2),
                afterstate_q=checkpoint.get("afterstate_q", False),
                history_features=checkpoint.get("history_features", False),
            )
            net.load_state_dict(checkpoint["model"])
            net.eval()
            game = Game()
            out = net(encode(game, history=net.history_features).unsqueeze(0))
            print(f"模型前向 OK: policy{out[0].shape} value{out[1].shape}")
        except Exception as e:
            ok = False
            print(f"!! 模型加载/前向失败: {e}")

    print("\n===== 冒烟结果:", "PASS" if ok else "FAIL", "=====")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
