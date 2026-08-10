import argparse
import random

from game.drop import DropDist, PRESETS
from game.env import Game
from game.console import render


def main():
    ap = argparse.ArgumentParser(description="合成小游戏 MVP")
    ap.add_argument("--dist", choices=sorted(PRESETS), default="uniform",
                    help="掉落分布预设: uniform/low/high/mid")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    if args.seed is not None:
        random.seed(args.seed)

    g = Game(DropDist.preset(args.dist))
    print("玩法: 输入 `a b` 把 a 列栈顶移到 b 列；3 个相同合成 n+1；9 消失得分；列高超过 7 即败。")
    print("命令: q 退出, n 新局, h 帮助")
    while True:
        render(g)
        if g.dead:
            print("游戏结束！最终得分:", g.score)
            raw = input("再来一局? [y/n]: ").strip().lower()
            if raw == "y":
                g = Game(DropDist.preset(args.dist))
                continue
            break
        try:
            raw = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if raw in ("q", "quit", "exit"):
            break
        if raw in ("n", "new"):
            g = Game(DropDist.preset(args.dist))
            continue
        if raw in ("h", "help"):
            print("输入 `a b` 将 a 列栈顶移到 b 列; q 退出; n 新局")
            continue
        parts = raw.split()
        if len(parts) != 2:
            print("格式: a b")
            continue
        try:
            src, dst = map(int, parts)
        except ValueError:
            print("请输入数字")
            continue
        try:
            g.move(src, dst)
        except ValueError as e:
            print(e)


if __name__ == "__main__":
    main()
