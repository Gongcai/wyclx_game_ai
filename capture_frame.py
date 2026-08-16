#!/usr/bin/env python
"""抓取指定游戏窗口的画面帧（grim + hyprctl）。

用法：
    .venv/bin/python capture_frame.py --out runs/records/board-frame.png
    .venv/bin/python capture_frame.py --title 一梦江湖 --out /tmp/frame.png

按窗口标题/类名在 Hyprland 里定位窗口（窗口移动也不怕），grim 抓窗口区域。
"""

import argparse
import json
import subprocess
import sys


def hyprctl_clients():
    out = subprocess.run(["hyprctl", "clients", "-j"], capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"hyprctl clients 失败: {out.stderr}")
    return json.loads(out.stdout)


def find_window(title=None, cls=None):
    wins = hyprctl_clients()
    for w in wins:
        if title and title in w.get("title", ""):
            return w
    for w in wins:
        if cls and cls in w.get("class", ""):
            return w
    return None


def grab_region(x, y, w, h):
    out = subprocess.run(
        ["grim", "-g", f"{x},{y} {w}x{h}", "-"],
        capture_output=True,
    )
    if out.returncode != 0 or not out.stdout:
        raise RuntimeError(f"grim 抓帧失败: {out.stderr.decode(errors='replace')}")
    return out.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--title", default="一梦江湖", help="窗口标题包含串")
    ap.add_argument("--class", dest="cls", default="steam_app_4277773963", help="窗口类名包含串")
    ap.add_argument("--out", default="runs/records/board-frame.png")
    args = ap.parse_args()

    win = find_window(args.title, args.cls)
    if win is None:
        print("未找到匹配窗口，当前所有窗口：")
        for w in hyprctl_clients():
            print(f"  title={w.get('title')!r} class={w.get('class')!r} "
                  f"at={w.get('at')} size={w.get('size')}")
        sys.exit(1)

    x, y = win["at"]
    w, h = win["size"]
    png = grab_region(x, y, w, h)
    open(args.out, "wb").write(png)
    print(f"窗口: title={win.get('title')!r} class={win.get('class')!r}")
    print(f"位置: ({x},{y}) 大小: {w}x{h}")
    print(f"已保存: {args.out}")


if __name__ == "__main__":
    main()
