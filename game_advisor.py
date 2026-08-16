#!/usr/bin/env python
"""人工操作辅助：全局 F8 截图，识别棋盘并打印 PUCT 操作建议。

程序不注入鼠标/键盘操作。Hyprland 只负责把全局 F8 转成 SIGUSR1，游戏
可以一直保持焦点；AI 建议输出在启动本程序的终端，并逐次写入 JSONL。
"""

import argparse
import copy
import json
import os
import select
import signal
import subprocess
import sys
import termios
import threading
import time
import tty

import cv2
import torch

from agents.dist import make_real_sampler
from agents.policy_value import PolicyValueNet, hidden_from_checkpoint
from agents.puct import puct_search
from capture_drops import Recognizer
from game import Game
from verify_capture import (
    CLEAR, board_rows, capture_live, cell_text, expected_frame_size,
)


DEFAULT_MODEL = "runs/puct-v3/real-h5-puct-r2-pv.pt"


class StateTracker:
    """按相邻人工截图维护游戏相位和模型需要的少量历史特征。"""

    def __init__(self, start_moves=None, max_merged=None):
        self.start_moves = start_moves
        self.initial_max = max_merged
        self.reset()

    def reset(self):
        self.moves = self.start_moves
        self.max_merged = self.initial_max or 1
        self.previous_flat = None
        self.previous_game = None
        self.current_empty_peak = 0
        self.recent_empty_peaks = [0, 0, 0]

    @staticmethod
    def initial_board(stacks):
        return stacks == [[1, 1] for _ in range(6)]

    def infer_human_actions(self, stacks):
        """枚举上个稳定状态的合法动作，找出能解释当前盘面的动作。"""
        if self.previous_game is None:
            return []
        matches = []
        for action in self.previous_game.legal_moves():
            sim = copy.deepcopy(self.previous_game)
            if sim.move(*action) and sim.stacks == stacks:
                matches.append(action)
        return matches

    def explain_transition(self, stacks, max_moves=3, frontier_cap=3000):
        """BFS 枚举 1..max_moves 步的合法序列解释盘面变化。

        掉落那一拍的盘面变化是"一步移动 + 6 个掉落 + 级联合并"，且玩家可能
        隔了几步才按一次 F8；单步解释会失败。sim.move 会在周期末按已记录的
        预告应用掉落，因此跨掉落的序列也能精确模拟。返回最短 (步数, 动作)。
        """
        if self.previous_game is None:
            return None
        frontier = [self.previous_game]
        for depth in range(1, max_moves + 1):
            nxt = []
            for game in frontier:
                for action in game.legal_moves():
                    sim = copy.deepcopy(game)
                    if not sim.move(*action):
                        continue
                    if sim.stacks == stacks:
                        return (depth, action)
                    nxt.append(sim)
            if len(nxt) > frontier_cap:
                break
            frontier = nxt
        return None

    def update(self, flat, stacks, preview, n_moves=0):
        changed = self.previous_flat is not None and flat != self.previous_flat
        if self.initial_board(stacks):
            self.reset()
            self.moves = 0
        elif self.moves is None:
            # 中途接入时，有预告只能对应可决策相位 2；无预告无法单帧区分。
            self.moves = 2 if preview is not None else 0
        else:
            if n_moves > 0:
                self.moves += n_moves
            elif changed:
                self.moves += 1
            if preview is not None and self.moves % 4 != 2:
                # 预告只在每轮第 2 步后可见：用它在任何一次按键上锚定相位。
                self.moves += (2 - self.moves) % 4

        board_max = max((value for stack in stacks for value in stack), default=1)
        preview_max = max(preview, default=1) if preview else 1
        self.max_merged = max(self.max_merged, board_max, preview_max)
        empty = sum(not stack for stack in stacks)
        self.current_empty_peak = max(self.current_empty_peak, empty)
        if changed and self.moves % 4 == 3:
            self.recent_empty_peaks = [
                self.current_empty_peak, *self.recent_empty_peaks[:2],
            ]
            self.current_empty_peak = empty
        self.previous_flat = list(flat)
        return changed

    def make_game(self, stacks, preview):
        game = Game(drop_sampler=make_real_sampler())
        game.stacks = [list(stack) for stack in stacks]
        game.moves = self.moves
        game.max_merged = self.max_merged
        game.preview = list(preview) if preview is not None else None
        game.dead = False
        game._afterstate_pending = False
        game.current_cycle_empty_peak = self.current_empty_peak
        game.recent_cycle_empty_peaks = list(self.recent_empty_peaks)
        game._hidden_preview_level = self.max_merged
        return game

    def remember(self, game):
        self.previous_game = copy.deepcopy(game)


def load_model(path, device):
    checkpoint = torch.load(path, weights_only=True, map_location=device)
    net = PolicyValueNet(
        hidden=hidden_from_checkpoint(checkpoint),
        value_outputs=checkpoint.get("value_outputs", 2),
        afterstate_q=checkpoint.get("afterstate_q", False),
        history_features=checkpoint.get("history_features", False),
    ).to(device)
    net.load_state_dict(checkpoint["model"])
    net.eval()
    return net, checkpoint.get("gamma", 0.99)


def stacks_from_flat(flat):
    return [
        [value for value in flat[col * 7:(col + 1) * 7] if value > 0]
        for col in range(6)
    ]


def append_jsonl(path, event):
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as file:
        file.write(json.dumps(event, ensure_ascii=False) + "\n")


def register_global_key(key, pid):
    # Hyprland 0.56 使用 Lua 配置解析器，legacy `hyprctl keyword bind` 不可用。
    # 用固定全局变量保存绑定对象；重启指导器时先禁用旧对象再注册新 PID。
    name = "CODEX_GAME_ADVISOR_BIND"
    script = (
        f'if _G.{name} then _G.{name}:set_enabled(false) end; '
        f'_G.{name}=hl.bind("{key}", hl.dsp.exec_cmd("kill -USR1 {pid}"))'
    )
    result = subprocess.run(
        ["hyprctl", "eval", script], capture_output=True, text=True,
    )
    if result.returncode != 0 or result.stdout.startswith("error"):
        raise RuntimeError(f"注册全局键失败: {result.stderr or result.stdout}")


def unregister_global_key(key):
    del key  # 固定全局对象已记录具体按键
    subprocess.run(
        [
            "hyprctl", "eval",
            "if _G.CODEX_GAME_ADVISOR_BIND then "
            "_G.CODEX_GAME_ADVISOR_BIND:set_enabled(false) end",
        ],
        capture_output=True, text=True,
    )


def print_advice(flat, stacks, preview, tracker, action, recognize_ms, search_ms,
                 warning, hotkey, human_actions=None):
    print(CLEAR, end="")
    print(
        f"回合状态: moves={tracker.moves} phase={tracker.moves % 4} "
        f"历史最高={tracker.max_merged}  识别={recognize_ms:.1f}ms "
        f"搜索={search_ms:.1f}ms"
    )
    if warning:
        print(f"警告: {warning}")
    print("      1  2  3  4  5  6")
    for row, values in enumerate(board_rows(flat), start=1):
        print(f"上{row}:  " + "  ".join(cell_text(value) for value in values))
    print("栈顶→栈底:")
    for col, stack in enumerate(stacks, start=1):
        print(f"  {col}: {stack}")
    print(f"预告: {preview if preview is not None else '无'}")
    if human_actions:
        text = ", ".join(f"{src + 1}→{dst + 1}" for src, dst in human_actions)
        prefix = "识别到人类上一步" if len(human_actions) == 1 else "人类上一步候选"
        print(f"{prefix}: {text}")
    print()
    if action is None:
        print("AI 建议: 无可用动作")
    else:
        print(f"AI 建议: 把第 {action[0] + 1} 列移动到第 {action[1] + 1} 列")
        print(f"           {action[0] + 1}  →  {action[1] + 1}")
    print()
    print(f"游戏保持焦点，按 {hotkey} 截图并获取下一条建议。")
    print("终端内按 r 重置回合计数，按 q 退出。")
    sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/capture.json")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--preview-background", default="runs/records/preview-empty.png")
    parser.add_argument("--preview-threshold", type=float, default=0.015)
    parser.add_argument("--global-key", default="F8")
    parser.add_argument("--no-global-key", action="store_true")
    parser.add_argument("--start-moves", type=int)
    parser.add_argument("--max-merged", type=int)
    parser.add_argument("--simulations", type=int, default=64)
    parser.add_argument("--depth", type=int, default=16)
    parser.add_argument("--trace", default="runs/advisor/advice.jsonl")
    parser.add_argument("--save", default="runs/records/advisor-latest.png")
    parser.add_argument("--frame", help="离线图片 smoke test")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as file:
        config = json.load(file)
    recognizer = Recognizer(config)
    expected_size = expected_frame_size(config)
    background = cv2.imread(args.preview_background) if os.path.exists(args.preview_background) else None
    net, gamma = load_model(args.model, args.device)
    tracker = StateTracker(args.start_moves, args.max_merged)

    def advise():
        if args.frame:
            image = cv2.imread(args.frame)
            source = args.frame
        else:
            image, source = capture_live(config)
        if image is None:
            raise RuntimeError("无法读取截图")
        if args.save and not args.frame:
            os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
            cv2.imwrite(args.save, image)
        started = time.perf_counter()
        flat = recognizer.board(image)
        preview = recognizer.preview_values(
            image, background=background,
            presence_threshold=args.preview_threshold,
        )
        recognize_ms = (time.perf_counter() - started) * 1000
        warning = None
        height, width = image.shape[:2]
        if expected_size and (width, height) != expected_size:
            warning = f"截图尺寸 {width}×{height} 与标定尺寸 {expected_size} 不同"
        if any(value < 0 for value in flat) or (preview and any(value < 0 for value in preview)):
            warning = "存在未识别格，已拒绝给出建议"
            action = None
            stacks = stacks_from_flat(flat)
            search_ms = 0.0
            human_actions = []
        else:
            stacks = stacks_from_flat(flat)
            board_changed = (
                tracker.previous_flat is not None and flat != tracker.previous_flat
            )
            is_reset = tracker.initial_board(stacks)
            human_actions = (
                tracker.infer_human_actions(stacks)
                if board_changed and not is_reset else []
            )
            explained = (1, human_actions[0]) if len(human_actions) == 1 else None
            if explained is None and board_changed and not is_reset:
                explained = tracker.explain_transition(stacks)
            if board_changed and not is_reset and explained is None:
                # 不再卡死：重同步到观测盘面继续给建议（相位靠预告在场与否
                # 在 update 里锚定），并落诊断日志便于追因（预告误读/动画中间态）。
                if warning is None:
                    warning = ("盘面变化无法逐步解释（多为掉落/连合成动画或多步连下），"
                               "已重同步、相位可能不准；建议异常时按 r 重置")
                append_jsonl(args.trace, {
                    "t": time.time(), "source": source, "status": "resync",
                    "board": flat, "stacks": stacks, "preview": preview,
                    "prev_board": tracker.previous_flat,
                    "prev_moves": tracker.moves,
                })
                n_moves = 0
            else:
                n_moves = explained[0] if explained else 0
            changed = tracker.update(flat, stacks, preview, n_moves=n_moves)
            game = tracker.make_game(stacks, preview)
            search_started = time.perf_counter()
            action = puct_search(
                game, net, args.device, args.simulations, args.depth,
                gamma=gamma, death_penalty=0.5, chance_samples=8,
                chance_widening=0.5, root_min_visits=2, safe_veto=True,
            )
            search_ms = (time.perf_counter() - search_started) * 1000
            append_jsonl(args.trace, {
                "t": time.time(), "source": source, "moves": tracker.moves,
                "phase": tracker.moves % 4, "changed": changed,
                "board": flat, "stacks": stacks, "preview": preview,
                "max_merged": tracker.max_merged,
                "human_action": list(human_actions[0]) if len(human_actions) == 1 else None,
                "human_action_candidates": [list(item) for item in human_actions],
                "advice": list(action) if action is not None else None,
            })
            tracker.remember(game)
        print_advice(
            flat, stacks, preview, tracker, action, recognize_ms, search_ms,
            warning, args.global_key, human_actions=human_actions,
        )

    if args.once:
        advise()
        return
    if not sys.stdin.isatty():
        raise RuntimeError("交互模式需要真实终端")

    requested = threading.Event()
    signal.signal(signal.SIGUSR1, lambda _sig, _frame: requested.set())
    bound = False
    fd = sys.stdin.fileno()
    previous_terminal = termios.tcgetattr(fd)
    try:
        if not args.no_global_key:
            register_global_key(args.global_key, os.getpid())
            bound = True
        tty.setcbreak(fd)
        print(CLEAR, end="")
        print(f"人工 AI 指导器已启动，模型: {args.model}")
        print(f"游戏保持焦点，按全局 {args.global_key} 截图并获取建议。")
        print("终端内按空格也可截图，按 r 重置，按 q 退出。")
        sys.stdout.flush()
        while True:
            if requested.is_set():
                requested.clear()
                try:
                    advise()
                except Exception as exc:
                    print(CLEAR + f"截图/建议失败: {exc}", flush=True)
            readable, _, _ = select.select([fd], [], [], 0.1)
            if not readable:
                continue
            key = os.read(fd, 1)
            if key in (b"q", b"Q", b"\x03"):
                break
            if key == b" ":
                requested.set()
            elif key in (b"r", b"R"):
                tracker.reset()
                print(CLEAR + "回合计数已重置。按 F8 截取新对局初始盘面。", flush=True)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, previous_terminal)
        if bound:
            unregister_global_key(args.global_key)
        print(CLEAR + "人工 AI 指导器已退出。", flush=True)


if __name__ == "__main__":
    main()
