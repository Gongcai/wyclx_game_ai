#!/usr/bin/env python
"""H5 网页版自动对战：Playwright 驱动研究版页面，AI 自动玩网页游戏。

零侵入：不修改游戏文件。每步从页面直读状态（window.__game + DOM），
用当前推荐 Policy/Value + PUCT 决策，再向 .grid-overlay 派发真实点击。
局终轨迹按 recorder 格式落盘，可直接用 replay_trace.py 校验
（真实引擎与本地模拟器逐盘面比对）。

用法：
    .venv/bin/python serve_game.py --port 8080 &
    .venv/bin/python auto_play_h5.py --url http://127.0.0.1:8080 --games 3
"""

import argparse
import json
import os
import time

import torch
from playwright.sync_api import sync_playwright

from agents.dist import make_real_sampler
from agents.policy_value import PolicyValueNet, hidden_from_checkpoint
from agents.puct import PuctTree, advance_tree, puct_search
from game import Game

DEFAULT_MODEL = "runs/puct-v3/real-h5-puct-r2-pv.pt"

READ_STATE_JS = """
() => {
  const game = window.__game;
  const boardEl = document.getElementById('game-board');
  if (!game || !boardEl) return null;
  const cols = [];
  for (let c = 0; c < 6; c++) {
    const colEl = boardEl.querySelector(`.column[data-col-idx="${c}"]`);
    if (!colEl) return null;
    const items = [];
    colEl.querySelectorAll('.rabbit-item').forEach(x => {
      const lv = parseInt(x.dataset.level), ri = parseInt(x.dataset.rowIdx);
      if (!isNaN(lv) && !isNaN(ri)) items.push([ri, lv]);
    });
    items.sort((a, b) => a[0] - b[0]);
    cols.push(items.map(x => x[1]));
  }
  const pv = [];
  document.querySelectorAll('#preview-area .preview-item').forEach(x => {
    const lv = parseInt(x.dataset.level);
    if (!isNaN(lv)) pv.push(lv);
  });
  return {
    board: cols,
    preview: pv.length === 6 ? pv : null,
    previewVisible: !!game.topPreviewVisible && pv.length === 6,
    remain: game.remainClickTimes,
    // 权威数据源：局终时 DOM 会被清理，gridRabbitItem 才是真实盘面
    gridData: game.gridRabbitItem.map(col => col.map(r => r.level)),
    composite: Number(game.compositeLevel) || 1,
    maxLevel: Number(game.compositeMaxLevelNum) || 0,
    score: Number(game.score) || 0,
    state: game.state,
    STATE_NORMAL: game.STATE_NORMAL,
    STATE_MOVING: game.STATE_MOVING,
    STATE_END: game.STATE_END,
    liftCol: (game.liftCol === undefined || game.liftCol === null) ? null : game.liftCol,
    popEnd: !!(document.querySelector('.pop-end.on')),
  };
}
"""


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


def phase_remain(moves):
    """已完成 moves 步后，H5 remainClickTimes 的期望值（首轮 3 步，其后每轮 4 步）。"""
    return 4 if moves % 4 == 3 else 3 - moves % 4


class LiveTracker:
    """跨步维护 moves 相位、历史最高合成与空列周期历史（r2 模型输入特征）。"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.moves = 0
        self.max_merged = 1
        self.current_empty_peak = 0
        self.recent_empty_peaks = [0, 0, 0]

    @staticmethod
    def initial_board(board):
        return board == [[1, 1] for _ in range(6)]

    def sync(self, board, preview, remain):
        """每步用观测校准；返回事件列表（resync/reset）供日志。"""
        events = []
        if self.initial_board(board):
            self.reset()
            events.append("new-game")
        board_max = max((v for col in board for v in col), default=1)
        preview_max = max(preview, default=1) if preview else 1
        self.max_merged = max(self.max_merged, board_max, preview_max)
        empty = sum(not col for col in board)
        self.current_empty_peak = max(self.current_empty_peak, empty)
        if remain is not None and remain != phase_remain(self.moves):
            # 复活会把回合节奏重置为 4；把相位对齐到期望 remain 的最小步数。
            for delta in range(4):
                if phase_remain(self.moves + delta) == remain:
                    self.moves += delta
                    events.append(f"resync+{delta}")
                    break
        if self.moves % 4 == 3:
            # 周期结束：空列峰值入历史（与 Game.resolve_afterstate 同步）。
            self.recent_empty_peaks = [
                self.current_empty_peak, *self.recent_empty_peaks[:2],
            ]
            self.current_empty_peak = empty
        return events

    def on_move_done(self):
        self.moves += 1

    def build_game(self, board, preview):
        game = Game(drop_sampler=make_real_sampler())
        game.stacks = [list(reversed(col)) for col in board]  # 底部->顶部 → 栈顶在前
        game.moves = self.moves
        game.max_merged = self.max_merged
        game.preview = list(preview) if preview is not None else None
        game.dead = False
        game._afterstate_pending = False
        game.current_cycle_empty_peak = self.current_empty_peak
        game.recent_cycle_empty_peaks = list(self.recent_empty_peaks)
        game._hidden_preview_level = self.max_merged
        return game


class BotTrace:
    """recorder 兼容轨迹（clicks/moves/drops/frames），供 replay_trace.py 校验。"""

    def __init__(self, trace_id):
        self.data = {
            "id": trace_id, "rule": "tuyr-v1", "source": "autoplay",
            "t_start": time.time(), "clicks": [], "moves": [], "frames": [],
            "drops": [], "status": "active",
        }
        self.last_preview = None

    def record_preview(self, preview):
        if preview and preview != self.last_preview:
            self.last_preview = list(preview)
            self.data["drops"].append(list(preview))
            self.data["frames"].append({
                "t": int((time.time() - self.data["t_start"]) * 1000),
                "kind": "preview", "preview": list(preview),
                "click_n": len(self.data["clicks"]),
            })

    def record_move(self, src, dst, board):
        self.data["clicks"] += [src, dst]
        self.data["moves"].append([src, dst])
        self.data["frames"].append({
            "t": int((time.time() - self.data["t_start"]) * 1000),
            "kind": "board", "board": [list(col) for col in board],
            "click_n": len(self.data["clicks"]),
        })

    def finish(self, status, game_stats):
        self.data["status"] = status
        self.data["game_stats"] = game_stats
        return self.data


def read_state(page):
    state = page.evaluate(READ_STATE_JS)
    if state is None:
        return None
    # 局终只认引擎自己的信号（结算弹层 / END 状态）。gridData 列高>7 只是
    # 合并动画进行中的瞬时现象（合并完成会回落），绝不能当死亡，否则会把
    # 活局误杀重载。
    reasons = []
    if state["popEnd"]:
        reasons.append("pop-end")
    if state["state"] == state.get("STATE_END"):
        reasons.append("state-end")
    state["merge_pending"] = bool(
        state.get("gridData") and any(len(col) > 7 for col in state["gridData"])
    )
    state["over"] = bool(reasons)
    state["over_reason"] = "+".join(reasons)
    return state


def wait_stable(page, args, timeout=6.0):
    """等动画结束：state==NORMAL 且盘面连续两次读数一致；局终立即返回。"""
    deadline = time.time() + timeout
    last_board, stable_since, last_state = None, None, None
    while time.time() < deadline:
        state = read_state(page)
        if state is None:
            time.sleep(0.15)
            continue
        last_state = state
        if state["over"]:
            return state
        if state.get("merge_pending"):
            # 合并动画进行中（某列瞬时>7）：等它落定再判稳定。
            last_board, stable_since = None, None
            time.sleep(0.15)
            continue
        if state["state"] != state["STATE_NORMAL"]:
            last_board, stable_since = None, None
            time.sleep(0.15)
            continue
        board = state["board"]
        if board == last_board:
            if stable_since is None:
                stable_since = time.time()
            if time.time() - stable_since >= 0.3:
                return state
        else:
            last_board, stable_since = board, None
        time.sleep(0.15)
    return last_state


def click_column(page, col):
    page.locator(f'.column[data-col-idx="{col}"] .grid-overlay').first.click(
        timeout=3000,
    )


def dismiss_popups(page):
    """关掉未登录时的研究版声明弹窗 / 服务器选择弹窗（练习模式无需登录）。"""
    for sel in (".local-login-close", ".pop-server.on .close"):
        loc = page.locator(sel).first
        try:
            if loc.count() and loc.is_visible():
                loc.click(timeout=2000)
        except Exception:
            pass
    # 兜底：直接摘掉遮罩类，防止残留弹层挡住棋盘点击。
    page.evaluate(
        "() => document.querySelectorAll('.pop-login.on, .pop-server.on')"
        ".forEach(p => p.classList.remove('on'))"
    )


def start_game(page, url):
    page.goto(url, wait_until="domcontentloaded")
    page.wait_for_selector(".btn-start1", timeout=15000)
    dismiss_popups(page)
    btn = page.locator(".btn-start1").first
    if btn.is_visible():
        btn.click(timeout=5000)
    page.wait_for_function(
        "() => window.__game && document.querySelectorAll('#game-board .column').length === 6",
        timeout=15000,
    )


def restart_game(page, url):
    """局终后开新局：优先走页面内按钮，失败则整页重载。"""
    try:
        close = page.locator(".pop-end.on .btn-close").first
        if close.count() and close.is_visible():
            close.click(timeout=3000)
            page.wait_for_selector(".btn-start1", timeout=5000)
            dismiss_popups(page)
            start = page.locator(".btn-start1").first
            if start.is_visible():
                start.click(timeout=5000)
            page.wait_for_function(
                "() => window.__game && window.__game.state === window.__game.STATE_NORMAL",
                timeout=10000,
            )
            return
    except Exception:
        pass
    start_game(page, url)


def append_jsonl(path, event):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as file:
        file.write(json.dumps(event, ensure_ascii=False) + "\n")


def play_one_game(page, url, net, gamma, args, game_idx, t0):
    trace = BotTrace(f"auto{game_idx}{int(time.time() * 1000) % 100000:05d}")
    tracker = LiveTracker()
    moves_done, n9_seen, warn = 0, 0, []
    state = None
    # 与页面逐步同步的模拟局：搜索在其上运行并跨步复用搜索树；
    # 揭示值（预告）从页面真实观测注入，盘面漂移时整体重建。
    sim = None
    tree = None

    def sync_sim(state):
        """确保 sim 与页面一致；返回用于搜索的 (game, tree)。"""
        nonlocal sim, tree
        stacks_page = [list(reversed(col)) for col in state["board"]]
        preview_visible = state.get("previewVisible") and state.get("preview")
        in_sync = sim is not None and not sim.dead and sim.stacks == stacks_page and (
            not preview_visible or sim.moves % 4 != 2 or sim.preview == state["preview"])
        if not in_sync:
            if sim is not None:
                warn.append("sim-resync")
            preview = state["preview"] if (preview_visible and tracker.moves % 4 == 2) else None
            sim = tracker.build_game(state["board"], preview)
            tree = PuctTree()
        return sim, tree

    def advance_sim(src, dst, settled):
        """页面确认动作后，把 sim 与搜索树推进到真实后继。"""
        nonlocal sim
        sim.move_afterstate(src, dst)
        chance = None
        if sim._afterstate_pending and not sim.dead and sim.moves % 4 == 2:
            settled_preview = settled.get("preview") if settled.get("previewVisible") else None
            if settled_preview:
                chance = settled_preview  # 本步揭示的真实预告值
        sim.resolve_afterstate(chance=chance)
        if tree is not None:
            advance_tree(tree, (src, dst), sim)

    def decide_and_click(state):
        for attempt in range(3):
            # 预告只在每轮第 2 步后可见（moves%4==2），其它相位读到属于残留，忽略。
            preview = state["preview"] if state.get("previewVisible") else None
            if preview is not None and tracker.moves % 4 != 2:
                preview = None
            if preview is None and state.get("previewVisible"):
                warn.append("preview-incomplete")
            events = tracker.sync(state["board"], preview, state["remain"])
            if events:
                warn.extend(events)
            # 悬空选择（上一次点击只完成了一半）先取消，避免动作错位。
            if state["liftCol"] is not None:
                click_column(page, state["liftCol"])
                state = wait_stable(page, args)
                if state["over"] or state["state"] != state["STATE_NORMAL"]:
                    return state, None
                continue
            trace.record_preview(preview)
            game, use_tree = sync_sim(state)
            action = puct_search(
                game, net, args.device, args.simulations, args.depth,
                gamma=gamma, death_penalty=0.5, chance_samples=8,
                chance_widening=0.5, root_min_visits=2, safe_veto=True,
                root_sequential_halving=args.halving, root_q_scale=args.q_scale,
                tree=None if args.no_tree_reuse else use_tree,
            )
            if args.debug:
                print(f"  [debug] moves={tracker.moves} preview={preview} "
                      f"legal={len(game.legal_moves())} action={action}", flush=True)
            if action is None:
                return state, None
            src, dst = action
            board_before = state["board"]
            click_column(page, src)
            # 源列"拿起"动画期间点击会被游戏忽略；liftCol 在点击瞬间就置位，
            # 真正的完成标志是状态机进入 'select'。
            deadline = time.time() + 2.0
            while time.time() < deadline:
                mid = read_state(page)
                if mid is None:
                    time.sleep(0.08)
                    continue
                if mid["over"] or mid["state"] == "select":
                    break
                time.sleep(0.08)
            time.sleep(0.1)
            click_column(page, dst)
            # 等盘面变化，确认这对点击被游戏接受。
            deadline = time.time() + 3.0
            while time.time() < deadline:
                nxt = read_state(page)
                if nxt is None:
                    time.sleep(0.12)
                    continue
                if nxt["over"] or nxt["board"] != board_before:
                    break
                time.sleep(0.12)
            settled = wait_stable(page, args)
            if settled is not None and settled["over"]:
                return settled, None
            if args.debug and settled is not None:
                print(f"  [debug] settled: state={settled['state']} over={settled['over']} "
                      f"changed={settled['board'] != board_before} liftCol={settled['liftCol']}",
                      flush=True)
            if settled is not None and not settled["over"] and \
                    settled["state"] == settled["STATE_NORMAL"] and \
                    settled["board"] != board_before:
                tracker.on_move_done()
                advance_sim(src, dst, settled)
                trace.record_move(src, dst, settled["board"])
                return settled, (src, dst)
            warn.append(f"click-fail@{src}->{dst}")
            state = settled if settled is not None else state
        return state, None

    while True:
        state = wait_stable(page, args)
        if state is None:
            warn.append("read-failed")
            break
        if state["over"]:
            break
        if state.get("merge_pending"):
            continue  # wait_stable 超时仍在合并中：继续等，不决策
        if state["state"] != state["STATE_NORMAL"]:
            continue
        if moves_done >= args.max_moves:
            break
        state, action = decide_and_click(state)
        if action is None:
            if state is not None and not state["over"]:
                warn.append("click-not-accepted" if moves_done else "no-action")
            if state is None or not state["over"]:
                break
            break
        moves_done += 1
        if moves_done % 20 == 0:
            print(f"  … 第 {moves_done} 步  得分 {state['score']} "
                  f"最高级 {state['maxLevel']} 用时 {time.time() - t0:.0f}s", flush=True)
        append_jsonl(args.move_log, {
            "t": time.time(), "game": trace.data["id"], "move_n": moves_done,
            "moves": tracker.moves, "phase": tracker.moves % 4,
            "remain": state["remain"], "action": list(action),
            "score": state["score"], "page_composite": state["composite"],
            "grid_heights": [len(c) for c in state.get("gridData") or []],
            "board": state["board"], "preview": state["preview"],
        })

    status = "complete" if state is not None and state["over"] else "abandoned"
    final = trace.finish(status, {
        "composite_max_level_num": state["maxLevel"] if state else 0,
        "score": state["score"] if state else 0,
        "max_level": state["composite"] if state else 1,
        "model": args.model,
    })
    os.makedirs(args.trace_dir, exist_ok=True)
    path = os.path.join(args.trace_dir, f"trace-{final['id']}.json")
    with open(path, "w", encoding="utf-8") as file:
        json.dump(final, file, ensure_ascii=False, indent=1)
    summary = {
        "game": final["id"], "status": status, "bot_moves": moves_done,
        "over_reason": state.get("over_reason") if state else None,
        "score": final["game_stats"]["score"],
        "max_level": final["game_stats"]["max_level"],
        "trace": path, "warnings": sorted(set(warn)),
    }
    print(f"局终: {json.dumps(summary, ensure_ascii=False)}", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--max-moves", type=int, default=500)
    parser.add_argument("--simulations", type=int, default=64)
    parser.add_argument("--depth", type=int, default=16)
    parser.add_argument("--halving", type=int, default=8,
                        help="根 sequential halving 候选数（配对评测部署配置）")
    parser.add_argument("--q-scale", type=float, default=2.0)
    parser.add_argument("--no-tree-reuse", action="store_true",
                        help="禁用搜索树跨步复用（默认启用，部署配置）")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--browser", choices=("edge", "chromium"), default="edge",
                        help="edge 用系统 microsoft-edge-stable，chromium 用 Playwright 自带")
    parser.add_argument("--revive", action="store_true",
                        help="溢出死亡时接受研究版复活弹窗（官方语义为直接局终，默认关闭）")
    parser.add_argument("--trace-dir", default="runs/autoplay")
    parser.add_argument("--move-log", default="runs/autoplay/moves.jsonl")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    net, gamma = load_model(args.model, args.device)
    print(f"模型 {args.model} 已加载（device={args.device}, gamma={gamma}）", flush=True)

    t0 = time.time()
    with sync_playwright() as pw:
        if args.browser == "edge":
            browser = pw.chromium.launch(
                executable_path="/usr/sbin/microsoft-edge-stable",
                headless=args.headless, args=["--no-sandbox"],
            )
        else:
            browser = pw.chromium.launch(headless=args.headless)
        # 页面在宽度 <700 且无 orientation 时会弹"推荐竖屏"遮罩挡住点击，用桌面视口。
        page = browser.new_page(viewport={"width": 1280, "height": 800})
        if not args.revive:
            page.on("dialog", lambda dialog: dialog.dismiss())
        else:
            page.on("dialog", lambda dialog: dialog.accept())
        start_game(page, args.url)
        for game_idx in range(args.games):
            if game_idx:
                restart_game(page, args.url)
                time.sleep(0.5)
            try:
                play_one_game(page, args.url, net, gamma, args, game_idx, t0)
            except Exception:
                import traceback
                print(f"[game {game_idx}] play_one_game 异常:", flush=True)
                traceback.print_exc()
        browser.close()
    print(f"完成 {args.games} 局，总用时 {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
