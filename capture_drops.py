#!/usr/bin/env python
"""实机掉落数据采集器：识别棋盘状态，记录每次掉落的 6 列掉落值。

依赖：configs/capture.json（calibrate_server.py 标定生成）、grim、hyprctl。

用法：
    .venv/bin/python capture_drops.py --config configs/capture.json --out runs/real_drops.jsonl
    .venv/bin/python capture_drops.py --interval 0.3 --max-cycles 500   # 采 500 个周期后退出

事件类型（JSONL 逐行）：
    preview   预告行变化（若实机有预告 UI）
    drop      检测到掉落应用，values 为 6 列掉落值（部分列可能 null=推断失败）
    new_game  棋盘重置（新对局）
    state     周期性的完整棋盘快照（校验/轨迹用）

Ctrl+C 正常退出（已落盘数据不丢）。
"""

import argparse
import json
import os
import subprocess
import sys
import time

import cv2
import numpy as np


def hyprctl_clients():
    out = subprocess.run(["hyprctl", "clients", "-j"], capture_output=True, text=True)
    return json.loads(out.stdout) if out.returncode == 0 else []


def find_window(title, cls):
    wins = hyprctl_clients()
    for w in wins:
        if title and title in w.get("title", ""):
            return w
    for w in wins:
        if cls and cls in w.get("class", ""):
            return w
    return None


def grab_region(x, y, w, h):
    out = subprocess.run(["grim", "-g", f"{x},{y} {w}x{h}", "-"], capture_output=True)
    if out.returncode != 0 or not out.stdout:
        return None
    return cv2.imdecode(np.frombuffer(out.stdout, np.uint8), cv2.IMREAD_COLOR)


def sample_color(img, x, y):
    """取帧内 (x,y) 中心 3x3 均值，返回 RGB。"""
    x, y = int(x), int(y)
    patch = img[max(0, y - 1):y + 2, max(0, x - 1):x + 2].reshape(-1, 3).mean(axis=0)
    return [float(v) for v in patch[::-1]]


class Recognizer:
    """按 config 颜色表做最近邻识别。"""

    def __init__(self, cfg):
        self.cells = cfg["cell_centers"]  # 42 个 (x, y)
        self.preview = cfg.get("preview")
        self.samples = []  # (v, r, g, b)
        self.samples_p = []  # 预告颜色（减淡效果，独立颜色表）
        for v, cols in cfg["colors"].items():
            for c in cols:
                self.samples.append((int(v), float(c[0]), float(c[1]), float(c[2])))
        if self.preview and self.preview.get("colors"):
            for v, cols in self.preview["colors"].items():
                for c in cols:
                    self.samples_p.append((int(v), float(c[0]), float(c[1]), float(c[2])))
        self.threshold = 90.0  # 超过该距离标 unknown
        # 预告颜色表（{v: [[r,g,b],...]}）：预告元素颜色与棋盘不同（减淡），用 RGB 最近邻识别
        self.preview_colors = cfg.get("preview_colors") or {}
        # 整格模板模式（优先）：整格归一化相关，颜色/字体/亮度无关
        self.tpl = cfg.get("templates")
        self.tpl_items = {}
        self.has_preview_templates = False
        self.tpl_gw = self.tpl_gh = 0.0
        self.tpl_pgw = self.tpl_pgh = 0.0
        if self.tpl:
            self.tw, self.th = self.tpl["size"]
            self.has_preview_templates = bool(self.tpl.get("items_p"))
            for k, items in (("b", self.tpl["items_b"]), ("p", self.tpl.get("items_p") or {})):
                self.tpl_items[k] = {
                    int(v): np.array(g, dtype=np.float32).reshape(self.th, self.tw, 3)
                    for v, g in items.items()}
            if not self.tpl_items["p"]:
                self.tpl_items["p"] = self.tpl_items["b"]
            c = self.tpl.get("cell", {})
            self.tpl_gw, self.tpl_gh = c.get("gw", 0.0), c.get("gh", 0.0)
            self.tpl_pgw, self.tpl_pgh = c.get("pv_gw", 0.0), c.get("pv_gh", 0.0)

    def classify(self, img, x, y, kind="b"):
        if self.tpl:
            return self._classify_tpl(img, x, y, kind)
        r, g, b = sample_color(img, x, y)
        pool = self.samples_p if kind == "p" and self.samples_p else self.samples
        best, bd = -1, 1e18
        for v, sr, sg, sb in pool:
            d = (sr - r) ** 2 + (sg - g) ** 2 + (sb - b) ** 2
            if d < bd:
                bd, best = d, v
        if bd > self.threshold ** 2:
            return -1
        return best

    def _classify_tpl(self, img, x, y, kind):
        items = self.tpl_items.get(kind)
        if not items:
            return -1
        gw = self.tpl_pgw if kind == "p" and self.tpl_pgw else self.tpl_gw
        gh = self.tpl_pgh if kind == "p" and self.tpl_pgh else self.tpl_gh
        if gw <= 0 or gh <= 0:
            return -1
        x0, y0 = int(x - gw / 2), int(y - gh / 2)
        x1, y1 = int(x + gw / 2), int(y + gh / 2)
        patch = img[max(0, y0):y1, max(0, x0):x1]
        if patch.shape[0] < 2 or patch.shape[1] < 2:
            return -1
        patch = cv2.resize(patch, (self.tw, self.th))
        a = patch.astype(np.float32)[:, :, ::-1].ravel()  # BGR->RGB 展平
        a = a - a.mean()
        best, bs = -1, 0.0
        for v, t in items.items():
            b = t.ravel() - t.ravel().mean()
            d = np.sqrt(float((a * a).sum() * (b * b).sum()))
            s = float((a * b).sum() / d) if d > 0 else 0.0
            if s > bs:
                bs, best = s, v
        return best if bs > 0.45 else 0

    def board(self, img):
        return [self.classify(img, x, y, "b") for x, y in self.cells]

    def preview_change_scores(self, img, background):
        """逐格比较当前预告区与“无预告”背景，返回 0~1 平均绝对差。

        复杂静态背景不需要被识别成一种颜色：直接减去同位置背景即可。
        """
        if not self.preview or background is None or img.shape != background.shape:
            return None
        if self.tpl_pgw and self.tpl_pgh:
            # 预告兔子只占格子中央一部分；框太大会被复杂背景稀释。
            width, height = self.tpl_pgw * 0.55, self.tpl_pgh * 0.55
        else:
            box = self.preview.get("box") or {}
            width = (box.get("x1", 0) - box.get("x0", 0)) / 6 * 0.55
            height = (box.get("y1", 0) - box.get("y0", 0)) * 0.55
        scores = []
        for x, y in self.preview["centers"]:
            x0, x1 = max(0, int(x - width / 2)), min(img.shape[1], int(x + width / 2))
            y0, y1 = max(0, int(y - height / 2)), min(img.shape[0], int(y + height / 2))
            current = cv2.GaussianBlur(img[y0:y1, x0:x1], (5, 5), 0)
            empty = cv2.GaussianBlur(background[y0:y1, x0:x1], (5, 5), 0)
            if current.size == 0 or empty.shape != current.shape:
                scores.append(1.0)
            else:
                scores.append(float(cv2.absdiff(current, empty).mean() / 255.0))
        return scores

    def preview_visible(self, img, background, threshold=0.015):
        """六格中至少四格明显偏离空背景，才判定预告整行出现。"""
        scores = self.preview_change_scores(img, background)
        if scores is None:
            return None, None
        return sum(score >= threshold for score in scores) >= 4, scores

    def preview_values(self, img, background=None, presence_threshold=0.015):
        if not self.preview:
            return None
        if background is not None:
            visible, _scores = self.preview_visible(img, background, presence_threshold)
            if visible is False:
                return None
        # 专用整格模板比单点颜色更抗复杂背景；没有模板才回退到颜色。
        if self.tpl and self.has_preview_templates:
            return [self.classify(img, x, y, "p") for x, y in self.preview["centers"]]
        if self.preview_colors:
            # RGB 最近邻（预告元素专用颜色，减淡效果已体现在颜色值里）
            out = []
            for x, y in self.preview["centers"]:
                r, g, b = sample_color(img, x, y)
                best, bd = -1, 1e18
                for v, cols in self.preview_colors.items():
                    for cr, cg, cb in cols:
                        d = (cr - r) ** 2 + (cg - g) ** 2 + (cb - b) ** 2
                        if d < bd:
                            bd, best = d, int(v)
                out.append(best if bd <= self.threshold ** 2 else -1)
            return out
        return [self.classify(img, x, y, "p") for x, y in self.preview["centers"]]


def simulate_insert(st, v):
    """在栈 st（st[0]=栈顶）顶部插入 v 并执行合并链，返回新栈。"""
    st = [v] + list(st)
    while True:
        if len(st) < 3:
            break
        top, k = st[0], 1
        while k < len(st) and st[k] == top:
            k += 1
        if k < 3:
            break
        del st[:k]
        nv = top + 1
        if nv > 8:
            break  # 合成 9 消失
        st.insert(0, nv)
    return st


def visual_to_stack(col, top_first):
    """视觉列（r0=框顶 … r6=框底）转栈（st[0]=栈顶）。
    top_first=True: 视觉顶部是栈顶（从底部堆起）；False: 视觉底部是栈顶。"""
    st = [v for v in col if v > 0]
    return st if top_first else st[::-1]


def infer_drop(prev_col, new_col, top_first, max_v=7):
    """反推掉落值：枚举 v，插入 prev 后模拟 == new。返回 v 或 None。"""
    prev_st = visual_to_stack(prev_col, top_first)
    new_st = visual_to_stack(new_col, top_first)
    hits = []
    for v in range(1, max_v + 1):
        if simulate_insert(prev_st, v) == new_st:
            hits.append(v)
    return hits[0] if len(hits) == 1 else (hits[0] if hits else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/capture.json")
    ap.add_argument("--out", default="runs/real_drops.jsonl")
    ap.add_argument("--interval", type=float, default=0.3, help="采集间隔秒")
    ap.add_argument("--max-cycles", type=int, default=0, help="采满 N 个掉落周期后退出（0=不限）")
    ap.add_argument("--state-every", type=float, default=2.0, help="完整快照间隔秒")
    ap.add_argument("--preview-background", default="runs/records/preview-empty.png")
    ap.add_argument("--preview-threshold", type=float, default=0.015)
    args = ap.parse_args()

    cfg = json.load(open(args.config))
    rec = Recognizer(cfg)
    preview_background = (
        cv2.imread(args.preview_background)
        if args.preview_background and os.path.exists(args.preview_background)
        else None
    )
    win_cfg = cfg.get("window", {})
    title, cls = win_cfg.get("title", "一梦江湖"), win_cfg.get("class", "steam_app_4277773963")

    out_f = open(args.out, "a", encoding="utf-8")
    def emit(ev):
        out_f.write(json.dumps(ev, ensure_ascii=False) + "\n")
        out_f.flush()

    print(f"采集器启动 config={args.config} out={args.out} interval={args.interval}s")
    print("等待游戏窗口…")
    prev_board = None
    prev_preview = None
    cycle = 0
    game_id = 0
    last_state_t = 0.0
    top_first = True  # 视觉顶部=栈顶（底部堆起），首次掉落时自动校验
    top_first_checked = False
    n_drop = 0

    while True:
        win = find_window(title, cls)
        if win is None:
            time.sleep(1.0)
            continue
        x, y = win["at"]
        w, h = win["size"]
        img = grab_region(x, y, w, h)
        if img is None:
            time.sleep(0.5)
            continue
        board = rec.board(img)
        preview = rec.preview_values(
            img, background=preview_background,
            presence_threshold=args.preview_threshold,
        )

        # 新对局检测：上一帧大部分有牌，本帧大部分空
        if prev_board is not None:
            prev_filled = sum(1 for v in prev_board if v > 0)
            filled = sum(1 for v in board if v > 0)
            if prev_filled >= 10 and filled <= 2:
                game_id += 1
                cycle = 0
                top_first_checked = False
                emit({"t": time.time(), "type": "new_game", "game": game_id})
                print(f"--- 新对局 #{game_id} ---")

        # 预告变化
        if preview is not None and prev_preview is not None and preview != prev_preview:
            emit({"t": time.time(), "type": "preview", "values": preview, "game": game_id})
            print(f"[预告] {preview}")

        # 掉落检测：变化列数 >= 4 视为掉落帧
        if prev_board is not None:
            changed = [c for c in range(6) if board[c * 7:(c + 1) * 7] != prev_board[c * 7:(c + 1) * 7]]
            if len(changed) >= 4:
                if not top_first_checked:
                    # 自动判定方向：统计"新增块"相对旧堆的位置
                    up = down = 0
                    for c in changed:
                        old = [v for v in prev_board[c * 7:(c + 1) * 7] if v > 0]
                        new = [v for v in board[c * 7:(c + 1) * 7] if v > 0]
                        if len(new) > len(old):
                            # 新增在视觉上方还是下方
                            for i, v in enumerate(board[c * 7:(c + 1) * 7]):
                                if v > 0 and prev_board[c * 7 + i] <= 0:
                                    up += i < 3.5
                                    down += i >= 3.5
                                    break
                    top_first = up >= down
                    top_first_checked = True
                    print(f"堆叠方向: {'视觉顶=栈顶' if top_first else '视觉底=栈顶'}")
                values = []
                for c in changed:
                    v = infer_drop(prev_board[c * 7:(c + 1) * 7], board[c * 7:(c + 1) * 7], top_first)
                    values.append(v)
                # 未变化列：掉落值 = 该列新顶（若有牌）；空列 = 掉落值（新块在底部？无法推断则 None）
                for c in range(6):
                    if c not in changed:
                        col = board[c * 7:(c + 1) * 7]
                        vals = [v for v in col if v > 0]
                        values.append(vals[0] if vals else None)
                cycle += 1
                n_drop += 1
                emit({"t": time.time(), "type": "drop", "game": game_id, "cycle": cycle,
                      "values": values, "unknown": [i for i, v in enumerate(values) if v is None],
                      "board": board})
                vs = " ".join("-" if v is None else str(v) for v in values)
                print(f"[掉落 {game_id}#{cycle}] {vs}  周期值分布: {sorted(v for v in values if v)}")
                if args.max_cycles and n_drop >= args.max_cycles:
                    print(f"已采满 {n_drop} 个周期，退出")
                    break

        now = time.time()
        if now - last_state_t >= args.state_every:
            last_state_t = now
            emit({"t": now, "type": "state", "game": game_id, "cycle": cycle, "board": board})

        prev_board, prev_preview = board, preview
        time.sleep(args.interval)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n已停止，数据在输出文件中")
