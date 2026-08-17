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
from pathlib import Path
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
        self.preview_cell_templates = {}
        self.preview_bg_lab = None
        self.board_subject_templates = {}
        self.board_bg_lab = None
        if self.tpl:
            self.tw, self.th = self.tpl["size"]
            preview_items = self.tpl.get("items_p") or {}
            self.has_preview_templates = bool(preview_items)
            for k, items in (("b", self.tpl["items_b"]), ("p", preview_items)):
                self.tpl_items[k] = {
                    int(v): np.array(g, dtype=np.float32).reshape(self.th, self.tw, 3)
                    for v, g in items.items()}
            if not self.tpl_items["p"]:
                # 预告兔子与棋盘同素材（半透明渲染），模板相关做过均值归一化、
                # 对亮度衰减免疫：没有专用预告模板时直接复用棋盘模板，
                # 远比单点颜色最近邻稳（2/3 两个米棕色仅差 ~30 色阶）。
                self.tpl_items["p"] = self.tpl_items["b"]
            c = self.tpl.get("cell", {})
            self.tpl_gw, self.tpl_gh = c.get("gw", 0.0), c.get("gh", 0.0)
            self.tpl_pgw, self.tpl_pgh = c.get("pv_gw", 0.0), c.get("pv_gh", 0.0)
            # 预告兔子形状模板：从棋盘模板抠出兔子本体（中值背景剥离 + 最大
            # 连通域）。与预告侧用同样的抠法，形状相关极性一致。
            self._tpl_rabbit_norm = {}
            self._tpl_rabbit_sq = {}
            for v, g in self.tpl["items_b"].items():
                if v == "0":
                    continue
                tbgr = np.array(g, np.float32).reshape(
                    self.th, self.tw, 3)[:, :, ::-1]
                med = np.median(tbgr.reshape(-1, 3), axis=0)
                dev = np.abs(tbgr - med).sum(2)
                m = (dev > 60).astype(np.uint8) * 255
                m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
                n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
                if n <= 1:
                    continue
                best = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                ys, xs = np.where(lab == best)
                rabbit = tbgr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
                a = cv2.resize(rabbit.astype(np.uint8), (48, 48)).astype(np.float32).ravel()
                a = a - a.mean()
                self._tpl_rabbit_norm[int(v)] = a
                self._tpl_rabbit_sq[int(v)] = float((a * a).sum())
            self._load_board_subject_templates()
        self._load_preview_cell_templates(cfg.get("preview_template_dir"))

    def _load_board_subject_templates(self):
        """从棋盘 0 级模板估计底色，建立只看牌面主体的模板。"""
        if not self.tpl_items.get("b") or 0 not in self.tpl_items["b"]:
            return
        empty_rgb = self.tpl_items["b"][0].astype(np.uint8)
        empty_bgr = empty_rgb.reshape(self.th, self.tw, 3)[:, :, ::-1]
        empty_lab = cv2.cvtColor(empty_bgr, cv2.COLOR_BGR2LAB).reshape(-1, 3)
        self.board_bg_lab = np.median(empty_lab, axis=0).astype(np.float32)
        for value, item in self.tpl_items["b"].items():
            if value == 0:
                continue
            image = item.astype(np.uint8).reshape(self.th, self.tw, 3)[:, :, ::-1]
            feature, _area_ratio = self._board_subject_feature(image)
            if feature is not None:
                self.board_subject_templates[value] = feature
        if any(value not in self.board_subject_templates for value in range(1, 9)):
            self.board_subject_templates = {}

    def _board_subject_feature(self, image):
        """提取棋盘牌面主体，忽略格子底色和小范围动画噪声。"""
        if image is None or image.size == 0 or self.board_bg_lab is None:
            return None, 0.0
        blurred = cv2.GaussianBlur(image, (3, 3), 0)
        lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB).astype(np.float32)
        distance = np.linalg.norm(lab - self.board_bg_lab, axis=2)
        mask = (distance > 7.0).astype(np.uint8) * 255
        mask[:1] = mask[-1:] = 0
        mask[:, :1] = mask[:, -1:] = 0
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        if count <= 1:
            return None, 0.0
        pick = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        x, y, width, height, area = stats[pick]
        area_ratio = float(area) / float(image.shape[0] * image.shape[1])
        if area < 100 or area_ratio < 0.06:
            return None, area_ratio
        subject_mask = (labels[y:y + height, x:x + width] == pick).astype(np.float32)
        subject = lab[y:y + height, x:x + width] - self.board_bg_lab
        subject *= subject_mask[:, :, None]
        subject = cv2.resize(subject, (48, 48)).ravel()
        subject_mask = cv2.resize(subject_mask, (48, 48)).ravel() * 20.0
        feature = np.concatenate((subject, subject_mask))
        norm = float(np.linalg.norm(feature))
        return (feature / norm if norm > 0 else None), area_ratio

    def _classify_board_subject(self, image):
        feature, _area_ratio = self._board_subject_feature(image)
        if feature is None:
            return 0
        scores = sorted(
            (float(np.dot(feature, item)), value)
            for value, item in self.board_subject_templates.items()
        )
        scores.reverse()
        top = scores[0]
        second_score = scores[1][0] if len(scores) > 1 else -1.0
        # 低置信度直接拒绝，避免 3/4 等相邻等级的猜测进入搜索。
        if top[0] < 0.65 or top[0] - second_score < 0.10:
            return -1
        return top[1]

    def _load_preview_cell_templates(self, directory):
        """载入 preview-<等级>-*.png 单格模板；0 级样本用于估计背景。"""
        if not directory:
            return
        paths = sorted(Path(directory).glob("preview-*.png"))
        labeled = []
        for path in paths:
            parts = path.stem.split("-")
            if len(parts) < 3 or not parts[1].isdigit():
                continue
            value = int(parts[1])
            image = cv2.imread(str(path))
            if image is not None and 0 <= value <= 7:
                labeled.append((value, image))
        backgrounds = [image for value, image in labeled if value == 0]
        if not backgrounds:
            return
        medians = []
        for image in backgrounds:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).reshape(-1, 3)
            medians.append(np.median(lab, axis=0))
        self.preview_bg_lab = np.median(np.array(medians), axis=0).astype(np.float32)
        for value, image in labeled:
            if value == 0:
                continue
            feature, _area_ratio = self._preview_subject_feature(image)
            if feature is not None:
                self.preview_cell_templates.setdefault(value, []).append(feature)
        # 等级不全时不能安全启用，否则缺失等级必然会被错分成已有等级。
        if any(value not in self.preview_cell_templates for value in range(1, 8)):
            self.preview_cell_templates = {}

    def _preview_subject_feature(self, image):
        """剥离米色动态背景，返回归一化主体特征和最大主体面积占比。"""
        if image is None or image.size == 0 or self.preview_bg_lab is None:
            return None, 0.0
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB).astype(np.float32)
        distance = np.linalg.norm(lab - self.preview_bg_lab, axis=2)
        mask = (distance > 9.0).astype(np.uint8) * 255
        mask[:2] = mask[-2:] = 0
        mask[:, :2] = mask[:, -2:] = 0
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        if count <= 1:
            return None, 0.0
        pick = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        x, y, width, height, area = stats[pick]
        area_ratio = float(area) / float(image.shape[0] * image.shape[1])
        # 干净背景里的飞鸟/边缘装饰实测远小于 5%，预告主体至少约 8%。
        if area_ratio < 0.05:
            return None, area_ratio
        subject_mask = (labels[y:y + height, x:x + width] == pick).astype(np.float32)
        subject = lab[y:y + height, x:x + width] - self.preview_bg_lab
        subject *= subject_mask[:, :, None]
        subject = cv2.resize(subject, (72, 72)).ravel()
        subject_mask = cv2.resize(subject_mask, (72, 72)).ravel() * 20.0
        feature = np.concatenate((subject, subject_mask))
        norm = float(np.linalg.norm(feature))
        return (feature / norm if norm > 0 else None), area_ratio

    def _classify_preview_cell(self, image):
        feature, _area_ratio = self._preview_subject_feature(image)
        if feature is None:
            return 0
        scores = sorted(
            (max(float(np.dot(feature, item)) for item in items), value)
            for value, items in self.preview_cell_templates.items()
        )
        scores.reverse()
        top = scores[0]
        second_score = scores[1][0] if len(scores) > 1 else -1.0
        if top[0] < 0.5 or top[0] - second_score < 0.08:
            return -1
        return top[1]

    def _preview_values_from_cells(self, image):
        box = self.preview.get("box") if self.preview else None
        if not box:
            return None
        x0, x1 = float(box["x0"]), float(box["x1"])
        y0, y1 = int(box["y0"]), int(box["y1"])
        cell_width = (x1 - x0) / 6.0
        values = []
        for col in range(6):
            left = max(0, int(x0 + col * cell_width))
            right = min(image.shape[1], int(x0 + (col + 1) * cell_width))
            patch = image[max(0, y0):min(image.shape[0], y1), left:right]
            values.append(self._classify_preview_cell(patch))
        if sum(value == 0 for value in values) >= 4:
            return None
        # 预告是整行同时出现；出现时个别空判定属于未识别，而非真实的 0。
        return [-1 if value == 0 else value for value in values]

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
        if kind == "b" and self.board_subject_templates:
            return self._classify_board_subject(patch)
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

    def _rabbit_from_diff(self, img, background, cx, cy, half=130, thr=80):
        """用与空背景的差分掩码抠预告格里的兔子本体（最大紧凑连通域）。

        thr=80：低于此为背景渲染噪声（<80），高于此为半透明兔子主体信号
        （80~150+），实测该值能把两者干净分开且不切碎兔子。
        """
        if background is None or img.shape != background.shape:
            return None
        x0, x1 = max(0, int(cx - half)), min(img.shape[1], int(cx + half))
        y0, y1 = max(0, int(cy - 70)), min(img.shape[0], int(cy + 70))
        current = img[y0:y1, x0:x1]
        empty = background[y0:y1, x0:x1]
        if current.size == 0 or current.shape != empty.shape:
            return None
        diff = cv2.absdiff(
            current.astype(np.int32), empty.astype(np.int32),
        ).sum(2)
        mask = (diff > thr).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(mask)
        compact, fallback = [], []
        for i in range(1, n):
            w, h, area = (stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT],
                          stats[i, cv2.CC_STAT_AREA])
            if area < 400:
                continue
            (compact if 0.5 <= w / max(1, h) <= 2.0 else fallback).append((area, i))
        pool = compact or fallback
        if not pool:
            return None
        pool.sort(reverse=True)
        _, pick = pool[0]
        ys, xs = np.where(lab == pick)
        return img[y0 + ys.min():y0 + ys.max() + 1, x0 + xs.min():x0 + xs.max() + 1]

    def _classify_rabbit(self, rabbit):
        """兔子本体形状相关（均值归一化，亮度/透明度无关）。

        2/3 两个米棕色的单点色差仅 ~30 色阶，颜色最近邻经常混；形状
        （耳形/脸型）是稳定判别特征。返回 (等级, 置信间隔)，不足门槛为 -1。
        """
        a = cv2.resize(rabbit, (48, 48)).astype(np.float32).ravel()
        a = a - a.mean()
        scores = sorted(
            ((float(np.dot(a, self._tpl_rabbit_norm[v])) /
              max(1e-9, np.sqrt(float((a * a).sum()) * self._tpl_rabbit_sq[v]))), v)
            for v in self._tpl_rabbit_norm)
        scores.reverse()
        top, second = scores[0], scores[1]
        if top[0] < 0.3 or top[0] - second[0] < 0.06:
            return -1
        return top[1]

    def preview_values(self, img, background=None, presence_threshold=0.015):
        if not self.preview:
            return None
        if self.preview_cell_templates:
            return self._preview_values_from_cells(img)
        if background is not None and img.shape != background.shape:
            # 标定坐标和空背景都属于固定帧尺寸。尺寸变化时继续差分不仅会
            # 触发 OpenCV 异常，识别结果本身也没有意义。
            return None
        if background is not None:
            visible, _scores = self.preview_visible(img, background, presence_threshold)
            if visible is False:
                return None
        # 有预告专用整格模板时优先使用。旧实现即使 items_p 已存在，也会先
        # 走棋盘兔子形状分支，导致新采的预告模板实际上从未生效。
        if self.tpl and self.has_preview_templates:
            return [self.classify(img, x, y, "p") for x, y in self.preview["centers"]]
        # 优先：兔子本体形状匹配（需要空背景参照）。比单点颜色抗光照/背景，
        # 解决 2/3 两个米棕色易混的问题。
        if background is not None and getattr(self, "_tpl_rabbit_norm", None):
            out = []
            for x, y in self.preview["centers"]:
                rabbit = self._rabbit_from_diff(img, background, x, y)
                out.append(self._classify_rabbit(rabbit) if rabbit is not None else -1)
            return out
        # 专用整格模板比单点颜色更抗复杂背景；没有模板才回退到颜色。
        if self.tpl and self.tpl_items.get("p"):
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
