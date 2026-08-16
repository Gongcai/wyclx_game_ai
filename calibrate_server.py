#!/usr/bin/env python
"""棋盘标定服务器：在浏览器里手动标注棋盘网格与牌面颜色。

用法：
    .venv/bin/python capture_frame.py --out runs/records/board-frame.png   # 先抓帧
    .venv/bin/python calibrate_server.py --frame runs/records/board-frame.png --port 8123
然后浏览器打开 http://127.0.0.1:8123/ 标注，点保存 → configs/capture.json
"""

import argparse
import json
import socket
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

FRAME = None
CONFIG_OUT = None


def merge_incremental(old, new):
    """把当前帧新采的颜色/整格模板合并进已有配置。

    原始点击坐标属于某一张具体截图，不能跨帧复用；可复用的是已经裁剪好的
    模板数组、颜色样本以及棋盘/预告布局。
    """
    if not old:
        return new

    def merge_samples(previous, current):
        merged = {str(level): list(samples) for level, samples in (previous or {}).items()}
        for level, samples in (current or {}).items():
            merged.setdefault(str(level), []).extend(samples)
        return merged

    new["colors"] = merge_samples(old.get("colors"), new.get("colors"))
    if old.get("preview_colors"):
        new["preview_colors"] = merge_samples(
            old.get("preview_colors"), new.get("preview_colors"),
        )

    old_preview = old.get("preview") or {}
    new_preview = new.get("preview") or {}
    if new_preview:
        new_preview["colors"] = merge_samples(
            old_preview.get("colors"), new_preview.get("colors"),
        ) or None

    old_templates = old.get("templates") or {}
    new_templates = new.get("templates") or {}
    if old_templates or new_templates:
        template = dict(old_templates or new_templates)
        template["size"] = new_templates.get("size", old_templates.get("size"))
        template["cell"] = new_templates.get("cell", old_templates.get("cell"))
        for key in ("items_b", "items_p"):
            items = dict(old_templates.get(key) or {})
            items.update(new_templates.get(key) or {})
            template[key] = items or None
        new["templates"] = template
        new["identify"] = "template"

    accumulated = new.setdefault("calib", {})
    accumulated["accumulated_template_levels"] = {
        "board": sorted((new.get("templates") or {}).get("items_b") or {}),
        "preview": sorted((new.get("templates") or {}).get("items_p") or {}),
    }
    return new


def build_config(frame_path, board, preview, samples, adjustments=None, dtemplates=None):
    img = cv2.imread(str(frame_path))
    if img is None:
        raise ValueError(f"无法读取帧图: {frame_path}")
    h, w = img.shape[:2]
    # 校验范围
    for k in ("x0", "y0", "x1", "y1"):
        v = board.get(k)
        if v is None:
            raise ValueError(f"board 缺少 {k}")
    if not (0 <= board["x0"] < board["x1"] <= w and 0 <= board["y0"] < board["y1"] <= h):
        raise ValueError(f"棋盘框越界: {board} 帧尺寸 {w}x{h}")
    # 样本颜色（中心 5x5 均值，BGR -> RGB）；kind=p 为预告样本（预告有颜色减淡效果，颜色不同）
    colors, colors_p = {}, {}
    for s in samples:
        x, y = int(s["x"]), int(s["y"])
        v = int(s["v"])
        kind = s.get("kind", "b")
        if not (0 <= x < w and 0 <= y < h):
            continue
        patch = img[max(0, y - 2):y + 3, max(0, x - 2):x + 3].reshape(-1, 3).mean(axis=0)
        r, g, b = [round(float(v)) for v in patch[::-1]]
        (colors_p if kind == "p" else colors).setdefault(v, []).append([r, g, b])
    if not colors and not colors_p and not dtemplates:
        raise ValueError("没有有效样本（至少要有数字模板或颜色样本）")
    if not colors and not dtemplates:
        raise ValueError("棋盘样本为空（请采样棋盘上的牌或数字模板）")
    adjustments = adjustments or {}
    # 42 格中心 + 6 预告格中心（应用手动微调偏移）
    def cells(box, cols, rows, kind):
        out = []
        for c in range(cols):
            for r in range(rows):
                key = f"{kind},{c},{r}" if kind == "b" else f"p,{c}"
                dx, dy = adjustments.get(key, [0, 0])
                out.append([
                    box["x0"] + (box["x1"] - box["x0"]) * (c + 0.5) / cols + dx,
                    box["y0"] + (box["y1"] - box["y0"]) * (r + 0.5) / rows + dy,
                ])
        return out
    centers_b = cells(board, 6, 7, "b")
    cfg = {
        "window": {"title": "一梦江湖", "class": "steam_app_4277773963"},
        "frame": str(frame_path),
        "frame_size": [w, h],
        "board": {k: round(float(board[k]), 1) for k in ("x0", "y0", "x1", "y1")},
        "cell_centers": centers_b,
        "preview": None,
        "colors": colors,
    }
    # 整格模板（识别方式：整格归一化相关，颜色/字体/亮度无关）
    templates = None
    if dtemplates:
        tw, th = 48, 32
        gw = (board["x1"] - board["x0"]) / 6
        gh = (board["y1"] - board["y0"]) / 7
        centers_p = []
        pv_gw = pv_gh = None
        if preview:
            centers_p = preview.get("points") or cells(preview, 6, 1, "p")
            pv_gw = (preview["x1"] - preview["x0"]) / 6
            pv_gh = preview["y1"] - preview["y0"]

        def crop_cell(cx, cy, w2, h2):
            x0, y0 = int(cx - w2 / 2), int(cy - h2 / 2)
            x1, y1 = int(cx + w2 / 2), int(cy + h2 / 2)
            patch = img[max(0, y0):y1, max(0, x0):x1]
            if patch.shape[0] < 2 or patch.shape[1] < 2:
                return None
            patch = cv2.resize(patch, (tw, th))
            return [int(v) for v in patch[:, :, ::-1].flatten()]  # BGR->RGB

        items_b, items_p = {}, {}
        for t in dtemplates:
            v = int(t["v"])
            kind = t.get("kind", "b")
            if kind == "p" and centers_p:
                cx, cy = centers_p[int(t["c"])]
                tpl = crop_cell(cx, cy, pv_gw, pv_gh)
                if tpl:
                    items_p.setdefault(v, tpl)
            else:
                cx, cy = centers_b[int(t["c"]) * 7 + int(t["r"])]
                tpl = crop_cell(cx, cy, gw, gh)
                if tpl:
                    items_b.setdefault(v, tpl)
        if not items_b and not items_p:
            raise ValueError("整格模板为空（样本无效）")

        templates = {
            "size": [tw, th],
            "items_b": items_b,
            "items_p": items_p or None,
            "cell": {"gw": round(gw, 1), "gh": round(gh, 1),
                     "pv_gw": round(pv_gw, 1) if pv_gw else None,
                     "pv_gh": round(pv_gh, 1) if pv_gh else None},
        }
    cfg["templates"] = templates
    cfg["identify"] = "template" if templates else "color"

    if preview:
        if preview.get("points"):
            # 用户手动放置的预告点（独立于棋盘列对齐），直接采用
            centers = [[float(p[0]), float(p[1])] for p in preview["points"]]
        else:
            centers = cells(preview, 6, 1, "p")
        cfg["preview"] = {
            "box": {k: round(float(preview[k]), 1) for k in ("x0", "y0", "x1", "y1")},
            "centers": centers,
            "colors": colors_p or None,  # 预告颜色（减淡效果），缺省用棋盘颜色
        }
    # 原始标定状态（供标定页面加载复用，免重复标定）
    cfg["calib"] = {
        "board": {k: round(float(board[k]), 1) for k in ("x0", "y0", "x1", "y1")},
        "preview_box": {k: round(float(preview[k]), 1) for k in ("x0", "y0", "x1", "y1")} if preview else None,
        "pv_points": preview.get("points") if preview else None,
        "adjustments": adjustments or {},
        "dtemplates": [{"c": t["c"], "r": t["r"], "v": t["v"], "kind": t.get("kind", "b")} for t in (dtemplates or [])],
        "samples": [{"x": s["x"], "y": s["y"], "v": s["v"], "kind": s.get("kind", "b")} for s in (samples or [])],
    }
    return cfg


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            body = Path(__file__).with_name("calibrate.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/config":
            p = Path(CONFIG_OUT)
            if not p.exists():
                self.send_response(404)
                self.end_headers()
                return
            cfg = json.loads(p.read_text())
            # 换了一张截图时只继承布局。旧 dtemplates/samples 的坐标指向旧帧，
            # 若直接套到新帧会把错误图块写进原等级模板。
            if cfg.get("frame") != str(FRAME) and cfg.get("calib"):
                cfg["calib"]["dtemplates"] = []
                cfg["calib"]["samples"] = []
                cfg["calib"]["incremental_new_frame"] = True
            body = json.dumps(cfg, ensure_ascii=False).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/frame.png":
            body = Path(FRAME).read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path != "/save":
            self.send_response(404)
            self.end_headers()
            return
        try:
            ln = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(ln))
            out = Path(CONFIG_OUT)
            out.parent.mkdir(parents=True, exist_ok=True)
            old_cfg = None
            if out.exists():
                # 保存前自动备份，误覆盖可恢复
                Path(str(out) + ".backup.json").write_bytes(out.read_bytes())
                try:
                    old_cfg = json.loads(out.read_text())
                except Exception:
                    old_cfg = None
            cfg = build_config(FRAME, data["board"], data.get("preview"), data.get("samples", []),
                               data.get("adjustments"), data.get("dtemplates"))
            cfg = merge_incremental(old_cfg, cfg)
            out.write_text(json.dumps(cfg, ensure_ascii=False, indent=1))
            body = json.dumps({"ok": True, "path": str(out)}).encode()
        except Exception as e:
            body = json.dumps({"ok": False, "error": str(e)}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", required=True, help="标定帧图（capture_frame.py 的输出）")
    ap.add_argument("--config", default="configs/capture.json")
    ap.add_argument("--port", type=int, default=8123)
    args = ap.parse_args()
    global FRAME, CONFIG_OUT
    FRAME, CONFIG_OUT = args.frame, args.config
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"标定服务器: http://127.0.0.1:{args.port}/")
    print(f"帧图: {FRAME}   保存到: {CONFIG_OUT}")
    print("Ctrl+C 退出")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
