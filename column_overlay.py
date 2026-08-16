#!/usr/bin/env python
"""在游戏窗口上方绘制 1~6 列编号。

普通模式下编号窗口不接收鼠标输入，点击会穿透到游戏。使用 ``--edit``
进入标定模式：拖动数字 1 或使用键盘微调首列位置，用 [ / ] 调整列间距。
"""

import argparse
import ctypes
import ctypes.util
import json
import os
import re
import subprocess
import sys
import tkinter as tk
from pathlib import Path


DEFAULT_CONFIG = "configs/column_overlay.json"
DEFAULT_CAPTURE_CONFIG = "configs/capture.json"


def read_json(path):
    try:
        with open(path, encoding="utf-8") as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def window_geometry(window_class, title):
    """从 XWayland 查询窗口根坐标，返回 (x, y, width, height)。"""
    result = subprocess.run(
        ["xwininfo", "-root", "-tree"], capture_output=True, text=True,
    )
    if result.returncode != 0:
        return None
    # 示例：0x... "标题": ("class" "class") 1332x748+2043+102
    size_re = re.compile(r"\s(\d+)x(\d+)([+-]\d+)([+-]\d+)\s")
    fallback = None
    for line in result.stdout.splitlines():
        if window_class and f'("{window_class}"' not in line:
            continue
        match = size_re.search(line)
        if not match:
            continue
        geometry = tuple(int(value) for value in (
            match.group(3), match.group(4), match.group(1), match.group(2),
        ))
        if title and f'"{title}"' in line:
            return geometry
        if geometry[2] > 100 and geometry[3] > 100:
            fallback = geometry
    return fallback


def hypr_window_info(window_class, title):
    """查询 Hyprland 逻辑坐标及窗口所在工作区。"""
    result = subprocess.run(
        ["hyprctl", "clients", "-j"], capture_output=True, text=True,
    )
    if result.returncode != 0:
        return None
    try:
        clients = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None
    fallback = None
    for client in clients:
        if client.get("class") != window_class:
            continue
        if client.get("title") == title:
            return client
        fallback = client
    return fallback


def set_placement_rule(info, x, y, edit):
    """先把 Tk 窗口放到游戏所在显示器，解决混合缩放坐标歧义。"""
    workspace = info.get("workspace", {}).get("id", 1)
    properties = (
        'name="codex-game-column-overlay",'
        'match={class="^(GameColumnOverlay|Gamecolumnoverlay)$"},'
        f'float=true,pin=true,workspace="{workspace} silent",'
        f'move="{x} {y}"'
    )
    if not edit:
        properties += ",no_focus=true"
    script = (
        "if _G.CODEX_COLUMN_OVERLAY_RULE then "
        "_G.CODEX_COLUMN_OVERLAY_RULE:set_enabled(false) end; "
        f"_G.CODEX_COLUMN_OVERLAY_RULE=hl.window_rule({{{properties}}})"
    )
    result = subprocess.run(
        ["hyprctl", "eval", script], capture_output=True, text=True,
    )
    if result.returncode != 0 or result.stdout.startswith("error"):
        raise RuntimeError(f"无法注册叠层窗口规则：{result.stderr or result.stdout}")


def unset_placement_rule():
    try:
        subprocess.run(
            [
                "hyprctl", "eval",
                "if _G.CODEX_COLUMN_OVERLAY_RULE then "
                "_G.CODEX_COLUMN_OVERLAY_RULE:set_enabled(false) end",
            ],
            capture_output=True, text=True,
        )
    except KeyboardInterrupt:
        pass


def defaults_from_capture(path, game_width):
    """把截图物理像素标定换算成游戏窗口逻辑像素。"""
    capture = read_json(path)
    frame_size = capture.get("frame_size") or []
    centers = capture.get("board", {}).get("cell_centers") or []
    if len(frame_size) < 2 or not frame_size[0] or len(centers) < 36:
        return {"first_x": 390, "first_y": 60, "gap": 136}
    scale = frame_size[0] / game_width
    column_x = []
    for col in range(6):
        points = centers[col * 7:(col + 1) * 7]
        column_x.append(sum(point[0] for point in points) / len(points) / scale)
    return {
        "first_x": round(column_x[0]),
        "first_y": 60,
        "gap": round(sum(
            column_x[index + 1] - column_x[index] for index in range(5)
        ) / 5),
    }


class NativeX11:
    """绕过 Tk 的多显示器坐标换算，直接操作 XWayland 窗口。"""

    def __init__(self):
        x11_name = ctypes.util.find_library("X11")
        xfixes_name = ctypes.util.find_library("Xfixes")
        if not x11_name or not xfixes_name:
            raise RuntimeError("系统缺少 libX11/libXfixes")
        self.x11 = ctypes.CDLL(x11_name)
        self.xfixes = ctypes.CDLL(xfixes_name)
        self.x11.XOpenDisplay.argtypes = [ctypes.c_char_p]
        self.x11.XOpenDisplay.restype = ctypes.c_void_p
        self.x11.XCloseDisplay.argtypes = [ctypes.c_void_p]
        self.x11.XSync.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.x11.XMoveWindow.argtypes = [
            ctypes.c_void_p, ctypes.c_ulong, ctypes.c_int, ctypes.c_int,
        ]
        self.x11.XQueryTree.argtypes = [
            ctypes.c_void_p, ctypes.c_ulong,
            ctypes.POINTER(ctypes.c_ulong), ctypes.POINTER(ctypes.c_ulong),
            ctypes.POINTER(ctypes.POINTER(ctypes.c_ulong)),
            ctypes.POINTER(ctypes.c_uint),
        ]
        self.x11.XQueryTree.restype = ctypes.c_int
        self.x11.XFree.argtypes = [ctypes.c_void_p]
        self.xfixes.XFixesCreateRegion.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
        ]
        self.xfixes.XFixesCreateRegion.restype = ctypes.c_ulong
        self.xfixes.XFixesSetWindowShapeRegion.argtypes = [
            ctypes.c_void_p, ctypes.c_ulong, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_ulong,
        ]
        self.xfixes.XFixesDestroyRegion.argtypes = [
            ctypes.c_void_p, ctypes.c_ulong,
        ]
        self.display = self.x11.XOpenDisplay(
            os.environ.get("DISPLAY", "").encode() or None,
        )
        if not self.display:
            raise RuntimeError("无法连接 XWayland DISPLAY")

    def top_window_id(self, window):
        """Tk 的 winfo_id 是内容窗口；沿父链找到根窗口下的外层窗口。"""
        current = ctypes.c_ulong(window.winfo_id()).value
        while True:
            root = ctypes.c_ulong()
            parent = ctypes.c_ulong()
            children = ctypes.POINTER(ctypes.c_ulong)()
            count = ctypes.c_uint()
            ok = self.x11.XQueryTree(
                self.display, current, ctypes.byref(root), ctypes.byref(parent),
                ctypes.byref(children), ctypes.byref(count),
            )
            if children:
                self.x11.XFree(children)
            if not ok or not parent.value or parent.value == root.value:
                return current
            current = parent.value

    def move(self, window, x, y):
        self.x11.XMoveWindow(self.display, self.top_window_id(window), x, y)
        self.x11.XSync(self.display, 0)

    def make_click_through(self, window):
        region = self.xfixes.XFixesCreateRegion(self.display, None, 0)
        # ShapeInput = 2；空 region 表示整个窗口均不接收输入。
        self.xfixes.XFixesSetWindowShapeRegion(
            self.display, self.top_window_id(window), 2, 0, 0, region,
        )
        self.x11.XSync(self.display, 0)
        self.xfixes.XFixesDestroyRegion(self.display, region)


class ColumnOverlay:
    def __init__(self, args):
        self.args = args
        self.geometry = window_geometry(args.window_class, args.title)
        self.hypr_info = hypr_window_info(args.window_class, args.title)
        if self.geometry is None or self.hypr_info is None:
            raise RuntimeError(f"没有找到游戏窗口：{args.title} ({args.window_class})")
        game_x, game_y, game_width, game_height = self.geometry
        del game_x, game_y, game_height

        defaults = defaults_from_capture(args.capture_config, game_width)
        saved = read_json(args.config)
        self.settings = {
            "first_x": int(saved.get("first_x", defaults["first_x"])),
            "first_y": int(saved.get("first_y", defaults["first_y"])),
            "gap": int(saved.get("gap", defaults["gap"])),
            "size": int(saved.get("size", 38)),
            "font_size": int(saved.get("font_size", 22)),
        }
        self.labels = []
        self.drag_origin = None
        self.last_geometry = None
        hypr_x, hypr_y = self.hypr_info["at"]
        size = self.settings["size"]
        set_placement_rule(
            self.hypr_info,
            hypr_x + self.settings["first_x"] - size // 2,
            hypr_y + self.settings["first_y"] - size // 2,
            args.edit,
        )
        self.root = tk.Tk(className="GameColumnOverlay")
        self.native = NativeX11()
        self._configure_window(self.root, 1)
        self.labels.append(self.root)
        for number in range(2, 7):
            window = tk.Toplevel(self.root, class_="GameColumnOverlay")
            self._configure_window(window, number)
            self.labels.append(window)

        self.reposition(force=True)
        # 窗口规则先把叠层归属到游戏的显示器，随后再应用各列精确坐标。
        self.root.after(300, lambda: self.reposition(force=True))
        self.root.after(500, self.follow_game)
        if args.edit:
            self.root.after(150, self.begin_edit)
            self.print_help()
        else:
            self.root.after(450, self.enable_click_through)

    def _configure_window(self, window, number):
        size = self.settings["size"]
        window.withdraw()
        window.overrideredirect(True)
        window.attributes("-topmost", True)
        window.attributes("-alpha", self.args.opacity)
        window.configure(background="#191919")
        canvas = tk.Canvas(
            window, width=size, height=size, highlightthickness=0,
            borderwidth=0, background="#191919",
        )
        canvas.pack()
        pad = 2
        canvas.create_oval(
            pad, pad, size - pad, size - pad,
            fill="#ffd23f", outline="#111111", width=2,
        )
        canvas.create_text(
            size / 2, size / 2, text=str(number), fill="#111111",
            font=("Sans", self.settings["font_size"], "bold"),
        )
        if self.args.edit and number == 1:
            canvas.bind("<ButtonPress-1>", self.drag_start)
            canvas.bind("<B1-Motion>", self.drag_move)
            canvas.bind("<ButtonRelease-1>", self.drag_end)
            canvas.bind("<Button-4>", lambda event: self.adjust_gap(event, -1))
            canvas.bind("<Button-5>", lambda event: self.adjust_gap(event, 1))
            canvas.bind("<MouseWheel>", self.mouse_wheel)
        window.deiconify()

    def begin_edit(self):
        self.root.focus_force()
        self.root.bind("<KeyPress>", self.on_key)

    def enable_click_through(self):
        for window in self.labels:
            self.native.make_click_through(window)

    def game_relative_position(self):
        game_x, game_y, _, _ = self.geometry
        return (
            game_x + self.settings["first_x"],
            game_y + self.settings["first_y"],
        )

    def reposition(self, force=False):
        first_x, first_y = self.game_relative_position()
        size = self.settings["size"]
        positions = []
        for index, window in enumerate(self.labels):
            center_x = first_x + index * self.settings["gap"]
            x = round(center_x - size / 2)
            y = round(first_y - size / 2)
            positions.append((x, y))
            # geometry() 在混合缩放多显示器上会错误减去显示器原点；先让
            # Tk 确认尺寸，再通过 XMoveWindow 使用真正的 X 根坐标移动。
            window.geometry(f"{size}x{size}+0+0")
            window.update_idletasks()
            self.native.move(window, x, y)
            window.lift()
        if force or positions != self.last_geometry:
            self.last_geometry = positions

    def follow_game(self):
        geometry = window_geometry(self.args.window_class, self.args.title)
        if geometry is not None and geometry != self.geometry:
            self.geometry = geometry
            self.reposition()
        self.root.after(500, self.follow_game)

    def drag_start(self, event):
        self.root.focus_force()
        self.drag_origin = (
            event.x_root, event.y_root,
            self.settings["first_x"], self.settings["first_y"],
        )

    def drag_move(self, event):
        if self.drag_origin is None:
            return
        x, y, first_x, first_y = self.drag_origin
        self.settings["first_x"] = first_x + event.x_root - x
        self.settings["first_y"] = first_y + event.y_root - y
        self.reposition()
        self.show_status()

    def drag_end(self, event):
        del event
        self.drag_origin = None
        self.save()

    def adjust_gap(self, event, direction):
        step = 10 if event.state & 0x1 else 1
        self.settings["gap"] = max(20, self.settings["gap"] + direction * step)
        self.reposition()
        self.show_status()
        self.save()

    def mouse_wheel(self, event):
        self.adjust_gap(event, -1 if event.delta > 0 else 1)

    def on_key(self, event):
        step = 10 if event.state & 0x1 else 1
        key = event.keysym
        if key == "Left":
            self.settings["first_x"] -= step
        elif key == "Right":
            self.settings["first_x"] += step
        elif key == "Up":
            self.settings["first_y"] -= step
        elif key == "Down":
            self.settings["first_y"] += step
        elif event.char in "[{":
            self.settings["gap"] = max(20, self.settings["gap"] - step)
        elif event.char in "]}":
            self.settings["gap"] += step
        elif key.lower() == "s":
            self.save()
            return
        elif key.lower() in {"q", "escape"}:
            self.save()
            self.root.destroy()
            return
        else:
            return
        self.reposition()
        self.show_status()

    def save(self):
        path = Path(self.args.config)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            json.dump(self.settings, file, ensure_ascii=False, indent=2)
            file.write("\n")
        print(f"\n已保存 {path}")

    def show_status(self):
        print(
            f"\r首列中心=({self.settings['first_x']}, {self.settings['first_y']})  "
            f"间距={self.settings['gap']}    ",
            end="", flush=True,
        )

    def print_help(self):
        print("列编号叠层：编辑模式")
        print("  拖动 1 / 方向键：移动全部编号（Shift=10 像素）")
        print("  在 1 上滚轮或 [ / ]：缩小/增大列间距（Shift=10 像素）")
        print("  鼠标调整会自动保存")
        print("  S：保存    Q/Esc：保存并退出")
        self.show_status()

    def run(self):
        self.root.mainloop()


def main():
    parser = argparse.ArgumentParser(description="在游戏窗口叠加 1~6 列编号")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--capture-config", default=DEFAULT_CAPTURE_CONFIG)
    parser.add_argument("--title", default="一梦江湖")
    parser.add_argument("--window-class", default="steam_app_4277773963")
    parser.add_argument("--opacity", type=float, default=0.88)
    parser.add_argument("--edit", action="store_true", help="允许拖动/键盘微调")
    args = parser.parse_args()
    if not 0.1 <= args.opacity <= 1.0:
        parser.error("--opacity 必须在 0.1 到 1.0 之间")
    try:
        ColumnOverlay(args).run()
    except KeyboardInterrupt:
        pass
    except (RuntimeError, tk.TclError) as error:
        parser.error(str(error))
    finally:
        unset_placement_rule()


if __name__ == "__main__":
    main()
