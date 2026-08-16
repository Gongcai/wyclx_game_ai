#!/usr/bin/env python
"""按空格抓取一次游戏窗口，并把识别棋盘打印到终端。

实时验证：
    .venv/bin/python verify_capture.py

离线验证现有图片：
    .venv/bin/python verify_capture.py --frame runs/records/board-frame.png --once

终端保持焦点即可；空格抓取，b 保存当前“无预告”背景，q 退出。
每次输出前会清空终端。
"""

import argparse
import json
import os
import sys
import termios
import time
import tty

import cv2

from capture_drops import Recognizer, find_window, grab_region


CLEAR = "\033[2J\033[H"


def clear_console():
    print(CLEAR, end="", flush=True)


def expected_frame_size(config):
    size = config.get("frame_size")
    if isinstance(size, list) and len(size) == 2:
        return int(size[0]), int(size[1])
    return None


def board_rows(flat):
    """标定顺序是逐列、每列从画面顶部到画面底部。"""
    return [[flat[col * 7 + row] for col in range(6)] for row in range(7)]


def cell_text(value):
    if value < 0:
        return "?"
    if value == 0:
        return "."
    return str(value)


def print_result(flat, preview, frame_shape, source, elapsed_ms, warning=None,
                 preview_scores=None, preview_threshold=None):
    clear_console()
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    height, width = frame_shape[:2]
    unknown = sum(value < 0 for value in flat)
    filled = sum(value > 0 for value in flat)
    preview_unknown = sum(value < 0 for value in preview) if preview is not None else 0
    print(f"识别时间: {now}    耗时: {elapsed_ms:.1f} ms")
    print(f"来源: {source}")
    print(
        f"画面: {width}×{height}    已占用: {filled}/42    "
        f"棋盘未识别: {unknown}    预告未识别: {preview_unknown}"
    )
    if warning:
        print(f"警告: {warning}")
    print()
    print("棋盘（与画面方向一致：上方在前，列 1~6 从左到右）")
    print("        1   2   3   4   5   6")
    print("      +---+---+---+---+---+---+")
    for row, values in enumerate(board_rows(flat), start=1):
        cells = "|".join(f" {cell_text(value):>1} " for value in values)
        print(f"上{row:<2}  |{cells}|")
        print("      +---+---+---+---+---+---+")
    print()
    print("各列栈（顶 → 底）")
    for col in range(6):
        values = [value for value in flat[col * 7:(col + 1) * 7] if value > 0]
        bad = any(value < 0 for value in flat[col * 7:(col + 1) * 7])
        suffix = "  [含未识别格]" if bad else ""
        print(f"  列 {col + 1}: {values}{suffix}")
    if preview is None:
        preview_text = "无（背景匹配）" if preview_scores is not None else "未判定"
    else:
        preview_text = "[" + ", ".join(cell_text(value) for value in preview) + "]"
    print(f"预告: {preview_text}")
    if preview_scores is not None:
        scores = ", ".join(f"{score:.3f}" for score in preview_scores)
        print(f"预告背景差分: [{scores}]  阈值={preview_threshold:.3f}")
    print()
    print("按空格重新抓取；无预告时按 b 保存背景；按 q 退出。")
    sys.stdout.flush()


def print_error(message):
    clear_console()
    print(f"抓取/识别失败: {message}")
    print()
    print("按空格重试；按 q 退出。")
    sys.stdout.flush()


def recognize_frame(recognizer, image, source, expected_size=None, save_path=None,
                    preview_background=None, preview_threshold=0.015):
    started = time.perf_counter()
    flat = recognizer.board(image)
    preview = recognizer.preview_values(
        image, background=preview_background,
        presence_threshold=preview_threshold,
    )
    preview_scores = recognizer.preview_change_scores(image, preview_background)
    elapsed_ms = (time.perf_counter() - started) * 1000
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        cv2.imwrite(save_path, image)
    height, width = image.shape[:2]
    warning = None
    if expected_size and (width, height) != expected_size:
        warning = (
            f"当前画面 {width}×{height} 与标定尺寸 "
            f"{expected_size[0]}×{expected_size[1]} 不同，坐标可能失准"
        )
    if preview_background is None and recognizer.preview:
        extra = "尚未标定无预告背景，请在预告隐藏时按 b"
        warning = f"{warning}；{extra}" if warning else extra
    print_result(
        flat, preview, image.shape, source, elapsed_ms, warning,
        preview_scores=preview_scores, preview_threshold=preview_threshold,
    )
    return not any(value < 0 for value in flat)


def capture_live(config):
    window_cfg = config.get("window", {})
    title = window_cfg.get("title", "一梦江湖")
    cls = window_cfg.get("class", "steam_app_4277773963")
    window = find_window(title, cls)
    if window is None:
        raise RuntimeError(f"找不到窗口：title 包含 {title!r} 或 class 包含 {cls!r}")
    x, y = window["at"]
    width, height = window["size"]
    image = grab_region(x, y, width, height)
    if image is None:
        raise RuntimeError("grim 未能抓取窗口区域")
    source = f"{window.get('title', '')} 位置=({x},{y}) 尺寸={width}×{height}"
    return image, source


def read_key():
    """raw 模式读取单键，并确保退出时恢复终端设置。"""
    fd = sys.stdin.fileno()
    previous = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return os.read(fd, 1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, previous)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/capture.json")
    parser.add_argument("--frame", help="读取图片而不是实时抓取窗口")
    parser.add_argument("--once", action="store_true", help="识别一次后退出")
    parser.add_argument(
        "--save", default="runs/records/verify-latest.png",
        help="每次实时抓帧覆盖保存到此路径；传空字符串可关闭",
    )
    parser.add_argument(
        "--preview-background", default="runs/records/preview-empty.png",
        help="无预告时的参考截图；交互模式按 b 生成",
    )
    parser.add_argument("--preview-threshold", type=float, default=0.015)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as file:
        config = json.load(file)
    recognizer = Recognizer(config)
    expected_size = expected_frame_size(config)

    def load_preview_background():
        if not args.preview_background or not os.path.exists(args.preview_background):
            return None
        return cv2.imread(args.preview_background)

    def run_once():
        try:
            if args.frame:
                image = cv2.imread(args.frame)
                if image is None:
                    raise RuntimeError(f"无法读取图片 {args.frame}")
                source = args.frame
                save_path = None
            else:
                image, source = capture_live(config)
                save_path = args.save or None
            recognize_frame(
                recognizer, image, source, expected_size=expected_size,
                save_path=save_path,
                preview_background=load_preview_background(),
                preview_threshold=args.preview_threshold,
            )
        except Exception as exc:
            print_error(str(exc))

    if args.once:
        run_once()
        return
    if not sys.stdin.isatty():
        raise RuntimeError("交互模式需要在真实终端中运行；离线检查请加 --once")

    clear_console()
    print("棋盘识别验证器已启动。")
    print("请保持游戏窗口完整可见，并保持当前终端焦点。")
    print("按空格抓取；无预告时按 b 保存背景；按 q 退出。")
    while True:
        key = read_key()
        if key in (b"q", b"Q", b"\x03"):
            clear_console()
            print("已退出棋盘识别验证器。")
            return
        if key == b" ":
            run_once()
        elif key in (b"b", b"B"):
            try:
                image, source = capture_live(config)
                os.makedirs(os.path.dirname(args.preview_background) or ".", exist_ok=True)
                cv2.imwrite(args.preview_background, image)
                clear_console()
                print(f"已保存无预告背景: {args.preview_background}")
                print(f"来源: {source}")
                print("请确认保存时画面上确实没有预告。")
                print("按空格开始验证；按 b 可重新标定；按 q 退出。")
            except Exception as exc:
                print_error(str(exc))


if __name__ == "__main__":
    main()
