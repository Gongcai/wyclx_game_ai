#!/usr/bin/env python
"""兔王争霸赛研究版服务器：静态托管游戏 + 收集对局轨迹。

用法：
    .venv/bin/python serve_game.py --port 8080            # 本地
    .venv/bin/python serve_game.py --port 80 --bind 0.0.0.0   # 服务器部署
"""

import argparse
import json
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = None
TRACES = None

CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".ttf": "font/ttf",
    ".mp3": "audio/mpeg",
    ".json": "application/json; charset=utf-8",
}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/":
            path = "/index.html"
        f = (ROOT / path.lstrip("/")).resolve()
        if ROOT.resolve() not in f.parents or not f.is_file():
            self._send(404, b"not found", "text/plain")
            return
        ctype = CONTENT_TYPES.get(f.suffix, "application/octet-stream")
        self._send(200, f.read_bytes(), ctype)

    def do_POST(self):
        if self.path != "/api/traces":
            self._send(404, b"not found", "text/plain")
            return
        try:
            ln = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(ln))
            items = data if isinstance(data, list) else [data]
            n = 0
            for it in items:
                if not isinstance(it, dict) or not it.get("clicks"):
                    continue
                # 增量上传时同一局会多次到达，按轨迹 id 覆盖而不是产生重复文件。
                trace_id = str(it.get("id", ""))
                if trace_id and trace_id.replace("_", "").replace("-", "").isalnum():
                    fn = f"trace-{trace_id}.json"
                else:
                    fn = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}.json"
                (TRACES / fn).write_text(json.dumps(it, ensure_ascii=False, indent=1),
                                         encoding="utf-8")
                n += 1
            self._send(200, json.dumps({"ok": True, "n": n}).encode(), "application/json")
        except Exception as e:
            self._send(200, json.dumps({"ok": False, "error": str(e)}).encode(),
                       "application/json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs/h5game", help="游戏静态目录")
    ap.add_argument("--traces", default="runs/traces", help="轨迹存放目录")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--bind", default="0.0.0.0")
    args = ap.parse_args()
    global ROOT, TRACES
    ROOT = Path(args.root)
    TRACES = Path(args.traces)
    TRACES.mkdir(parents=True, exist_ok=True)
    print(f"🐰 兔王争霸赛研究版: http://{args.bind}:{args.port}/")
    print(f"   轨迹目录: {TRACES.resolve()}")
    ThreadingHTTPServer((args.bind, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
