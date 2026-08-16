# H5 游戏研究版部署指南（兔王争霸赛 · 轨迹采集）

## 组成

| 文件 | 说明 |
|---|---|
| runs/h5game/ | 完整游戏（HTML/CSS/JS/图片/字体，全本地化，离线可玩） |
| runs/h5game/recorder.js | 轨迹录制器（零侵入：不改游戏逻辑，只观察点击与 DOM） |
| serve_game.py | 服务器：静态托管 + POST /api/traces 收集轨迹（纯标准库，无需 venv） |
| replay_trace.py | 轨迹重放校验器（H5 规则模拟器，确定性回放） |
| runs/traces/ | 轨迹存放目录（每局一个 JSON） |

## 本地试玩

    .venv/bin/python serve_game.py --port 8080
    浏览器打开 http://127.0.0.1:8080/   → 点"开始"（练习模式，无需登录）

游戏规则与 H5 原版完全一致（源码级：6×7、3合1、9消失、4步一轮、真实掉落表）。

## AI 自动玩（auto_play_h5.py）

    .venv/bin/python serve_game.py --port 8080 &
    .venv/bin/python auto_play_h5.py --url http://127.0.0.1:8080 --games 3

- Playwright 驱动 Chromium 打开页面，自动关声明弹窗、点"开始"
- 每步直读 `window.__game` + DOM（盘面/预告/相位/remain），r2 模型 PUCT
  决策后点击 `.grid-overlay`（点源列 → 等拿起动画进入 select 态 → 点目标列）
- 局终自动检测并重开下一局；`--revive` 可选接受研究版复活弹窗（默认局终）
- 轨迹按 recorder 格式写入 `runs/autoplay/`（页面内录制器也会照常上传一份），
  用 `replay_trace.py` 校验：实测 113 步 0 非法、快照 100% 一致
- 依赖 venv 内 playwright（浏览器需 `python -m playwright install chromium`）

## 录制内容（recorder.js）

- clicks：每次点击的列（两个一组 = 一次移动 src→dst，含无效点击，重放时校验）
- frames：棋盘/预告快照（每步移动/掉落/预览刷新后，350ms 稳定延迟）
- drops：每轮预告 6 个值（预告=实际掉落，源码确认）
- 局终自动存入 localStorage，批量 POST /api/traces（失败留待下次补传）

## 部署到服务器

    # 本地打包
    tar czf tuyr-h5.tar.gz runs/h5game serve_game.py replay_trace.py

    # 服务器（Linux，需 python3）
    mkdir -p /opt/tuyr && tar xzf tuyr-h5.tar.gz -C /opt/tuyr
    cd /opt/tuyr/runs && mkdir -p traces
    nohup python3 /opt/tuyr/serve_game.py --port 80 --bind 0.0.0.0 --root /opt/tuyr/runs/h5game --traces /opt/tuyr/runs/traces > /var/log/tuyr.log 2>&1 &

    # systemd（推荐）
    [Unit] Description=Tuyr H5
    [Service] ExecStart=/usr/bin/python3 /opt/tuyr/serve_game.py --port 80 --root /opt/tuyr/runs/h5game --traces /opt/tuyr/runs/traces
    Restart=always
    [Install] WantedBy=multi-user.target

    # 反向代理（nginx，域名+HTTPS 可选）
    server { listen 80; server_name game.example.com;
             location / { proxy_pass http://127.0.0.1:80; } }

## 数据回收与校验

    # 服务器轨迹拷回本地 runs/traces/
    scp -r user@server:/opt/tuyr/runs/traces/ runs/traces/

    # 校验（动作合法性 + 快照一致性 + 得分/最高级统计）
    .venv/bin/python replay_trace.py runs/traces/*.json --summary

    # 单局详情
    .venv/bin/python replay_trace.py runs/traces/xxx.json

## 数据格式（每局 JSON）

    {
      "id": "xxx", "rule": "tuyr-v1",
      "t_start": 1734..., "t_end": ...,
      "clicks": [1, 3, 2, 5, ...],          # 点击列序列
      "frames": [                            # 快照（click_n = 当时点击数）
        {"t": 0, "kind": "init", "board": [[1,1],...], "preview": [...]},
        {"t": 100, "kind": "board", "board": [...], "click_n": 2},
        {"t": 300, "kind": "preview", "preview": [1,3,2,5,4,6], "click_n": 6}
      ],
      "drops": [[1,3,2,5,4,6], ...]          # 每轮掉落值
    }
    board: 每列 底部->顶部 的等级数组（6 列 × N）

## 注意

- 练习模式无需登录；竞赛模式需要登录（部署版建议隐藏竞赛按钮，或保持原样玩家可自行登录）
- 轨迹上传失败会自动留在浏览器 localStorage，下次访问补传
- 一局随机玩 ~20 步、认真玩 100-300 步；目标攒 500+ 局
