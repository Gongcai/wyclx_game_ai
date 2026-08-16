# 实机掉落数据采集（工具已就绪）

更新时间：2026-08-13（工具完成，等待小游戏界面）
目标：从实机游戏画面采集真实掉落序列 → 拟合分布 / 简单形-复杂形 regime 建模。

## 环境（已验证可用）

- Hyprland + XWayland；grim/hyprctl 可用；CUDA 可用（RTX 4060，权限已放开）
- 游戏：Steam Proton 运行"一梦江湖"窗口（class=steam_app_4277773963，标题=一梦江湖）
- 抓屏链路：capture_frame.py 动态定位窗口（窗口移动/换屏都不怕）→ grim 区域抓帧 ✅
- 识别链路：opencv 固定点位颜色最近邻（calibrate 生成 42 格 + 预告格 + 颜色表）✅
- 掉落推断：帧间变化列数≥4 判定掉落帧；simulate_insert 反推每列掉落值（2000 次随机测试 0 失败）✅

## 使用步骤（用户操作）

### 1. 抓标定帧（在小游戏棋盘界面时执行）
    .venv/bin/python capture_frame.py --out runs/records/board-frame.png
   （自动找"一梦江湖"窗口抓帧；务必确认抓到的画面是棋盘，可打开 PNG 检查）

### 2. 网页标定
    .venv/bin/python calibrate_server.py --frame runs/records/board-frame.png
   浏览器打开 http://127.0.0.1:8123/
   ① 棋盘框模式：在图上拖拽画出 6×7 棋盘矩形（贴合棋盘四边）
   ② 预告框模式：若有"下次掉落预告"UI，画出其区域（6 格横条）；没有可跳过
   ③ 采样模式：点击棋盘里的牌，输入等级 1-8（0=背景）；每级采 2-3 个，背景 1-2 个
      （也支持点击格子后用键盘 0-8 直接定级）
   ④ 预览识别：检查每格识别数字是否正确，错格重新采样
   ⑤ 保存 → 生成 configs/capture.json

### 3. 开录（在棋盘界面开录，玩 30-60 分钟）
    .venv/bin/python capture_drops.py --out runs/real_drops.jsonl
   - 预告变化 / 掉落应用 / 新对局 自动记录；Ctrl+C 停止
   - 掉落值反推失败时该列记 null（unknown 字段标注），事后可重分析
   - 每 2s 存一次完整棋盘快照（state 事件）→ 兼做实机轨迹数据

### 4. 数据回传 → 我做分布拟合
   分析 runs/real_drops.jsonl：边缘分布 vs 假设分布、简单形/复杂形 regime 检测（HMM）

## 注意

- 抓帧/采集期间游戏窗口大小不要变（位置可变，动态定位）；窗口最小化时抓不到帧会重试
- 识别失败的格子（unknown）可能因颜色未采样：回到标定器补采样该色，重新保存即可
- 实机是否有"预告 UI"未知：有则 preview 事件是掉落值的主数据源（预告=实际），
  无则依赖 drop 帧反推
- 采集间隔默认 0.3s；若玩家操作极快导致帧间夹多个动作（反推失败率高），调小到 0.15-0.2s
