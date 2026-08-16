# wyclx_game_ai

用强化学习玩"合成小游戏"（兔王争霸赛：6 列 × 7 高栈、3 合 1 升级、9 消失得分）的研究项目。

当前主线：**真实掉落分布 + 搜索自博弈蒸馏**。从游戏 H5 版源码提取真实掉落表
（按合成等级分 8 档），在此分布上用启发式教师 → PUCT 自博弈逐轮蒸馏出
Policy/Value 网络，配部署级搜索（afterstate chance node + 树复用 +
sequential halving 根选择）。推荐模型 `real-h5-puct-r2`。

## 实战表现

- 网页版自动对战（Playwright 驱动）：单局最高 346 步 / 27,940 分 / 21 个 9
- 本地基准（同配置 30 局）：存活期望 174 步，吞吐 44 个 9 / 千步
- 全部轨迹经确定性重放校验（0 非法、逐盘面一致）

## 目录

| 路径 | 说明 |
|---|---|
| `game.py` | 游戏规则模拟器（与 H5 源码逐转移对齐） |
| `agents/` | DQN、Policy/Value、PUCT 搜索、启发式策略 |
| `train_policy_value.py` | 等变 Policy+Value 蒸馏 |
| `generate_puct_demos.py` | PUCT 自举示范生成 |
| `eval_paired_puct.py` | 同种子配对评测（模型筛选标准） |
| `auto_play_h5.py` | AI 自动玩网页版（浏览器驱动） |
| `game_advisor.py` | 实机截图操作建议（人工辅助模式） |
| `replay_trace.py` | 轨迹确定性重放校验器 |
| `docs/progress.md` | 实验记录与推荐配置 |
| `PLAN.md` | 设计文档（中文） |

## 快速开始

```bash
python test_game.py                       # 规则自测
python train.py --dist uniform            # 训练 DQN 基线
python eval_paired_puct.py --models ...   # 配对评测
```

环境：Python 3.12 + torch，见 `AGENTS.md`。

## 版权说明

本仓库**不包含**游戏的任何官方素材、源码或资源文件（版权归原版权方所有）。
`serve_game.py` / `auto_play_h5.py` 面向的本地研究版游戏目录（`runs/h5game/`，
已 gitignore）需自行获取，仅供个人研究使用，不得商业用途或公开传播。

## License

MIT（见 [LICENSE](LICENSE)）
