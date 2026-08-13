# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

训练 AI 玩"合成小游戏"（6 列×7 高栈、3 合 1 升级、9 消失得分）的强化学习项目。目标是比赛 30 分钟限时内最大化合成 9 的吞吐（死亡可无限重开）。设计文档是 `PLAN.md`；`AGENTS.md` 的「游戏规则与代码陷阱」一节列出了历史 bug 和易错点，**修改 `game.py` / 编码 / 奖励逻辑前必读**。阶段二/三（`bridge/` 画面识别、实机部署）尚未实现。代码注释、日志、文档均为中文。

## 环境与命令

- Python 3.12 + torch（cu130）只在 `.venv/` 里：一律用 `.venv/bin/python` 运行，系统 python 没有 torch。无 requirements.txt / pyproject。
- `runs/` 已 gitignore，存放所有训练产物与示范数据。

```bash
.venv/bin/python test_game.py        # 测试（不是 pytest，纯脚本 run_all()，输出 "all tests passed"）
.venv/bin/python main.py             # 控制台人机对战（列号显示 1~6，内部 0 基）
```

### DQN 训练与评测

```bash
.venv/bin/python train.py --dist uniform|high|low|mid [--n-envs 8] [--steps 1000000] [--run-id <dist>-main]
.venv/bin/python train.py --dist <同分布> --resume <run_id> --steps <总步数>   # 断点续训，需同超参同 seed
.venv/bin/python eval.py --model runs/<id>/best.pt --dist uniform --n 1000     # 固定种子对比 agent/beam/greedy/random
```

产物写入 `runs/<run_id>/`：best.pt / final.pt / checkpoint.pt / metrics.jsonl（逐行 JSONL）。SIGINT/SIGTERM 时自动落盘 checkpoint。

### 搜索 + 蒸馏管线（当前主线）

```bash
# 1. beam search 生成示范（长局用 --continue-after-score --max-moves 500）
.venv/bin/python generate_demos.py --dist uniform --episodes 100 --workers 4

# 2. 训练 Policy+Value 网络（等变列编码，可混合多个 --demos，beam 硬标签 + PUCT 软标签）
.venv/bin/python train_policy_value.py --demos runs/demos/<name>.pt [--init-model ...] [--history-features]

# 3. PUCT 自举：保存根节点 30 动作访问分布与长局 n9 回报，回到第 2 步迭代
.venv/bin/python generate_puct_demos.py --model runs/pv/<ckpt>.pt --dist high [--save-q-targets]

# 4. 比赛吞吐评测：跨死亡重开，固定动作预算统计每千步 9 数并估算 30 分钟成绩
#    推荐配置（树复用 + 降深，见 docs/progress.md）：
#    --puct-tree-reuse --puct-depth 16 --puct-simulations 128 --puct-chance-samples 8
.venv/bin/python eval_competition.py --policy puct --pv-model runs/pv/<ckpt>.pt --dist high [--json-out ...]
#   --policy 可选 agent|beam|mcts|puct|greedy|random；agent 时加 --model 与 --arch mlp|equivariant

# 5. 多模型配对对比（同种子同局面，支持增量追加到 --json-out）
.venv/bin/python eval_paired_puct.py --models LABEL=ckpt.pt LABEL2=ckpt2.pt --json-out runs/paired.json
```

其他：`train_bc.py`（行为克隆，仅诊断用）、`train_dagger.py`（DAgger 修复闭环分布偏移）、`train_afterstate_q.py`（afterstate Q 头微调，需 `--init-model`）、`analyze_human_strategy.py`。

## 架构

数据流是一个自举循环：**beam search 示范 → Policy+Value 蒸馏 → PUCT 搜索 → PUCT 示范再蒸馏**，最终用 `eval_competition.py` 的吞吐指标选型。

- `game.py` — 唯一的环境实现。`Game.move(src, dst)` 之外还有 `Game.move_afterstate()`，把确定性玩家动作与随机掉落事件分离（供显式 chance node 用）。`Game.events` 记录每次 move 产生的合并事件（move 开始时清空，必须在 move 后立即读取），奖励计算依赖它。
- `agents/dist.py` + `configs/drop_dists.json` — 掉落分布采样器。**核心方法论是 model selection**：真实掉落分布未知，在 uniform/high/low/mid 多个假设分布下各训一套，同超参同种子对比，实机对拍择优；新增分布加进该 JSON。
- `agents/dqn.py` — DQN 智能体 + 状态编码 `encode()`（390 维：6×7×9 one-hot 网格 + 12 维元特征：周期相位/预告/掉落上限）。编码是 DQN 与 Policy+Value 共用的输入约定，改动需同步。
- `agents/search.py` — 基于真实 `Game` 深拷贝的 beam search（示范教师）；未知掉落用可观察盘面构造确定性模拟种子，禁止读真实 RNG 隐藏状态。
- `agents/policy_value.py` — 等变列编码网络，联合预测：动作分布（policy）、未来窗口折扣 9 数（value）、距下一个 9 的步数、短期死亡风险。是 PUCT 的先验与叶评估。
- `agents/puct.py` — PUCT 搜索。默认在每条模拟里独立采样未知掉落（root-sampling）；`chance_samples > 0` 时切换为显式 afterstate chance node（Stochastic MuZero 式，配 progressive widening）；另支持 Gumbel 根 sequential halving。`--puct-death-penalty` 从叶价值扣除死亡风险，代表丢失成熟盘面重经历冷启动的机会成本。
- `agents/mcts.py`、`agents/baselines.py`、`agents/human_strategy.py` — 随机 MCTS、random/greedy 基线、人类策略规则。
- 评测哲学：`eval_competition.py` 分开统计每局首个 9 的步数成本与同局后续 9 的间隔，用于判断"继续经营成熟盘面 vs 主动死亡重开"的盈亏平衡；`eval_paired_puct.py` 做同种子配对比较以降低方差。

## 最易踩的坑（完整列表见 AGENTS.md）

- `Game.stacks[c][0]` 是**栈顶**（不是 `[-1]`）。移动是源栈顶整段相同元素一起走；合并判定在入栈后、death 判定在合并之后。
- 掉落周期 4 步：`moves % 4 == 3` 采样进 `preview`，`% 4 == 0` **原样应用 preview**，不得重新采样。
- 采样器签名 `sampler(rng, max_merged=None)`，`Game._sample()` 必须把 `self.max_merged` 传进去，否则掉落 cap 失效。
- 死亡时栈可临时高 8，`encode()` 按 7 截断。
- 列置换数据增强时（`train_policy_value.py`），30 维动作分布必须随列同步重排。
