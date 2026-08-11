# 合成小游戏 AI：训练与实机部署规划

## 1. 背景与目标

一个 MMO 内的小游戏：6 列栈，每列高 7，栈顶进出。玩家把一列栈顶的整段相同元素移到另一列栈顶；任一列栈顶 ≥3 个相同元素即全部合成一个 n+1（9 消失并得分）；每 4 步所有列各掉落一个随机元素（第 3 步后预告）。任一堆超高 7 即失败。

**目标**：比赛限时 30 分钟，死亡可立即重开且不限制局数；按 30 分钟内累计合成 9 的数量排名。训练和部署最终优化单位墙钟时间的合成吞吐，而不是单局存活或首次得分。已知人类最高约 150~160，第一名约 750。

**核心难点**：真实游戏的掉落分布未知（实测后期小值极少，无法大样本验证）→ 采用 **model selection**：在不同假设分布下训练多个智能体，实机对拍择优。

**速度约束**：动画不可跳过，AI 实机上限约 2~3 step/s（人类鼠标操作约 1 step/s），因此 30 分钟估算默认按 0.4 秒/步，几十毫秒级搜索推理可以接受。

## 2. 已确认规则（控制台 MVP 已实现并验证）

| 规则 | 实现 |
|---|---|
| 栈模型 | 6 列 × 高 7，栈底在下，栈顶进出；整段相同元素一起出栈 |
| 合并 | 栈顶连续 ≥3 个相同值全部合成 n+1；9 得分(+9)后消失；合成后再查（链式） |
| 掉落 | 每 4 步、6 列各掉一个，值 ∈ [1, min(7, 已合成最大值)]；第 3 步后六列各自预告，预告即实际掉落值 |
| 死亡 | 任意列高度 > 7（合并先于死亡判定）；允许自杀式操作 |
| 分布 | 可配置权重，预设 uniform / low / high / mid |

## 3. 阶段一：训练

### 3.1 环境接口（`game.py`，已就绪）

- `Game.reset() / Game.move(src, dst) / Game.legal_moves() / Game.observe()`
- **栈表示：stacks[列][0] = 栈顶**，控制台列号显示 1~6（输入时内部转 0 基）
- 动作空间 6×5=30（含自杀）；非法动作仅"源列空/同列"
- 环境带事件钩子 `Game.events`：每次动作产生的合并值（含掉落合并），供奖励计算
- 掉落分布由 `drop_sampler(rng, max_merged)` 注入（`agents/dist.py`，支持 cap 与任意权重）
- 状态编码（`agents/dqn.py`）：6×7 网格 one-hot（值 1~8 → 类别 0~7，类别 8 预留防越界）+ 元特征（周期相位 4、六列预告/7、预告存在标志、掉落上限 min(7,max_merged)/7）

### 3.2 奖励（已确认方案 B，按事件实现）

- 合成 9：游戏得分 `+9`；训练奖励 `+10`
- 中间合成：`--merge-w` × 新值（默认 `1.0`）
- 每步：−0.05
- 死亡：−5（引导 AI 优先存活）

### 3.3 算法

- **首选 DQN**（`agents/dqn.py`）：Double DQN + 目标网络 + 经验回放（+ 优先回放，如训练不稳定再加）
  - 网络：MLP(390 → 256 → 256 → 30)，输入 one-hot 网格 + 元特征，输出 30 维 Q
  - ε-greedy：1.0 → 0.05（默认 20 万步线性衰减）
  - 全动作可执行（自杀合法），用 legal_mask 屏蔽"源空/同列"动作
- **基线对照**（`agents/baselines.py`）：随机合法动作、贪心（模拟每个合法动作取即时收益+栈高惩罚）
- **搜索基线**：`eval.py` 已提供基于真实 `Game` 深拷贝的 beam search（`--search-depth`、`--beam-width`）；已公开掉落预告按真实值规划，未知未来掉落使用仅由可观察盘面构造的确定性模拟种子，禁止读取真实 RNG 隐藏状态。它计算较慢，但可用于验证可达性和生成示范轨迹。
- **随机 MCTS**：`agents/mcts.py` 在未知掉落处用独立随机样本构造机会结果，已公开 preview 保持不变；`eval_competition.py --policy mcts` 可与 beam 比较首个 9 成本、后续 9 间隔和每千步吞吐。
- **示范生成**：`generate_demos.py --dist uniform --episodes 100 --workers 4` 并行运行 beam search，仅保存达到目标分数的完整 transition（状态、动作、奖励、下一状态、mask、终止标志）至 `runs/demos/`。
- **长局示范**：增加 `--continue-after-score --max-moves 500` 后，首个 9 不再终止轨迹，并逐步保存 `n9`/`n9_steps`，用于学习成熟盘面的后续合成成本。
- **Policy+Value**：`train_policy_value.py` 使用等变列编码，联合预测 beam/PUCT 动作、未来窗口内折扣 9 数量、距下一个 9 的步数和短期死亡风险，作为后续 PUCT 的先验与叶节点评估。
  - 可一次传入多个 `--demos`；首个 9 后的成熟状态默认使用更高 policy 权重，重点优化后续 9 的边际成本
  - `--high-tile-policy-weight` 对已有 7/8 的关键盘面额外加权；`--death-horizon` 和 `--death-w` 控制死亡风险目标及损失权重
  - 单局极值候选使用覆盖整局的长 horizon，并以 `--elite-score-threshold` / `--elite-policy-weight` 提高高分长局的策略权重；该目标与比赛吞吐模型分开筛选
- **PUCT**：`agents/puct.py` 在每条模拟中独立采样未知掉落，用 Policy head 提供合法动作先验、Value head 评估叶节点；`--puct-death-penalty` 从叶节点价值扣除预测死亡风险，表示丢失成熟棋盘并重新经历冷启动的机会成本。通过 `eval_competition.py --policy puct --pv-model ...` 测试比赛吞吐。
- **Stochastic MuZero afterstate 实验**：`Game.move_afterstate()` 将确定性玩家动作与随机周期事件分离；PUCT 可用 `--puct-chance-samples N` 建立显式 chance node，并以 `--puct-chance-widening 0.5` 渐进扩展随机结果。默认 0 保持原 root-sampling。第一阶段只替换搜索树，在 16 局留出集上尚未超过旧搜索；后续需联合训练 afterstate Q/chance-aware policy，而不是直接替换部署配置。
- **PUCT 自举**：`generate_puct_demos.py` 保存根节点 30 动作访问分布和实际长局 `n9` 回报；`train_policy_value.py` 可混合 beam 硬标签与 PUCT 软标签，并在列置换增强时同步重排完整动作分布。
- **行为克隆**：`train_bc.py --demos runs/demos/uniform-beam.pt` 按局划分训练/验证示范，保存的权重可直接传给 `eval.py --model`；由于闭环分布偏移，当前仅作为诊断，不作为 DQN 初始化。
- **DQfD 式训练**：新训练可加 `--prefill-demos runs/demos/uniform-beam.pt`；示范保存在独立 expert 池，每个 batch 固定按 `--expert-ratio` 抽样，并叠加 `--expert-bc-w` 动作监督损失，避免被在线失败 replay 稀释。
- **DAgger**：`train_dagger.py` 让当前策略进入自己的状态分布，并逐步降低教师执行概率；每个访问状态都由无 oracle beam 标注后加入聚合数据，用于修复纯行为克隆的闭环分布偏移。
- **升级路线**：beam search 示范预填充 → DQN 微调；DQN 平台期后再考虑 MCTS（+价值网络，AlphaZero 式单人）

### 3.4 训练工程

- `train.py`：多环境并行（--n-envs 8），每步训练，`runs/<run_id>/` 存 best.pt / final.pt / metrics.jsonl
- `eval.py`：固定种子 N 局，对比 agent / beam / greedy / random，输出平均得分、存活步数、合成 9 数
- `eval_competition.py`：跨死亡自动重开，在固定动作预算下统计每千步合成 9，并结合实机单步耗时和本地决策延迟估算 30 分钟成绩
  - 分开统计每局首个 9 所需步数和同局后续 9 的间隔，用于判断继续经营成熟盘面与主动死亡重开的盈亏平衡点
- 死亡局编码保护：栈可临时高 8，编码槽位按 7 截断

### 3.5 多分布 model selection

- `configs/drop_dists.json`：预置 uniform / high / low / mid + 自定义权重
- 每个分布训一个 agent，同超参同种子对比
- **筛选**：实机（阶段三）对拍，选真实得分最高的模型

## 4. 阶段二：桥接层（画面 → 状态 → 动作）

目标：把真实游戏画面转成 `Game` 同构的状态序列，并把 AI 动作转成屏幕点击。

```
屏幕 → [截图] → [锚定校正] → [数字识别] → 状态序列 → AI 决策
                                                        ↓
                    游戏窗口 ← [鼠标点击] ← 动作 → 点击坐标换算
```

### 4.1 锚定截图区域（calibration.json）

- 一次校准生成配置文件：`{anchor: {x, y}, grid_origin: {x, y}, cell_w, cell_h, preview_origin, score_origin, ...}`
- 校准工具 `bridge/calibrate.py`：截图预览 + 人工点击"游戏区域左上角 / 网格角点"等参考点，自动计算格子尺寸
- 运行时按锚点 + 区域尺寸裁剪（支持窗口位移后重新校准）

### 4.2 锚定识别点（自动校正）

- 画面清晰 → 用固定锚点特征自动校验：识别网格边框 / 列分隔线 / 列底基准线
- 若锚点匹配失败（窗口移动、缩放、遮挡）→ 重新对齐或报警人工介入

### 4.3 数字识别（状态序列）

- 数字 1~8 清晰：优先**模板匹配**（截取每个格子，与数字模板比对，简单稳定）
- 备选：pytesseract OCR / 像素连通域特征
- 输出：6 列 × 7 行数值 + 预告行 + 分数 → 与 `Game.stacks` 完全同构
- 每回合**闭环校验**：识别结果必须自洽（列高、值的合法性、与上回合变化的因果一致性），不一致则重截重识别，仍失败则暂停报警

### 4.4 动作执行

- 移动 = 点击源列栈顶坐标，再点击目标列栈顶坐标（坐标 = grid_origin + 列偏移）
- 点击后用状态校验是否生效，确认后再进入下一步（防丢帧）

### 4.5 与训练联动的注意点

- 实机每步的实际掉落分布可被记录（日志），**用于反推真实分布**，反哺 model selection
- 桥接层只负责读写状态，不感知 AI 内部结构

## 5. 阶段三：实机评测与迭代

1. 校准 + 桥接层冒烟测试（人工对拍画面与状态序列）
2. 部署候选 agent 各打 N 局，记录得分分布、存活步数、掉线率
3. 择优 + 用实机掉落日志修正分布假设 → 重训/微调（可循环）
4. （可选）MCTS / beam search 升级

## 6. 里程碑

| # | 内容 | 产出 |
|---|---|---|
| M1 | 训练管线 | DQN + 基线 + 评测脚本，跑通训练曲线 |
| M2 | 多分布训练 | 各分布 agent + 离线对比报告 |
| M3 | 桥接层 | capture / calibrate / recognize / act，状态闭环校验 |
| M4 | 实机评测 | 对拍报告 + 实机掉落日志 + model selection 结论 |

## 7. 目录结构

```
.
├── game.py           # 环境核心（唯一游戏实现，含测试 test_game.py）
├── agents/           # 算法与基线
│   ├── dqn.py        # DQN + 状态编码 + mask
│   ├── baselines.py  # random / greedy
│   └── dist.py       # 掉落分布采样器
├── train.py          # 训练入口
├── eval.py           # 离线评测
├── configs/
│   └── drop_dists.json
├── bridge/           # 桥接层（阶段二）
│   ├── calibrate.py  # 锚点校准工具
│   ├── capture.py    # 截图/锚定
│   ├── recognize.py  # 数字识别
│   └── act.py        # 点击执行
├── runs/             # 训练产物（gitignore）
└── PLAN.md
```
