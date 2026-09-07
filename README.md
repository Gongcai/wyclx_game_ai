# 基于真实掉落分布与搜索自博弈蒸馏的"兔儿爷"游戏智能体

> 用强化学习攻克网易《一梦江湖》内置合成小游戏**"兔儿爷"**（兔王争霸赛：
> 6 列 × 7 高栈、3 合 1 升级、9 消失得分、每 4 步一轮随机掉落）。
> 核心路线：**从游戏源码提取真实掉落分布 → 启发式教师冷启动 → PUCT 自博弈逐轮蒸馏
> → 部署级搜索配置 → 网页版全自动对战 → N-Tuple 价值函数 + Beam 搜索**。

## 摘要

"兔儿爷"是一个高随机性的合成类游戏：每轮 6 个掉落值独立采样、分布随合成等级
变化，玩家只能在部分相位看到下轮掉落预告。本项目经历 **DQN 假设分布对拍 →
搜索蒸馏 → 人类启发式 → 真实分布建模 → N-Tuple 价值 + Beam 搜索** 五个阶段。

当前最强是阶段五：`ntuple-ext-eps`（共享 N-Tuple 查表价值函数，5.6 MB、纯 CPU、
无神经网络、无树搜索）配 beam 搜索，在无步数上限的同种子配对评测（32 seed）中
打出**单局 72.1 个 9 / 平均存活 1026 步 / 单局最高 309 个 9**，对阶段四的
PUCT 部署模型（9.6 个 9 / 199 步）配对差 **+62.5 [+40.0, +88.0]**。阶段四模型
（`real-h5-puct-r2`，71 万参数等变 Policy/Value 网络 + 显式随机 PUCT）仍是已
落地的网页版实战方案，实测**单局 346 步 / 27,940 分 / 21 个 9**，本地基准 30 局
存活期望 174 ± 105 步、吞吐 44 个 9/千步，网页与本地表现经统计检验不可区分。

## 1. 问题与挑战

- **随机性主导**：预知未来掉落（诊断实验）可让每千步 9 数 +20%、存活 4~5 倍
  ——不确定性是最大瓶颈，也决定了必须建模分布而非背板。
- **部分可观测**：掉落预告只在每轮第 2 步后可见，其余相位只能靠分布推演。
- **长局困境**：死亡源于"7 高列 + 0 空列"死亡螺旋，在死亡前 10~100 步就已
  形成；价值头无法从短局数据预测长局后果（鸡生蛋问题）。

## 2. 方法演进（训练历史）

### 阶段一：DQN 假设分布对拍（已到平台）

在 uniform / high / low / mid 四个假设掉落分布下各训一个 DQN（model
selection 思路，实机对拍择优）。结论：单局存活天花板 ~170 步 / 6.5 个 9，
对搜索侧/价值侧的所有干预（死亡惩罚扫描、死亡头校准、人类规则塑形、安全
否决、长 horizon 价值）全部鲁棒——**结构性瓶颈，调参无用**。容量测试（h128
vs h256 从零重训）同样证伪。

### 阶段二：搜索 + 蒸馏管线（high 假设分布线）

beam search 示范 → 等变 Policy/Value 蒸馏 → **显式随机 afterstate PUCT**
（chance node + progressive widening）自博弈 → Gumbel 改进策略蒸馏
（`gumbel48`，128 配对 seed +0.81 显著超越教师）。关键增益项：**搜索树跨步
复用**（64 配对 seed +0.98）与根 sequential halving。同时确立方法论：
同强度自蒸馏一轮即平台化、离线指标与闭环表现无关、小样本配对结论必须扩到
≥64 seed。

### 阶段三：人类启发式（强独立策略，但当不了教师）

可解释评估函数 + 浅层 beam 的启发式以 35.8 个 9/千步超过当时的训练模型
（决策还快 30%）。但三次实验证明**启发式先验无法通过蒸馏转移给神经网络**
（policy-only 蒸馏 −1.53、从零重训 −2.34、行为克隆全面弱于搜索学到的先验）。
预知模式诊断（foresee-d12：43 个 9/千步）量化了随机性的上限收益，指明方向。

### 阶段四：真实掉落分布线（当前主线）

从游戏 H5 版源码直接读出**真实掉落分布表**：按合成等级（compositeLevel）
分 8 档、每次采样独立、预告即实际掉落值。与本地模拟器逐转移对齐后重建训练线：

```text
启发式教师示范（128 局 / 22,565 步 / 887 个 9）
  → real-h5-pv（纯教师蒸馏，value MAE 1.30）
  → PUCT(s64,d16) 自示范 16 局 → r1（局均 10.69 个 9）
  → r1 自示范 48 局（511 个 9，单局最高 36）→ r2 ★推荐模型
```

关键配方：PUCT 示范权重全量（2.0，早期实验压到 0.25 反而锁死在教师水平）。
r2 的离线 MAE（1.41）比被淘汰的实验更差但闭环更强。r3 及混合/价值头变体
经 5 组配对评测全部证伪——平台化在真实分布线复现，下一级增益需要更强教师
（sims 128+ 或推演辅助）或更大网络。

### 阶段五：N-Tuple 价值 + Beam 搜索（当前最强）

共享 N-Tuple 查表价值函数（`agents/ntuple.py`）+ afterstate 稀疏 TD(λ)，部署时
配 beam 搜索（`agents/ntuple_search.py`，每 ply 共享一次 chance 采样，避免"幸运
预告线"跨条目复利）。三个决定性发现：

1. **Beam 质量由 V 强度门控，而非搜索深度**。V 弱时（1-ply 20.68）等算力下与
   PUCT 打平（+0.69、+0.97，CI 跨零）；把 V 续训到 1-ply 42+ 后，质量对搜索
   深度 6~20 完全不敏感，可用最浅最快的 d6w12（131ms/步）支配 PUCT。
2. **持续探索（ε 下限 0.10）是唯一有效杠杆**。同一 V、同一 λ，只把 ε 从 0.01
   抬到 0.10，1-ply 每千步 3M 步内从 19.6 涨到 39.4 并继续爬到 42+；ε 收敛到
   0.01 的对照线同期只涨到 29.6 且后段趋平。λ 从 0.6 提到 0.8 确认有害
   （1.8M 步死平在 22.7）。这与阶段二"喂自身策略窄分布会退化"是同一失败模式。
3. **`best.pt` 是按单次 eval 最高分挑的，有系统性虚高**。训练期 eval 噪声约
   ±4.5%，在 150 次 eval 上取 argmax 必然挑到 +2.5σ 的点（42.35 的"纪录"真实
   水平只有 39.35）。无上限配对评测证实 `final.pt` 与 `best.pt` 等价
   （+4.44，CI 跨零），**部署应取 `final.pt`**。

无步数上限的同种子配对评测（32 seed，dist=real，搜索统一 d8w12rw24）：

| 模型 | 每局 9 | 平均存活步数 | 每千步 9 | 单局最高 |
|---|---|---|---|---|
| **`eps-final`（8M）** | **72.12** | **1025.8** | 70.31 | 286 |
| `eps-best`（7.74M） | 67.69 | 968.9 | 69.86 | 309 |
| `l2-ext`（上一代 V） | 18.38 | 310.2 | 59.23 | 70 |
| `PUCT r2`（阶段四部署） | 9.62 | 198.6 | 48.46 | 21 |

配对差（bootstrap 95% CI）：eps-final 对 l2-ext **+53.75 [+29.66, +81.12]**、
对 PUCT r2 **+62.50 [+39.97, +88.00]**，均显著；eps-final 与 eps-best
**+4.44 [-25.75, +35.09]** 不显著（两者等价）。

## 3. 部署配置与成果

### 当前推荐：N-Tuple V + Beam（阶段五）

模型 `runs/ntuple-ext-eps/final.pt`（5.6 MB 查表），搜索 `d6w12rw24`
（depth 6 / width 12 / root width 24），**纯 CPU、131ms/步、延迟平坦**。
尚未接入网页版与实机链路，本地配对评测相对阶段四为 +62.5 个 9/局。

### 已落地：PUCT r2（阶段四）

推荐搜索参数：`simulations=64 depth=16 c=1.5 death_penalty=0.5
chance_samples=8 chance_widening=0.5 root_min_visits=2
root_sequential_halving=8 root_q_scale=2.0 tree_reuse=true safe_veto=true`

| 指标 | 数值 |
|---|---|
| 本地基准（30 局，期望 ± 标准差） | 存活 174 ± 105 步；得分 11,375 ± 9,728；7.7 ± 8.3 个 9 |
| 吞吐 | 44 个 9 / 千步；单步决策 ~70ms |
| 网页实战单局最高 | **346 步 / 27,940 分 / 21 个 9** |
| 一致性校验 | 轨迹 0 非法、逐盘面与真实引擎 100% 一致（确定性重放） |

两套方案的选择依据是**基础设施**而非"算法更强"：CPU-only / 无 GPU / 要延迟
稳定选 N-Tuple + beam；已有 GPU 神经管线且要求 <100ms 决策选 PUCT sims64。
注意 PUCT 长局会因树复用把单步延迟推到 758ms（超出 500ms 容忍），N-Tuple 不会。

自动化闭环：`auto_play_h5.py` 用 Playwright 驱动浏览器直读游戏内部状态、
模拟点击全自动对战；`game_advisor.py` 提供实机 F8 截图建议（人工辅助模式）。
所有对局（人类/AI）以统一轨迹格式落盘，`replay_trace.py` 确定性重放校验。

## 4. 关键教训

1. **离线指标 ≠ 闭环表现**：value MAE 更好的模型闭环更差（两次验证）。
2. **同强度自蒸馏一轮即平台**：增益要么来自更强目标（更深搜索），要么来自
   结构改变（更大网络/真实分布）。
3. **启发式是强独立策略但不是教师**：行为转移三连败。
4. **局终判定要用引擎信号**：合成动画进行中引擎数据会瞬时出现"列高 8"，
   误当死亡信号会把好局处决（本项目网页版存活虚低一倍的真凶）。
5. **评测纪律**：配对同种子 + ≥64 局 + bootstrap CI，单批最高值不作数。
6. **评测的步数上限会系统性压缩强模型的差距**：300 步上限下新模型只测出
   16.7 个 9/局，放开上限后是 72.1——差 4 倍，且撞线率 76% 意味着大部分局
   被拦腰截断。所有带步数上限的"显著胜利"都只是真实差距的下限。
7. **每局 9 数的方差 ≈ 均值（变异系数 100%）**，且几乎全部来自存活时长
   （corr(9 数, 步数) ≈ 1.000，局长是重尾分布）。单局没有信息量；模型间的
   细微差异（如 `best.pt` vs `final.pt`）需要约 1560 个配对 seed 才能检出，
   应直接判定为等价。低方差的"每千步"口径更适合模型筛选，"每局 9 数"
   用于衡量部署能力。

## 5. 目录

| 路径 | 说明 |
|---|---|
| `game.py` | 规则模拟器（与 H5 源码逐转移对齐） |
| `agents/` | DQN / 等变 Policy-Value / PUCT（chance node、树复用、halving）/ 启发式 / **N-Tuple 查表价值 + beam 搜索** |
| `train.py` · `train_policy_value.py` | DQN 训练 / 搜索示范蒸馏 |
| `train_ntuple.py` · `eval_ntuple.py` | N-Tuple afterstate TD(λ) 训练 / 评测（CPU 可跑） |
| `generate_puct_demos.py` | PUCT 自举示范生成 |
| `eval_paired_puct.py` · `eval_paired_ntuple.py` | 同种子配对评测（模型筛选标准），后者对比 N-Tuple 与 PUCT |
| `plot_ntuple_progress.py` | N-Tuple 训练/评测数据可视化（输出到 `runs/charts/`） |
| `auto_play_h5.py` | 网页版全自动对战 |
| `game_advisor.py` | 实机截图操作建议 |
| `replay_trace.py` | 轨迹确定性重放校验 |
| `docs/progress.md` | 完整实验记录与推荐配置 |
| `PLAN.md` · `AGENTS.md` | 设计文档 / 环境说明 |

## 6. 快速开始

**模型下载**：推荐模型 `real-h5-puct-r2-pv.pt`（2.8 MB，自包含检查点）在
[Release v1.0.0](https://github.com/Gongcai/wyclx_game_ai/releases/tag/v1.0.0)，
放到 `runs/puct-v3/` 即可（或用 `--model` 指定路径）。

```bash
python test_game.py                        # 规则自测（输出 all tests passed）
python train.py --dist uniform             # DQN 基线
python generate_puct_demos.py --model ...  # 自举示范
python train_policy_value.py --demos ...   # 蒸馏
python eval_paired_puct.py --models A=... B=...  # 配对评测

# N-Tuple Network + afterstate TD(lambda)（当前最强，CPU 可跑）
# 关键配方：--eps-end 0.10（保持持续探索；用默认 0.01 会显著变慢并提前趋平）
python train_ntuple.py --dist real --steps 8000000 --gamma 0.99 --trace-lambda 0.6 \
  --alpha 0.05 --eps-end 0.10 --merge-w 1.0 --n9-w 30 --death-w 15 --run-id ntuple-ext-eps
python eval_ntuple.py --model runs/ntuple-ext-eps/final.pt --dist real --n 1000

# N-Tuple + beam 与 PUCT 的同种子配对评测（无步数上限，勿加 --max-moves 300）
python eval_paired_ntuple.py \
  --ntuple-models eps-final=runs/ntuple-ext-eps/final.pt l2ext=runs/ntuple-l2-ext/best.pt \
  --ntuple-overrides eps-final:8:12 l2ext:8:12 --search-root-width 24 \
  --puct-model runs/puct-v3/real-h5-puct-r2-pv.pt \
  --episodes 32 --seed 895000 --max-moves 10000 --simulations 128 \
  --json-out runs/eval-paired-ntuple-epsvsl2ext-nocap.json
```

环境：Python 3.12 + torch（cu130），依赖位于 `.venv/`，详见 `AGENTS.md`。

### 训练环境说明

本项目在 **Arch Linux** 上完成训练与全部实验（Hyprland/Wayland 桌面，
NVIDIA GPU + CUDA）。其中**实机画面采集与指导链路**（`game_advisor.py`
等）依赖 Linux 桌面特性：游戏本体通过 **Steam Proton** 运行 Windows 客户端，
抓屏走 Hyprland + `grim`，全局热键由 Hyprland 转发——这部分在
Windows / macOS 上**不一定能复原**。训练、模拟器评测与网页版自动对战
（`auto_play_h5.py`，仅需 Playwright）理论上跨平台，但未在其他系统上验证过。

## 版权说明

本仓库**不包含**游戏的任何官方素材、源码或资源文件（版权归原版权方所有）。
本地研究版游戏目录（`runs/h5game/`，已 gitignore）需自行获取，仅供个人研究
使用，不得商业用途或公开传播。

## License

MIT（见 [LICENSE](LICENSE)）
