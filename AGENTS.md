# AGENTS.md

训练 AI 玩"合成小游戏"（6 列×7 高栈、3 合 1 升级、9 消失得分）的强化学习项目。设计文档是 `PLAN.md`（中文），实验记录与当前推荐模型/搜索参数在 `docs/progress.md`，阶段二/三（`bridge/` 画面识别、实机部署）**尚未实现**。

## 环境

- Python 3.12 + torch（cu130），只在 `.venv/` 里：用 `.venv/bin/python` 运行，系统 python 没有 torch。
- 无 requirements.txt / pyproject，依赖只在 venv 内。
- `runs/` 已 gitignore；代码注释、日志、文档均为中文。

## 命令

- `python main.py` — 控制台人机对战（列号 1~6，内部 0 基）
- `python test_game.py` — 测试（**不是 pytest**，纯脚本 `run_all()`，输出 "all tests passed"）
- `python train.py --dist uniform|high|low|mid [--n-envs 8] [--steps 1000000]` — 训练，产物写入 `runs/<run_id>/`（best.pt / final.pt / checkpoint.pt / metrics.jsonl，JSONL 逐行）。默认 run_id 是 `<dist>-<时间戳>`，实跑一般显式 `--run-id <dist>-main`
- 断点续训：`python train.py --dist <同分布> --resume <run_id> --steps <总步数>` — 从 `runs/<run_id>/checkpoint.pt` 恢复（权重/optimizer/replay/步数/ε/best 全量），**需与中断时同超参同 seed**；SIGINT/SIGTERM 时自动落盘再退出，每轮 eval 后也落盘
- `python eval.py --model runs/<id>/best.pt --dist uniform --n 1000` — 固定种子对比 agent / greedy / random

### 搜索 + 蒸馏管线（当前主线，详见 CLAUDE.md 与 docs/progress.md）

- `python generate_demos.py --dist high --episodes 100 --workers 4 [--continue-after-score --max-moves 500]` — beam search 示范
- `python train_policy_value.py --demos runs/demos/<name>.pt [--init-model ...]` — 等变 Policy+Value 蒸馏（beam 硬标签 + PUCT 软标签可混合）
- `python generate_puct_demos.py --model runs/demos/<pv>.pt --dist high` — PUCT 自举示范（根访问分布 + 长局 n9 回报）
- `python eval_competition.py --policy puct --pv-model runs/demos/<pv>.pt --dist high` — 跨死亡重开的比赛吞吐评测
- `python eval_paired_puct.py --models A=a.pt B=b.pt --json-out runs/xxx.json --max-new 16` — 同种子配对对比，逐局原子写入，可断点续跑；模型筛选看配对均值差 + bootstrap 95% CI，不看单批最高值

## 游戏规则与代码陷阱（易错）

- `Game.stacks[c][0]` 是**栈顶**（不是 `[-1]`）。
- 动作空间 6×5=30，自杀式移动合法；非法仅"源列空 / 同列"。**移动是源栈顶整段相同元素一起出栈/入栈**（如栈顶 `[1,1]` 两格一起走，不是单格），入栈后合并判定在目标列；death 判定在合并之后（可因自杀/掉落溢出死亡）。
- 掉落周期 4 步：`moves % 4 == 3` 时采样 6 个值固定进 `preview`（预告即实际掉落值），`moves % 4 == 0` 时**原样应用 preview**，应用时不要重新采样（曾实现成两次采样导致预告与实际不一致）；单次掉落可能触发合并（合并事件也进 `events`）。
- 掉落值规则：`1 ≤ v ≤ min(7, max_merged)`，可等于 7。采样器签名是 `sampler(rng, max_merged=None)`——**`Game._sample()` 必须把 `self.max_merged` 传进去，否则 cap 失效**（曾漏传导致掉落一直全范围 1~7，属历史 bug）。默认 sampler 需兼容两参。
- `Game.events` 每次 `move()` 开始前被清空：**必须在 move 后立即读取**。奖励计算在 `train.py` 的 `step_reward()`：每步 −0.05，合成 9 +10，中间合成 `--merge-w`（默认 1.0）×值，死亡 −5；另有 `--pair-bonus`（默认 0.3）/`--empty-bonus`（默认 0.2）势函数塑造：`r += γΦ(s') − Φ(s)`，Φ = 系数×(栈顶成对列数 `top_pair_count` + 空列数 `empty_col_count`)，policy-invariant，只转移信用不改变最优策略。
- 死亡时栈可临时高 8，`encode()` 按 7 截断（`agents/dqn.py`）。
- DQN 输入：6×7×9 one-hot（值 v 用类别 v-1，类别 8 预留防越界，共 378 维）+ 元特征 12 维（周期相位 4 + 六列预告/7 + 预告存在标志 + 掉落上限 min(7,max_merged)/7）= 390 维；`_sample` 相关 cap 逻辑改动注意与 `agents/dist.py` 一致（权重从 `configs/drop_dists.json` 读）。replay 存 CPU 张量，`learn()` 时搬到 device。

## 方法论

核心方法是 **model selection**：在 `configs/drop_dists.json` 预置的 uniform / high / low / mid 多个假设分布下各训一个 DQN，实机对拍择优。新增分布应加进该 JSON，并用同超参同种子训练对比。
