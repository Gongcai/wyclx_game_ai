# 8 卡平台：长局 bootstrap 部署

目标：打破 hidden=128 训练平台期，用更大网络 + 长局自我对弈迭代出更强的 Policy+Value。
本地搜索侧实验已到平台（见 docs/progress.md），这是结构性杠杆。

## 0. 平台准备（海光 DCU）

- 海光 DCU 需要 DTK/ROCm 版 torch（`torch_dcu`）。安装后 `torch.cuda.is_available()`
  应返回 True（DCU 映射到 cuda 命名空间）。系统 CUDA torch 不行。
- 迁移验证：`python deploy/smoke_dcu.py --model runs/demos/high-halving8-gumbel48-pv.pt`
  → 应输出 PASS（设备数 ≥8、test_game.py 通过、模型前向 OK）。
- 代码侧已就绪：所有脚本支持 `--device cuda:N`；checkpoint 加载自动推断 hidden
  （`hidden_from_checkpoint`），hidden 128/256/512 模型通用。

## 1. 多卡长局自我对弈（数据生成）

```bash
python launch_selfplay.py \
    --model runs/demos/high-halving8-gumbel48-pv.pt \
    --gpus 8 --episodes-per-gpu 50 --max-moves 2000 \
    --out runs/demos/long-bootstrap-r1.pt
```

- 每个 GPU 一个 worker（`--device cuda:i`、seed 错开 10 万），完成后合并。
- `--max-moves` 逐轮拉长：r1=2000，之后 4000/8000…（当前模型存活 ~174 步，
  长局数据从存活尾部分布获得；模型变强后局才真正变长——这就是 bootstrap 循环）。
- 耗时预估：每局 ~2000 步 × 0.14s ≈ 5 分钟；每 GPU 50 局 ≈ 4 小时；8 卡并行
  = 400 局/轮。

## 2. 更大网络训练

```bash
python train_policy_value.py \
    --demos runs/demos/long-bootstrap-r1.pt runs/demos/high-long-64.pt \
           runs/demos/high-long-64-s1000.pt runs/demos/high-halving8-selfplay48.pt \
    --hidden 256 \
    --mature-policy-weight 2.0 --high-tile-policy-weight 1.0 \
    --elite-score-threshold 10 --elite-policy-weight 2.0 \
    --death-horizon 32 --death-w 0.5 \
    --out runs/demos/high-h256-bootstrap-r1-pv.pt
```

- 从零训练（无 `--init-model`），编码器/策略/价值全学。旧血缘的 policy_head_only
  微调已证实平台化，不再使用。
- 数据混合：新长局数据 + 旧的长局示范（保证覆盖面）。
- 训练快：hidden 256 在 64GB 卡上 batch 256 × 100 epochs 分钟级。

## 3. 评测与选型

```bash
python eval_paired_puct.py \
    --models gumbel48=runs/demos/high-halving8-gumbel48-pv.pt \
             h256=runs/demos/high-h256-bootstrap-r1-pv.pt \
    --tree-reuse-models gumbel48 h256 \
    --episodes 128 --seed 700000 --max-moves 500 \
    --simulations 128 --depth 16 --chance-samples 8 \
    --chance-widening 0.5 --root-min-visits 2 --death-penalty 0.5 \
    --json-out runs/eval-paired-h256-r1.json --max-new 64
```

- 配对差 95% CI 为正（≥64 seed）才替换推荐模型。
- 若 h256 通过：用 h256 做下一轮 self-play（回到步骤 1），`--max-moves` 拉长，
  `--hidden` 可再加到 512。

## 4. 循环节奏

```
self-play(8卡, 长局) → 训练(hidden 256, 从零) → 配对评测 → 通过则替换并拉长
```

- 每轮约 1 天（4h self-play + 训练 + 评测）。
- 停止条件：连续两轮配对差 CI 跨零（平台化）。

## 注意事项

- 训练/评测脚本的 `--device` 默认 `cuda:0`；多卡只用于 self-play 并行，
  训练单卡 64GB 足够（hidden 512 也远小于 64GB）。
- 海光 DCU 的 `torch.cuda` 命名空间与 CUDA 一致，代码无需改。
- 产物全部在 `runs/`（已 gitignore）。
