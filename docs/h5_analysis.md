
## 本地 game.py 与 H5 官方规则对齐（2026-08-15）

基于 H5 源码分析（docs/h5_analysis.md），本地 game.py 已对齐官方规则：

| 维度 | 对齐后 |
|---|---|
| 初始布局 | 每列 [1,1]（H5 initRabbits），h5_style=False 可回退旧版 |
| 开局预告 | reset 即采样第一轮隐藏预告；第 2、6、10…步后才公开 |
| 掉落节奏 | moves%4==3（第 3、7、11... 步）先刷新下一轮隐藏预告、再应用已知预告 |
| 掉落分布 | 默认 make_real_sampler()（REAL_TABLE 8 张表按合成等级） |
| 死亡后 | 不再应用/刷新预告（H5 gameOver 即终止） |
| 得分语义 | `score` 对齐官方（2~8 级合成加 `2^level`）；`n9_count` 对齐 `compositeMaxLevelNum` |

同步适配：
- replay_trace.py H5Game：死亡后不掉落；9 不计入显示分
- agents/puct.py _unsafe_preview_actions：检查窗口 moves%4==2（原第 4 步语义）
- agents/heuristic.py：预告应用步改为 moves%4==3
- train.py / eval.py：新增 --dist real（真实分布）
- test_game.py：验证“预抽但延迟公开”及预抽等级锁定

验证：
- test_game.py 全部通过
- 30 局随机对拍（game.py vs H5Game，同动作同掉落序列）：状态/死亡/合成等级逐局一致

注意：cycle_search 依赖旧时序（moves%4==3 采样）已失效，该方向早已证伪不推荐使用。
# H5 版源码分析报告（兔王争霸赛，一梦江湖 H5）

来源：https://jianghu.163.com/h5/20251212/hctey/
