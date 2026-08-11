主要内容
1. 作者和团队信息
作者：Julian Schrittwieser, Ioannis Antonoglou, Thomas Hubert, Karen Simonyan, Laurent Sifre, Simon Schmitt, Arthur Guez, Edward Lockhart, Demis Hassabis, Thore Graepel, Timothy Lillicrap, David Silver
团队：DeepMind
主要贡献者：Demis Hassabis（DeepMind创始人，也是AlphaGo/AlphaZero项目负责人）, David Silver（AlphaGo/AlphaZero项目主要负责人）。这两位在强化学习和博弈AI领域享有盛誉，领导DeepMind团队取得了许多突破性成果。DeepMind 在强化学习领域处于领先地位，代表性成果包括 AlphaGo、AlphaZero、WaveNet 等。
2. 背景和动机
发表时间：2020年发表于Nature。
研究问题：如何在不了解环境动态的情况下，让智能体在复杂环境中进行有效规划？
问题背景：
传统的规划算法依赖于对环境动态的精确了解（例如，游戏规则或精确的模拟器）。
在现实世界的许多问题中（例如，机器人、工业控制），环境动态是复杂且未知的。
Model-based RL（基于模型的强化学习）旨在通过学习环境动态模型来解决这个问题，但之前的研究在视觉丰富的领域（如Atari游戏）中效果不佳。
Model-free RL（无模型的强化学习）在视觉丰富的领域表现出色，但在需要精确和复杂预测的领域（如围棋、象棋）中效果不佳。
3. 相关研究
Model-based RL：
传统方法：构建状态转移模型（预测下一个状态）和奖励模型（预测转移过程中的预期奖励）。
基于像素建模：直接对像素级别的观测流进行建模，但计算成本高，且容易受到建模误差的影响。
Value equivalent models（价值等价模型）：学习一个抽象的MDP模型，使得在该模型中进行规划等价于在真实环境中进行规划。Predictron、TreeQN、Value Iteration Networks、Value Prediction Networks等都属于这个范畴。
不足之处：
传统的Model-based RL方法需要手动设计状态表示，这限制了智能体的灵活性和泛化能力。
基于像素建模的方法计算成本高，难以扩展到大规模问题。
之前的Value equivalent models，例如Value Prediction Networks，没有策略预测，搜索仅利用价值预测。
4. 核心思路
MuZero的核心思想是：学习环境模型，只预测对规划最直接相关的方面，即奖励、策略和价值函数。

灵感来源：
AlphaZero在围棋、象棋等游戏中的成功。
Value equivalent models 的思想。
具体做法：
表示函数（representation function）：将观测（例如，图像）转换为隐藏状态。
动态函数（dynamics function）：以迭代的方式更新隐藏状态，输入前一个隐藏状态和假设的下一个动作，预测即时奖励和新的隐藏状态。
预测函数（prediction function）：根据隐藏状态预测策略（例如，要采取的行动）和价值函数（例如，预测的赢家）。
关键创新：
端到端训练：模型完全以端到端的方式进行训练，目标是准确估计奖励、策略和价值函数。
抽象状态表示：隐藏状态不需要捕获重建原始观测所需的所有信息，也不需要匹配环境的真实状态。相反，隐藏状态可以以任何与预测当前和未来价值和策略相关的方式来表示状态。
内部规则：智能体可以在内部创建规则或动态，从而实现最准确的规划。

5. 方案与技术
整体框架：MuZero 包含三个主要组件：表示函数、动态函数和预测函数。
模型：
表示函数
：将历史观测信息
编码成一个初始隐藏状态
。
动态函数
：给定前一个隐藏状态
和一个动作
，预测即时奖励
和新的隐藏状态
。
预测函数
：给定隐藏状态
，预测策略
和价值函数
。
搜索：使用 Monte-Carlo Tree Search (MCTS) 算法，该算法与 AlphaZero 的搜索算法类似，但经过泛化，允许单智能体域和中间奖励。MCTS 在每个内部节点使用当前模型参数
生成的策略、价值和奖励估计。
训练：
模型的所有参数都经过联合训练，以准确匹配每个假设步骤
的策略、价值和奖励，以及在
个实际时间步后观察到的相应目标值。
类似于 AlphaZero，改进的策略目标由 MCTS 搜索生成；第一个目标是最小化预测策略
和搜索策略
之间的误差。
改进的价值目标通过玩游戏或 MDP 生成。与 AlphaZero 不同，MuZero 允许长episode，通过从搜索值
启动
步到未来，使用折扣和中间奖励：
。
奖励目标就是观察到的奖励；因此，第三个目标是最小化预测奖励
和观察到的奖励
之间的误差。
最后，还添加了一个 L2 正则化项，从而得到总体损失：
，其中
,
和
分别是奖励、价值和策略的损失函数。
针对性：
抽象状态表示：通过学习抽象的状态表示，MuZero 可以忽略不相关的环境细节，从而提高规划效率。
MCTS：MCTS 算法可以有效地探索潜在的行动序列，并选择最有希望的行动。
端到端训练：通过端到端训练，MuZero 可以优化模型的各个组件，以实现最佳的规划性能。
6. 实验与结论
实验设计：
Atari：在 57 个 Atari 游戏中评估 MuZero 的性能。Atari 游戏是测试 AI 技术的经典环境，其中基于模型的规划方法历来表现不佳。
围棋、象棋和将棋：在围棋、象棋和将棋中评估 MuZero 的性能，无需任何游戏规则知识。
结果：
Atari：MuZero 在 57 个 Atari 游戏中实现了新的state-of-the-art，超过了之前最好的无模型方法 R2D2，并在所有游戏中超过了之前最好的基于模型的方法 SimPLe。
围棋、象棋和将棋：MuZero 在围棋、象棋和将棋中达到了超人的性能，与提供游戏规则的 AlphaZero 算法相匹配。
重要发现：
MuZero 可以在不了解环境动态的情况下，在复杂环境中进行有效规划。
MuZero 的抽象状态表示可以提高规划效率。
MuZero 的 MCTS 算法可以有效地探索潜在的行动序列。
MuZero 的端到端训练可以优化模型的各个组件，以实现最佳的规划性能。
实验如何支撑结论：
Atari 实验证明了 MuZero 在视觉复杂环境中的有效性。
围棋、象棋和将棋实验证明了 MuZero 在需要精确和复杂预测的环境中的有效性。
通过对比实验，作者证明了 MuZero 的各个组件（抽象状态表示、MCTS、端到端训练）对整体性能的贡献。
7. 贡献
MuZero 证明了基于学习模型的规划方法在复杂环境中的可行性。
MuZero 的抽象状态表示为未来的研究提供了一个新的方向。
MuZero 的成功经验可以应用于其他领域的规划问题，例如机器人、工业控制等。
8. 不足
模型准确性：虽然 MuZero 在许多环境中取得了令人印象深刻的结果，但在模型不准确的情况下，其性能可能会受到影响。尤其是在Atari游戏中，作者也提到，MuZero在Atari上的规划能力不如Go，这可能是因为Atari游戏的模型不准确性更高。
计算成本：MuZero 的训练过程需要大量的计算资源，这限制了其在资源受限环境中的应用。
可解释性：MuZero 的抽象状态表示缺乏可解释性，难以理解智能体的决策过程。
对于MuZero在Atari游戏中，即使只进行一次模拟，其性能也很好，这意味着策略网络已经学会了内化搜索的好处。 那么，是否可以进一步减少搜索的计算量，从而提高整体效率？
QA
Q1：MuZero 和 AlphaZero 的主要区别是什么？
MuZero 是对 AlphaZero 的扩展，主要区别在于 MuZero 不依赖于对环境动态的先验知识。具体来说：

AlphaZero 需要知道游戏规则来进行状态转移、判断合法动作和判断游戏结束。
MuZero 则通过学习一个内部模型来模拟环境动态，从而可以在不了解游戏规则的情况下进行规划。
Q2：MuZero 的“隐藏状态”有什么作用？它需要包含环境的全部信息吗？
MuZero 的隐藏状态是模型学习到的抽象表示，它不需要包含环境的全部信息。其主要作用是：

支持对未来奖励、策略和价值的预测。
允许智能体自主学习对规划最有用的状态表示，从而提高规划效率。
Q3：为什么 MuZero 要预测奖励、策略和价值函数这三个量？
这三个量是规划最直接相关的。通过准确预测这三个量，MuZero 可以在不了解环境动态的情况下进行有效规划。

奖励：直接反映了智能体与环境交互的结果。
策略：指导智能体选择下一步行动。
价值函数：评估当前状态的长期收益。
Q4：MuZero 在 Atari 游戏中取得了很好的效果，但作者也提到，在 Atari 上的规划能力不如围棋。这是为什么？
这可能是因为 Atari 游戏的模型不准确性更高。

围棋的规则相对简单，模型更容易学习到精确的动态。
Atari 游戏的视觉复杂性更高，模型难以准确预测环境的未来状态。
Q5：MuZero Reanalyze 是如何提高样本效率的？
重新分析旧轨迹：使用最新的模型参数重新运行 MCTS，为过去的轨迹生成更好的策略目标。
使用目标网络：使用基于近期参数的目标网络，为价值函数提供更稳定、更新的 n 步自举目标。
伪代码
下面给出一个示例性代码，展示如何在一个简化的 5×5 围棋环境上，按照 MuZero 所提出的主要思路来实现：

import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

#────────────────────────────────────────────────────────────────────────────
# (A) 5×5 简化围棋环境
#────────────────────────────────────────────────────────────────────────────
class GoEnv5x5:
    """
    一个极简化的 5×5 围棋环境示例。
    仅用于演示 MuZero 的整体结构，不保证完全符合围棋规则。
    棋盘编码:
        0 代表空
        1 代表黑子
        -1 代表白子
    当前玩家:  1 表示黑方, -1 表示白方
    
    结束条件（示例化简）:
    - 当前玩家无合法落子点时，游戏立即结束，另一方获胜，获胜方得到 +1，失败方得到 -1。
    - 也可以自行设定走子上限或其他结束判定。
    
    注意: 这里为教学演示，很多围棋细节没有实现(例如吃子、打劫、气的计算)
    """
    def __init__(self):
        self.size = 5
        self.board = np.zeros((self.size, self.size), dtype=np.int32)
        self.current_player = 1  # 先手为黑
        self.done = False
        # 最简单的：如果一方无合法动作，另一方获得 +1
        self.reward_black = 0.0
        self.reward_white = 0.0
        self.num_steps = 0  # 用来限制步数或做统计

    def reset(self):
        """重置环境，返回初始观测"""
        self.board[:] = 0
        self.current_player = 1
        self.done = False
        self.reward_black = 0.0
        self.reward_white = 0.0
        self.num_steps = 0
        return self.get_observation()

    def get_observation(self):
        """
        返回对当前盘面和当前玩家的观测。
        对 MuZero 而言，这里可以是原始 board + 当前玩家标志。
        做简单处理：将当前玩家贴到最后 +1 个平面
        """
        obs = np.stack([
            (self.board == 1).astype(np.float32),   # 黑子平面
            (self.board == -1).astype(np.float32),  # 白子平面
        ], axis=0)  # shape = [2, 5, 5]
        # 再额外添加一个平面表示当前是不是黑方(1.0)或白方(0.0)
        current_player_plane = np.full((1, self.size, self.size),
                                       1.0 if self.current_player == 1 else 0.0,
                                       dtype=np.float32)
        obs = np.concatenate([obs, current_player_plane], axis=0)  # [3, 5, 5]
        return obs  # 这就是本例的“原始观测”

    def step(self, action):
        """
        执行动作 action:
        - action 为 0 ~ 24 的整数，表示在 5×5 棋盘上的位置
        返回: (obs, reward, done, info)
        """
        if self.done:
            # 如果已经 done, 再调用step就直接原样返回
            return self.get_observation(), 0.0, True, {}

        self.num_steps += 1

        # 判断动作是否是合法落子
        row = action // 5
        col = action % 5
        if self.board[row, col] != 0:
            # 非法: 当前位置已经有子了
            # 在围棋中这通常是不允许的，这里直接结束并且当前玩家惩罚
            # 最简单做法：判当前玩家输
            if self.current_player == 1:
                self.reward_white = 1.0
            else:
                self.reward_black = 1.0
            self.done = True
        else:
            # 合法落子
            self.board[row, col] = self.current_player

            # 检查一下当前玩家是否还有继续下子的可能
            # (针对下一回合)
            # 如果下一回合无任何空位可下，则下一回合玩家判负
            next_player = -self.current_player
            legal_moves = self.get_legal_moves(next_player)
            if len(legal_moves) == 0:
                # 下一回合无棋可下，则当前玩家胜
                if self.current_player == 1:
                    self.reward_black = 1.0
                else:
                    self.reward_white = 1.0
                self.done = True

            # 翻转玩家
            self.current_player = next_player

        # 计算本回合奖励
        # 可以把黑方奖励记为 reward_black, 白方记为 reward_white
        # 在 step 函数里只返回当前刚下完子的那位玩家的奖励即可
        if self.current_player == -1:
            # 表示黑子刚下完
            reward = self.reward_black
        else:
            # 表示白子刚下完
            reward = self.reward_white

        obs_next = self.get_observation()
        return obs_next, reward, self.done, {}

    def get_legal_moves(self, player):
        """
        返回给定 player (1 or -1) 所有合法的动作(可落子的坐标)
        """
        legal_actions = []
        for r in range(self.size):
            for c in range(self.size):
                if self.board[r, c] == 0:
                    # 简化处理：只要空就能下
                    legal_actions.append(r * 5 + c)
        return legal_actions


#────────────────────────────────────────────────────────────────────────────
# (B) 神经网络结构：表示函数 hθ、动态函数 gθ、预测函数 fθ 等
# 在实践中通常会把它们合并到一个网络里，通过不同的输入输出区分。
#────────────────────────────────────────────────────────────────────────────
class MuZeroNet(nn.Module):
    """
    一个合并版的网络:
    hθ: 表示函数,  将观测(3,5,5) -> 初始隐藏状态 s0
    gθ: 动态函数,  将(s_{k-1}, a_k) -> (r_k, s_k)
    fθ: 预测函数,  将 s_k -> (p_k, v_k)
    这里简单地把隐藏状态 s 的形状也设成同样的(通道数可以不同), 
    并将动作 a 嵌入到一个向量后拼到 s 上, 然后经过卷积网络得到下一隐藏状态。

    注意: 这里只是演示，并没有做太深的模型。
    """
    def __init__(self, board_size=5, hidden_ch=64, action_dim=25):
        super().__init__()
        self.board_size = board_size
        self.action_dim = action_dim
        self.hidden_ch = hidden_ch

        # ───────── 表示函数 hθ ─────────
        # 输入: obs shape = [3,5,5], 输出: hidden state s0 shape = [hidden_ch,5,5]
        self.rep_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=hidden_ch, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_ch, hidden_ch, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # ───────── 动态函数 gθ ─────────
        # 输入: (s_{k-1}, a_k)
        # a_k one-hot -> shape=[action_dim]
        # 我们把它 reshape 成 [action_dim,1,1] 并在 spatial 上广播后拼到 s_{k-1} 维度
        # 然后再做卷积得到 s_k
        # 同时要输出一个立即奖励 r_k (标量), 这里做一个 MLP 提取
        # 为了简化, 我们把 s_{k-1} 先 global pooling => embed + a_k => MLP => r_k
        self.dyn_conv = nn.Sequential(
            nn.Conv2d(hidden_ch + 1, hidden_ch, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_ch, hidden_ch, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.dyn_reward_head = nn.Sequential(
            nn.Linear(hidden_ch * board_size * board_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # 输出标量，表示即时奖励
        )

        # ───────── 预测函数 fθ ─────────
        # 输入: s_k shape = [hidden_ch,5,5]
        # 输出: p_k (动作分布), v_k (价值)
        # p_k 大小=25
        # v_k 标量
        self.policy_head = nn.Sequential(
            nn.Conv2d(hidden_ch, 32, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * board_size * board_size, action_dim)
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(hidden_ch, 32, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * board_size * board_size, 1)
        )

    def representation(self, obs):
        """
        表示函数 hθ
        obs shape = [3,5,5]
        return s0 shape = [hidden_ch,5,5]
        """
        x = self.rep_conv(obs)
        return x

    def dynamics(self, s, a):
        """
        动态函数 gθ
        输入:
           s shape = [batch_size, hidden_ch,5,5]
           a shape = [batch_size]     (离散动作下标)
        输出:
           r shape = [batch_size, 1] (标量奖励)
           s_next shape = [batch_size, hidden_ch,5,5]
        """
        bs = s.shape[0]
        # 先将a做one-hot
        a_onehot = torch.zeros(bs, self.action_dim, dtype=s.dtype, device=s.device)
        a_onehot[range(bs), a] = 1.0
        # 把 a_onehot reshape => [bs,1,1,action_dim] 或者 broadcast
        a_onehot = a_onehot.view(bs, self.action_dim, 1, 1)
        # 这里为了简单，直接在 spatial 上广播 => [bs,1,5,5]
        # 但 action_dim=25, 做法: 我们可以只取 a_onehot 的某个plane
        # 此处仅做演示，真正实现可以把 a 嵌入成通道后拼接
        # 简化: 只要把 a 的 scalar index 变成 1 plane? 这里直接做一个小技巧
        a_plane = a.view(bs, 1, 1, 1).float().expand(-1, 1, s.shape[2], s.shape[3])
        # 拼接: [bs, hidden_ch+1,5,5]
        x = torch.cat([s, a_plane], dim=1)
        x = self.dyn_conv(x)
        # 用 x 来预测奖励 r
        # 先把 x flatten
        x_flat = x.view(bs, -1)
        r = self.dyn_reward_head(x_flat)
        return r, x

    def prediction(self, s):
        """
        预测函数 fθ
        输入:
          s shape = [bs, hidden_ch,5,5]
        输出:
          p => [bs, action_dim]    动作的 logit (策略分布)
          v => [bs, 1]            价值
        """
        p = self.policy_head(s)
        v = self.value_head(s)
        return p, v.squeeze(-1)


#────────────────────────────────────────────────────────────────────────────
# (C) MCTS 搜索 (极简版本)
#────────────────────────────────────────────────────────────────────────────
class Node:
    """MCTS 树节点，用以存储统计信息"""
    def __init__(self, prior):
        self.prior = prior  # 来自网络预测的先验 p(a)
        self.visit_count = 0
        self.value_sum = 0
        self.children = {}
        self.hidden_state = None
        self.reward = 0

    @property
    def q_value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count


def softmax_sample(logits, temperature=1.0):
    """ 根据 logits 用softmax采样一个动作 """
    probs = torch.softmax(logits / temperature, dim=-1).cpu().numpy()
    return np.random.choice(len(probs), p=probs)


class MCTS:
    def __init__(self, network: MuZeroNet, action_dim=25, c_puct=1.0):
        self.network = network
        self.action_dim = action_dim
        self.c_puct = c_puct

    def run_mcts(self, root, legal_moves, n_simulations=30):
        """
        在给定根节点 root(其 hidden_state 已经由 env 或 representation 给出) 上
        进行若干次模拟，从而更新树中各个节点的访问次数、价值等信息
        """
        for _ in range(n_simulations):
            node = root
            search_path = [node]

            # 1) Selection
            while node.children:
                # 根据 UCB 或 pUCT 公式选子节点
                best_action, node = self.select_child(node)
                search_path.append(node)

            # 2) Expand & Evaluate (如果不是终止节点)
            parent = search_path[-2] if len(search_path) >= 2 else None
            if parent is not None:
                # 用动态函数 gθ 拓展出子节点 hidden state
                a = best_action
                r, s_next = self.network.dynamics(parent.hidden_state.unsqueeze(0),
                                                  torch.tensor([a], device=parent.hidden_state.device))
                r = r[0].item()
                s_next = s_next[0]
                # 用预测函数 fθ 得到 policy logits p, value v
                p_logits, v = self.network.prediction(s_next.unsqueeze(0))
                p_logits = p_logits[0]
                v = v.item()
                # 合法动作mask
                p_logits_np = p_logits.detach().cpu().numpy()
                mask = np.zeros_like(p_logits_np, dtype=bool)
                mask[legal_moves] = True
                # 对非法动作设为很小值
                p_logits_np[~mask] = -1e9
                prior_prob = torch.softmax(torch.from_numpy(p_logits_np), dim=-1).numpy()
                
                node.hidden_state = s_next
                node.reward = r
                # 为每个可能的 a 创建一个子节点
                for a2 in range(self.action_dim):
                    if a2 in legal_moves:
                        node.children[a2] = Node(prior_prob[a2])
                # v 的值回溯
                self.backup(search_path, v)
            else:
                # 如果只有根节点, 也可以直接用其变化
                pass

    def select_child(self, node: Node):
        """使用 pUCT 公式在 node.children 中选出最优动作"""
        best_score = -float('inf')
        best_action = None
        best_child = None

        sum_n = sum(child.visit_count for child in node.children.values()) + 1e-8
        for action, child in node.children.items():
            u = self.c_puct * child.prior * math.sqrt(sum_n) / (1 + child.visit_count)
            score = child.q_value + u
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child

    def backup(self, search_path, value):
        """
        将最终的价值value返传给路径上的节点
        """
        for node in reversed(search_path):
            node.value_sum += value
            node.visit_count += 1
            # 模拟双人对抗时可翻转value,本例就简单不翻转
            # 这里如果是零和游戏(黑白对抗),可做 value = -value
        return


#────────────────────────────────────────────────────────────────────────────
# (D) 将上述组件整合：自对弈 + MCTS + 训练
#────────────────────────────────────────────────────────────────────────────
class MuZeroAgent:
    def __init__(self, board_size=5, action_dim=25, lr=1e-3):
        self.board_size = board_size
        self.action_dim = action_dim
        self.network = MuZeroNet(board_size=board_size, action_dim=action_dim)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self.mcts = MCTS(self.network, action_dim=action_dim, c_puct=1.0)
        self.replay_buffer = deque(maxlen=10000)

    def select_action(self, obs, legal_moves, temperature=1.0, n_sim=30, device="cpu"):
        """
        给定 obs(当前观测), 用表示函数得到 root hidden state,
        再在 root 上 run_mcts, 最后根据访问次数或策略分布选一个动作
        """
        obs_t = torch.from_numpy(obs).unsqueeze(0).to(device)
        # hθ 得到 s0
        with torch.no_grad():
            s0 = self.network.representation(obs_t)  # shape=[1,hidden_ch,5,5]
        # 创建根节点
        root = Node(prior=1.0)
        root.hidden_state = s0[0]  # shape=[hidden_ch,5,5]
        # 用预测函数算一下 root 的 policy, v (用来初始化 children)
        with torch.no_grad():
            p_logits, v = self.network.prediction(s0)
            p_logits = p_logits[0]  # [action_dim]
            v_val = v.item()
        # 筛选合法动作
        p_logits_np = p_logits.cpu().numpy()
        mask = np.zeros_like(p_logits_np, dtype=bool)
        mask[legal_moves] = True
        p_logits_np[~mask] = -1e9
        prior_prob_root = torch.softmax(torch.from_numpy(p_logits_np), dim=-1).numpy()
        for a in legal_moves:
            root.children[a] = Node(prior_prob_root[a])

        # 在 root 上进行 MCTS
        self.mcts.run_mcts(root, legal_moves, n_simulations=n_sim)

        # 根据访问次数(或贪心)来选动作
        visits = np.array([root.children[a].visit_count if a in root.children else 0
                           for a in range(self.action_dim)])
        if temperature < 1e-8:
            # 贪心
            a = np.argmax(visits)
        else:
            # 按访问次数做多项式采样
            probs = visits ** (1.0/temperature)
            probs_sum = np.sum(probs)
            if probs_sum < 1e-8:
                # 可能所有动作都没有访问, fallback随机
                a = random.choice(legal_moves)
            else:
                probs /= probs_sum
                a = np.random.choice(len(probs), p=probs)

        return a, visits

    def store(self, obs, action, reward, obs_next, done):
        """存储到replay buffer以备训练"""
        self.replay_buffer.append((obs, action, reward, obs_next, done))

    def train_step(self, batch_size=16, device="cpu"):
        """
        一个简单的训练过程示例:
        并未完全展现 MuZero 全流程(我们这里没有做多步展开 K=5, 只做单步TD范例)。
        真实MuZero会额外训练动态函数和后续价值。
        """
        if len(self.replay_buffer) < batch_size:
            return
        
        batch = random.sample(self.replay_buffer, batch_size)
        obs_batch = []
        action_batch = []
        reward_batch = []
        obs_next_batch = []
        done_batch = []

        for (o, a, r, o2, d) in batch:
            obs_batch.append(o)
            action_batch.append(a)
            reward_batch.append(r)
            obs_next_batch.append(o2)
            done_batch.append(d)

        obs_batch = torch.from_numpy(np.array(obs_batch)).float().to(device)
        action_batch = torch.tensor(action_batch, dtype=torch.long, device=device)
        reward_batch = torch.tensor(reward_batch, dtype=torch.float, device=device)
        obs_next_batch = torch.from_numpy(np.array(obs_next_batch)).float().to(device)
        done_batch = torch.tensor(done_batch, dtype=torch.float, device=device)

        # 1) 通过表示函数 hθ 得到 s, 通过预测函数 fθ 得到 p,v
        s0 = self.network.representation(obs_batch)            # [bs, hidden_ch,5,5]
        p_logits, v_est = self.network.prediction(s0)          # p_logits => [bs,25], v_est => [bs]
        
        # 2) 通过动态函数 gθ 得到 r 预测  (这里只 roll 一步)
        r_pred, s_next_pred = self.network.dynamics(s0, action_batch)  # r_pred => [bs, 1], s_next_pred => [bs, hidden_ch,5,5]
        
        # 3) BootStrap 下一步的价值
        with torch.no_grad():
            p_logits_next, v_next = self.network.prediction(s_next_pred)
            v_next = v_next * (1.0 - done_batch)  # 如果下一步done则价值=0

        # TD目标(非常简化的版本)
        target_v = reward_batch + v_next

        # 损失函数:
        # (a) value loss
        value_loss = nn.MSELoss()(v_est, target_v)
        # (b) reward loss
        reward_loss = nn.MSELoss()(r_pred.squeeze(-1), reward_batch)
        # (c) policy loss (在这里为了演示我们只做 "交叉熵 + 真实动作的 one-hot")
        # 真实MuZero中的policy target来自MCTS访问次数
        policy_target = torch.zeros_like(p_logits)
        policy_target[range(batch_size), action_batch] = 1.0
        policy_loss = -(policy_target * torch.log_softmax(p_logits, dim=-1)).sum(dim=-1).mean()

        loss = value_loss + reward_loss + policy_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


#────────────────────────────────────────────────────────────────────────────
# (E) 主流程：对战 + 训练
#────────────────────────────────────────────────────────────────────────────
def main_muzero_5x5_go(num_episodes=1000, device="cpu"):
    env = GoEnv5x5()
    agent = MuZeroAgent(board_size=5, action_dim=25, lr=1e-3)

    for eps in range(num_episodes):
        obs = env.reset()
        done = False
        episode_data = []

        while not done:
            legal_moves = env.get_legal_moves(env.current_player)
            action, visits = agent.select_action(obs, legal_moves,
                                                 temperature=1.0,
                                                 n_sim=30,
                                                 device=device)
            obs_next, reward, done, _ = env.step(action)
            # 存储(s,a,r,s')
            agent.store(obs, action, reward, obs_next, done)
            obs = obs_next

        # 每回合结束做若干次训练
        for _ in range(10):
            loss_val = agent.train_step(batch_size=16, device=device)

        if (eps+1) % 50 == 0:
            print(f"Episode {eps+1}, last loss={loss_val}")

    print("训练结束！可尝试用 agent 与环境进行自对弈。")


if __name__ == "__main__":
    main_muzero_5x5_go(num_episodes=200, device="cpu")  # 可根据机器性能调整回合数
