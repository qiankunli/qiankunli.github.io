---

layout: post
title: 语言模型强化学习基础：从策略梯度到 PPO
category: 架构
tags: MachineLearning
keywords: rl, ppo, importance sampling, on-policy, off-policy

---

<script>
  MathJax = {
    tex: {
      inlineMath: [['$', '$'],['$$', '$$']], // 支持 $和$$ 作为行内公式分隔符
      displayMath: [['$$', '$$']], // 块级公式分隔符
    },
    svg: {
      fontCache: 'global'
    }
  };
</script>
<script async src="/public/js/mathjax/es5/tex-mml-chtml.js"></script>

## 简介

* TOC
{:toc}

这篇虽然会多次借用 LLM / token 生成来帮助理解，但它讨论的仍然是**通用 RL 细节**。一些基础概念：
- $\pi$（Policy，策略）：即LLM模型
- $\theta$（Parameter，参数）：即模型参数
- s（State，交互状态）：即上文，初始状态即为$s_1$
- a（Action，交互行为）：即输出的token，可以简单理解为每个字符。（实际上一个字不等于一个token）
- $\tau$（Trajectory，轨迹）：$\tau = \{ s_1,a_1,r_1,s_2,a_2,r_2,...,s_T,a_T,r_T \}$

reward怎样变成各个 Token 的更新：Reward ==> advantage ==> loss

## 从Policy-Based到在 policy gradient

Policy-Based 直接学习 $\pi_\theta$，那用什么指标衡量”策略有多好”？怎么优化这个指标？$J(\theta)$ 就是北极星——目标很简单：找到让 $J(\theta)$ 最大的参数 $\theta$。

$$
J(\theta)=\mathbb{E}_{\tau_\theta}\left[\sum_{t=0}^{\infty}\gamma^t r_t\right]
$$

$J(\theta)$是所有可能轨迹回报的期望——按策略$\pi_\theta$ 跑无数次，取平均。不同的策略$\pi_\theta$  产生不同的轨迹分布，因此$J(\theta)$ 不同。优化目标就是找到让 $J(\theta)$ 最大（即平均回报最高）的参数 $\pi_\theta$。怎么让$J(\theta)$变大？深度学习里最经典的招数：沿着梯度方向走。

$$
\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)
$$

但 $\nabla_\theta J(\theta)$怎么算？目标函数里有一个期望 E——理论上要求把所有可能的轨迹都跑一遍然后取平均。现实中可能的轨迹数量是天文数字，不可能全部跑遍。就好比想知道全校学生的平均身高——不可能量遍每一个人，但可以随机抽 100 个人来估计。抽 100 个人算出来的样本均值，就是对真实均值的抽样估计。策略梯度用的也是同一个思路：跑几条轨迹，用这几条轨迹的梯度平均，来估计真实的 $\nabla_\theta J(\theta)$。这就是策略梯度定理出场的地方， $\nabla_\theta J(\theta)$可以被转化为一个**可以用采样来估计的形式**

$$
\nabla_\theta J(\theta)
=
\mathbb{E}_{\pi_\theta}
\left[
\sum_t
\nabla_\theta \log \pi_\theta(a_t \mid s_t)\cdot G_t
\right]
$$

如果一个动作导致了好的结果（$G_t$ 大），就增加再做这个动作的概率；如果导致了坏的结果（$G_t$ 小），就降低它的概率。为什么不直接写成 $\nabla_\theta \pi_\theta(a_t \mid s_t)\cdot G_t$，非要多一个 $\log$？这是一个数学技巧，叫做对数导数技巧（Log-Derivative Trick）。根据链式法则：

$$
\nabla_\theta \log \pi
=
\frac{\nabla_\theta \pi}{\pi}
$$

这个“除以 $\pi$”的操作恰好抵消了期望计算中隐含的 $\pi$ 因子，让整个公式变得干净且可计算。从工程角度看，概率 $\pi$ 在 $(0,1)$ 之间，直接对概率求梯度可能产生极小的数值，影响训练稳定性。$\log$ 把 $(0,1)$ 映射到 $(-\infty,0)$，梯度数值更稳定。

### 从轨迹回报降到单步训练信号

策略梯度定理告诉了我们梯度的形式（理论公式）。REINFORCE 就是这个定理最朴素的实现——用蒙特卡洛采样来估计期望/$G_t$。所谓蒙特卡洛，就是通过随机试验，用采样结果来估计一个难以直接算出的量。比如，你想知道模型回答“天空为什么是蓝色的？”平均能得多少分。理论上需要枚举所有可能的回答：

$$
J(\theta)
=
\sum_y \pi_\theta(y\mid x)\,R(x,y)
$$

也就是：每种回答的**出现概率 × 得分**，再加起来。但回答的可能性太多，无法枚举。于是按policy的概率分布随机生成 5 个回答，评分分别是：

```text
回答 1：80 分
回答 2：20 分
回答 3：80 分
回答 4：80 分
回答 5：20 分
```
用样本平均估计期望：

$$
J(\theta)
\approx
\frac{80+20+80+80+20}{5}
=
56
$$

这就是一次 Monte Carlo estimation。**概率已经体现在“哪些回答更容易被采到”里面了，所以这里直接求平均，不需要再乘一次回答概率。**


算法流程：

1. 用当前策略 $\pi_\theta$ 跑完一个完整的 episode，记录每一步的状态、动作和奖励。
2. 对每一步，计算从那一步到 episode 结束的累计回报：
  $$
  G_t=\sum_{k=t}^{T}\gamma^{k-t}r_k
  $$
3. 用采样来估计梯度：
  $$
  \nabla_\theta J
  \approx
  \sum_t
  \nabla_\theta\log\pi_\theta(a_t\mid s_t)\cdot G_t
  $$

4. 沿梯度方向更新参数：
  $$
  \theta\leftarrow\theta+\alpha\nabla_\theta J
  $$

这里有一个隐含要求：*用来计算梯度的样本，必须来自当前策略$\pi_\theta$*。策略梯度公式中的期望是 $\mathbb{E}_{\tau\sim\pi_\theta}[\cdots]$，它的意思是：用来计算梯度的样本，必须来自当前策略把智能体实际采到的所有动作按经验频次加权平均——当前策略经常选的动作，在平均里占的权重就高；很少选的动作，权重就低。举个例子。假设当前策略在某个状态下：有 80% 的概率选”向左”；有 20% 的概率选”向右”。智能体跑了 100 步，大约 80 次选了”向左”，20 次选了”向右”。公式按这个 80/20 的频次来加权平均梯度，这就是 $\mathbb{E}_{\pi_\theta}$做的事。但如果这 100 步数据来自旧策略——旧策略在这个状态下有 60% 概率选”向左”，40% 选”向右”——数据里实际会有60 次”向左”、40 次”向右”。实际频次（60/40）和公式期望（80/20）已经不一致，用这种数据算出来的梯度就是错的⸺”向左”的贡献被低估，”向右”的贡献被高估。



因此，用来估计梯度的轨迹也应由同一个策略 $\pi_\theta$ 采样。用$\pi_\theta$ 生成一批轨迹并更新参数后，策略变成了
$\pi_{\theta'}$；原来的轨迹仍服从旧策略 $\pi_\theta$ 的分布，不再严格符合新公式中的 $\tau\sim\pi_{\theta'}$。如果继续不加修正地复用这些数据，得到的就不再是新策略梯度的无偏估计。这就是Policy Gradient / REINFORCE 属于 on-policy 方法的原因：通常需要不断重复“用当前策略采样 → 更新策略 → 重新采样”。



### 从绝对回报降到相对好坏

REINFORCE 能工作，但高方差让它几乎不可用。$G_t$ 是从时刻 t 到 episode 结束的累积回报——它包含了这段路径上的所有随机性。同一个动作，不同的采样轨迹可能给出截然不同的 $G_t$：
1. 好运气，后续每步都恰好拿了高分，$G_t$很大
2. 坏运气，后续每步都恰好拿了低分，$G_t$很小

策略梯度用 $G_t$ 来判断”这个动作好不好”——但$G_t$的波动意味着，同一个好的动作可能因为运气差而被惩罚，同一个差的动作可能因为运气好而被奖励。这就像用一次考试的成绩来判断一个学生的水平——考砸了不代表学得差，可能只是那天状态不好。好在策略梯度定理有一个奇妙的性质：可以在梯度估计中减去一个不依赖于动作的”基线”，既不改变梯度的期望方向，又能大幅降低方差。

此外，REINFORCE 要跑完整个 episode 才能更新。因为 $G_t$ 需要从时刻 t 到 episode 结束的所有奖励。不跑到终点，就不知道$G_t$ 的完整值。这就像一部电影——不到最后一幕，无法给出公正的评价。这也暗示了一个优化方向：如果能用一个更稳定的估计来替代 $G_t$，就不必等到 episode 结束才能更新。这个”更稳定的估计”从哪来？答案是用 V (s) 作为基线，把”绝对回报”变成”相对回报”。

### 怎样估计 Advantage

基线不改变期望，回顾策略梯度定理：

$$
\nabla_\theta J(\theta)
=
\mathbb{E}_{\pi_\theta}
\left[
\sum_t
\nabla_\theta\log\pi_\theta(a_t\mid s_t)\cdot G_t
\right]
$$

现在把 $G_t$ 换成 $G_t-b(s_t)$，其中 $b(s_t)$ 是一个只依赖状态、不依赖动作的函数：

$$
\nabla_\theta J(\theta)
=
\mathbb{E}_{\pi_\theta}
\left[
\sum_t
\nabla_\theta\log\pi_\theta(a_t\mid s_t)
\cdot
\left(G_t-b(s_t)\right)
\right]
$$

基线可以是任何不依赖动作的函数。最简单的选择是常数（比如所有 episode 的平均回报）。常数基线在无状态赌博机中就有用，但它无法区分不同状态。Williams 在 1992 年提出REINFORCE 时， 更新式中已经允许从奖励信号里减去一个不依赖当前动作的基线项。这样做的关键性质是： 只要基线不随当前动作改变， 它不会改变策略梯度的期望方向， 但可能显著降低采样估计的方差。后来，Sutton、McAllester、Singh 和 Mansour 在 policy gradient theorem 中把策略梯度写成更清楚的形式：策略更新可以看作 $\nabla_\theta \log \pi_\theta(a\mid s)$ 乘上某种“动作好坏”的估计。这个估计可以是完整回报 $G_t$，也可以是动作价值 $Q^\pi(s,a)$，还可以减去一个状态相关的 baseline。理论分析表明，当 $b(s)=V^\pi(s)$ 时，方差缩减效果接近最优。直觉上，$V^\pi(s)$ 恰好回答了“从这个状态出发，按当前策略做，平均能拿多少分”——用它做基线，更新信号变成了“实际表现比平均好了多少”。把 $G_t-V(s_t)$ 这个量叫做优势（Advantage）：

$$
A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s)
$$

优势函数 $A^\pi(s,a)$是策略梯度方法中最重要的概念之一。它回答的不是”这个动作有多好”，而是”这个动作比平均水平好了多少”。这个”相对好坏”的信号，比 $G_t$ 的”绝对回报”信号稳定得多。本节说的 Value Baseline，就是用一个价值网络近似 $V(s)$，再用：

$$
A_t=G_t-V(s_t)
$$

实际操作中，V (s) 由一个额外的神经网络（价值网络）来估计：是“用 learned value function 作为 baseline。它已经很接近下一章的 Critic， 但仍然保留 REINFORCE 的特点： 必须等一个完整 episode 结束后， 用 Monte Carlo 回报 Gt 来更新。

## 从 Value Baseline 到 Actor-Critic

优势函数的理论定义是 $A=Q-V$，但实际中通常不直接计算 $Q$（也算不出来，只能估计）。从定义出发，经过一步展开就能得到一个更实用的形式。从

$$
A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s)
$$

开始。动作价值函数的定义是：

$$
Q^\pi(s,a)
=
\mathbb{E}
\left[
R_{t+1}+\gamma V^\pi(S_{t+1})
\mid
S_t=s,A_t=a
\right]
$$

这个期望的含义：在状态 s 做了动作 a 之后，拿到的即时奖励加上下一状态的价值。如果只取一次采样（**不走完整个 episode**，也不对所有可能转移求平均），就得到 Q 的一步估计：

$$
Q(s,a)\approx r+\gamma V(s')
$$

其中 r是这一步实际拿到的奖励，$s'$ 是这一步实际到达的下一状态。把这个近似代入优势函数定义：

$$
A(s,a)
=
Q(s,a)-V(s)
\approx
r+\gamma V(s')-V(s)
$$

右边就是 TD Error/TD 估计：

$$
A(s,a)\approx r+\gamma V(s')-V(s)=\delta
$$

用 TD Error $\delta=r+\gamma V(s')-V(s)$ 就能替代 $G_t-V(s)$ 作为优势估计：

$$
\nabla_\theta J
\approx
\nabla_\theta\log\pi_\theta(a_t\mid s_t)\cdot\delta
$$

用 TD Error 替代 Gt 作为策略梯度的信号，有两个好处：
1. 不需要等 episode 结束（MC 估计）——每走一步就能更新（Gt 需要跑完一整局，这是MC 方法的限制）
2. 方差更低——$\delta$ 只涉及一步的随机性（Gt 涉及整条轨迹的随机性）

要计算$\delta= r+\gamma V(s')-V(s)$，你需要知道$ V(s')$和$V(s)$，但在真实问题中，V 是未知的——需要一个网络来估计它。这个网络就是 Critic。Actor 和 Critic 共享输入（状态 s），但输出不同：Actor 输出动作概率分布，Critic 输出价值标量。它们通过优势函数
$A \approx \delta$ 协作：Critic 给出评估，Actor 根据评估调整行为。

李宏毅老师课程里的下面两张图，正好对应这一步变化。第一张图用同一状态 $s_t$ 出发的多条未来轨迹说明 $V(s_t)$：它是这些未来回报在当前策略下的平均水平。因此，一次实际采样得到的 $G_t$ 减去 $V(s_t)$，就能判断动作 $a_t$ 比平均水平好还是差。

![](/public/upload/machine/rl_version352.jpg)

第二张图进一步把完整轨迹回报 $G_t$ 换成“一步真实奖励 $r_t$ + 下一状态的估计价值 $V(s_{t+1})$”，从而得到一步 Advantage。原图省略了折扣因子，相当于取 $\gamma=1$；按照本文记号，完整形式是 $r_t+\gamma V_\phi(s_{t+1})-V_\phi(s_t)$。

![](/public/upload/machine/rl_version4.jpg)

### Actor 与 Critic 如何同步更新

```text
Actor-Critic 数据流

状态 s
  │
  ├──→ Actor（策略网络）
  │      π(a|s) → 选择动作 a
  │                   │
  │                   ▼
  │              执行动作 a
  │                   │
  │                   ▼
  │           环境返回 r 和 s'
  │
  ├──→ Critic（价值网络）
  │      V(s)  ─────────┐
  │      V(s') ─────────┤
  │                     ▼
  │          δ = r + γV(s') - V(s)
  │                     │
  │                     ▼
  │      Actor 更新：θ ← θ + α∇log π(a|s) · δ
  │      Critic 更新：V(s) ← V(s) + αδ
  │
  └──→ 进入下一状态，重复以上过程
```
Critic 本质上是价值函数 $V(s)$ 的神经网络实现，它学习“每个状态值多少分”。Actor 则是策略 $\pi(a\mid s)$ 的神经网络实现，它根据 Critic 提供的评估调整行为。工程上，它们既可以是两张完全独立的网络，也可以共享前面的特征提取层、只保留各自的输出头；职责边界始终不变：Actor 输出动作分布，Critic 输出价值标量。

Actor-Critic 是一个通用骨架，后续算法的差别主要来自“评估信号怎么估计”和“策略网络怎么更新”。名称上也容易混淆：Advantage Actor-Critic 描述的是用 Advantage 更新 Actor 的方法族；A2C 通常指多个 worker 同步采样和更新的实现，A3C 中的第一个 A 则是 Asynchronous，表示异步并行。同步或异步属于采样与更新架构，不会改变这里的一步 Advantage 公式。

### Critic 如何学习

Critic 用参数为 $\phi$ 的价值网络 $V_\phi(s)$ 近似当前策略的状态价值 $V^\pi(s)$。这里特意把参数写成 $\phi$：Actor 的参数是 $\theta$，负责决定“怎么行动”；Critic 的参数是 $\phi$，负责预测“从当前状态出发，按现有策略继续走，平均还能得到多少回报”。因此 Critic 的训练本质上是一个回归问题，关键在于给 $V_\phi(s_t)$ 构造什么目标值。

最直接的办法是把一条 episode 走完，用实际得到的完整回报 $G_t$ 作为监督信号。这就是 Monte Carlo（MC）目标：

$$
y_t^{\mathrm{MC}}=G_t,
\qquad
L_{\mathrm{Critic}}^{\mathrm{MC}}(\phi)
=
\left(G_t-V_\phi(s_t)\right)^2
$$

![](/public/upload/machine/rl_mc.jpg)

$G_t$ 来自真实采样，不依赖 Critic 对未来的猜测；代价是必须等 episode 结束，而且整段轨迹的随机性都会进入 $G_t$，方差较大。

另一种办法只观察一步：当前拿到奖励 $r_t$，下一状态之后的长期回报交给 Critic 自己估计。这就是 Temporal Difference（TD）目标：

$$
y_t^{\mathrm{TD}}
=
r_t+\gamma(1-d_t)\operatorname{sg}\!\left(V_\phi(s_{t+1})\right)
$$

$$
\delta_t
=
y_t^{\mathrm{TD}}-V_\phi(s_t),
\qquad
L_{\mathrm{Critic}}^{\mathrm{TD}}(\phi)=\delta_t^2
$$

其中 $d_t$ 表示这一步之后 episode 是否终止；终止时 $d_t=1$，下一状态的价值按 $0$ 处理。$\operatorname{sg}$ 表示 stop gradient：计算目标值时可以使用 $V_\phi(s_{t+1})$，但这条目标分支不参与本次反向传播，否则网络会同时移动“预测值”和“靶子”。

![](/public/upload/machine/rl_td.jpg)

TD 使用了 bootstrap：用当前学到的 $V_\phi(s_{t+1})$ 估计尚未发生的未来。它每走一步就能更新，方差通常更低；如果 Critic 尚未学准，这个目标也会带入估计偏差。

同一个 $\delta_t$ 会同时出现在两张网络的更新中，但职责不同：

- Critic 最小化 $\delta_t^2$，让自己的价值预测更准；
- Actor 把 $\delta_t$ 当作一步 Advantage 样本，判断刚才的动作比当前策略的平均水平好多少。

Actor 对应的可微损失可以写成：

$$
L_{\mathrm{Actor}}(\theta)
=
-\log\pi_\theta(a_t\mid s_t)\operatorname{sg}(\delta_t)
$$

这里也要切断 $\delta_t$ 到 Critic 的梯度。Actor 根据评估结果调整动作概率，Critic 则通过自己的回归损失学习评估；两者各自优化自己的参数。

在 LLM 的 RLHF 场景中，Reward Model 和 Critic 很容易被混为一谈。Reward Model 负责提供环境反馈，回答“这段回答最终有多好”；Critic 学习这个反馈在当前策略下的长期期望，回答“生成到当前前缀时，后面平均还能得到多少回报”。常见做法是让中间 token 的 $r_t$ 为 $0$ 或较小的 KL 惩罚，在最后一个 token 加上 Reward Model 对完整回答的评分。

Critic 不会凭空创造奖励，它的训练目标最终仍来自环境或 Reward Model。反过来，只有 Reward Model 也可以采用 MC、REINFORCE 或 GRPO 一类不带 Critic 的方法，只是通常要承受更高方差和更粗的 credit assignment。PPO 同时使用两者，是因为 Reward Model 定义优化方向，Critic 则帮助得到更稳定、更细粒度的 Advantage。

| Critic 目标 | 未来信息来自哪里 | 更新时机 | 典型特点 |
| --- | --- | --- | --- |
| MC：$G_t$ | 实际走完的后续轨迹 | episode 结束后 | 偏差小、方差大 |
| TD：$r_t+\gamma V_\phi(s_{t+1})$ | 一步真实奖励 + Critic 估计 | 每一步 | 方差小，但会引入 bootstrap 偏差 |

MC 看到了完整未来，却容易受整条轨迹的随机性影响；一步 TD 更新及时，却把一步之后的未来全部交给了尚在学习的 Critic。接下来的 GAE 会在这两端之间连续取值。

### GAE（Generalized Advantage Estimation）：在 MC 与 TD 之间折中

GAE（Generalized Advantage Estimation，广义优势估计）的基本材料仍然是一步 TD error。对于采样得到的一段长度为 $T$ 的轨迹，先计算：

$$
\delta_t
=
r_t+\gamma(1-d_t)V_\phi(s_{t+1})-V_\phi(s_t)
$$

只使用 $\delta_t$，相当于把一步之后的未来全部交给 Critic。GAE 会继续吸收后续的 TD error，并让距离越远的信号权重越小：

$$
\hat A_t^{\mathrm{GAE}(\gamma,\lambda)}
=
\sum_{l=0}^{T-t-1}(\gamma\lambda)^l\delta_{t+l}
$$

把求和展开，可以更直观地看到最终奖励如何逐步传回较早的动作：

$$
\hat A_t
=
\delta_t
+\gamma\lambda\delta_{t+1}
+(\gamma\lambda)^2\delta_{t+2}
+\cdots
$$

$\lambda$ 控制对远期 TD error 的信任程度：

- $\lambda=0$ 时，$\hat A_t=\delta_t$，就是一步 TD Advantage。它依赖 Critic 较多，通常方差较低、bootstrap 偏差较大。
- $\lambda=1$ 时，如果终止状态价值为 $0$，中间的价值项会逐项抵消，最终得到 $\hat A_t=G_t-V_\phi(s_t)$，回到 MC Advantage。它依赖实际轨迹更多，偏差较小、方差较大。
- $0<\lambda<1$ 时，GAE 混合了不同长度的未来信息，在偏差和方差之间折中。

实现时无需为每个 $t$ 重复求和。将公式改写成递归形式，从轨迹末尾向前扫描一次即可：

$$
\hat A_t
=
\delta_t
+\gamma\lambda(1-d_t)\hat A_{t+1}
$$

```python
gae = 0.0
for t in reversed(range(T)):
    nonterminal = 1.0 - done[t]
    delta = reward[t] + gamma * nonterminal * value[t + 1] - value[t]
    gae = delta + gamma * lam * nonterminal * gae
    advantage[t] = gae

value_target = advantage + value[:-1]
```

在 rollout 结束后，代码会把当时记录的 `value`、算出的 `advantage` 和 `value_target` 当作固定训练数据，不让梯度穿过这些量。随后两张网络分别使用：

$$
L_{\mathrm{Actor}}(\theta)
=
-\mathbb{E}_t
\left[
\log\pi_\theta(a_t\mid s_t)\hat A_t
\right]
$$

$$
\hat R_t=\hat A_t+V(s_t),
\qquad
L_{\mathrm{Critic}}(\phi)
=
\mathbb{E}_t
\left[
\left(\hat R_t-V_\phi(s_t)\right)^2
\right]
$$

由于每个时间步对应不同的状态——在 LLM 中就是不同的 token prefix——Critic 给出的 $V_\phi(s_t)$ 不同，TD error 或 GAE 得到的 $\hat A_t$ 通常也不同。Actor 用各自的 $\hat A_t$ 加权每个 action/token 的策略梯度，因此 Advantage estimation 在提供更新信号的同时，也实现了 action/token 级的 temporal credit assignment；这里的 $\hat A_t$ 是基于 Critic 的估计信号，并不是对每个 token 真实因果贡献的精确识别。

因此，GAE 同时连接了两条训练线：$\hat A_t$ 告诉 Actor 每个动作相对平均水平好多少，$\hat R_t$ 则成为 Critic 下一轮要拟合的价值目标。在 LLM 的稀疏奖励场景中，中间 token 的即时奖励往往为 $0$，GAE 仍能通过后续 $\delta$ 的加权累积，把终点反馈逐步传回前面的 token。更准确地说，完成 credit assignment 的是“Critic 提供相邻状态价值 + GAE 累积多步 TD error”这一整套机制，Critic 单独只负责提供价值估计。

至此，Advantage “怎么算”已经解决。演进思路

```
R(τ)
  ↓ 只评价整条轨迹，credit assignment 很粗
G_t
  ↓ 每一步只使用它之后的 reward
G_t - b(s_t)
  ↓ 减去 baseline，降低方差
G_t - Vφ(s_t)
  ↓ 用价值网络作为状态相关 baseline
δt = rt + γVφ(st+1) - Vφ(st)
  ↓ 用一步 bootstrap，不必完全依赖 MC
ÂtGAE = Σl (γλ)^l δt+l
  ↓ 在 MC 与 TD 之间折中
交给 Actor 更新策略
```

PPO 之前的主线，可以压缩成一句话：

> 如何把一条轨迹末尾得到的黑盒 Reward，逐步变成低方差、细粒度、可以通过 backprop 更新策略的训练信号。

核心公式演进是：

```text
R(τ)
  ↓ 只评价整条轨迹，credit assignment 很粗
G_t
  ↓ 每一步只使用它之后的 reward
G_t - b(s_t)
  ↓ 减去 baseline，降低方差
G_t - Vφ(s_t)
  ↓ 用价值网络作为状态相关 baseline
δt = rt + γVφ(st+1) - Vφ(st)
  ↓ 用一步 bootstrap，不必完全依赖 MC
ÂtGAE = Σl (γλ)^l δt+l
  ↓ 在 MC 与 TD 之间折中
交给 Actor 更新策略
```

展开来看是三条连续的问题线。

1. 梯度怎么来？RL 优化的是：$J(\theta)=\mathbb{E}_{\tau\sim\pi_\theta}[R(\tau)]$，但离散动作的采样不可导，Reward 也在计算图之外。通过 log-derivative trick，把梯度改写成：
    $$
    \nabla_\theta J(\theta)
    =
    \mathbb{E}
    \left[
    \nabla_\theta\log\pi_\theta(a_t\mid s_t)
    \cdot \text{评价信号}
    \right]
    $$

    这样 Reward 无需求导；真正承接梯度的是 action 的 log-prob。
2. “评价信号”怎样估得更准？REINFORCE 最初用完整回报：$\text{评价信号}=G_t$，但它方差大、必须等待 episode 结束。因此逐步引入：
    1. Reward-to-go：用 $G_t$ 代替整条轨迹共享的 $R(\tau)$，改善 credit assignment。
    2. Baseline：用 $G_t-b(s_t)$ 表达相对好坏，降低方差。
    3. Value Baseline：令 $b(s_t)=V_\phi(s_t)$，得到 MC Advantage：$\hat A_t=G_t-V_\phi(s_t)$
    4. Actor-Critic：用 Critic 估计状态价值，通过一步 TD 得到：$\delta_t=r_t+\gamma V_\phi(s_{t+1})-V_\phi(s_t)$，Actor 把 $\delta_t$ 当作 Advantage；Critic 通过价值回归让 $V_\phi$ 越来越准。
    5. GAE：一步 TD 方差低但依赖 Critic，MC 依赖真实轨迹但方差高。GAE 用 $\lambda$ 在两者之间折中：$\hat A_t^{\mathrm{GAE}}=\sum_l(\gamma\lambda)^l\delta_{t+l}$，到这里，已经解决了“每个 action 应该获得多大的更新信号”。
3. 下一步的问题是：同一批 on-policy 数据能否多训练几轮，同时又不让新策略偏离采样数据时的旧策略太远？这正是 PPO 要处理的问题。

## 从 Actor-Critic 到 PPO

PPO 早在 2017 年就已提出，[原论文](https://arxiv.org/abs/1707.06347)在模拟机器人运动、Atari 游戏等任务上进行了验证，初衷是解决通用强化学习中“如何高效、稳定地更新策略”的问题，LLM RLHF是这套方法在语言生成任务上的应用。

### Actor-Critic 尚未解决的两个问题

策略梯度最核心的直觉：如果优势为正，就提高这个动作的概率；如果优势为负，就降低这个动作的概率。但这个直觉落到实际训练中，还有两个问题需要解决。
1. 一批数据能用几轮？ 原始策略梯度要求数据来自当前策略。参数一更新，这批 rollout 就变成了旧策略的数据，再用就会产生偏差。
2. 策略每次能改多少？优势估计来自有限样本，带有噪声。如果某个动作碰巧表现很好，普通策略梯度可能把这个动作的概率调得过高，下一轮采样分布跟着剧烈变化，训练就会震荡。

针对这两个问题，PPO 的做法是：让同一批经验可以多学几轮(采样很贵)，但每一轮都要限制新策略不要离旧策略太远。 它不改变Actor‑Critic 的基本分工，只是在”怎么更新 Actor”这一步加了一套更稳的规则。PPO 是一种训练策略网络的方法。它最后会写成 loss，是因为 PyTorch 优化器只能根据一个可微的标量目标做反向传播。需要把”复用旧数据”和”限制策略变化”这两个训练要求，翻译成可以 loss.backward() 的数学表达式。

### 原始策略梯度的 On-Policy 约束

真实语言模型的回答数量巨大，不可能把每个回答都生成、评分，再算：

$$
J(\theta)
=
\sum_y
\pi_\theta(y\mid x)\,R(x,y)
$$

于是只能**抽一些回答，用样本估计梯度**。Policy Gradient 给出一种做法：

$$
\nabla_\theta J(\theta)
=
\mathbb E_{y\sim\pi_\theta}
\left[
R(x,y)\nabla_\theta\log\pi_\theta(y\mid x)
\right]
$$

这里最需要盯住的，不是 Reward，也不是 log，而是：$\boxed{y\sim\pi_\theta}$，**按照当前策略抽样，然后对这些样本的梯度信号直接平均，才能得到这个公式对应的估计**。**抽样方式本身，就是计算公式的一部分**。它不是随便找些数据来填进去。如果直接平均旧样本，实际计算的是：


$$
\underbrace{\mathbb E_{\mathrm{old}}}_{\text{旧样本决定的比例}}
\left[A^{\mathrm{new}}\nabla\log\pi_{\mathrm{new}}\right]
$$

想要的是：

$$
\underbrace{\mathbb E_{\mathrm{new}}}_{\text{新策略对应的比例}}
\left[A^{\mathrm{new}}\nabla\log\pi_{\mathrm{new}}\right]
$$

旧样本每条记录有效，$A_{new}$ 和 $ \log\pi_{\mathrm{new}}$都对，不能说$E_{\mathrm{old}}\left[A^{\mathrm{new}}\nabla\log\pi_{\mathrm{new}}\right]$能直接代表当前策略(的期望 Reward)。Loss 能计算、能下降，只说明它自己的数学目标在被优化，不自动证明任务目标在改善。SFT 通常在固定数据集上优化目标；Policy Gradient 的目标涉及策略自己产生的分布，而这个分布随参数更新而变化。所以旧数据中的 Response、Reward 可以不变，但它代表的生成频率/数据分布（理论上应该对一个x 的所有可能y 计算reward），可能已经不适合当前策略。

因此，用来估计梯度的轨迹也应由同一个策略 $\pi_\theta$ 采样。用 $\pi_{\mathrm{old}}$ 收集一批 rollout 并更新参数后，策略已经变成 $\pi_\theta$，原来的数据仍然服从 $\pi_{\mathrm{old}}$ 的分布。直接把它当作新策略的数据继续训练，估计就会产生偏差。

![](/public/upload/machine/rl_policy_gradient.jpg)

原始 Policy Gradient / REINFORCE 因而属于 on-policy 方法：通常需要不断重复“用当前策略采样 → 更新策略 → 重新采样”。PPO 希望让刚刚采集的同一批 rollout 多训练几轮，首先要处理新旧策略之间的数据分布差异。


sl(prompt,response,label) 是稳定的，rl (prompt,response,reward ) 也是稳定的，比如针对“为什么天空是蓝色的”，假设模型只能生成两种回答，评分规则始终不变：

| 回答 | 内容示意 | Reward |
|---|---|---|
| A | 提到空气对蓝光的散射更强，但解释不完整 | 80 |
| B | 更准确、完整地解释瑞利散射及相关细节 | 100 |


1. 旧模型采集一批数据，旧模型生成 A 的概率是 80%，生成 B 的概率是 20%。让它独立回答 100 次，假设这次恰好得到：

    ```
    80 条：(为什么天空是蓝色的, A, 80)
    20 条：(为什么天空是蓝色的, B, 100)
    ```
    平均奖励是84。
2. 训练之后，模型变好了，当前模型生成 A 的概率降到 20%，生成 B 的概率升到 80%。这时重新采样 100 次，预期大约得到：
    ```
    20 条：(为什么天空是蓝色的, A, 80)
    80 条：(为什么天空是蓝色的, B, 100)
    ```
    当前模型的期望奖励是96。注意：A 还是 80 分，B 还是 100 分，改变的是模型生成两种回答的比例。
3. 问题不是任何一条记录错了，而是：旧数据描绘的是“经常答 A、偶尔答 B”的模型；现在要优化的却是“偶尔答 A、经常答 B”的模型。策略梯度也需要求平均，不过平均的不是 Reward 本身，而是每条样本提供的更新信号：

   $$
   g_y= \nabla_\theta\log\pi_\theta(response\mid prompt)
   \cdot \hat A
   $$

   直接平均旧的 100 条样本，得到的仍然是：$
  \hat g_{\mathrm{old\ data}}=\frac{80g_A+20g_B}{100}
  =0.8g_A+0.2g_B$
  而当前策略对应的期望梯度是：$g_{\mathrm{current}}=0.2g_A+0.8g_B$。单条信号可以算对，但汇总时的比例不对，整体梯度就可能有偏。
4. On-policy 思路：重新采样。让当前模型生成一批新回答，自然得到接近 20/80 的比例，再据此估计更新方向。
5. 利用 Off-policy 数据的办法之一：重要性采样。保留旧数据，但调整每条记录的权重：

    | 旧数据 | 每条权重：当前概率／旧概率 | 加权后的总计数 |
    |---|---|---|
    | 80 条 A | $0.2/0.8=0.25$ | $80\times0.25=20$ |
    | 20 条 B | $0.8/0.2=4$ | $20\times4=80$ |

    这样，旧数据虽然没有重新生成，但在计算平均时，贡献比例变成了当前策略对应的 20/80。在这个简化例子中，加权平均奖励就是 96。(prompt, response, reward) 记录“发生过什么”，而策略概率决定“这些事情在当前模型下有多常发生”。RL 的期望目标需要同时考虑这两者。

### 重要性采样：从旧分布估计新分布

普通 Monte Carlo Sampling 是：想估计分布 $p$ 下的平均值，就从 $p$ 采样，再直接求平均。重要性采样（Importance Sampling）则是：想估计分布 $p$ 下的平均值，但从另一个分布 $q$ 采样，再加权求平均。

$$
\mathbb E_{x\sim p}[f(x)]
=
\mathbb E_{x\sim q}
\left[
\frac{p(x)}{q(x)}f(x)
\right]
$$

“Importance” 从哪里来？ 在典型用法中，我们会设计 $q$，让它更多地采到“对期望贡献大”的区域。例如某种罕见事件虽然出现概率低，但影响特别大，就可以刻意多采一些这种事件，再通过权重修正“采多了”的影响。因此，整个名字可以理解为：把采样机会更多地分配给重要区域，再用权重保持估计目标不变。而在 PPO 的语境里，$q$ 已经是采集数据的旧策略，并不是专门为了“重要区域”设计的。我们主要利用这个方法的另一面——对已有的旧分布样本加权，估计目标分布下的期望。

先看普通采样平均。回到策略梯度公式：

$$
\nabla_\theta J(\theta)
=
\mathbb{E}_t
\left[
\nabla_\theta \log \pi_\theta(a_t \mid s_t)\hat{A}_t
\right].
$$

公式里的被积函数是

$\nabla_\theta \log \pi_\theta(a_t \mid s_t)\hat{A}_t$，把它简记为 $f(a_t)$，问题变成：要估计新策略 $\pi_\theta$ 下这个量的期望，理想写法是：$\mathbb{E}_{a\sim\pi_\theta}[f(a_t)]$，如果手里有新策略采出来的动作，直接对这些 $f(a_t)$ 求平均即可。但现在没有新策略样本，只有旧策略 $\pi_{\mathrm{old}}$ 采出来的动作。直接平均会有偏差，因为旧策略和新策略对同一个动作的偏好不同。

重要性采样的做法是：旧样本不是垃圾——它们毕竟是真实交互得到的，只是被旧策略“采偏”了。每个样本要乘一个修正权重，把旧策略的分布“翻译”成新策略的分布。这个权重就是新策略选择该动作的概率与旧策略选择该动作的概率之比：

$$
\frac{\pi_\theta(a\mid s)}
{\pi_{\mathrm{old}}(a\mid s)}.
$$

这样，新策略下的期望可以改写成：

$$
\mathbb{E}_{a\sim\pi_\theta}[f(a_t)]
=
\mathbb{E}_{a\sim\pi_{\mathrm{old}}}
\left[
\frac{\pi_\theta(a\mid s)}
{\pi_{\mathrm{old}}(a\mid s)}
f(a_t)
\right].
$$

如果新策略比旧策略更喜欢这个动作，权重大于 $1$，这个样本对新策略的贡献应该被放大；如果新策略不太会选这个动作，权重小于 $1$，这个样本的贡献应该被缩小。这一步来自期望变换：把新策略下的期望，改写成旧策略分布下的加权平均。在 PPO 代码里，`collect_rollout` 在采样时存下了 `old_logprobs`。

### 策略比率：新旧策略如何看待同一动作

重要性采样的修正权重 $\pi_\theta/\pi_{\mathrm{old}}$ 需要同时知道新旧策略的概率，计算上并不直观。PPO 先把这个比率拆成两个部分来理解：**同一个动作，新策略给它多大概率，旧策略给它多大概率。**假设在某个状态 $s_t$，旧策略和新策略对三个动作的概率如下：

| 动作 | 旧策略概率 $\pi_{\mathrm{old}}(a\mid s_t)$ | 新策略概率 $\pi_\theta(a\mid s_t)$ | 比率 $r=\pi_\theta/\pi_{\mathrm{old}}$ |
| --- | ---: | ---: | ---: |
| 左 | 0.50 | 0.25 | 0.5 |
| 右 | 0.25 | 0.50 | 2.0 |
| 停 | 0.25 | 0.25 | 1.0 |

旧策略采样到“右”，新策略比旧策略更喜欢这个动作（比率为 $2.0$），这个样本对新策略的贡献应该被放大。旧策略采样到“左”，新策略已经不太想选它了（比率为 $0.5$），这个样本的权重应该降低。

写成公式，这个权重就是策略比率（Policy Ratio）：

$$
r_t(\theta)
=
\frac{\pi_\theta(a_t\mid s_t)}
{\pi_{\mathrm{old}}(a_t\mid s_t)}
$$

这里的 $a_t$ 指旧策略当时真实采到的动作。$r_t=1$ 表示新旧策略对它的概率相同；$r_t>1$ 表示新策略更倾向选它；$r_t<1$ 表示新策略在回避它。代码中通过 log 概率之差的指数计算，避免直接除两个很小的概率：

$$
r_t(\theta)
=
\exp\left(
\log\pi_\theta(a_t\mid s_t)
-
\log\pi_{\mathrm{old}}(a_t\mid s_t)
\right)
$$


### PPO-Clip：限制过大的策略更新

重要性采样让旧数据可以用于新策略，但它本身有一个根本弱点：重要性权重 $r_t$  没有上界。

策略更新时，重要性采样中的 $f(a_t)$ 对应动作的优势 $\hat A_t$。把策略比率和优势放在一起，得到未裁剪的代理目标（Surrogate Objective）：

$$
L^{\mathrm{IS}}(\theta)
=
\mathbb{E}_t
\left[
r_t(\theta)\hat A_t
\right]
$$

$\hat A_t$ 决定更新方向，$r_t$ 决定旧样本在新策略下的权重。这个目标在 $r_t$ 接近 $1$ 时比较可信；但目标函数没有限制 $r_t$ 偏离 $1$ 的幅度。如果某个好动作的 $r_t$ 被不断推大，策略会迅速远离采样数据时的旧策略，旧数据的可信度也随之下降。

PPO-Clip 的做法是先把策略比率裁剪到 $[1-\varepsilon,1+\varepsilon]$：

$$
\overline r_t(\theta)
=
\operatorname{clip}
\left(
r_t(\theta),1-\varepsilon,1+\varepsilon
\right)
$$

再在未裁剪目标和裁剪目标之间取更保守的一项：

$$
J^{\mathrm{CLIP}}(\theta)
=
\mathbb{E}_t
\left[
\min
\left(
r_t(\theta)\hat A_t,
\overline r_t(\theta)\hat A_t
\right)
\right]
$$

例如 $\varepsilon=0.2$ 时，裁剪区间是 $[0.8,1.2]$。裁剪的作用取决于优势的正负：

| Advantage | 希望 ratio 怎样变化 | 生效的裁剪边界 |
| --- | --- | --- |
| $\hat A_t>0$，好动作 | 增大 $r_t$，提高动作概率 | $r_t>1+\varepsilon$ 后不再奖励继续增大 |
| $\hat A_t<0$，坏动作 | 减小 $r_t$，降低动作概率 | $r_t<1-\varepsilon$ 后不再奖励继续减小 |
| $\hat A_t=0$ | 不调整 | 目标值为 $0$ |

外层的 $\min$ 保证只裁掉沿优势方向走得过远的更新。如果策略向不利方向移动，未裁剪项仍然保留梯度，把策略拉回正确方向。因此，PPO-Clip 并不是把实际的 ratio 强制锁死在 $[1-\varepsilon,1+\varepsilon]$，而是让代理目标不再鼓励超出边界的有利变化。

代码里通常把两个目标分别记作 `surr1` 和 `surr2`。由于公式写的是需要最大化的目标，而 PyTorch 优化器默认最小化 loss，所以 Actor 的策略损失取负：

```python
ratio = torch.exp(new_logprobs - old_logprobs)
surr1 = ratio * advantages
surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
actor_loss = -torch.min(surr1, surr2).mean()
```

重要性采样让旧策略采集的数据可以用于更新新策略，Clip 则限制新策略不要离旧策略太远。两者合在一起，才构成 PPO-Clip 的策略更新规则。

### PPO Loss 的完整构成

PPO-Clip 只定义了 Actor 的更新目标。完整的 PPO 仍然沿用 Actor-Critic：Actor 学习 policy，Critic 学习 value function，同时用 entropy 避免 policy 过早变得确定。为了交给 optimizer，这三个目标通常合成一个 scalar loss：

$$
\mathcal{L}_{\mathrm{PPO}}(\theta,\phi)
=
\mathcal{L}_{\mathrm{policy}}^{\mathrm{CLIP}}(\theta)
+c_v\mathcal{L}_{\mathrm{value}}(\phi)
-c_H\mathcal{H}[\pi_\theta]
$$

其中 $\theta$ 是 Actor parameters，$\phi$ 是 Critic parameters，$c_v$ 和 $c_H$ 分别控制 Value Loss 与 Entropy Bonus 在总 loss 中的权重。三项承担的职责不同：

1. **Clipped Policy Loss** 更新 Actor。上一节的 $J^{\mathrm{CLIP}}(\theta)$ 是需要最大化的 surrogate objective，而 optimizer 默认执行 gradient descent，因此 Policy Loss 取它的相反数：

    $$
    \mathcal{L}_{\mathrm{policy}}^{\mathrm{CLIP}}(\theta)
    =-J^{\mathrm{CLIP}}(\theta)
    $$

    它让 positive-Advantage action 的概率上升、negative-Advantage action 的概率下降，并用 Clip 限制单次 policy update 的幅度。

2. **Value Loss** 更新 Critic。Critic 预测的 $V_\phi(s_t)$ 要逼近 rollout 与 GAE 计算出的 value target $\hat V_t^{\mathrm{target}}$，因此这一项可以直接使用 regression loss：

    $$
    \mathcal{L}_{\mathrm{value}}(\phi)
    =
    \mathbb{E}_t
    \left[
    \left(V_\phi(s_t)-\hat V_t^{\mathrm{target}}\right)^2
    \right]
    $$

    Critic 越准确，下一轮得到的 Advantage estimate 通常也越可靠。许多实现还会对新旧 value prediction 的差值做 clipping，以免 Critic 在一次 update 中变化过大，但它仍然属于 Value Loss。

3. **Entropy Bonus** 作用于 Actor，鼓励 policy 保留 exploration。对于离散动作，policy entropy 为：

    $$
    \mathcal{H}[\pi_\theta]
    =
    -\mathbb{E}_t
    \left[
    \sum_a \pi_\theta(a\mid s_t)\log\pi_\theta(a\mid s_t)
    \right]
    $$

    总 loss 中使用 $-c_H\mathcal{H}[\pi_\theta]$。最小化 loss 时，提高 entropy 会让这一项变小，从而阻止 policy 过早收敛到少数动作。

这三项并不共享同一种“误差”含义：Policy Loss 是从 expected return 推导出的 surrogate loss，Value Loss 是 value prediction error，Entropy Bonus 是 regularization。它们作为同一个 optimization problem 中的三个目标，通过 weighted sum 合成为 optimizer 所需的 scalar objective；系数 $c_v$ 和 $c_H$ 用来平衡三种 gradient 对 parameter update 的影响。Actor 与 Critic 使用独立 optimizer 时也可以分别反向传播，但 loss 的逻辑构成不变。

### PPO 的完整训练循环

把前面的模块合在一起，一轮 PPO 训练可以整理为：

1. Actor 根据动作概率分布与环境交互，Critic 同时估计状态价值。
2. `collect_rollout` 记录状态、动作、奖励、价值估计，以及采样时的 `old_logprobs`。
3. `compute_gae` 根据奖励和价值估计计算每一步的 Advantage 与 value target。
4. 在同一批 rollout 上训练 $K$ 轮：Clipped Policy Loss 更新 Actor，Value Loss 更新 Critic，Entropy Bonus 保持 exploration；三项加权得到 PPO Loss。
5. 更新完成后丢弃这批 rollout，用新策略重新采样并重复以上过程。

```python
model = SFT_model

for iteration in training:
    # rollout
    responses = model.generate(prompts)
    old_logprobs = model.logprob(prompts, responses)
    old_values = model.value(prompts, responses)
    rewards = reward_model(prompts, responses)
    advantages, value_targets = compute_gae(rewards, old_values)

    # PPO update
    for epoch in range(K):
        new_logprobs, values, entropy = model.evaluate(prompts, responses)
        ratio = exp(new_logprobs - old_logprobs)
        policy_loss = PPO_clip_loss(ratio, advantages)
        value_loss = mse_loss(values, value_targets)
        entropy_bonus = entropy.mean()
        loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_bonus
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

第 $0$ 轮更新开始时，新旧参数相同，因此 `new_logprobs == old_logprobs`、$r_t=1$。执行一次 `optimizer.step()` 后，模型参数和 `new_logprobs` 都会变化；再次使用同一批 responses 时，策略比率就开始反映新旧策略的差异。

PPO 通常仍被归类为 on-policy 算法。它只对当前策略刚刚采集的 rollout 做有限次数的复用，随后便使用更新后的策略重新采样；它不会像带 replay buffer 的 off-policy 算法那样长期使用来自任意历史策略的数据。策略比率提供数据复用的数学基础，Clip 则阻止新策略在这几轮更新中离采样策略太远。

## 李宏毅老师对Policy Gradient的讲解

[The Definitive Guide to Policy Gradients in Deep ReinforcementLearning:Theory, Algorithms and Implementations](https://arxiv.org/pdf/2401.13662) Policy Gradient 论文综述。


[DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://zhuanlan.zhihu.com/p/20530204146)

### 从轨迹回报到 Advantage

回到前文的 REINFORCE 公式，首先，它并没有告诉我们轨迹中某个单独的动作到底好不好，其次会发现 $ R(\tau) $ 即reward恒为正的情况，那会导致一直在增加任何token的输出概率。我们希望调小概率时，reward应该是个负的。直观的想法，如果我见过一批reward，减去他们的均值就好了，这个方法叫REINFORCE with baseline，让reward有正有负，这对rl的训练效率至关重要。这个方法搞出来的均值baseline不一定是最好的，我们想知道的是此刻状态对应的价值，均值没有这样的物理意义，是不是可以用一个NN模型来预估出这个价值呢？这就是Actor-Critic方法。Actor就是你的动作概率模型，Critic就是用一个NN在算这个baseline，或者我们叫他value-base的model。如果不用一个model来估计状态的价值，还有什么好办法？那你就基于每一条原始样本生成一组序列，用他们的reward均值作为baseline，这种方法叫self-critic，它利用了蒙特卡洛方法代替了TD error。

![](/public/upload/machine/reinforce_baseline.png)

我们实际操作中是用sample的方式来训练，这就导致某些项实际上因为没被sample到(只是没被采样到，并不代表它们不好)而导致输出概率下降（实际ground truth是要提升）。所以我们希望引入一个baseline（b）**让reward不是恒为正**。公式变成如下：

$$
\nabla \overline{R}_\theta = \frac{1}{N} \sum_{n=1}^{N} \sum_{t=1}^{T_n} (R(\tau^n) - b) \nabla \log p_\theta(a_t^n \mid s_t^n)
$$

通常我们可以将baseline设置为reward的期望值，即 $ b \approx E[R(\tau)] $。

那为什么可以“随手”减一个 baseline、结果还对？因为**减 baseline 不改变梯度的期望（无偏），只改变方差**——前提是 $b$ 不依赖于当前要评估的那个 action（可以依赖状态 $s$）。一行就能证：

$$
E_{a \sim p_\theta}[\nabla \log p_\theta(a \mid s)\cdot b] = b\sum_a p_\theta(a \mid s)\,\nabla \log p_\theta(a \mid s) = b\sum_a \nabla p_\theta(a \mid s) = b\,\nabla \sum_a p_\theta(a \mid s) = b\,\nabla 1 = 0
$$

两个关键步：$p_\theta \cdot \nabla \log p_\theta = \nabla p_\theta$（对数求导反过来用），以及 $\sum_a p_\theta(a \mid s) = 1$ 求梯度后恒为 0。所以 baseline 这一项对期望梯度的贡献恒为 0，怎么减都不会把梯度“减偏”。

这条结论很重要，但前提也很重要：**只有当 baseline 不依赖当前被评估的 action 时，减 baseline 才不会改变策略梯度的期望。**用状态价值 $V(s)$ 作 baseline，并在更新 actor 时将它视为常量，满足这个条件。GRPO 的组内均值则包含当前样本自己的 reward，因此严格说并不满足“与当前 action 无关”；如果只减组内均值，梯度期望会出现约 $(G-1)/G$ 的常数缩放，使用组内标准差归一和 clip 后还会有进一步偏差。所以 GRPO 的组内 Advantage 不能简单称为“完全无偏”，它是用有限组采样换取无需训练 Critic 的工程折中。

我们知道最终输出的是一个序列 $ \tau $，且在算reward时是以 $ \tau $ 的粒度计算的（episode-level reward）（PS：**reward只能告诉我们哪个$ \tau $ 更好**）。即使整体的reward是正的，也不意味着序列中的每一个action都是有收益的（如：说了一串废话，最后才说对结果）。因此，更合理的做法是我们需要**给每一个action合适的credit**（action/token-level reward）。

首先，我们会有一些假设（注意：并不一定什么情况下都适用，应根据具体情况使用不同的reward function）：

1. reward应单独为每个action计算（前面的）

    $$ 
    R(\tau^n) \rightarrow \sum_{t'=t}^{T_n} r_{t'}^n \quad \text{\# 计算当前action后所有reward的总和作为当前action的reward} 
    $$

2. 越快完成任务应越重要，距离越远贡献越小

    $$  
    R(\tau^n) \rightarrow \sum_{t'=t}^{T_n} r_{t'}^n \rightarrow \sum_{t'=t}^{T_n} \gamma^{t'-t} r_{t'}^n \quad \text{\# } \gamma \text{为时间衰减函数} 
    $$ 

实际上 $R(\tau^n) - b$  这一项其实是在算在某个state下执行某个action比执行其他action有多好，也就是我们常说的Advantage Function，可以表示为 $A^\theta(s_t, a_t)$ ，因此综上公式可以写作：

$$ 
\nabla \overline{R}_\theta \approx \frac{1}{N} \sum_{n=1}^{N} \sum_{t=1}^{T_n} A^\theta(s_t, a_t) \nabla \log p_\theta(a_t^n \mid s_t^n)
$$ 
$$ 
= E_{(s_t, a_t) \sim \pi_\theta} [A^\theta(s_t, a_t) \nabla \log p_\theta(a_t^n \mid s_t^n)]
$$ 

前文提到 each training data is weighted by $R(\tau^n)$，从上述公式我们看到each action is weighted by  $A^\theta(s_t, a_t)$. 此时所有$a_t$的一样$A_t$，都是$R(\tau^n) - b$。

在 LLM RL 中，奖励通常是一个 scalar reward（ sequence-level ），然而，由于序列级似然的数值范围极大，且由此带来的梯度估计具有极高的方差，Sequence-level Objective 很难优化。LLM RL 算法通常采用 Token-level Objective。$\mathcal{J}^{\text{token}}(\theta) \approx \mathcal{J}^{\text{seq}}(\theta)$，提高 LLM RL 稳定性的方法可以理解为：如何维持这一近似的有效性。

### 两个核心问题：Baseline 与 Credit Assignment

换个角度看，整篇的主线其实就一条**加工链**：把原始 reward 加工成真正用来更新策略的信号 $A_t$。

$$\underbrace{r / R(\tau)}_{\text{绝对、给定}}\ \xrightarrow{\ \text{减 baseline}\ }\ \text{相对好坏}\ \xrightarrow{\ \text{credit 分摊}\ }\ \underbrace{A_t}_{\text{相对、单步}}$$

下面这两个问题，正好对应加工的两步——**减 baseline（相对化）**与 **credit assignment（分摊到每一步）**；各家算法的差别，也都落在这两步怎么做上。
1. 绝对优势问题(没有减去baseline导致方差大) 。不同轨迹的 $R(\tau_i)$ 是不同的，而且我们是采样了一些轨迹，如果我们采样的轨迹 $R(\tau_i)$ 都是正的，把轨迹中对应的动作都提高，但一些我们没有采样到，是正确答案的轨迹对应的动作就得降低，因为，我们不希望奖励总是正的，而是要减去一个基线（平均回报），表示相对优势。只有一个动作相比相对优势是正的，我们才提高其动作的概率，如果相比相对优势是负向的，我们要降低其动作的概率。
  1. 价值评估函数 ==> 入状态价值函数和动作价值函数 ==> 优势价值函数=动作价值函数-状态价值函数
  1. 绝对优势 ==> 引入baseline, baseline实现很多，PPO（value模型），GRPO（组内平均值）,REINFORCE++（batch内平均值）
2. 信用分配问题 (Credit Assignment Problem)。信用分配粒度过粗（细粒度的reward分配）。一个轨迹（序列）的整体的回报很高，不能说明他的每一个动作都好，我们对轨迹中所有的动作都使用相同的回报是否合理？我们希望对一个动作能有两种方法衡量其价值：1）评估动作当下的影响（单步回报）；2）又能体现动作对后续策略轨迹的长远效益。（轨迹整体回报）
  1. 有两种典型的方法：蒙特卡洛法（MC）和时序差分法（ TD）
    1. Monte Carlo 方法采用最直观的评价方式：一句话最终好 → 这句话里所有 token 都好；一句话最终坏 → 所有 token 都坏，如果整句话得了 +10 分，那每个 token 都应该更常出现；如果得了 -10 分，每个 token 都应该更少出现。每个 token 获得相同的梯度信号，无法区分贡献度。
    2. TD 引入一个新角色：Value Function（价值函数）V(s)，也叫 Critic（评论家），它是一个预测未来 reward 的模型（通常是另一个神经网络）。它的定义是：V(s) = "从当前上下文 s 开始，我预计这句话最终能拿多少分？"
    3. GAE

以最简单的 TD(0)（TD是一个算法家族名） 为例，最简单的 TD(0) 算法定义 TD 误差 (TD Error)：
$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

其中：

1. $V(s_t)$：生成这个 token 之前，我预计未来能得多少分
2. $r_t$：生成这个 token 立即获得的奖励（LLM 场景下通常为 0）
3. $\gamma V(s_{t+1})$：生成这个 token 之后，我现在预计未来能得多少分（$\gamma$ 是折扣因子，通常接近 1）
4. $\delta_t$：实际 vs 预期的差距

TD的关键点在于：每个 token 的责任被局部化了，每个 token 不再为整句话的最终命运负责，只为"我这一步有没有让情况变好"负责。
1. TD 没有用最终 reward！
2. TD 没用整条轨迹！
3. 而是只用了当前估计和下一步估计
4. 让误差被"局部化"在一步之内

但是需要注意的是，TD 不能直接用于策略更新！TD 误差 $\delta_t$ 是用来训练 Critic（价值函数）的：$V(s_t) \leftarrow V(s_t) + \alpha \cdot \delta_t$，但我们真正要做的是更新策略（Policy）——改变 LLM 生成每个 token 的概率分布。我们还需要把 TD 思想转化为策略梯度信号。这就是 Advantage 的作用：把TD 思想用在策略更新上。Advantage 不是问"这个状态有多好"（Value），而是问："在这个状态下，选择这个 token 比平均水平好多少？"

$$
A(s_t, a_t) = Q(s_t, a_t) - V(s_t) = "选这个token的价值" - "这个状态的平均价值"
$$

1. $Q(s_t, a_t)$：在状态 $s_t$ 下选择 token $a_t$ 之后的期望总奖励（Action-Value Function）
2. $V(s_t)$：在状态 $s_t$ 下，不管选什么 token，平均能得到的期望奖励（State-Value Function）

Advantage 的意义：
1. A > 0：这个选择比平均水平好 → 增加概率
2. A < 0：这个选择比平均水平差 → 降低概率
3. A ≈ 0：这个选择就是平均水平 → 不需要调整

梯度演进

1. REINFORCE 的梯度：$\nabla \log \pi(a_t|s_t) \times G_t$
2. 当我们把它写成更通用的形式，可以用任何"优势估计"来替代 $G_t$：$\nabla \log \pi(a_t|s_t) \times A_t$
3. 在 Monte Carlo / REINFORCE 中：$A_t = G_t - V(s_t) = \text{（实际获得的总奖励）} - \text{（之前预期的总奖励）}$
4. 在 TD-based 算法中：$A_t = \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$
5. 有了 Advantage，策略梯度更新变成：$\nabla \log \pi(a_t|s_t) \times A(s_t,a_t)$

### Advantage 估计的 Bias-Variance 权衡

RL 里到处在“估计”一个没法直接算出来的量——最典型的就是“在当前局面下，未来到底能拿多少分”（价值 $V$）。未来还没发生、还带运气成分，所以只能拿手头的样本去估计它。既然是估计，“准不准”要拆成**两个互相独立的维度**：有没有偏（bias）、稳不稳（variance）。很多人一上来就把它们混在一起，这是第一个坎。

把“估计”想象成反复做很多次、每次给出一个数：

- **无偏（unbiased）**：这些数**平均下来正好等于真实值**。单看某一次可能偏高偏低，但没有系统性地偏向某个方向，误差会互相抵消。
- **有偏（biased）**：这些数**平均下来系统性地偏高（或偏低）**，而且这个偏差再多做几次也消不掉——因为方法本身锚错了地方。

关键：**“无偏”不等于“这一次准”**。无偏说的是方法没有系统性偏向，是对**方法**的评价，不是对**单次结果**的评价。

打靶比喻最直观（真实值 = 靶心，每次估计 = 一发子弹）：

| | 方差小（稳） | 方差大（散） |
|---|---|---|
| **无偏** | 全打在靶心——理想 | 弹着点乱飞，但平均落点在靶心，多打几枪取平均会逼近靶心 |
| **有偏** | 枪枪打在一起，但整体偏在角落，再多打也到不了靶心 | 又散又偏——最差 |

带数字的小例子：假设某学生这类考试的真实平均分 = 70（靶心，但你不知道）。

- **无偏、高方差**：随便抽他一次成绩当估计，这次 45、下次 92，单看不靠谱，但抽足够多次取平均 → 收敛到 70。它没偏，只是抖。
- **有偏、低方差**：干脆每次都报“上学期记录的 65 分”，永远 65、超级稳，但系统性低了 5 分，做一万次还是 65。它很稳，但偏了。

放到 RL 里（具体推导见下篇与主线篇），这正是 MC 与 TD 的区别：

- **MC（蒙特卡洛）**：用整条轨迹真实拿到的回报 $G_t$，没掺任何猜测 → **无偏**；但把一整条轨迹的随机都加进来 → **高方差**。
- **TD**：用“当前奖励 + 对下一步价值的估计 $V(s')$”，而 $V(s')$ 是还没训准的网络猜的 → **有偏**；但只看一步 → **低方差**。

所以 RL 里很多估计方法都在这条 **bias-variance（偏差-方差）权衡**轴上挑位置：完整 on-policy 轨迹上的 MC 通常无偏但高方差；TD / critic 通过 bootstrap 用一定偏差换取更低方差。GRPO 的组内均值、标准差归一和 clip 还会引入额外的估计偏差，不能简单归入“无偏但高方差”。这条轴会在下篇的 baseline / Actor-Critic / GAE 里反复出现。

而且**这条轴的最优落点会随任务的“长短”移动**——2025 年长程 Agentic 任务起来后，这点变得很显眼：

- **短任务**（数学题、单测这类几十到几千 token、可验证的）：轨迹较短、最终得分清晰，省掉 critic 的 GRPO 路线常有较好的工程性价比，但组内估计、归一化和 clip 并不保证严格无偏。
- **长程任务**（几十上百步、动辄上十万 token 的 agent 轨迹）：方差随轨迹长度不断累积，“整条输出共用一个优势”又把信用分配摊得极粗，这时往往值得把 critic 请回来（PPO 那一路）压方差、给每一步更细的信号。

所以“短任务爱用 GRPO、长任务回 critic”，本质不是谁取代谁，而是 bias-variance 天平随 horizon 换了落点。这个“天平”的完整版（GRPO/PPO 在长短程下的机制差异）见 [LLM Post-Train 中的强化学习]({% post_url 2025-02-18-llm_rl %})；通用 RL 的 critic / GAE 推导见上文。

