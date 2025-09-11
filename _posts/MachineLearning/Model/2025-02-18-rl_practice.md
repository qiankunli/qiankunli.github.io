---

layout: post
title: rl从Policy Gradient（策略梯度）到PPO到GRPO
category: 架构
tags: MachineLearning
keywords:  rl

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


SFT是针对Token级别的反馈（对token进行优化和反向传播，每一个token相同的贡献），而RL是针对整个答案文本的评价（对整个语句进行优化和反向传播）。
1. sft的优化逻辑。逐 token 模仿学习（偏向语言建模），
    1. Loss 计算方式：常用交叉熵损失（Cross-Entropy Loss）。
        $$
        \text{Loss} = -\frac{1}{N} \sum_{t=1}^{N} \log P_{\theta}(y_t \mid y_{<t}, x)
        $$
        $y_t$是真实token，$y<t$ 是前面生成的 token，N 是 token 数量。
    2. 反向传播：整个 response 的所有 token loss 加起来（通常是 求和 或 平均），这样 response 的 loss 就是一个 标量，再进行反向传播。**所以每个 token 的贡献是均等的**。
    2. 例如输入 “User: 你好\nAssistant: 我很好”，在监督微调时，模型预测 "我" 的概率与标注答案比对，计算 loss。预测 "很" 的概率与真实答案比对，再计算 loss……依次类推。训练过程就是让模型学会模仿人工答案的逐 token 分布。
2. RLHF优化逻辑。整句优化（偏向人类偏好）。
    1. 训练目标不是让模型模仿 token，而是让整个 句子/输出序列 的质量更高（即 reward 更大）。
    2. Loss 计算方式：用 Policy Gradient (策略梯度) 方法，比如 PPO。梯度更新时，reward 对 整条输出序列 贡献一个分数。每个 token 的更新权重不再是均等的，而是由策略概率分布 + reward 信号共同决定，比如ppo每个 token 的更新强度 = 优势 $A_t$ × 概率比 $r_t(\theta)$。如果 token 在新策略里更被偏好（$r_t$ > 1），且 reward 较高，则这个 token 的梯度被放大。如果 token 导致 reward 低，则会被削弱。每个 token 并不是直接用 cross-entropy，而是有权重的负 log 概率：
        $$
        \text{loss}_t = - A_t \cdot \log \pi_{\theta}(y_t \mid y_{<t})
        $$
        $A_t$ 是 advantage（整条序列的 reward 减 baseline），$\pi_{\theta}$是当前策略概率，每个 token 的“loss”本质上是 带权重的负 log 概率。
    3. 反向传播：把整个序列（甚至整个 batch）的所有 token loss 求和/平均，反向传播 → 更新参数。不是逐 token 单独更新，而是 序列/batch 一次性反向传播。
3. 在实现层面，无论 SFT 还是 PPO，都会对每个 token 算 loss，然后反向传播梯度。区别只是：SFT 的 loss 是“平均的负 log prob”，而 PPO/GRPO/GSPO 的 loss 在“负 log prob”前面多了一个权重（advantage/reward）。

## PPO(Proximal Policy Optimization)

在 LLM 应用环境中，我们应用 PPO 算法，是把 LLM 当成智能体，但**什么是环境呢？似乎不像下围棋、玩游戏这种传统 RL 场景中那样容易定义，奖励从何而来呢？那我们就训练一个 RM 来充当这样角色**(RM 扮演的是「环境」)，它最主要的目标就是给 LLM 这个智能体以外部的 「奖励信号」，这个奖励代表了 LLM 的决策（输出响应）有多符合人类的期望或偏好。

其中 Critic 的作用就是计算 优势函数 (Advantage Function)，**从而减少策略梯度估计的方差**，使训练更稳定、高效。**RM 是 外部的奖励信号**，是外部环境给与智能体的真实响应——虽然在 LLM 的这个场景里，我们没有特别准确的外部环境建模，退而求其次用另一个训练好的 RM 模型来代替了——而 **Critic 是智能体内心对自己答案的评价**。打个不准确的比方，你做一套卷子，Critic 是你自己检查自己的答案，给出的自我评价；而 RM 是老师用标准答案给你打分。这样看来，**不要 Critic 是不是也行？**无非就是我自己「莽」一点，自己不评估自己的答案，反正 RM（环境）会给我反馈，牵引我改进。确实可以，GRPO其实在 Actor-Critic 框架之前，RL 算法就是这样的，不要「基线」了而已。代价就是方差比较大，训练不稳定。其实是通过另一种更简单的「估算基线」的方法，取代了 Critic：就是采样多次，用 RM 评价的平均值来充当这个「基线」。**Critic 不是提供额外的奖励来源，而是通过学习预测未来的期望回报，提供了一个动态的基准，用来校准 RM 提供的原始奖励信号**，生成更稳定、信息量更大的 Advantage 信号，从而稳定并加速 PPO 的训练。GRPO其实是通过另一种更简单的「估算基线」的方法，取代了 Critic：就是采样多次，用 RM 评价的平均值来充当这个「基线」。

[为什么ppo算法中引入critic model可以降低方差？](https://zhuanlan.zhihu.com/p/1903970440885540257)经典的蒙特卡洛策略梯度算法，例如REINFORCE，通过直接优化参数化策略进行学习，但其核心缺陷在于梯度估计的高方差。这种高方差主要源于其对完整样本轨迹回报的依赖，导致策略更新过程不稳定且收敛效率低下。为解决这一关键问题，PPO算法引入了Critic（评论家）模型作为一项核心改进。Critic通过学习状态价值函数充当一个动态的基线（baseline），其目标正是显著降低策略梯度估计的方差，从而提升学习的稳定性和效率。
1. 什么是高方差？为什么它是个问题？想象一下，你正在努力学习如何投篮。如果你仅仅根据多轮投篮后整场比赛的输赢（类似于 RL 中的蒙特卡洛回报）来判断，那么你对每一次投篮的反馈就会非常嘈杂。也许你投出了一个好球，但你的队伍由于其他因素仍然输了比赛。或者，也许一个运气球进了，尽管你的姿势并不好。这与像 REINFORCE 这样的基本策略梯度方法中发生的情况类似。它们基于在整个回合（episode）中累积的总奖励来估计动作的“好坏”。即使动作或环境的随机性只有微小的变化，这些奖励也可能会剧烈波动。反馈（梯度估计）中的这种高方差意味着：
    1. 不稳定的更新： 策略可能会在每次更新时被随机地推向不同的方向。
    2. 缓慢的收敛： 从嘈杂的信号中辨别出动作的真实、潜在质量需要很长时间。
2. Critic 登场：理性的声音。评估 Actor 所采取的动作，或者更常见地，它估计一个特定状态有多好。它学习一个价值函数，通常是状态价值函数 $V(s)$，表示智能体从状态 $s$开始并遵循当前策略可以获得的预期未来总奖励。这个由 Critic 学到的 $V(s)$充当了一个智能的**基线 (baseline)**。

PPO 不仅仅着眼于原始的、嘈杂的回报$G_t$（从时间$t$开始的折扣奖励之和），而是利用 Critic 的基线来计算所谓的**优势函数 (Advantage function)** $A(s_t,a_t)$。其核心思想是确定在状态$s_t$下采取动作 $a_t$比该状态下的平均预期要好多少。估算优势的一种常见方法是：$A(s_t,a_t) \approx G_t - V(s_t)$，这里：
1. $G_t$，从状态 $s_t$开始直到回合结束所收到的实际（采样）累积折扣奖励。
2. $ V(s_t)$，Critic 对从状态  $s_t$开始的预期累积折扣奖励的估计。

然后，PPO（以及一般的 Actor-Critic 方法）中的策略梯度更新使用这个优势：鼓励 Actor 采取导致正优势的动作，而不鼓励采取导致负优势的动作。

直观理解：为什么减去基线有效？把它想象成按曲线给考试评分。教授不仅仅看你的原始分数$G_t$，还会考虑班级在该难度考试中的平均分$ V(s_t)$。你的“优势”就是你比平均水平好（或差）多少。这种相对衡量通常比原始分数更能稳定地反映你的理解程度，因为原始分数可能会受到考试异常简单或困难的影响。类似地，从 $G_t$中减去 $ V(s_t)$
1. 中心化奖励： 如果智能体处于一个通常较好的状态（$ V(s_t)$较高），即使是一个不错的动作也可能导致回报$G_t$ 很高，但不会远高于 $ V(s_t)$。优势 $A(s_t,a_t)$会很小，正确地表明该动作对于那个好状态来说只是“平均水平”。
2. 减少状态价值方差的影响： 某些状态本质上比其他状态更有价值。通过减去状态的价值，我们将学习信号集中在动作本身的后果上，而不是状态的内在价值。如果一个好状态下的所有动作都会导致高回报，那么仅使用 $G_t$无法清楚地区分哪个动作是最佳的。优势函数有助于做出这种区分。
这使得学习信号（优势）的噪声更小，更专注于动作的相对质量，从而带来更稳定和高效的学习。

**方差降低背后的数学原理**。策略梯度定理允许我们将预期总奖励$J(\theta)$关于策略参数 $\theta$的梯度写为：

$$
\nabla_{\theta}J(\theta) = \mathbb{E}_{\tau \sim \pi_{\theta}} [ \sum_{t=0}^{T-1} \psi_t \nabla_{\theta} \log \pi_{\theta}(a_t | s_t) ]
$$

在这里，$ \psi_t$ 是衡量在状态 $ s_t $ 下动作 $ a_t $“好坏” 的某种指标。在 REINFORCE 中，$\psi_t = G_t $（从时间 $t$ 开始的回报）。在像 PPO 这样的 Actor-Critic 方法中，$\psi_t = A(s_t, a_t) \approx G_t - V(s_t)$


[从Policy Gradient到REINFORCE++，万字长文梳理强化学习最新进展](https://mp.weixin.qq.com/s/mGlObqTANspHGkujzCmY5A)

PPO另一个版本，目标函数

$$
\mathrm{objective}(\phi)={E}_{(x,y)\sim D_{\pi_{\phi}^{\mathrm{RL}}}}[r_{\theta}(x,y)- \beta\log(\pi_{\phi}^{\mathrm{RL}}(y \mid x) / \pi^{\mathrm{SFT}}(y \mid x))]
+ \gamma {E}_{x \sim D_{\mathrm{pretrain}}} [\log(\pi_{\phi}^{\mathrm{RL}}(x))]
$$

1. 第一项 ${E}_{(x,y)\sim D_{\pi_{\phi}^{\mathrm{RL}}}}[r_{\theta}(x,y)]$ ，这里 $(x,y)\sim D$ ，是我们用于RLHF训练的数据集，输入(x)，y是模型对应的输出。目标函数是奖励模型r_θ(x,y)在训练数据上的期望值。所以这里是试图最大化之前训练的奖励模型预测的奖励。
2. 第二项，同样，x是提示，y是模型输出。 $\pi_{\phi}^{\mathrm{RL}}(y \mid x)$ 是我们正在训练的当前模型的预测概率，而 $\pi^{\mathrm{SFT}}(y \mid x)$ 是我们开始时的基础模型的预测概率，令$p=\pi_{\phi}^{\mathrm{RL}}(y \mid x)$ ，$q=\pi^{\mathrm{SFT}}(y \mid x)$，$- \beta {E}_{(x,y)\sim D_{\pi_{\phi}^{\mathrm{RL}}}}\log(p / q)]$，这个期望值是两个分布p和q之间的Kullback-Leibler散度（KL散度），它表示两个分布的差异程度。通过对差异施加惩罚，确保在训练模型时，其输出概率与基础模型中的输出概率（预训练+指令微调后的模型）保持相近。
3. 最后一项 $\gamma {E}_{x \sim D_{\mathrm{pretrain}}} [\log(\pi_{\phi}^{\mathrm{RL}}(x))]$，这里 $x\sim D_{pretrain}$ 是回到预训练数据集 $D_{pretrain}$ ，这一项形式上和之前下一个token的预测采用的损失函数是一样的，只是乘以一个常数$\gamma $添加这一项的目的是进行RLHF时，保持在预训练数据上预测下一个token的良好性能。

这个就是PPO，"proximal"是因为我们保持接近基础模型，"policy optimization"是因为在强化学习中，模型的输出概率被称为模型的策略。

[浅谈 RL 里面的 KL 散度](https://zhuanlan.zhihu.com/p/26370587517)个人认为，RL与SFT的区别在于，SFT是token level的0/1奖励，RL是句子level的离散奖励。当然，RL也可以往过程奖励（PRM）或者规则奖励（rule-based）去走。往过程奖励走，无非是引入一个sub-sentence level的监督信号，介于整句（或者说答案）与单个词之间的监督信息。往规则走，是希望整体系统不要被Reward Model所束缚，如果数据质量足够高+基座足够优秀，那么就不需要花里胡哨的reward形式，直接使用rule-based reward就行。这里的reward大多是（-1、0.2、1）这种三段式设计，本质上和SFT的0/1是差不多的。如果我们对（-1, 0.2, 1）做一次softmax，那么就变成了（0.08, 0.26, 0.64）。从某个视角来说，也算是one-hot label的平滑形式。大家都喜欢说，RL是泛化的，SFT是记忆的。我觉得，之所以造成这种现象，是因为RL学的比较难，所以聚焦于方法论，而SFT学的比较简单，那么就容易掉入过拟合陷阱，也就是SFT记住但无泛化。正是因为RL学的比较难，那么RL的acc涨的是比较慢的，好处但是就是，真的往解题的技巧上学。


[深挖PPO，聊聊前身TRPO](https://zhuanlan.zhihu.com/p/1908671543476749557) PPO的目标函数与Reinforce算法的目标函数还是有一定差异的，尤其是它采用了 $\pi_{\theta_{old}}$ 来采样数据并在目标函数中引入重要性采样 $\frac{\pi_{\theta}}{\pi_{\theta_{old}}}$
 ，这其实并不直观，我们明明推导的是基于蒙特卡洛采样的Policy Gradient，怎么目标函数中有这么多不同的策略网络呢？为什么采样的策略和执行更新的策略不一样呢？为什么长得这么不一样的目标函数也可以算作蒙特卡洛采样呢？这篇文章就要来梳理这个问题，首先从TRPO说起。TRPO(Trust Region Policy Optimization)和PPO(Proximal Policy Optimization)的作者是同一人，TRPO也是PPO的前身工作。

### 补充

一个典型的 PPO 算法流程是这样的：
![](/public/upload/machine/ppo.png)
actor model 的输入是prompt（今天天气怎么样？），输出是response（今天天气很好，适合出去玩。）。reference model的输入是prompt + actor_response，Reference Model 通过前向传播，为每个 token 计算概率分布。假设 Reference Model 的输出概率分布如下：
1. 对于 token "今天"，概率分布为 `[0.4, 0.3, 0.2, 0.1]`，其中 "今天" 的概率为 0.4。
2. 对于 token "天气"，概率分布为 `[0.1, 0.5, 0.3, 0.1]`，其中 "天气" 的概率为 0.5。
3. 对于 token "很好"，概率分布为 `[0.2, 0.3, 0.4, 0.1]`，其中 "很好" 的概率为 0.3。
4. 对于 token "适合"，概率分布为 `[0.3, 0.2, 0.4, 0.1]`，其中 "适合" 的概率为 0.4。
5. 对于 token "出去"，概率分布为 `[0.1, 0.2, 0.6, 0.1]`，其中 "出去" 的概率为 0.6。
6. 对于 token "玩"，概率分布为 `[0.2, 0.3, 0.4, 0.1]`，其中 "玩" 的概率为 0.4。
计算每个 token 的对数概率：
1. "今天" 的对数概率：ln(0.4)
2. "天气" 的对数概率：ln(0.5)
3. "很好" 的对数概率：ln(0.3)
4. "适合" 的对数概率：ln(0.4)
5. "出去" 的对数概率：ln(0.6)
6. "玩" 的对数概率：ln(0.4)
整个回复的参考对数概率为这些对数概率的总和：`ln(0.4) + ln(0.5) + ln(0.3) + ln(0.4) + ln(0.6) + ln(0.4)`

PS： 这个算是回答了 $p_{\theta}(a_t \mid s_t)$ 或 $\pi_{\theta}(a_t \mid s_t)$如何算

## Group Relative Policy Optimization(群体相对策略优化)

![](/public/upload/machine/ppo_grpo.jpg)

图中有以下几个关键点：
1. 没有 Value Model（仅保留 Reward Model 作为监督的依据） 和输出 v（value）
2. 同一个 q 得出了一组的 o（从 1 到 G）
3. 计算 A（Advantage） 的算法从 GAE 变成了 Group Computation
4. KL 散度计算不作用于 Reward Model，而是直接作用于 Policy Model

GRPO舍弃了传统PPO算法中的Critic模型(通常与策略模型大小相同)部分，转而通过直接从群体得分中估算baseline。在训练大语言模型llm时，一个最大的问题是中间状态很难评估（PPO的critic model总是试图精细地预测每个步骤的价值），由于语言生成是一个自回归式的序列决策过程，我们很难直接判断某个中间状态的好坏，——它是否能最终生成一个高质量答案，往往是不确定的。这就带来了一个核心问题：PPO中使用的critic model（即计算价值函数value function 用的模型）到底有没有用？它的意义有多大？准不准？这都是很难确定的。所以，PPO中critic model的意义虽然存在，但它的准确性是个大问题，限制了整体方法的可靠性。相比之下，GRPO采取了一种截然不同的思路：它直接让llm多次sample，生成完整的响应，然后用显式的奖励来评价这些最终结果。正因为一把梭哈，直接生成结果，**完全跳过了对中间状态的评估**，直接聚焦于完整输出的质量。既然中间状态这么难评估，那就干脆不评估，生成结果出来后自然可以通过设计好的奖励机制来判断好坏。这样一来，GRPO省去了预测中间状态价值的麻烦，直接依赖最终输出的奖励信号。更重要的是，这种方式可以通过显式设计reward奖励信号，而且必然是有效的。因为奖励是针对完整响应计算的，这种清晰的反馈比PPO中模糊的价值函数预测要可靠得多。

具体来说，对于每一个问题 $ q $ ，GRPO会从旧的策略模型参数 $  \pi_{\theta_{\text{old}}} $  中采样一组输出 $ \{o_1, o_2, o_3, \ldots, o_G\} $，然后通过最大化GRPO目标函数以优化当前策略模型参数 $ \pi_\theta $。

辅助理解：不同的策略模型 $\pi$ 实际上是一个模型在不同参数阶段下的版本。

具体可以按如下理解，
- $ \pi_{\theta_{\text{old}}} $：上一轮模型参数的模型，可以理解为 $ \pi_\theta $ 上一个iteration的模型。
- $ \pi_\theta $：最新的模型参数的模型（正在更新的）。
- $ \pi_{\theta_{\text{ref}}} $：初始模型参数。

原文公式如下图所示：

$$ 
\mathcal{J}_{\text{GRPO}}(\theta) = E [ q \sim P(Q), \{o_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}(O \mid q) ]
$$ 

$$ 
\frac{1}{G} \sum_{i=1}^G ( \min ( \frac{\pi_\theta(o_i \mid q)}{\pi_{\theta_{\text{old}}}(o_i \mid q)} A_i, \text{clip} ( \frac{\pi_\theta(o_i \mid q)}{\pi_{\theta_{\text{old}}}(o_i \mid q)}, 1-\varepsilon, 1+\varepsilon ) A_i ) - \beta D_{KL} ( \pi_\theta \mid \mid \pi_{\text{ref}} ) ),
$$ 

$$ 
D_{KL} ( \pi_\theta \mid \mid \pi_{\text{ref}} ) = \frac{\pi_{\text{ref}}(o_i \mid q)}{\pi_\theta(o_i \mid q)} - \log \frac{\pi_{\text{ref}}(o_i \mid q)}{\pi_\theta(o_i \mid q)} - 1,
$$ 

$ \varepsilon $ 控制学习步长上限，保证每一轮学习不至于与上一轮偏移过大。主要是防止策略崩溃。

$ \beta $ 控制原始能力偏移惩罚的程度。主要是缓解灾难性遗忘的问题。

[DeepSeek的GRPO算法是什么？ - 梦想成真的回答 - 知乎](
https://www.zhihu.com/question/10766825126/answer/113322446718)最核心的就是$\frac{\pi_\theta(o_i \mid q)}{\pi_{\theta_{\text{old}}}(o_i \mid q)}A_i$
1. q 表示此次的问题
2. $o_i$ 表示旧的policy model 的第i个输出策略
3. $\pi_\theta(o_i \mid q)$ 表示给定问题q，policy model 输出$o_i$的概率
4. $\pi_{\theta_{\text{old}}}(o_i \mid q)$ 表示给定问题q，旧的上一轮梯度下降后的 policy model 输出$o_i$的概率

进而，我们知道了$\frac{\pi_\theta(o_i \mid q)}{\pi_{\theta_{\text{old}}}(o_i \mid q)}$其实就是policy 模型学习过程中的偏移程度，很明显，loss中要最大化$\frac{\pi_\theta(o_i \mid q)}{\pi_{\theta_{\text{old}}}(o_i \mid q)}$，然这不能无上限的 maximize，因为PPO其实是希望模型一点一点学，每一次不能偏移过多，因此clip项就是在做梯度剪裁，KL散度也在进行正则。$\frac{\pi_\theta(o_i \mid q)}{\pi_{\theta_{\text{old}}}(o_i \mid q)}$越大，比如说大于1，那么说明新的策略模型更倾向于输出$o_i$这个策略。

我们再说优势函数 (Advantage Function) $A_i$，R1中非常简单直接，就是reward score 做个标准化，得到的$A_i$就是第i个决策在多个决策中相比较baseline 好多少或者坏多少。如果$A_i$>0，那么就是正向激励，否则，$A_i$< 0 就是负向激励。

通过如下公式计算出：

$$ 
A_i = \frac{r_i - \text{mean} ( \{r_1, r_2, \cdots, r_G\} )}{\text{std} ( \{r_1, r_2, \cdots, r_G\} )}.
$$ 

我们现在可以讲解 $\frac{\pi_\theta(o_i \mid q)}{\pi_{\theta_{\text{old}}}(o_i \mid q)}A_i$ 为啥要把这两项乘到一起？其实原因很简单，$\frac{\pi_\theta(o_i \mid q)}{\pi_{\theta_{\text{old}}}(o_i \mid q)}$就是第i个输出的策略倾向，$A_i$可以理解为一种激励。如果新的策略模型$\pi_\theta(o_i \mid q)$比旧的策略模型$\pi_{\theta_{\text{old}}}(o_i \mid q)$更加希望输出$o_i$ 的策略，并且优势函数$A_i$> 0，也就是获得的reward score好于平均值，那么当然值得鼓励了，所以最大化这一项没有问题。如果$A_i$< 0，说明使用这种策略反倒效果不好，此时$\frac{\pi_\theta(o_i \mid q)}{\pi_{\theta_{\text{old}}}(o_i \mid q)}A_i$是一个负数，最大化负数不就是减少$\frac{\pi_\theta(o_i \mid q)}{\pi_{\theta_{\text{old}}}(o_i \mid q)}$，也就是减少输出 $o_i$的策略的概率吗？对于一堆候选策略，当你做的比别人好，我鼓励你，当你做的比别人差，我尽量减少这种情况再次出现，模型就一步一步朝着更好的优化方向走了。PS：$A_i$ 是Response 粒度的Advantage，**如何转为每个 token 的loss？**计算新策略和旧策略的概率比，将概率比与相对优势相结合$\frac{\pi_\theta(o_i \mid q)}{\pi_{\theta_{\text{old}}}(o_i \mid q)}A_i$，并加上 KL 正则项，算作每个 token 的损失。

![](/public/upload/machine/grpo.jpg)

接下来说clip，$ \text{clip} ( \frac{\pi_\theta(o_i \mid q)}{\pi_{\theta_{\text{old}}}(o_i \mid q)}, 1-\varepsilon, 1+\varepsilon )A_i $，clip( ) 这一项表示把 $ \frac{\pi_\theta(o_i \mid q)}{\pi_{\theta_{\text{old}}}(o_i \mid q)}$ 限制在 $1-\varepsilon$ 到 $1+\varepsilon$ 之间，类似梯度剪裁。$\varepsilon$ 是一个超参数。很简单，GRPO并不希望新的策略模型更新的太快，本身强化学习就不好训练，容易训飞。

GRPO希望 $ D_{KL} ( \pi_\theta \mid \mid \pi_{\text{ref}} ) = \frac{\pi_{\text{ref}}(o_i \mid q)}{\pi_\theta(o_i \mid q)} - \log \frac{\pi_{\text{ref}}(o_i \mid q)}{\pi_\theta(o_i \mid q)} - 1 $ 尽量变小一点。需要注意的是，这并不是标准的KL散度，让 $ D_{KL} ( \pi_\theta \mid \mid \pi_{\text{ref}}) $ 变小就是让 $ \frac{\pi_{\text{ref}}(o_i \mid q)}{\pi_\theta(o_i \mid q)} $ 变小。 $\pi_{\text{ref}}(o_i \mid q) $ 是上一个epoch训练好的 $\pi_\theta(o_i \mid q)$，新的epoch中保持不变，而 $\pi_\theta(o_i \mid q) $ 是新的epoch正在训练的策略模型。这意味着这个kl变体损失项是为了让分子和分母基本上相同，当x=1的时候（参考 $ x-lnx-1 $ 的曲线），kl变体损失项最小为0。

我们稍微总结下GRPO：GRPO去掉了value model，仅仅只训练policy model，并且使用reward score标准化的方式作为baseline，让模型找到更优的解法，为了更稳定的训练，clip项和KL散度变种都是辅佐让模型不要训飞，一步步慢慢学的手段。


[可视化看GRPO学到了哪些东西](https://zhuanlan.zhihu.com/p/1895568864848356926)
1. 训练后的模型推理能力依然极度依赖base model。输出概率的改变大部分都发生在连接词、推理流程影响的词，而在单步公式、数字、推理这些基本能力上，模型输出几乎还是在沿用base模型认为最好的路径，说明这些基础能力几乎完全来自base model。这里还做了一个训练时间太长，已经训练崩溃了的checkpoint的可视化，可以看到一旦离base模型太远，正常输出都没法进行下去了，会出现重复、乱码等情况。同时可以看到，这个崩溃的实验其实还没有完全崩溃，输出的前面一段还是正常的，只是训练到了后期，才开始出现模式坍塌。从这个实验的观察可以推出，RL是不能远离base模型太多的，一旦把模型训练偏得多了一些，模型的输出就会崩溃，中断。导致无法输出最终答案，所有rollout的reward分数都是0，模型就没有正确的方向了。总之，目前的Math RL是强依赖一个非常强大的base model：不同大小的模型，相同算法RL后，提升的上限有很大的不同，RL可以通过激发base模型的长推理模式，充分发挥base模型的潜在推理能力。这意味着，**想要获得最好的推理效果，就必须要在最好的base模型上面训练**。
2. base模型中本来概率很高的token，依然在GRPO中被不断加强，在训练后likelihood变得更接近1，也就是进行了过度自信的优化。这样的优化对模型的生成准确率能力基本是没有帮助的，但是会导致对模型的修改，产生额外的学习税。并且在token的prob都被推高的情况下，也会影响模型的探索的可能，例如prob在比较高的情况下，例如0.99时，在同一个context下，做16次rollout还有15%的可能出现非最高的token，相当于每7个这种token，rollout 16次中会有一个非最高token被探索，但是如果prob推高至0.999，概率就会降低为1.6%。这些token的探索就会几乎不存在。
2. 从模型的演变角度来说，**对一个大模型做RL，是从base/SFT模型作为ref开始，逐渐训练远离这个ref模型，同时获得越来越高的reward的过程**。但是与此同时，RL又不能离ref模型太远，否则轻则无法继续训练、重则模型崩溃，而且离ref模型渐行渐远的过程，模型会忘掉前面学过的知识，降智交智商税。在这两层看似矛盾的要求下，要想RL训练的好，大家提出了很多tricks。

    1. 第一种是让训练过程对ref模型修改的又快又好，这样可以最大限度保留原始模型的能力。
    2. 另一个思路是在模型训练的过程中，持续恢复智商。
    目前证明有效的都是要让训练又快又好的tricks，而对于恢复智商类的tricks目前基本是没有用的。主要原因我认为是这些策略虽然初衷是好的，但是在LLM的训练过程中起到了拖后腿的作用，无脑将模型往回拉。

rollout 是一个强化学习专用词汇，指的是从一个特定的状态按照某个策略进行一系列动作和状态转移，在 LLM 语境下，“某个策略”就是 actor model 的初始状态，“进行一系列动作”指的就是推理，即输入 prompt 输出 response 的过程。[跟着 verl 的代码梳理GRPO](https://www.zhihu.com/question/10766825126/answer/1899519409401357289)

## GSPO

[详解Qwen3-GSPO和DeepSeek-GRPO两大强化学习算法的区别](https://zhuanlan.zhihu.com/p/1932791271363154917)Qwen3 GSPO强化学习方法，把奖励计算从token级别 改成了sequence 级别，解决了LLM 过多的关注token导致训练不稳定的问题。PS：建议细读，文章给的例子非常惊喜。

GRPO 的Loss的计算过程
1. 算优势：每个句子的奖励减去组内平均奖励，得到相对优势（好句子为正，差句子为负），也就是A。
2. 当前模型生成 token 的概率 ÷ 旧模型生成相同 token 的概率（importance ratio），也就是 $\pi_\theta \mid \pi_{\theta_{old}} $
3. 概率比剪辑：把概率比限制在 [0.8, 1.2] 之间，防止更新太激进
4. 算基础损失：对每个 token 计算 概率比*优势=  $\pi_\theta \mid \pi_{\theta_{old} * A}，再加负号。让好句子（优势大）的损失为小，坏句子（优势小）的损失大。PS：token-level的加权优势。
5. 加正则项（可选）：加入 KL 散度，防止模型和初始版本差太远
6. 总损失：Group的所有损失的平均值

GRPO 和 GSPO Loss的计算关键差异

其中最主要的差异在重要性计算那一步，GRPO是计算每个token的概率比

```
log_ratio = per_token_logps - old_per_token_logps  # 每个token的log概率差
log_importance_weights = log_ratio  # 保留token级粒度
coef_1 = torch.exp(log_importance_weights)  # 每个token的概率比
```

而GSPO是计算整个句子的平均概率比

```
log_ratio = per_token_logps - old_per_token_logps  # 每个token的log概率差
# 按句子平均：总log概率差 / 有效token数（避免padding影响）
log_importance_weights = (log_ratio * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)
log_importance_weights = log_importance_weights.unsqueeze(-1)  # 扩展为(batch_size, 1)
coef_1 = torch.exp(log_importance_weights)  # 整个句子的平均概率比
```
GSPO 把奖励，优化，和加权的“尺度”统一了（都是sequence）

GRPO（token 级） 更关注每个 token 的精细优化，适合需要精准控制生成细节的任务（如机器翻译、代码生成），但可能受异常 token 影响较大。GSPO（sequence 级） 通过句子级平均平滑了 token 级波动，训练更稳定，适合以句子整体质量为导向的任务（如摘要生成、问答），但牺牲了部分 token 级优化精度。所以GSPO 还有一个GSPO-token的版本，是为了去解决多轮对话中一些，特殊token的生成效果问题。

## 其它

[强化学习在LLM训练中的作用的思考](https://zhuanlan.zhihu.com/p/1892911327959291306) 建议细看下，不可避免的要与sft 对比。

1. RL方法共享相同的更新原则：即通过对目标函数进行梯度更新来调整模型参数。最简单的形式表示为：$\theta \leftarrow \theta + \alpha\nabla J$。其中 $\theta$代表模型参数，$\alpha$是学习率，$\nabla J$是目标（通常是期望奖励）的梯度。然而，这个梯度的计算方式以及包含哪些项在不同方法之间可能有很大差异。
2. 近端策略优化(PPO)是一种策略梯度方法，它在优化目标的同时，确保策略更新与之前的策略保持"近端"。它通过计算概率比率来实现：$r(\theta) =\frac{\pi_\theta(a \mid s)}{\pi_{\theta_{\text{old}}}(a \mid s)}$，这个比率然后乘以优势估计（通常使用广义优势估计或GAE计算），并应用裁剪操作以防止更新过于偏离旧策略。由此对精心设计的目标进行更新，使得策略变化较大时提供稳定性。
3. 人类反馈的强化学习（RLHF）就是在PPO方法的基础上集成了人类偏好数据的一种方法。首先使用人类评注者提供的成对的比较或评分来训练奖励模型。随后的RL阶段使用这个奖励信号来优化模型，通常将其与PPO的技术如裁剪和KL散度惩罚结合起来，以确保渐进的更新。
4. DeepSeek-R1的GRPO进一步修改了这一思想，消除了对之前PPO单独价值函数的使用。不依赖于状态价值的外部估计，而是就每个提示词生成一组回复，标准化得到的奖励分之后来计算群体相对优势，简化了架构并减少了计算开销，同时仍能捕获组内回复的差异性。



