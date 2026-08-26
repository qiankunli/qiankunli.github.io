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

如果你想先建立整体框架，建议先看[从长期回报、Credit Assignment 到 PPO]({% post_url 2026-04-06-rl_mainline %})。这篇保留细节、公式和推导，主线篇则以问题主线和演化关系为主。经典书 https://github.com/walkinglabs/hands-on-modern-rl


这篇虽然会多次借用 LLM / token 生成来帮助理解，但它讨论的仍然是**通用 RL 细节**，重点在策略梯度、Advantage、Actor-Critic、GAE、PPO 这些方法本身。如果你更想专门看 RL 放到 LLM post-train 之后的对应关系，可以再读[LLM Post-Train 中的强化学习：PPO、GRPO 与 GSPO]({% post_url 2025-02-18-llm_rl %})。

PS：单单是rl 强化学习本身的推导就是一门很厚的知识，搞明白actor-critic 就有百分之七八十了，之后是ppo 针对actor-critic 的一些调整，可以直接看[PPO理论推导+代码实战](https://zhuanlan.zhihu.com/p/13467768873)

这篇保留的是细节、推导和 LLM / PPO / GAE 相关内容。如果你想先看主线，比如 Agent / Environment / MDP / Bellman / MC / TD / value-based / policy-based 是怎么串起来的，可以先读[从长期回报、Credit Assignment 到 PPO]({% post_url 2026-04-06-rl_mainline %})。


一些基础概念：
- $\pi$（Policy，策略）：即LLM模型
- $\theta$（Parameter，参数）：即模型参数
- s（State，交互状态）：即上文，初始状态即为$s_1$
- a（Action，交互行为）：即输出的token，可以简单理解为每个字符。（实际上一个字不等于一个token）
- $\tau$（Trajectory，轨迹）：$\tau = \{ s_1,a_1,r_1,s_2,a_2,r_2,...,s_T,a_T,r_T \}$

## 换个视角：RL 本质上是一个 gradient estimator

不管是 backprop 还是 RL，它们的最终目的都是修改 DNN 的 parameters $\theta$。先厘清一件事：**backprop 本身不是一种 learning algorithm，它的作用其实是求导（differentiation）**。给定一个可微的 loss $L(\theta)$，backprop 通过 chain rule 算出 $\nabla_\theta L$，然后我们把这个 gradient 交给 optimizer（SGD、Adam）去更新 $\theta$。

在监督学习（比如 CNN）里，这个流程非常直观，因为 data 和 $\theta$ 无关——不管训练时 params 怎么变，图片本身不会变，所以我们可以直接对着固定的 dataset 做 backprop。

而 RL 的目标函数是 $J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)]$，我们想最大化策略 $\pi_\theta$ 所能 sample 到的 trajectory 的 average returns。看起来也是"算个目标、对 $\theta$ 求导、交给 optimizer"，但真要动手求导，会撞上两道监督学习里根本不存在的坎。

1. 数据分布本身随 $\theta$ 一起动（监督学习里数据是钉死的）。把两边的目标摊开对比就清楚了。监督学习的 loss 长这样：

    $$L(\theta) = \frac{1}{N}\sum_{i=1}^{N} \ell\big(f_\theta(x_i),\, y_i\big)$$

    这里的数据 $(x_i, y_i)$ 来自一个**固定的、和 $\theta$ 无关**的 dataset。所以求梯度时，求和号外面那一圈样本纹丝不动，梯度只落在里面的 $f_\theta$ 上：$\nabla_\theta L = \frac{1}{N}\sum_i \nabla_\theta\, \ell\big(f_\theta(x_i), y_i\big)$。一句话——对着一批不变的题，只调整你的答题函数。

    而 RL 的目标，把期望按定义展开，其实是对所有可能的轨迹求和：

    $$J(\theta) = \sum_{\tau} P(\tau;\theta)\, R(\tau)$$

    关键在于轨迹出现的概率 $P(\tau;\theta)$ 里带着 $\theta$。逻辑链路是：$\theta$ 变 → 策略 $\pi_\theta$ 变 → 每一步采的 action 变 → 环境随之给出的下一个 state 也变 → 整条轨迹 $\tau$ 的分布 $P(\tau;\theta)$ 跟着变。也就是说，**你"训练数据"（采样到的轨迹）的分布，随着你优化 $\theta$ 一起在漂**。这就是后面会反复出现的 on-policy 困境：上一轮采的数据，参数一更新就"过期"了。落到求导上，这一项绕不过去——你必须对 $P(\tau;\theta)$ 本身求导，而不能像监督学习那样把数据分布当常数提到求和号外面。

2. 采样这一步不可导，链式法则在这里断了。

    就算我们下决心去对 $P(\tau;\theta)$ 求导，还有第二道坎。策略网络的实际工作方式是：forward 先算出一个动作的概率分布（比如 logits 过 softmax），然后**从这个分布里"掷骰子"抽一个具体的 action** $a \sim \pi_\theta(\cdot\mid s)$。（顺带一提，**softmax 有平移不变性——logit 的绝对高度无意义，只有 logit 之间的差值有意义**：所有 logit 同时加一个常数，$\text{softmax}$ 出来的分布完全不变。这也是常说"模型权重的绝对大小意义不大、相对值才重要"的一个来源。）

    监督学习的前向过程，从头到尾都是矩阵乘、激活函数这类**可微算子**的堆叠，所以 chain rule 能一路从 loss 流回每个参数。**但"从分布里采样"这一步不是可微算子**：它的输入是一组概率，输出是一个被随机抽中的离散动作。你把 $\theta$ 挪动一丁点，这个被抽中的动作**要么不变、要么突然跳到另一个值**，中间没有平滑的 $\frac{\partial a}{\partial \theta}$ 可言。形象点说，采样就像在计算图正中间插了个 `np.random.choice`，梯度流到这儿就断了，没法像监督学习一样直接 backprop 穿过去。

所以 RL 真正要解决的是：**如何去 estimate $\nabla_\theta J(\theta)$？这就是 policy gradient 的由来。**解法是一个很巧的数学技巧——log-derivative（似然比）技巧。它基于一个对任意可微概率 $P$ 都成立的恒等式（来自 $\nabla_\theta \log P = \frac{\nabla_\theta P}{P}$ 这条求导链式法则的反用）：

$$\nabla_\theta P(\tau;\theta) = P(\tau;\theta)\,\nabla_\theta \log P(\tau;\theta)$$

把它代进 $\nabla_\theta J(\theta) = \sum_\tau \nabla_\theta P(\tau;\theta)\, R(\tau)$：

$$\nabla_\theta J(\theta) = \sum_\tau P(\tau;\theta)\,\nabla_\theta \log P(\tau;\theta)\, R(\tau) = \mathbb{E}_{\tau \sim \pi_\theta}\big[R(\tau)\,\nabla_\theta \log P(\tau;\theta)\big]$$

这一步同时拆掉了上面两道坎：

- **拆第一道坎**：右边重新变回了一个"对 $\pi_\theta$ 的期望"，于是可以**用采样去估计**——拿当前策略实际跑出来的几条轨迹求平均，就是对这个期望的无偏估计，不必再纠结"分布在漂"。
- **拆第二道坎**：再把 $\log P(\tau;\theta)$ 展开，一条轨迹的概率 = 环境的状态转移概率 × 策略每步的动作概率，而**环境转移概率不含 $\theta$、求导后整项消失**，只剩下 $\sum_t \nabla_\theta \log \pi_\theta(a_t\mid s_t)$。这一项是策略网络输出的 log 概率，**完全可微**，backprop 能正常穿过去——我们绕开了"对采样本身求导"，改成对"采到的动作的 log 概率"求导。

于是就得到著名的 REINFORCE：$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}[R(\tau)\nabla_\theta \log \pi_\theta(\tau)]$，通过 sampling 若干 trajectories 来 estimate 这个 expected value，得到一个关于 gradient 的“近似”。直觉上它在干一件很朴素的事：**对你真正采到的那些动作，按它最终带来的回报 $R$ 的好坏，去调高或调低它们的 log 概率**——好动作让它以后更容易被采到，差动作压下去。这也正是"用 A 来表示你有多希望 s 时采取动作 a"的数学出处：公式里的 $R(\tau)$ 就是那个权重 A，A 估得越准，这个梯度就越靠谱。

说清楚这一层，关系就明确了：在监督学习里，backprop 直接精确地给出 gradient；但在 RL 里，我们没法直接求这个 gradient，所以需要 policy gradient 这类方法先 estimate 出一个 gradient，然后把这个近似值交给**和监督学习完全一样的 optimizer（Adam）**去更新 $\theta$。换句话说，**RL 的作用本质上是一个结构复杂的 gradient estimator**，给 optimizer 提供优化信息；而它的那些子问题（credit assignment、exploration、variance reduction）全部是为了让最终输出的那个 gradient vector 更准。

## 有偏与无偏：RL 估计里的两个维度

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

所以“短任务爱用 GRPO、长任务回 critic”，本质不是谁取代谁，而是 bias-variance 天平随 horizon 换了落点。这个“天平”的完整版（GRPO/PPO 在长短程下的机制差异）见 [LLM Post-Train 中的强化学习]({% post_url 2025-02-18-llm_rl %})；通用 RL 的 critic / GAE 推导见下文。

## Policy Gradient（策略梯度）

李宏毅老师对Policy Gradient的讲解，最原始的Policy Gradient 直接$A_t=G_t$

![](/public/upload/machine/rl_lhy_policy_gradient.jpg)

[The Definitive Guide to Policy Gradients in Deep ReinforcementLearning:Theory, Algorithms and Implementations](https://arxiv.org/pdf/2401.13662) Policy Gradient 论文综述。


[DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://zhuanlan.zhihu.com/p/20530204146)

Reward Function（奖励函数定义，即输出序列$\tau$ 能获得的奖励），用于评估某个状态/**动作序列**的好坏：

$$
R(\tau) = \sum_{t=1}^{T} r_t
$$

因为actor输出有一定的随机性，即对与同一个$s_t$，actor不一定每次都输出$a_t$，自然each $\tau$ has a probability to be sampled（PS：采样出来的数据分布不一定是真实的数据分布）. 这个probability用$p_\theta(\tau)$ 或 $p(\tau \mid \theta)$ 表示，连带$R(\tau)$也是随机的（所以不能单纯用标量，要计算期望），所以模型参数 $\theta$ 下的Expected Reward（期望奖励）表示为sum over all possible trajectory：

$$
\overline{R}_\theta = \sum_{\tau} R(\tau) p_\theta(\tau)
$$

综上，我们希望调整模型参数 $ \theta $ 使这个期望奖励越大越好，因此可得Policy Gradient公式如下，期望做gradient ascent最大化期望奖励：

$$
\nabla \overline{R}_\theta = \sum_{\tau} R(\tau) \nabla p_\theta(\tau)
$$

其中 $R(\tau)$ 来自environment 反馈（it can even be a black box），跟$\theta $没关系，所以做gradient的时候只对$p_\theta(\tau)$ 做gradient即可。

**我们分别来看 $R(\tau)$ 和 $p_\theta(\tau)$ 可以被约等为什么样子**。

### 从轨迹奖励到action得失

$$
\nabla \overline{R}_\theta = \sum_{\tau} R(\tau) \nabla p_\theta(\tau)
= \sum_{\tau} R(\tau)  p_\theta(\tau) \frac {\nabla p_\theta(\tau)}{p_\theta(\tau)}

= \sum_{\tau} R(\tau) p_\theta(\tau) \nabla \log p_\theta(\tau) \quad \text{\# Note: } \nabla f(x) = f(x) \nabla \log f(x)
$$

![](/public/upload/machine/rl_rt.jpg)

直接对$p_\theta(\tau)$ 求导无法计算，$R(\tau) p_\theta(\tau)$是期望形式，可以通过采样足够多的轨迹来估计。

$$
\sum_{\tau} R(\tau) p_\theta(\tau) \nabla \log p_\theta(\tau) \approx \frac{1}{N} \sum_{n=1}^{N} R(\tau^n) \nabla \log p_\theta(\tau^n) \quad \text{\# 实际上就是N个sample轨迹近似期望}
$$

接下来的问题是如何计算 $\nabla \log p_\theta(\tau^n)$， 其中，模型参数$\theta$ 下生成序列$\tau$ 的概率如下：

$$
p(\tau \mid \theta) = p_\theta(\tau) = p(s_1) p_\theta(a_1|s_1) p(s_2|s_1, a_1) \ldots = p(s_1) \prod_{t=1}^{T} p_\theta(a_t|s_t) p(s_{t+1}|s_t, a_t)
$$

轨迹概率可以分解为动作条件概率的连乘，对数概率可以拆成每一步的 log 概率之和。

![](/public/upload/machine/rl_pt1.jpg)

忽略掉跟 $\theta$ 无关的项（ 环境无法作用gradient 所以可以移除）

![](/public/upload/machine/rl_pt2.jpg)


$$
\nabla \overline{R}_\theta = \frac{1}{N} \sum_{n=1}^{N} \sum_{t=1}^{T_n} R(\tau^n) \nabla \log p_\theta(a_t^n | s_t^n) 
$$

$$
\nabla \log p_\theta(a_t \mid s_t)
= \frac{\nabla p_\theta(a_t \mid s_t)}{p_\theta(a_t \mid s_t)}
$$

$\nabla p_\theta(\tau)$ 转换为 $\nabla p_\theta(a_t^n | s_t^n)$ 之后可以通过实际采样到的 (s, a) 对来计算。

其中(用到了对数求导)，分母$p_\theta(a_t \mid s_t)$ 体现了重要性比重：小概率但被采样到的动作，对梯度更新影响会更大（因为这说明策略应该更重视它）。避免了在采样过程中采到很多奖励值很低但是出现频次高的动作，造成模型对这种低奖励值高频次动作的偏好。 整体而言，相当于用$p_{\theta}(a_t \mid s_t)$做了某种归一化。

![](/public/upload/machine/rl_pg.jpg)

直观理解：在某个state（上文）下执行某个action（token）使得最后整个输出$ \tau$ 的reward是正的时候，我们应该**增加这个输出的几率**，反之减少。each training data is weighted by $R(\tau^n). $ PS：rl就是，判断哪个输出更好，把这个输出的概率提高，花活在于提高多少。可以给action model 每个token算loss 了。进而**是不是可以理解为rlhf和sft的反馈粒度都是token？**

**以上是REINFORCE 算法的核心更新公式，在此基础上，可以引入 baseline（不影响期望，减少方差）、优势函数（A3C, PPO 等方法），改进收敛效果**。

### $R(\tau)$可以被约等为什么样子

如果仔细看上述公式，首先，它并没有告诉我们轨迹中某个单独的动作到底好不好，其次会发现 $ R(\tau) $ 即reward恒为正的情况，那会导致一直在增加任何token的输出概率。我们希望调小概率时，reward应该是个负的。直观的想法，如果我见过一批reward，减去他们的均值就好了，这个方法叫REINFORCE with baseline，让reward有正有负，这对rl的训练效率至关重要。这个方法搞出来的均值baseline不一定是最好的，我们想知道的是此刻状态对应的价值，均值没有这样的物理意义，是不是可以用一个NN模型来预估出这个价值呢？这就是Actor-Critic方法。Actor就是你的动作概率模型，Critic就是用一个NN在算这个baseline，或者我们叫他value-base的model。如果不用一个model来估计状态的价值，还有什么好办法？那你就基于每一条原始样本生成一组序列，用他们的reward均值作为baseline，这种方法叫self-critic，它利用了蒙特卡洛方法代替了TD error。

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

## actor-critic

[PPO理论推导+代码实战](https://zhuanlan.zhihu.com/p/13467768873) 建议细读。

Actor-Critic架构为什么要有Critic呢？这就涉及强化学习的算法稳定性问题。与监督学习（SL）相比，RL实际上是很难稳定的一类训练机制。大致的原因如下：
1. RL本身是处理动态系统的最优控制问题，而SL是处理一个静态优化问题。动，就比静更难处理。
2. 加上RL的数据非稳态，Env-agent交互机制的数据采集量少，这使得梯度计算的方差更大，方差一大就容易偏离预期目标，算法就容易跑飞了。主流的强化学习算法是怎么解决这一问题的呢？加上Critic，使用State-value function或者Action-value function稳定策略梯度的计算过程。更高级一些的算法是采用Advantage Function，也就是加上了Baseline，增加梯度计算的稳定性。这是AC算法总是优于REINFORCE算法的原因之一。
3. 如果没有Critic，PPO只能使用蒙特卡洛的完整轨迹回报（高方差）或单纯依赖即时奖励（短视），导致训练低效甚至失败。Critic的引入使得PPO能通过时序差分（TD）学习高效地估计价值，平衡偏差与方差。
PS： 配套的**actor-critic 架构就得有actor model 和critic model**


### 为何引入critic

A用 $A=G_t$, $A=G_t^{\prime}$,$A=G_t^{\prime}-b$（此处b 是一个恒定值） 这类公式直接算不太靠谱， **所以就想着用一个network 来估计baseline**，即$V^\theta(s)$。它是一个Value function（也就是critic network），输入是s（注意不是`(s,a)`），输出是一个scalar，表示针对 actor $\theta$，the discounted cumulated reward expects to be obtained after seeing s。PS: 就好比仅看当前棋局（不是判断走下一步）就给出输赢的概率。

critic 如何用在训练actor上？将critic scalar/score作为baseline。A用 $G_t^{\prime} - V^\theta(s)$来表示。

![](/public/upload/machine/rl_version351.jpg)

$V^\theta(s)$ 可以认为看到s后所有动作a带来的return均值

![](/public/upload/machine/rl_version352.jpg)

这里有个问题，$G_t^{\prime}$ 只是s 采取动作$a_t$ 后某一个动作序列(sample)的return，可能这个sample 特别好或特别坏，所以不能充分反应$a_t$ 的好坏，因此把$G_t^{\prime}$ 换一下，用平均减平均，A用 $r_t + V^\theta(s_{t+1}) - V^\theta(s)$来表示。也就是用 advantage 的 Actor-Critic（习惯写作 A2C，Advantage Actor-Critic）。这里要避免一个常见误称：A3C 的第一个 A 是 Asynchronous，指的是多个 worker 异步并行地采样和更新，是一种工程上的并行架构——跟这里这个单步优势公式没有关系，套不套异步是另一回事。

![](/public/upload/machine/rl_version4.jpg)

![](/public/upload/machine/rl_ac.jpg)

actor 和critic都是一个network，他们输入都是一样的，都要理解s（棋谱、游戏、llm），actor 输出action，critic 输出scalar，它们network的前大半一般是一样的。

reward shaping。到目前，rl 的过程就是收集一系列`<s,a,r>`，对r进行整理后得到一系列`<s,a,A>`，之后就可以训练actor。但我们特别担心一个情况（sparse reward问题）：大多数时候$r_t$都是0（人生很多时候何尝不是这样）。此时要（除了env真正的reward之外）提供一些额外的reward（如何定义reward 需要domain knowledge）。就好比孩子study原本env $r_{t+1} = -1$ 孩子不开心，你通过给他一个棒棒糖，改变了study的reward。

![](/public/upload/machine/rl_reward_shaping.jpg)

PS：此时每个$a_t$的$A_t$不同了，$A_t = R_t - V(s_t)$


**把前面两种 advantage 写法连起来：它们其实是同一个东西的两端。** 先把记号补全：动作价值 $Q(s,a)$ = 在状态 $s$ 选了动作 $a$ 之后的期望回报；状态价值 $V(s)$ = 在 $s$ 下不指定动作时的平均期望回报。优势函数就是两者之差：

$$A(s,a) = Q(s,a) - V(s)$$

含义是“选这个动作”比“这个状态的平均水平”好多少。再由 Bellman，$Q(s,a) = E[\,r + \gamma V(s')\,]$，代进去：

$$A(s,a) = \underbrace{r + \gamma V(s') - V(s)}_{=\ \delta_t,\ \text{即 TD error}}$$

也就是说，**上面那个一步形式 $r+\gamma V(s')-V(s)$ 并不是另起炉灶，它就是单步优势，而且恰好等于 TD 误差 $\delta_t$——这是个恒等式，不是巧合**（严格讲 $A=E[\delta_t]$，单次采样就用 $\delta_t$ 这一个样本去估它）。这顺带解释了一件容易绕晕的事：训练 critic 用的 $\delta_t$，和更新 actor 用的单步 advantage，为什么长得一模一样——它们本就是同一个量，只是一个拿去回归 $V$、一个拿去加权 $\nabla\log\pi$。

于是前面两种 advantage 落在同一根轴的两端：

- $A = G_t^{\prime} - V(s)$：用真实采样的整条 return $G_t^{\prime}$ → **蒙特卡洛端，无偏但高方差**；
- $A = r + \gamma V(s') - V(s) = \delta_t$：只看一步、其余全靠 $V$ 估计 → **TD 端，低方差但有偏**（偏就偏在 bootstrap 的那个还不准的 $V$ 上）。

这正是上篇「有偏与无偏」那条 bias-variance 轴。而下一节的 GAE，就是拿一个 $\lambda$ 在这条 MC↔TD 轴上滑动、取一个折中。

### GAE
GAE是在 Actor-Critic 里，继续优化 Advantage 的估计。

TD偏差大但方差小，Monte Carlo偏差小但方差大。GAE 把多步 TD 残差按权重叠起来，在 bias / variance 之间调平衡，目的是在估计优势函数时，降低方差、同时控制偏差。PS：在只有终点奖励（如 0/1）的情况下，稳定、高效地把最终奖励“分摊”到每一个 token 的决策上。在 PPO 中，真正用来更新 policy 的不是 reward，而是 Advantage。如何估计每个action的$A_t$。

先定义一步 TD error：

$$
\delta_t = r_t + \gamma V_{\pi}(s_{t+1}) - V_{\pi}(s_t)
$$

GAE 不把单个 $\delta_t$ 直接当作最终结果，而是把从当前位置开始的多步 TD error 按指数权重累加：

$$
\hat A_t^{GAE(\gamma,\lambda)} = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}
$$

$$
\delta_{t+l} = r_{t+l} + \gamma V_{\pi}(s_{t+l+1}) - V_{\pi}(s_{t+l})
$$

当 $\lambda=0$ 时，$\hat A_t^{GAE}$ 退化成一步 TD error：$r_t + \gamma V_{\pi}(s_{t+1}) - V_{\pi}(s_t)$。

有限 episode 中，终止之后的 $\delta$ 视为 0。当 $\lambda=1$ 且终止状态的价值取 0 时，TD error 的加权和会望远镜式消去中间的 $V$，得到蒙特卡洛形式 $G_t-V_{\pi}(s_t)$。因此 $\lambda$ 控制 GAE 在一步 TD 与完整 return 之间的 bias-variance 折中。记这种优势估计为 $A_t^{GAE}(s_t,a_t)$，新的策略梯度调整为：

$$
\nabla J(\pi_{\theta}) = \underset{\tau \sim \pi_{\theta_{old}}}{E_t} \left[ \frac{\pi_{\theta}(a_t | s_t)}{\pi_{old}(a_t | s_t)} A_{\pi}^{GAE}(s_t, a_t) \nabla log \pi_{\theta}(a_t | s_t) \right]
$$
由于 $\nabla f(x) = f(x) \nabla log f(x)$，上式可以改写为：
$$
\nabla J(\pi_{\theta}) = \underset{\tau \sim \pi_{\theta_{old}}}{E_t} \left[ \frac{\nabla \pi_{\theta}(a_t | s_t)}{\pi_{old}(a_t | s_t)} A_{\pi}^{GAE}(s_t, a_t) \right]
$$
我们的优化目标变成：
$$
\arg\max_{\pi_{\theta}} J(\pi_{\theta}) = \underset{\tau \sim \pi_{\theta_{old}}}{E_t} \left[ \frac{\pi_{\theta}(a_t | s_t)}{\pi_{\theta old}(a_t | s_t)} A_{\pi}^{GAE}(s_t, a_t) \right]
$$

### 如何学习critic/critic_loss 演化过程

critic模型可以提供可靠的token级别的中间价值估计(critic用来计算$v(s_t)$的，作为一个baseline)，在RLHF场景，因为PRM比较难，那么存在下面的计算reward的方式：
1. 对于t如果是中间token，那么其通常r=0（或者为了防止模型崩坏加的微小 KL 散度惩罚）。
2. 对于t是结尾的token，那么其通常等于奖励模型打出的得分。

critic模型的另外一个作用是：可以将最后的 Reward 分配到中间步骤（Token-level）。他是如何做到的呢？假设已经有了一个训练好的critic模型，它可以判断当前已经生成的句子的好坏程度。对于中间token的优势，上文提到我们有GAE来计算：假设简略版本的GAE是这
$$ A_t \approx r_t + \gamma V_{\pi}(s_{t+1}) - V_{\pi}(s_t) $$

由于中间token的奖励是0（或者微小的负的kl散度），$\gamma =1$，$0 < \lambda < 1$。那么

$$ A_t \approx V_{\pi}(s_{t+1}) - V_{\pi}(s_t) $$

因此一个好的critic模型可以：可以将最后的 Reward 分配到中间步骤（Token-level）

说如何计算/训练$V^\theta(s)$？
1. Monte-Carlo(MC) based approach。 $V^\theta(s)$ 与 $G_t^{\prime}$ 越接近越好。**约束是你得拿到完整的episode**（比如你得玩完整场游戏，因为要算$G_t$）。
   ![](/public/upload/machine/rl_mc.jpg)
2. Temporal-difference(TD) approach。$V^{\theta}(s_t) - \gamma V^{\theta}(s_{t+1}) $与$r_t$越接近越好。适合无法拿到完整的episode的场景。
   ![](/public/upload/machine/rl_td.jpg)

[PPO理论推导+代码实战](https://zhuanlan.zhihu.com/p/13467768873) 不如这里详细。

关于critic_loss 第一想法是：
$critic\_loss =(R_t + \gamma * V_{t+1} - V_{t})^2 $

最小化TD Error（TD误差就是预测误差：实际观察到的和预测的差距）就可以训练一个预测状态价值的Critic model，critic优化是：

$$
\arg \min_{V_{\pi}} L(V_{\pi}) = E_t[(r_t + \gamma V_{\pi}(s_{t+1} - V_{\pi}(s_t))^2)]
$$

Reward Model提供环境的基础反馈信号（即$r_t$），是Critic学习的输入。Critic Model将即时奖励转化为长期价值估计，指导策略优化方向。
1. 只有Reward Model：只能提供即时奖励信号（如生成完整句子后的总分数），但无法评估每个 token 或部分响应的长期价值，使策略更倾向于选择长期回报更高的动作。
2. 只有Critic Model：若奖励函数未知（如逆强化学习），Critic无法凭空学习价值函数。
在PPO中，Reward Model提供基础的真实反馈，而Critic Model将其转化为长期价值估计，两者缺一不可。



## online/offline policy

在强化学习中，策略可以根据它们与数据生成策略的关系被分类为 on-policy 或 off-policy。这两种方法在处理经验数据和更新策略时有所不同，
1. On-policy 方法直接从目标策略（即当前学习和评估的策略）中采样数据，它要求学习算法和行为策略是一致的，即生成数据的策略必须是当前优化的策略。让模型自己生成轨迹，然后给个评分，比如用当前模型对一批问题生成回答，然后根据这些回答的质量（奖励）来调整模型参数，让它下次说得更好。这里的数据和模型是 “同步” 的 —— 数据来自 “现在的模型”，优化的也是 “现在的模型”。the agent learned and the agent interacting with the environment is the same.PS： 就好比帅哥追美女的某些招式/经验对普男反而有害。
    ![](/public/upload/machine/rl_policy_gradient.jpg)
    也就是在传统supervisor learning中，training data都是事先准备好的，无论跑多少次Epoch 都是一批training data，而对与rl 来说，参数要更新多少次，training data 就要更新多少次，用本轮的$\theta$ 输出的`<s,a>` 计算`<s,a,A>`来更新$\theta$，**体现在代码上就是training data产生在for循环之内**，也是为何rl比较耗时的原因。

    
    当前策略 $\pi_\theta$-> 生成rollout→计算奖励→更新参数$\pi_{\theta^\prime}$ →新策略→生成新rollout→ …
    
2. Off-policy 方法允许从与目标策略不同的行为策略中采样数据（找外面现成的答案，让模型学着模仿）。比如训练时用的数据不是当前模型生成的，而是来自 “别人”（比如人类专家写的回答、更强的模型生成的回答，或者过去版本的模型留下的回答）。虽然这些回答不是当前模型自己说的，但可以帮它更快学到正确的模式。the agent learned and the agent interacting with the environment is different.

on-policy好比自己走万里路，自己从自己的经验中学习，但问题在于，反馈太稀疏了。无论轨迹多长，一个训练回合只给一个评分。比如模型生成了 1000 个 token 的解题过程，最后告诉你错了，但到底是哪一步错了？是运算顺序错了，还是算术本身错了？不知道。这种稀疏反馈让 RL 效率很低。off-policy 类似于从别人的经验中学习，最常见的做法就是 SFT，用精心标注的数据集训练。这些数据可以来自表现很好的老师模型。但这有个问题，学生学到的是老师常遇到的上下文，不是自己会遇到的上下文。一旦学生早期犯了一个老师不会犯的错误，后面的状态就跟训练时看到的越来越远，这就是 exposure bias。这个问题在长序列上尤其严重。还有个问题，学生可能学到了老师的风格和自信，但不一定学到了老师的准确性。这就像学生抄了老师的解题步骤，但不懂为什么这么解，换道题就懵了。

on-policy 和 off-policy 的区别，主要体现在对数据的使用上，off-policy 的训练效率会明显更高一些。一方面，off-policy 可以不等所有的 response 生成完毕，就启动模型训练；另一方面，off-policy 可以多次使用同一条数据，提高数据利用率，on-policy 则不可以。 off-policy 的模型快速熵坍缩（ on-policy缓慢熵坍缩）。防止熵坍缩：加入熵 loss 和 clip higher 。

![](/public/upload/machine/on_off_policy.png)


PPO叫Proximal Policy Optimization，就是揉和了online/offline policy。actor to train has to know its different from the actor to interact（产生training data的actor）.

Exploration,采集trainning data时可以给actor 加一些随机性，不必每次都是`s1 => actor ==> a1`。

![](/public/upload/machine/rl_exploration.jpg)

### 重要性采样

先不说rl，前文提到，我们可以通过足够的采样的均值来近似一个分布的期望。 
$$
E_{x \sim p}[ f(x)] \approx \frac{1}{N} \sum_{i=1}^{N} f(x^i)
$$


当我们有两个分布$p(x)$和$q(x)$，但是又无法直接从 $p(x)$采样，但可以从$q(x)$采样时，我们可以这么描述$x \sim p(x)$下 
的期望：

$$
E_{x \sim p}[f(x)] = \int f(x)p(x) dx = 
\int  f(x)  \frac{p(x)}{q(x)} q(x) dx = E_{x \sim q}[\frac{p(x)}{q(x)} f(x)]
$$

注意从 $E_{x \sim p}$ 换成了 $ E_{x \sim q}$，通过一个 权重修正，把在 q(x) 下采样的数据“重加权”为好像来自 p(x) 的数据。其中
$w(x) = \frac{p(x)}{q(x)}$
就叫重要性权重（importance weight），表示在 q(x) 下采样到的数据，并不都“同等重要”地代表目标分布 p(x)。

重要性权重 w(x) 调整了哪些样本更“重要”。如果某个样本在目标分布 p(x) 里比在行为分布 q(x) 更可能出现（即 p/q > 1），那它就被赋予更高的权重；反，如果 p/q < 1，它就被减弱。这个过程就叫 Importance Sampling（ IS,按重要性来采样/加权）。

套一下上面的公式

$$
\nabla \overline{R}_\theta = E_{{\tau \sim p_\theta(\tau)}} [ R(\tau) \nabla \log p_\theta(\tau)] = 

E_{\tau \sim p_{\theta'}(\tau)} [\frac{p_{\theta}(\tau)}{p_{\theta'}(\tau)} R(\tau) \nabla \log p_\theta(\tau)]
$$

$ratio = \frac{p_{\theta}(\tau)}{p_{\theta'}(\tau)}$就是 importance weight。

具体到action 粒度

![](/public/upload/machine/rl_imptance_sample.jpg)

在实践中，我们为了降低采样成本，提升训练效率（采样是训练所需的，主要是不想对新的策略采样到的轨迹再计算奖励和优势），我们希望对得到的一批“经验”进行多次训练，过程如下：
1. 假设某次更新完毕后，我们得到策略 $\pi_{old}$
2. 我们用$\pi_{old}$和环境交互，得到一批经验数据（主要是状态价值、优势、回报）。
3. 我们将把这一批回合数据重复使用k次：即我们先把这批数据喂给 $\pi_{old}$，更新得到$\pi_{\theta_0}$，我们再把同一批数据喂给$\pi_{\theta_0}$，更新得到$\pi_{\theta_1}$；以此类推，做k次更新后，我们得到$\pi_{\theta}$。
4. 我们管这个过程叫off-policy（产出数据的策略和用这批数据做更新的策略不是同一个）。
5. 在这k次更新后，我们令$\pi_{old} = \pi_{\theta}$。重复上面的过程，直到达到设定的停止条件为止。

但是在我们训练的过程中，由于策略已经发生了改变，采样出来的分布已经变了据此我们应该将新的策略梯度调整为：

$$
\nabla J(\pi_{\theta}) = \underset{\tau \sim \pi_{\theta_{old}}}{E_t} [ \frac{\pi_{\theta}(a_t | s_t)}{\pi_{old}(a_t | s_t)} A_{\pi}(s_t, a_t) \nabla log \pi_{\theta}(a_t | s_t) ]
$$

**这里要澄清一个容易自相矛盾的说法。** 前面说 off-policy 是“用别人/旧策略产生的数据”（人类专家、更强模型、很旧的历史策略），可上面又把“同一批数据复用 K 次”也叫 off-policy，听着打架。其实 on/off-policy 不是非黑即白，而是一条谱，关键看：**产生数据的行为策略，离当前要更新的目标策略有多远。**

- PPO 复用的那批数据来自 $\pi_{old}$，而 $\pi_{old}$ 只是“几步梯度之前的自己”，分布只漂了一点点。所以 PPO 本质上仍是**近似 on-policy**：importance ratio $\frac{\pi_\theta}{\pi_{old}}$ 只是把这点小漂移修正回来，clip 则把漂移摁在一个小区间里、不让它越漂越远。正因如此，PPO 通常被归为 on-policy（或 near-on-policy），而不是真正的 off-policy。
- **真正的 off-policy**，是行为策略可以离目标策略任意远的那种——replay buffer、人类示范、别的模型的输出（也就是前文那个定义）。这时分布差异大，要么靠完整的重要性采样硬扛，要么干脆换一套机制（如 Q-learning / DQN 的经验回放）。

一句话：“复用 K 次”只让 PPO **轻微**偏离 on-policy，而 clip 的作用是限制 surrogate objective 继续奖励过大的有利变化。它不是把 ratio 硬限制在 $[1-\varepsilon,1+\varepsilon]$：当 $A_t>0$ 时主要裁掉 ratio 过大的方向，当 $A_t<0$ 时主要裁掉 ratio 过小的方向；如果策略朝不利方向移动，仍会保留梯度把它拉回来。

重要性采样恒等式严格要求目标分布的支撑集被行为分布覆盖；两个分布差得太大时主要问题是估计方差会很高。PPO 用 clip 控制 surrogate objective 的更新幅度，一些实现还会额外使用 KL penalty，但 KL 并不是 PPO-Clip 目标的必选项。

```
model = SFT_model

for iteration in training:
    # rollout
    responses = model.generate(prompts)
    logprob_old = model.logprob(prompts, responses)
    rewards = reward_model(prompts, responses)
    advantages = compute_advantage(rewards)

    # PPO update
    for epoch in range(K): 
        logprob_new = model.logprob(prompts, responses)
        ratio = exp(logprob_new - logprob_old) 
        loss = PPO_loss(ratio, advantages)
        optimizer.step() // model参数更新，model_old ==> model_new
```

1. Epoch 0: 此时 weights_new == weights_old，所以 ratio 为 1。
2. `Optimizer.step()`: 权重由 $w$ 变为  $w - \eta \cdot \nabla L$。
3. Epoch 1: 当你再次调用 model.forward，模型内部使用的是更新后的权重，产出的 logprob/logits 自然就变了
    1. 如果这次更新是成功的，模型会倾向于给那些 advantage（优势）为正的 responses 分配更高的概率。
    2. 于是，同一个 responses 在新模型下的 logprob_new 就会变大。

`ratio = exp(logprob_new - logprob_old) `正是 PPO 能够进行 K 次迭代的关键。如果没有这个比率：一旦 `optimizer.step()` 更新了权重，之前的 responses 就不再是由当前模型/model_new 产生的了（分布发生了偏移），按理说这些数据就该作废。有了这个比率：它在数学上补偿了“旧数据”和“新模型”之间的分布差异，使得我们可以对同一批数据反复学习 K 次，极大地提高了计算效率。


## 总结一下

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

## 缺点

rl最大的缺点：效率极低。模型往往要把整个任务完整跑一遍，等到最后才知道自己做得对不对。只有在这一次尝试彻底结束后，它才能收到一个简单的结果信号。想象一下，一个系统花了几十步、上百步去写代码、下棋或解题，最后得到的只是一个模糊的提示：成功还是失败。它不知道哪一步做得好，也不清楚哪一步出错，只能靠反复试验去猜。往往要重复上千次，才能从偶然的成功里提炼出一点有价值的经验。**低效的根源在于，它几乎只看结果**。模型在完成一项复杂任务时，无论中间经历了多少误判和偶然，只要最后成功，就会被判定为正确。它学到的不是理解，而是取巧。强化学习让模型更会迎合奖励，却未必更聪明。现实世界的目标是模糊的、多维度的，根本无法被单一的奖励函数概括。Karpathy 更看重的，是让模型在每一步中学会理解。不是完成任务后才得到结果，而是在过程中就能意识到自己哪里做得好、哪里需要调整。这需要更细致的过程监督与反思机制，让模型能像人一样边做边学。人类的智慧并非来自被奖励，而是来自对错误的体察与对过程的理解。
