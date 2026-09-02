---

layout: post
title: 从 Expected Return 到 Surrogate Loss：RL 如何接入 Backprop
category: 架构
tags: MachineLearning
keywords: rl, policy gradient, gradient estimator, surrogate loss, backprop

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

经典强化学习用 expected return 描述“什么是更好的策略”，神经网络训练则要求一个可微的 scalar loss。本文解释两者之间缺失的桥梁：如何从 expected return 推导 gradient estimator，再把更新方向编码成可以交给 autograd 和 optimizer 的 surrogate loss。

## Backprop 只负责 Differentiation

不管是 backprop 还是 RL，它们的最终目的都是修改 DNN 的 parameters $\theta$。先厘清一件事：**backprop 本身不是一种 learning algorithm，它的作用其实是求导（differentiation）**。给定一个可微的 loss $L(\theta)$，backprop 通过 chain rule 算出 $\nabla_\theta L$，然后我们把这个 gradient 交给 optimizer（SGD、Adam）去更新 $\theta$。

在监督学习（比如 CNN）里，这个流程非常直观，因为 data 和 $\theta$ 无关——不管训练时 params 怎么变，图片本身不会变，所以我们可以直接对着固定的 dataset 做 backprop。

## Expected Return 为什么不能直接 Backprop

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

RL 真正想优化的是期望 Reward，但奖励信号通常不能直接对策略参数 $\theta$ 求导。例如 Reward 可能来自人类打分、代码测试是否通过、最终答案是否正确，这些都是计算图之外的黑盒反馈。即使 Reward 来自一个神经网络形式的 Reward Model，Reward Model 本身虽然可微，但“策略参数 $\theta$ → 离散采样 token → Reward Model 得分”这条链路仍被离散采样操作截断；而且 Reward Model 通常作为冻结的外部评估器使用，并不会把梯度直接传回 Policy。

## Policy Loss 是 Surrogate Loss，而不是 Prediction Error

更专业地说，监督学习提供的是 **target-based supervision（目标值监督）/ direct supervision（直接监督）**。例如预测身高时，label $y=179$ 和 prediction $\hat y_\theta=180$ 位于同一个 prediction space（预测空间），单位也相同，因此可以直接定义 discrepancy。严格来说，$y-\hat y_\theta$ 是 residual（残差），再经过平方或绝对值才成为 loss，例如 $L(\theta)=(y-\hat y_\theta)^2$。label 本身不需要可微；关键是 loss 对模型输出 $\hat y_\theta$ 可微，而且这个 error signal 同时给出了“错多少”和“往哪个方向改”，梯度可以沿着 $L\rightarrow\hat y_\theta\rightarrow\theta$ 传回去。

RL 提供的则通常是 **evaluative feedback（评价性反馈）**：action / trajectory 位于 behavior space（行为空间），Reward 位于另一个 utility space（效用空间）。Reward 不是与 action 对齐的 ground-truth target，二者不能像 $y-\hat y$ 那样直接相减；它只评价已经采样出来的行为“有多好”，并不直接说明正确 action 是什么、哪个 action 导致了结果，以及每一步应该改多少。因此 Reward 也常被称为 scalar feedback、outcome-level feedback 或 bandit feedback。把这种整体评价转换成每个 action / token 的优化信号，需要额外解决 credit assignment，并通过 policy gradient 间接完成。

关键是：**Reward 对策略参数不可直接求导，但模型生成某条轨迹的概率是可微的。策略梯度不计算 $\frac{\partial R}{\partial\theta}$，而是用 Reward / Advantage 加权已采样 action 的 log-prob，从而估计期望 Reward 对模型参数的梯度。**对于一条已经采样出来的固定轨迹，$R(\tau)$ 可以看作外部给定的标量；参数 $\theta$ 改变的是这条轨迹再次出现的概率 $P(\tau;\theta)$。

这一区别也解释了为什么 **Policy Loss 通常不是 prediction error**。RL 真正想最大化的是

$$J(\theta)=\mathbb{E}_{\tau\sim\pi_\theta}[R(\tau)],$$

它是策略的 **true objective**：数值直接表示当前策略的 expected return。但由于轨迹需要经过离散 sampling 和外部 environment 才能产生，$J(\theta)$ 不能像监督学习里的 $L_{\text{MSE}}=(y-\hat y_\theta)^2$ 那样，直接构造一条从 ground-truth target 到 model output 的可微计算图。Policy network 输出的是 action distribution，而 Reward / Advantage 是对已采样行为的 evaluative feedback；两者既不在同一个 semantic space，也不存在可以直接相减的“正确答案”。

因此，policy gradient 的首要任务不是先发明一个 loss，而是估计 $\nabla_\theta J(\theta)$。给定当前 policy 采集的一个 minibatch $\mathcal B$，所需的 gradient direction 可以估计为：

$$
\hat g
=
\frac{1}{|\mathcal B|}
\sum_{t\in\mathcal B}
\nabla_\theta\log\pi_\theta(a_t\mid s_t)\hat A_t
\approx
\nabla_\theta J(\theta)
$$

其中 $\hat A_t$ 是不参与求导的 weighting signal；在最原始的 REINFORCE 中可以直接使用 return $G_t$，加入 baseline 或 Critic 后则使用 Advantage estimate。为了让 autograd 计算出这个方向，工程上把它编码成一个可微的 scalar **surrogate loss**：

$$
\mathcal L_{\mathrm{PG}}(\theta)
=
-\frac{1}{|\mathcal B|}
\sum_{t\in\mathcal B}
\operatorname{stop\_gradient}(\hat A_t)
\log\pi_\theta(a_t\mid s_t)
$$

这个 loss 与目标 gradient 的关系是：

$$
\nabla_\theta\mathcal L_{\mathrm{PG}}(\theta)=-\hat g,
\qquad
\theta
\leftarrow
\theta-\alpha\nabla_\theta\mathcal L_{\mathrm{PG}}(\theta)
=
\theta+\alpha\hat g
$$

在 PyTorch 中，`stop_gradient` 对应 `.detach()`。一段最小实现是：

```python
log_probs = policy_dist.log_prob(actions)
policy_loss = -(log_probs * advantages.detach()).mean()

optimizer.zero_grad()
policy_loss.backward()
optimizer.step()
```

这里的 `actions` 已经由 policy sampling 得到，计算图不需要穿过 sampling；`log_probs` 保留了 policy parameters 到已采样 action 概率的可微路径。`advantages.detach()` 让 Advantage 只作为 weighting signal，不让 Policy Loss 沿这条路径修改 Critic。`backward()` 负责计算 gradient，真正修改 parameters 的是 `optimizer.step()`。

因此，RL 的转换顺序是：**先从 expected return 推导出 policy 应该沿哪个 gradient direction 更新，再把这个方向编码成方便 autograd 计算的 surrogate loss。**这里真正承接 gradient 的是 model output 的 log-prob；$\hat A_t>0$ 时，gradient descent 会提高已采样 action 再次出现的概率，$\hat A_t<0$ 时则会降低它的概率。

Policy Loss 的数值通常不表示“policy 错了多少”，单独看也未必有直观意义；它主要是让 `loss.backward()` 产生所需 gradient 的计算载体。**这并不意味着 RL 里的 loss 可以随意构造**：surrogate objective 必须有数学依据，它的 gradient 要能够估计或近似 true objective 的 gradient。

PPO-Clip 又向前走了一步。相对原始的 on-policy Policy Gradient，它有意修改 surrogate objective：超出 clipping range 的有利更新不再继续提供 gradient。这样得到的 gradient estimator 对原始 objective 一般带有 bias，但可以限制过大的 policy update，用少量 bias 换取更稳定的 optimization。

与之相对，Critic 的 Value Loss 更接近普通 regression loss：return target 和 $V_\phi(s)$ 都位于 utility space，可以用 MSE 衡量 prediction error。也就是说，同一个 Actor-Critic 系统里，Policy Loss 主要负责提供有用的 update direction，Value Loss 则确实在拟合一个 target value。

## Log-Derivative Trick：构造 Score-Function Estimator

下面的 log-derivative trick（似然比技巧），就是从 true objective 推导 gradient estimator，并说明这个 surrogate loss 为什么成立。它基于一个对任意可微概率 $P$ 都成立的恒等式（来自 $\nabla_\theta \log P = \frac{\nabla_\theta P}{P}$ 这条求导链式法则的反用）：

$$\nabla_\theta P(\tau;\theta) = P(\tau;\theta)\,\nabla_\theta \log P(\tau;\theta)$$

其中 $\nabla_\theta\log P(\tau;\theta)$ 称为 score function，因此这类方法也叫 **score-function estimator** 或 **likelihood-ratio estimator**。$\log$ 的作用不是装饰公式：它的导数产生 $1/P(\tau;\theta)$，恰好抵消 expectation 中隐含的 $P(\tau;\theta)$，从而把“对 probability distribution 求导”改写成“在该 distribution 下求期望”。

把它代进 $\nabla_\theta J(\theta) = \sum_\tau \nabla_\theta P(\tau;\theta)\, R(\tau)$：

$$\nabla_\theta J(\theta) = \sum_\tau P(\tau;\theta)\,\nabla_\theta \log P(\tau;\theta)\, R(\tau) = \mathbb{E}_{\tau \sim \pi_\theta}\big[R(\tau)\,\nabla_\theta \log P(\tau;\theta)\big]$$

一条 trajectory 的 probability 可以进一步分解为：

$$
P(\tau;\theta)
=
\rho_0(s_0)
\prod_{t=0}^{T-1}
\pi_\theta(a_t\mid s_t)
P(s_{t+1}\mid s_t,a_t)
$$

其中 $\rho_0$ 是 initial-state distribution，$P(s_{t+1}\mid s_t,a_t)$ 是 environment transition probability。只有 $\pi_\theta(a_t\mid s_t)$ 含有 policy parameters $\theta$，所以：

$$
\nabla_\theta\log P(\tau;\theta)
=
\sum_{t=0}^{T-1}
\nabla_\theta\log\pi_\theta(a_t\mid s_t)
$$

这也是 Policy Gradient 可以在未知 environment model 下训练的关键原因：训练仍然需要与 environment interaction 来采样 trajectory，但不需要事先知道 transition probability，也不需要让 gradient 穿过 environment。

这一步同时拆掉了上面两道坎：

- **拆第一道坎**：右边重新变回了一个"对 $\pi_\theta$ 的期望"，于是可以**用采样去估计**——拿当前策略实际跑出来的几条轨迹求平均，就是对这个期望的无偏估计，不必再纠结"分布在漂"。
- **拆第二道坎**：gradient 不再穿过 sampling 去计算 $\frac{\partial a_t}{\partial\theta}$，而是对已采样 action 的 log-prob 求导。这一项完全可微，可以直接交给 backprop。

把 trajectory probability 展开到每一步，并用从时刻 $t$ 开始的 return $G_t$ 做 credit assignment，就得到：

$$
\nabla_\theta J(\theta)
=
\mathbb E_{\tau\sim\pi_\theta}
\left[
\sum_t
\nabla_\theta\log\pi_\theta(a_t\mid s_t)G_t
\right]
$$

再减去不依赖 action 的 baseline，期望 gradient 不变；把 $G_t-b(s_t)$ 写成 Advantage estimate $\hat A_t$：

$$
\nabla_\theta J(\theta)
=
\mathbb E_{\tau\sim\pi_\theta}
\left[
\sum_t
\nabla_\theta\log\pi_\theta(a_t\mid s_t)\hat A_t
\right]
$$

这就是前面 minibatch gradient estimator 的理论来源。

## REINFORCE：从 Gradient Estimator 到 Parameter Update

使用 sampled trajectories 和 $G_t$ 来近似上面的期望，就是 REINFORCE。直觉上它在干一件很朴素的事：**对真正采到的 action，按照它后续带来的 return 调高或调低其 log-prob**——高回报 action 以后更容易被采到，低回报 action 的概率则被压低。加入 baseline、Critic 和 GAE 后，weight 从 $G_t$ 逐步变成更低方差的 Advantage estimate $\hat A_t$，但 gradient estimator 的基本结构不变。

监督学习可以从固定 data 上的 prediction error 直接得到 gradient；RL 则要先从 sampled interaction 中估计 gradient。REINFORCE 产生的估计值最终仍然交给**和监督学习相同的 optimizer（Adam）**更新 $\theta$。credit assignment、exploration 和 variance reduction 等问题，都会影响这个 gradient estimator 的质量。
