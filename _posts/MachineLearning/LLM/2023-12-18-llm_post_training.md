---

layout: post
title: 大模型Post-Training
category: 架构
tags: MachineLearning
keywords: llm rhlf

---

<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

* TOC
{:toc}

## 简介

[关于post-training和一些思考](https://mp.weixin.qq.com/s/dvyvKExTl5t9aQDPd4hDJg)在GPT刚问世的时候，业界对于RLHF的作用还存在质疑。许多公司发现，仅仅使用SFT就足以满足他们的需求。甚至在Meta内部，对于是否应该使用RL技术也存在分歧。但是随着DPO等算法的推出，以及开源社区更多经验的分享，业界逐渐接受了RLHF这一训练环节的重要性。学术界提出了各种XPO方法，而工业界也提出了多种PPO的替代算法。逐渐认识到RLHF的价值：RLHF能够在各种评测榜单中显著提高模型分数（刷分利器），在定制化这块也很有前景，对聊天风格的转换这一点很重要，它使得OpenAI发布的mini小模型都能在Arena中排前三，这一点也许正是国内大模型欠缺的。总结来说post-training的转变可以参照下图，可以明显的看出alignment阶段规模的扩大。

![](/public/upload/machine/post_training.jpg)

[ChatGPT训练三阶段与RLHF的威力](https://mp.weixin.qq.com/s/20IcxdGAKTngREp7h29ojw)

![](/public/upload/machine/llm_rhlf.jpg)

1. 预训练模型是一个未加控制的“怪物”，因为其训练数据来源于对互联网内容的无差别抓取，其中可能包括点击诱导、错误信息、政治煽动、阴谋论或针对特定人群的攻击等内容。
2. 在使用高质量数据进行微调后，例如StackOverflow、Quora或人工标注，这个“怪物”在某种程度上变得可被社会接受。
3. 然后通过RLHF进一步完善微调后的模型，使其更符合客户的需求，例如，给它一个笑脸。
你可以跳过这三个阶段中的任何一个阶段。例如，你可以直接在预训练模型的基础上进行RLHF，而不必经过SFT（Supervised Fine-Tuning，监督微调）阶段。然而，从实证的角度来看，将这三个步骤结合起来可以获得最佳性能。预训练是资源消耗最大的阶段。对于InstructGPT模型，预训练阶段占据了整体计算和数据资源的98%。可以将SFT和RLHF视为解锁预训练模型已经具备、但仅通过提示难以触及的能力。

[RLHF 的故事：起源、动机、技术和现代应用](https://mp.weixin.qq.com/s/cRixcz6VeZ-C4D-IpXsNmw) 未细读。

[解析大模型中的Scaling Law](https://mp.weixin.qq.com/s/7Zdi8z84grl1BO1k7DGpUQ)在大模型的研发中，通常会有下面一些需求：
1. 计划训练一个10B的模型，想知道至少需要多大的数据？
2. 收集到了1T的数据，想知道能训练一个多大的模型？
3. 老板准备1个月后开发布会，能用的资源是100张A100，那应该用多少数据训一个多大模型最终效果最好？
4. 老板对现在10B的模型不满意，想知道扩大到100B模型的效果能提升到多少？

## RLHF流程

强化学习（Reinforcement Learning, RL）是一种机器学习方法，模型通过与环境的交互来学习决策策略。模型在每一步的选择中会得到奖励或惩罚，目标是最大化长期的累积奖励。在自然语言处理（NLP）中，强化学习可以用于优化模型的输出，使其更符合期望的目标。

RL包含行动、 环境、观察、奖励机制等模块，奖励机制是RL 具有特色的模块，在奖励机制出现之前，众多机器学习算法是通过损失函数的梯度更新来进行模型学习的，这种损失函数优化效果带来的是模型直接收益反馈，然而不同于传统机器学习任务的单一任务分析，针对复杂环境的分析以及任意动作的带来的奖励反馈极为动态，比如我们在驾驶场景，方向盘多转动5度所带来的奖励收益是极为复杂的，这也让众多传统机器学习算法无法对上述任务进行建模。如何设计良好的奖励机制，是强化学习系统算法建模之前就要想清楚的问题。RLHF的做法是不再像原有RL依赖机器计算奖励反馈，而是利用人工计算奖励反馈，所以该算法框架才被定义为基于人类反馈的强化学习框架。 

![](/public/upload/machine/rlhf_workflow.jpg)

[图解大模型RLHF系列之：人人都能看懂的PPO原理与源码解读](https://mp.weixin.qq.com/s/mhPJzhQvPJlAWsO2nW9BHg)

[RLHF——让大模型对齐人类偏好](https://mp.weixin.qq.com/s/UGLifcfq9SmjARYk9D3kDQ)预训练主要针对补全能力，但不一定是“有用”的补全。RLHF优化模型所涉及的三个步骤
1. 指令微调（SFT）：模型会模仿其训练数据，使用精选的人类回答数据集来微调预训练的大语言模型以应对各种查询。**这让模型获得了优异的指令理解和意图识别能力**，模型的输出也更符合人类的期待，胜过通用文本生成模型，**弥补了 LLMs预测下一个单词目标与用户遵循指令目标之间的差距**，指令的作用是约束模型的输出，使其符合预期的响应特征或领域知识，为人类干预模型的行为提供一个通道。PS： chat 模型就是SFT 过的模型
    1. 指令微调SFT（Supervised fine-tuning）的数据集是问答对，即（prompt，answer）对，prompt我们可以理解为指令或问题，answer就是针对该指令或问题的高质量答案。SFT就是在预训练模型基础上利用这些人工标注的数据进一步微调
    2. IFT可以算作SFT的一个子集，或者说先驱步骤，IFT的主要目的是让模型适应并听从人类的指令，比如当指令prompt出现"summarize"时，模型就应该知道现在的任务是总结任务。经过IFT之后，模型学会了听从指令，但是其生成的内容却不一定安全可靠。所以为了提升大模型的帮助性、降低有害性，人们会继续做SFT，通过高质量的数据给模型展示无害的、有帮助性的回答，规训模型的生成内容。
2. 奖励模型训练(RW)：使用一个包含人类对同一查询的多个答案打分的数据集训练一个奖励模型。或者说，就是一个打分模型，标注者对大量的SFT模型输出进行投票，哪个更好，哪个更差，由此创建了一个由比较数据组成的新数据集。相比监督微调，这种方法的优势在于不需要标注者编写回答，只需要为模型生成的几个回答打分，大幅提高了标注效率。
    1. 训练RM的数据集包含同一提示的不同输出，其中query表示提示信息或者说指令信息，chosen为标注后排序分数较高的答案，即针对提示选择的答案；rejected为标注后排序分数较低的答案，即针对提示拒绝的答案。
        ```json
        {
            "query": "联合国总部在哪里？",
            "chosen": "联合国总部大楼位于纽约曼哈顿东侧，属于xxx",
            "rejected": "联合国的15个专门机构都没有设在总部，然而，xx"
        }
        ```
    2. 训练RM是一个排序任务，不是直接对文本标注分数来训练奖励模型，**因为不同的研究人员对同一个句子可能有不一样的评分，这样会导致大量的噪声出现，如果改成排序，则会大大降低噪声**。不同的排名结果将被归一化为用于训练的标量奖励值。针对query，输入chosen和rejected答案，训练目标尽可能的使得chosen答案和rejected答案的差值更大。
    2. 奖励模型可以利用预训练模型进行初始化，或者也可以进行随机初始化。训练奖励模型的基本目标是获得一个模型，该模型接收一系列的文本，之后返回每个文本对应的标量奖励，该奖励会在数字值的大小上代表人类偏好，越大表示越接近人类偏好，越小表示越脱离人类偏好。
    3. 对于rm模型来说，采用sft模型进行参数初始化，将原来的lm输出层替换成一个线性全连接层，在接受提示和响应作为输入后，输出一个标量奖励值。在训练过程中，采用pair-wise方法进行模型训练，即对于同一个提示内容x来说，比较两个不同的回答$y_w$和$y_l$之间的差异，假设$y_w$在真实情况下好于$y_l$，那么希望$x+y_w$经过模型后的分数比$x+y_l$经过模型后的分数高，反之亦然。
3. RLHF 训练/rlhf-ppo：人类反馈强化学习/近端策略优化算法（PPO），根据 RW 模型的奖励反馈进一步微调 SFT 模型。
    1. 设计对齐模型的优化目标：这个优化目标不仅考虑到奖励模型的得分，也尽量让对齐模型参数更新后输出的分布不要偏移sft模型太远，防止模型越训越差。
    2. 我们让**对齐模型**根据prompt自生成回答，并采用训练好的奖励模型对回答进行打分，对齐模型会根据评分结果不断调整自己的输出分布。

总结一下**RLHF=rm+ppo**：我们通过比较容易获得的公开无标签数据，来训练一个大语言模型/预训练模型，然后，通过人工编写的问答对，来生成高质量的监督对话数据，来优化大语言模型的对话能力。在得到了这个优化后模型（sft model）之后，标注者便在给定问题上可以基于模型生成的答案，对回答进行排序，并用排序数据训练一个reward model对回答的结果排序打分，用来评估回答的质量。最后，也是强化学习中最重要的一步，就是用你的“奖励模型”来提升 SFT model的效果。PS：在得到一个sft model之后，如何进一步优化sft model？一种办法是准备更多的“问题回答对“，但这个成本很高，再一个准备的多了，也可能会有价值观等问题，所以干脆训练一个专门的reward model来做这个事儿，用它来对sft model 生成的内容打分，进而继续“微调”sft model。这个很像家长、老师会告诉我们做事的正确答案，但是教的不多，到社会上，没人告诉你对错，只能通过别人的脸色、反应来判断自己做的对错。

[RLHF 为什么不直接对 loss 进行梯度下降来求解？](https://mp.weixin.qq.com/s/Qxue1q9n9q06HLg_ijjqRw)

[大模型对齐技术，各种什么O：PPO,DPO, SimPO,KTO,Step-DPO, MCTS-DPO,SPO](https://mp.weixin.qq.com/s/pE_sSlaGUfKNM9EaBLR-cg) 推荐细读。PS：建议捋一下。

## 迭代

RLHF 是一个完整技术框架，PPO 仅仅是其中强化学习算法模块的一种实现方式。人类反馈构造出偏好对齐数据集以便训练RM，正因为有了RM，才能让强化学习有了发挥的空间，让sft后的模型有了进一步的提升。但偏好对齐一定需要RL嘛？偏好对齐一定需要人类反馈吗？偏好对齐一定需要训练RM嘛？偏好对齐一定需要大量偏好样本么？

### PPO

利用PPO算法，根据RW模型的奖励反馈进一步微调 sft model。经过强化学习后，LLM 给出的回答会越来越逼近那些在奖励模型中得分比较高的回答。包含actor model、reference模型/ref_model、reward model和critic model。actor model是我们想通过强化学习微调的大模型，但是强化学习过程很容易把模型训练“坏”，因此需要另外一个不会参数更新的 ref_model来当作标的，别让actor model跑偏太远。**为什么PPO不直接使用reward model?**虽然reward model可以提供每个状态或状态动作对的即时奖励信号，但它并不能直接提供对应的价值估计。奖励信号只反映了当前动作的即时反馈，critic model的作用是估计状态或状态动作对的长期价值，也称为状态值函数或动作值函数。critic model能够学习和预测在当前状态下采取不同动作所获得的累积奖励，它提供了对策略改进的指导。PPO算法使用critic model的估计值来计算优势函数，从而调整策略的更新幅度，使得更有利于产生更高长期回报的动作被选择。PS： actor model 和 ref_model是同一个模型的两个副本，reward model和critic model也是同一个模型的两个副本，且起源都是base model。[拆解大语言模型RLHF中的PPO算法](https://mp.weixin.qq.com/s/y7o9F9vz8dv609ee6xqYtw) 原理与代码并重，值得细读。

### DPO（Direct Preference Optimization）

[人人都能看懂的DPO数学原理](https://mp.weixin.qq.com/s/aG-5xTwSzvHXN4B73mfKMA)

在训练奖励模型的过程中，我们就已经在考虑“什么回答是好的，什么回答是不好的”这个问题了。而对齐模型依然是在考虑这个问题。所以，我们能不能避开奖励模型的训练，直接一步到位训练对齐模型呢？
1. RLHF算法包含奖励模型(reward model)和策略模型(policy model，也称为演员模型，actor model)，基于偏好数据以及强化学习不断迭代优化策略模型的过程。RLHF常使用PPO作为基础算法，整体流程包含了4个模型，且通常训练过程中需要针对训练的actor model进行采样，因此训练起来，稳定性、效率、效果不易控制。
2. 在实际rlhf-ppo的训练中，存在【显存占据大】、【超参多】、【模型训练不稳定】等一系列问题。所以，在考虑“一步到位训练对齐模型”的过程中，我们是不是也能顺手做到绕过强化学习，采用一个更简单的方式（比如类似于sft）来使用偏好数据训练对齐模型呢？
2. DPO算法不包含奖励模型和强化学习过程，直接通过偏好数据进行微调，将强化学习过程直接转换为类似SFT过程，因此整个训练过程简单、高效，**主要的改进之处体现在于损失函数**。DPO算法仅包含RLHF中的两个模型，即演员模型(actor model)以及参考(reference model)，且训练过程中不需要进行数据采样。DPO算法的目的是最大化奖励模型(此处的奖励模型即为训练的策略)，使得奖励模型对chosen和rejected数据的差值最大，进而学到人类偏好。

偏好数据，可以表示为三元组(提示语prompt, 良好回答chosen, 一般回答rejected)。

## Self-Play RL

[LLM的范式转移：RL带来新的 Scaling Law](https://mp.weixin.qq.com/s/JPfgF6UtgIYwWXwNQHOoqQ)2018 年，Lex Fridman 邀请 Ilya 来 MIT 客座讲一节课，Ilya 选择的主题是 RL 和 self-play，因为他认为这是通往 AGI 的路上最关键的方法之一。Ilya 在讲座中用一句话概括了强化学习：让 AI 用随机路径去尝试一个新的任务，如果效果超出预期，就更新神经网络的权重让 AI 记得多使用成功的实践，然后开始下一次尝试。这个概括中可以看到强化学习和其他 AI 范式的重要区别，经典三大范式（监督学习、非监督学习、强化学习）中只有强化学习的假设是让 AI 进行自主探索、连续决策，这个学习方式最接近人类的学习方式，也符合我们想象中的 AI agent 应该具备的自主行动能力。强化学习的核心在于"探索"（Explore）和"利用"（Exploit）之间的权衡。LLM 在"利用"现有知识上做到了现阶段的极致，而在"探索"新知识方面还有很大潜力，RL 的引入就是为了让 LLM 能通过探索进一步提升推理能力。在实现 RL 的过程中，有两个核心组件。他们之间一直在反复交互，agent 在环境中执行 action，并且根据环境的变化评估 reward：

1. Environment：AI 探索完成任务的环境，当 Alphago 下围棋时，环境就是 19x19 的棋盘。环境会发生变化，AI 会从环境变化中收到 reward value 判断过去的那一系列探索是否有明显的收益，例如距离下围棋胜利是否更接近了。
2. Agent：agent 会根据对环境的观测和感知来输出一个动作，目标是得到更高的 reward。agent 这个概念最早就是来自强化学习。
PS: Environment ==> reward ==> agent ==> action ==> environment ==> reward ==> agent ==> action ==> ...
如果把这里的 agent 主体换成 LLM，那么会在探索的过程中做很多 LLM inference。因此这里 RL 在 LLM 中应用的思路本质是用 inference time 换 training time，来解决模型 scale up 暂时边际收益递减的现状。LLM 直接生成是可以类比系统 1 的慢思考。而 RL 就为 LLM 带来了系统 2 慢思考。

[万字长文解析OpenAI o1 Self-Play RL技术路线](https://mp.weixin.qq.com/s/wJURZu61LoxiXHtKUT3jIA)OpenAI o1 可以通过 Self-Play 的方式提升模型 Reasoning 的能力，o1 是怎么实现这样的能力呢，纯粹从推理态来看是 inference time thinking 做到的，就是在回答用户问题之前，模型会陷入一个思考的过程。逐步思考，提出假设，并且反思，以实现 Reasoning 能力。这里面的 thinking 流程是模型和其他大模型最大的不同，在这中间经历了相当长时间的思考阶段。思考的内容，目前在 ChatGPT 的客户端中可以做了隐藏（防止被蒸馏）。虽然不清楚背后实现的具体逻辑，但是从目前已有的接口来看，o1 至少已经能够实现：提出假设，验证思路，反思过程这三种主要的逻辑推理能力。

大语言模型的主要学习策略从 RLHF 的巨大成功之后，也出现过摇摆。以 next token prediction 作为代表的 Behavior Clone 思路主要的手段是预训练和 SFT 为主的，主要强调从海量知识中自监督学习加上专家数据的示教。但是这一条路径遇到了很大的困难，我们如今已经几乎耗尽了几乎所有互联网上所有的语料，但是极强的智能也没有出现。同时 SFT 作为 Behavior Clone 的上限是比较低的，大多数情况下需要堆叠大量高质量语料，成本几乎成为了垂直领域难以负担的问题。更大的问题在于 SFT 几乎无法囊括负例的示教，对于 trial-n-error 的自我博弈智能来说，只能利用其中比例极低的正例。所以祖师爷 John Schulman 的 PPO 加上 RLHF 力挽狂澜，把 GPT-3 拉出黑暗，直接进化到 InstructGPT，用人类反馈进行建模引爆了整个领域。但是我们**现在又到了一个十字路口，大模型看起来好像是一个死记硬背的书呆子，推理能力迟迟没有见到突飞猛进的变化**，大模型 Self-Play 能否通过部分领域示教数据，模型通过自我博弈持续提升策略？这里面需要有两个先决条件：Generator 和 Verifier 都要足够强。语言和游戏在这个方面是截然相反的，游戏中的行为生成是困难的而价值评判是简单的：对于路边看棋大爷下好一步棋很难，但是判断这一步下的好不好他还是可以的。语言模型生成行为是容易的，但是判断生成的好坏是困难的，1B 的模型都可以滔滔不绝证明哥德巴赫猜想，但是判断每一步是否正确却非常困难。

这一切正在悄然改变，Reward 数据正在越变越多，作为 Verifier 的 Reward Model（RM）也在变得越来越强。我们看到了越来越多的证据，新的的 scaling 趋势呈现在了生成式 RM 上2。这种 Reward Model 相比于传统的方法来说，对于大语言模型的判别已经不是一锤子买卖了。它更像是人类标注员的思路，对问题和答案会和传统生成式模型一样也能够进行 CoT。他会对于一个问题和答案，首先按照生成式模型的方法给出自然语言的判断，然后再给出 RL 所需要的标量数值，彻底摆脱了判别式 RM 中 BT 假设的枷锁。所以随着 Reward Model 思考的深入，其准确度也会不断上涨。同时更重要的是，verifer 和 generator 之间也可以通过信息密度更高的自然语言的方式进行互动。相当于 RM 监督 policy 的时候，不仅告诉了每条答案的评分还详细给出了错误的原因。这种以自然语言作为交互模式的对抗 + 合作的模式可以随着计算资源的增长获得明显的增长（推演的更多，反思的更细）。其中的对抗是，大语言模型要经历生成更好的回答让 RM 无法挑出问题，而 RM 也要自己增长能力以发现大语言模型的更多漏洞。合作则在于，最终两者的博弈并不是零和的，两者的同步增长会使得我们的大语言模型拥有真正的长思考能力，并有机会往全领域泛化。

那么第二个问题是：Verifier 判别出来的正例和负例是不是同时能够利用起来，答案是比较正面的。而且强化学习中，引入负例可以更有效地提升大语言模型的推理强度。数据利用效率更是达到了仅使用正例的八倍，这个结论是非常好理解的，对于推理来说一个巨大的采用空间内，做错的可能性在起初要大大高于能够做对的概率。如果无法充分利用负例的数据价值，学习效率就会大打折扣。

从推理时的较为确定的 Self-Play 方式出发，我们可以反向推演一下 o1 的可能技术路线（声明一下这些都是推演）。假设 Generator 和 Verifier 是两个相互配合的模型，部署的时候使用两个模型组成的系统，那么就可以使用 actor-critic 的方式加 TD-error 来更新 generator model 和 verifier model。PS： 细节可以看原文，有点get 到 self-play 啥意思了。

LLM scaling up 的边际收益开始递减，用 RL self-play + MCTS 提升 LLM 推理能力成为下一个技术范式。在新范式下，LLM 领域的 scaling law 会发生变化：计算量变大仍会带来模型智能的提升，但会从模型参数量变大，转移到 inference-time compute 增加，也就是模型进行更多 RL 探索。要让 RL 算法能够在连续推理任务上做到最好，理解 self-play + MCTS 的思路是最重要的。放到 LLM 语境下，self-play 是让 LLM 同时扮演一个或多个 agent model 去做推理任务，并由另一个 LLM 作为 reward model 来给出打分评价，一定次数后更新 LLM 权重让其多记住做得好的推理方式。Self-play RL 是要在好的策略上持续探索，怎么定义“好”就尤其重要。因此， Reward model（奖励模型） 是 RL 中最关键的模块之一，有两个关键的卡点是需要解决的，那就是 reward model 的泛化性和连续性。Self-play RL 在棋牌、电子游戏、数学竞赛上之所以有效，是因为这些领域都有明确的胜负标准，可以作为 reward model 的基础。有了 LLM 的 in-context learning，我们相信代码、数学是可以通过 LLM + self-play RL 来持续进步的。根据 The information 报道，strawberry 目前能力最强的领域就在 math 和 code 上，Sonnet 3.5 在代码的提升也是很好的佐证。这两个领域具有准确、快迭代的评判标准，使得模型能够获得明确的反馈：我们可以把 code script 放进 Python Interpreter/ compiler（如果不成功，报错信息也能帮助 AI 自己去发现和理解错误在哪里），把 math proof 放进 Lean（Lean 是一种编程语言，通过计算机验证数据定理，广泛用在 AI 形式化数学证明中帮助 AI 理解数学题），就能自动验证其准确性。

**reward model 对其他领域的泛化性**并不明确。物理、医药有明确的标准答案，但需要很长的实验验证周期。法律、金融的问题往往没有通用解法，很难用通用的 reward model 实现。文字创意领域的 reward 很多时候不符合马尔可夫模型，也就是其 reward 常常会有跳变。一本好的小说、剧本，会讲究反转，试想 LLM next-token prediction 到一个反转之前其 reward 函数还很低，一个精彩的反转让 reward 函数突然大幅提升，self-play RL 很难捕捉这个突然的变化。**因此这里孕育着新范式下的第二个创业机会：垂直领域的 reward model**。而要让 reward function 能捕捉到更多的信号，在垂直领域之外泛化，最重要的方向就是怎么用好 LLM 作为 reward model，并同时输出数字和文字评估。


LLM as a PRM （process reward model）：通往泛化的重要路线。在传统 RL 中，我们按照最终结果评分，其评分模型称为 ORM（outcome reward model）；而通过专门训练 LLM 成为 process verifier ，新的评分模型叫做 PRM，往往是使用娇小 LLM fine-tune 得到。PRM （Process reward model） 是奖励好的推理步骤，而不仅仅是正确的结果。这更接近人类的学习和推理方式。而且在 process supervision 过程中，reward 的形式也不止限于数值，文字评价也可以作为指导模型继续行动的 reward。

[左脚踩右脚真能上天？谷歌提出自我纠正RL提升LLM](https://mp.weixin.qq.com/s/OnPPizOC1Ae63WjhNdLYNA)光靠模型自己给自己打分，较难获得新的信息增量提升效果，要么需要多个模型，要么依赖于更高级的模型或其他形式的监督（如RAG提供外部输入）。但**最强的那个模型要怎么继续进步呢**？光靠模型自己反思真的不行吗？谷歌DeepMind团队提出多轮在线强化学习方法 SCoRe，完全使用自生成的数据进行训练，不需要任何外部反馈模型/信号，就能实现内在自我修正策略以“即时”修正自己的错误，提升了模型的推理能力。我们理性的看，这种“自我反思”要有效果，有一个重要前提，是当前LLMs已经压缩了和问题答案相关的信息/知识，但还无法很灵活的使用，需要更高效的策略帮它提取&运用起来。

## 技术

RLHF开源框架主要有DeepspeedChat、Trlx、ColossalAI-Chat，同时在这些框架中会包括一些常用的节省GPU资源，加快训练速度的框架例如Accelerate、PEFT等。在整个RLHF的优化训练中，少则涉及2个模型，多则涉及4个模型（**base-model,sft-model,reward-model,ppo-model**），超参数较多，训练优化存在较多不确定性。还有一个需要关注的问题，就是RLHF的优化训练耗时较多，少则半月，多则数月才会训练完成，训练资源成本较多。

[一键式 RLHF 训练 DeepSpeed Chat（一）：理论篇](https://mp.weixin.qq.com/s/t5lT1NIZ6TysfgJks7kYKA)ChatGPT模型的训练是基于InstructGPT论文中的RLHF方式。这与常见的大语言模型的预训练和微调截然不同，目前仍缺乏**一个支持端到端的基于人工反馈机制的强化学习（RLHF）的规模化系统**，为使RLHF训练真正普及到AI社区，**DeepSpeed-Chat应运而生**。
[一键式RLHF训练 DeepSpeed Chat（二）：实践篇](https://mp.weixin.qq.com/s/M3odD3dR2bPOar2ZIUsABg) 值得细读

[PAI-ChatLearn ：灵活易用、大规模 RLHF 高效训练框架（阿里云最新实践）](https://mp.weixin.qq.com/s/f7rkS1bFo0ZtJ7pptYbBQQ)

[开启训练之旅: 基于Ray和vLLM构建70B+模型的开源RLHF全量训练框架](https://mp.weixin.qq.com/s/Vi_SIvDqpveaPaP8C5KTyA) DeepSpeedChat和LLaMA Factory这些框架往往是基于 ZeRO 等并行方式，将 RLHF 算法中的四个模型切片后放到同一个 GPU 上。在模型规模越来越大的今天，这种调度方式无法满足 70B+ 甚至仅 13B+模型的全量 RLHF 训练，必须通过合并 Actor Critic 模型或者使用 LoRA 等方式妥协内存使用量。而这些PEFT的方式往往意味着模型效果的妥协。

TRL 是由大名鼎鼎的Transformer 针对强化学习专门设计的，旨在打造一个针对大语言模型开展强化学习所用到的全栈类库。提供三阶段训练框架，包括微调训练阶段的SFTTrainer、RM模型训练阶段的RewardTrainer、以及强化学习训练阶段使用的PPOTrainer。 PS：对应训练的LLM的Trainer

### 数据集格式


## 数据准备

sft 
1. 主要数据格式：QA问答
2. 适用类型：微调的主要场景，垂直领域应用私有化重点能力

```
# 带有历史的QA
{
    "instruction": "你是谁？",
    "input": "",
    "output": "我是一个由XXX自主训练的生成式自然语言处理模型，名字为GPT，专门用于回答运维相关问题。",
    "history":[
        [
            "你好",
            "你好呀！请问有什么需要帮助的地方？"
        ]
    ]
}
# 不带历史的QA提问
{
    "instruction": "你是谁？",
    "input": "",
    "output": "我是一个由XXXX自主训练的生成式自然语言处理模型，名字为GPT，专门用于回答运维相关问题。"
}
```

rm
1. 主要数据格式：带有排序的QA方案
2. 适用类型：用于SFT后的模型能力增强，训练奖励模型协助对模型结果进行优化符合人类预期

```
{
    "instruction": "我需要举办6人晚宴。你能帮我推荐三道不包括坚果或海鲜的菜吗？",
    "input": "",
    "output": [
      "好的，我将为您推荐三道不包含坚果或海鲜的菜。这三道菜都是容易做且好吃的家常菜，希望您能满意。\n1. 青椒土豆丝\n2. 西红柿炒鸡蛋\n3. 土豆炖豆角",
      "尖椒干豆腐，土豆丝，红烧肉",
      "如果是六个人，就点八个菜，一般是荤素搭配，凉菜+热菜，两三个素凉，一个肉凉，以及两三个素热，两三个肉热。"
    ],
    "history": []
  }
# 和sft的主要差异在于output中，需要排序给出对应的回答，依次从好到差这样排序
```

DPO\PPO 直接偏好优化
1. 主要数据格式：
2. 使用类型：直接跳过对应的RM训练过程中，利用数据来完成强化学习操作

```

{
    "instruction": "解释为什么下面的分数等于 1/4\n4/16",
    "input": "",
    "output": [
      "分数 4/16 等于 1/4，因为分子和分母都可以被 4 整除。将顶部和底部数字都除以 4 得到分数 1/4。",
      "1/4 与 1/4 相同。"
    ]
}
# 在output中有两个答案，分别表示choosen和reject来表示数据是否接收。
```

[基于 LoRA 的 RLHF: 记一次不太成功但有趣的百川大模型调教经历](https://mp.weixin.qq.com/s/4dt3XiLnZN7Q17VHz3lsng) 非常经典。 PS：**大模型统一的一个好处是input字段统一，进而数据集格式统一**。这不像以前的专有模型，input字段各式各样。数据集格式是什么样子，就侧重训练模型哪些方面的能力。
1. sft 数据集/Instruction 数据集。
    ![](/public/upload/machine/sft_dataset.jpg)
2. sft训练之后的大概效果是这样的：
    ```
    输入: "你是谁开发的啊"
    原始 baichuan-7B: "我就是你，我是你自己。(自性)"
    ChatBaichun-HC3: "我是一个计算机程序，由一个人或一群人编写。我的目的是帮助人们解决问题和回答问题。"
    ```
3. rl 数据集，这个是训练reward model 用的，不是训练sft model 用的。
    ![](/public/upload/machine/rl_model.jpg)

### 代码

SFT是指令学习， 而RM和PPO合在一起用于RLHF的对齐， 先做SFT，再做RM，最后做PPO

![](/public/upload/machine/rhlf_ppo.jpg)

```python
# 训练
experience_list = []
for i in range(epoch):
# 生成模型所需的input_ids
    input_ids = tokenizer.batch_encode_plus(prompt_list, return_tensors="pt",...)["input_ids"]
    experience = make_experience(args, actor_model, critic_model,ref_model，reward_model,input_ids, ...)  
    experience_list.append(experience)                                      
    mr = np.mean(np.array(mean_reward))
    actor_model.train()
    critic_model.train()
    ppo_step = update_model(args, experience_list, actor_model, actor_optimizer, critic_model,critic_optimizer, tb_write, ppo_step)                          
# 模型保存
actor_model.save_pretrained(os.path.join(args.output_dir, "checkpoint-{}".format(ppo_step)))
tokenizer.save_pretrained(os.path.join(args.output_dir, "checkpoint-{}".format(ppo_step)))
def make_experience(args, actor_model, critic_model, ref_model，reward, reward_model, input_ids, generate_kwargs):
    actor_model.eval()
    critic_model.eval()
    with torch.no_grad():
        # 获取prompt内容长度
        prompt_length = input_ids.shape[1]
        # 使用动作模型通过已有提示生成指定内容，其中：seq_outputs为返回序列，包含prompt+生成的answer
        seq_outputs, attention_mask = actor_model.generate(input_ids, **generate_kwargs)
        # 通过动作模型和原始模型同时计算生成结果对应的log_probs
        action_log_probs = actor_model(seq_outputs, attention_mask)
        base_action_log_probs = ref_model(seq_outputs, attention_mask)
        # 通过评判模型计算生成的answer的分值
        value, _ = critic_model(seq_outputs, attention_mask, prompt_length)
        value = value[:, :-1]
        # 通过奖励模型计算生成奖励值，并对奖励值进行裁剪
        _, reward_score = reward_model.forward(seq_outputs, attention_mask, prompt_length=prompt_length)
        reward_clip = torch.clamp(reward_score, -args.reward_clip_eps, args.reward_clip_eps)
        # reward_clip = reward_score
        # 对动作模型和原始模型的log_probs进行kl散度计算，防止动作模型偏离原始模型
        kl_divergence = -args.kl_coef * (action_log_probs - base_action_log_probs)
        rewards = kl_divergence
        start_ids = input_ids.shape[1] - 1
        action_mask = attention_mask[:, 1:]
        ends_ids = start_ids + action_mask[:, start_ids:].sum(1)
        batch_size = action_log_probs.shape[0]
        # 将奖励值加到生成的answer最后一个token上
        for j in range(batch_size):
            rewards[j, start_ids:ends_ids[j]][-1] += reward_clip[j]
        # 通过奖励值计算优势函数
        advantages, returns = get_advantages_and_returns(value, rewards, start_ids, args.gamma, args.lam)
    experience = {"input_ids": input_ids, "seq_outputs": seq_outputs, "attention_mask": attention_mask,
                "action_log_probs": action_log_probs, "value": value, "reward_score": reward_score,
                "advantages": advantages, "returns": returns}
    return experience
def update_model(args, experience_list, actor_model, actor_optimizer, critic_model, critic_optimizer, tb_write,ppo_step):
    # 计算actor模型损失值
    actor_loss = actor_loss_function(experience["action_log_probs"][:, start_ids:],...)       
    # actor模型梯度回传，梯度更新
    actor_loss.backward()  
    actor_optimizer.step()
    actor_optimizer.zero_grad()
    # 计算critic模型损失值
    # critic模型梯度回传，梯度更新          
```

[阿里PAI-ChatLearn：大规模 Alignment高效训练框架正式开源](https://zhuanlan.zhihu.com/p/717112741) 未细读

## 小结 

流程与技术放在一起如下图

![](/public/upload/machine/llm_tech.png)

[BaiChuan2技术报告细节分享&个人想法](https://mp.weixin.qq.com/s/H6gbh8f9EEXQohjUN8bMDQ) 当产出一个新的模型结构时，可以从以上视角、阶段来一一对比。PS： 提示词在不同模型之间，还是有细微差别的，那想要提示词效果比较好，自然需要了解对应模型的细节。

[大模型RLHF中PPO的直观理解](https://zhuanlan.zhihu.com/p/672420633) 未读，有几个图不错。