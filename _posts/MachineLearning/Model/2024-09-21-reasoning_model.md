---

layout: post
title: 推理LLM梳理
category: 技术
tags: MachineLearning
keywords: reason model

---

* TOC
{:toc}

<script>
  MathJax = {
    tex: {
      inlineMath: [['$', '$']], // 支持 $和$$ 作为行内公式分隔符
      displayMath: [['$$', '$$']], // 块级公式分隔符
    },
    svg: {
      fontCache: 'global'
    }
  };
</script>
<script async src="/public/js/mathjax/es5/tex-mml-chtml.js"></script>

## 简介

张俊林：目前可以提高模型效果的Scaling方法，按照性价比由高到低排序的话: Test time Scaling Law> RL Scaling Law>预训练阶段Scaling Law(数据不够了，只能推大模型尺寸)。如果哪天RL Scaling Law和Test Time Scaling Law到了天花板，又没有找到新的性价比更合算的Scaling law，也不是说模型效果就提不上去了，大家仍然可以回归预训练阶段的Scaling Law，没有新数据也没关系，推大模型尺寸规模就可以，效果仍然会上升。然后RL阶段Scaling 的天花板随之升高，然后可以再去Scale RL和Test Time，就进一步得到智商更高的大模型。如果这成立，那意味着AGI的解决方案已经完整了？

## 扯扯闲篇

当大家在网上探索o1是如何训练时，肯定会看到以下几个热点词：
1. Test/Inference-Time scaling law，通过增加推理阶段的算力提升模型的推理能力
2. Post Training，通过后训练提升模型的推理能力
3. PRM/ORM：基于过程/结果的奖励模型
4. CoT：思维链
5. 强化学习、self-play（自我博弈）与MCTS（使用蒙特卡洛搜索树寻找最佳答案）

### 什么是Test/Inference-time Scaling Law

openai 发现大模型预训练阶段通过增加算力、数据、参数的方式提升模型性能的的scaling law，一般称为pretrain-time scaling law。但是随着人类的知识用完，再增加另外两个数据也带不来模型性能的提升了，预训练的scaling law见顶。o1 证明了如果增加推理阶段的计算量可以提升模型性能。证明了一种趋势，即在预训练时花费更少的计算量，同时增大推理阶段的计算量。

[OpenAI o1模型的本质优势是什么？ - 猛猿的回答 - 知乎](https://www.zhihu.com/question/667055619/answer/3864887300)设想一下，当我们手里有一个基础模型（我们称其为generator），但是这个模型的逻辑推理能力（比如解数学题的能力）较差时，我们该怎么改进它？再说的具体点，不考虑数据集相关的成本，假设我手头的gpu算力（FLOPs）是有限的，我该怎么利用它，能让我的模型最终能推理出更好的结果？一个比较直接的想法是：把算力花在它的pretain阶段，给模型注入更多数理逻辑的预训练知识。例如用更好、更多的代码数学等数据，或者扩展模型的参数规模。这个做法启发自大家都很熟悉的scaling law（更具体地说是pretrain-time scaling law）。但是，当我们研读openai o1的技术报告时，我们会发现，它把这个算力更多地用在了2个地方：

1. 用在了rlhf的训练上（post training）
2. 用在了模型的推理阶段上（Test/Inferece）
正如pretrain scaling law受到模型参数和训练数据的影响一样，**Test/Inferece scaling law也必然受某些因素影响，而这些因素是什么，又是怎么影响的？**不过等等，此时你肯定想问：
1. 一般来说，一个模型的效果是由它的训练阶段决定的，所以如果这里说通过pretrain或者post training来提升模型的推理能力，我都能理解。但是inference阶段是怎么提升模型的推理能力的？你说的把算力用在inference阶段到底是什么意思？
2. post training和inferece是两种独立的提升模型推理能力的方法吗？它们可以结合在一起使用吗？

把算力用在inference阶段，也就是说，在不变动pretrain阶段的情况下，只通过推理等层面的优化，来提升模型最后的生成效果。这里又分成两种情况。
1. 优化推理输入：prompt。这个方法大家应该非常熟悉了。例如，原来你的模型吃一个问题，直接吐给你回答。但是现在为了让模型能更好模拟人类的思考方式，你希望【模型在步步思考后再给出回答，也就是模型的生成结果里包含思考步骤+答案】，那么你可以选择在prompt中给模型相应的例子，或者在多轮对话中引导模型think step by step，来实现这个目标。你的prompt给的越细节，你的多轮引导给的越多，模型或许就能产出更好的结果。比如DSPy 的APE。难点是，要么**一个问题给一个这样的prompt，要么所有问题共用一个这样的prompt**。
2. 优化推理输出：revise output distribution。可是，优化推理输入的方法还是不够直接。难道对于每一个问题，我需要精心设计prompt，或者手动诱导模型think step by step才行。**所以能不能让模型吃下一个问题后，自动化地去做CoT的过程呢？**也就是说，现在我们希望模型在吃下一个问题后，能自主产生以下输出：
`attempt1 -> attempt2 -> attempt3 -> ...-> attempti -> answer`，其中，每个attempt包含“多个中间步骤steps + 最终结果”，也就是它在模拟人类的思考过程：先做一次attempt，然后发现问题，在此基础上在做别的attempt，直到找到最终答案。那么我要怎么让模型做到这点呢，一个直观的方法就是，如果我有：
`problem -> attempt1 -> ... -> attempti -> answer` 这种带标签的数据，那我不就能直接训练了？训练的方法也有很多，例如：
1. 我直接做sft，把最正确的attempt放在输入序列最后，当作label进行训练即可
2. 我用类似rlhf的方法，先有一个奖励模型，它能对每一个思考步骤做评估，然后利用这个评估结果，指引模型步步搜索，每一步都找到最佳的思考步骤，最后不就能找到答案了？
PS：Inference-time体现在，当用户输入一个问题之后，o1要花费更长的时间进行「思考」，其实也就是在生成最终答案之前，先生成了很多reasoning tokens。所以就得训练llm不要直接给答案（心直口快），得养成step by step（甚至带上反思）的”本能“

这两种解法，仅从训练方法上来说，都可以算成是post-training，也就是我们通过把算力花在post-training上来提升模型的逻辑推理能力。可是，本文的标题不是【把算力花在inference上】吗？inference在哪里呢？我们再重新端详这2种解法：
1. 假设我们使用解法1或者解法2 post training好了模型，现在我们拿它做推理。模型吃一个问题，产出一系列中间结果和答案，但是你能保证，这些中间结果和答案一定是最好的吗？
2. 所以此时，一方面，我们可以考虑优化推理阶段，即使用一个能够评估中间步骤的verifier，在推理时指引模型搜索出最佳答案。例如，我们对一个问题采样多个attempts链，从中找最好的。或者在单个attempts中找到最好的attempt，诸如此类。
3. 而另一方面，我们可以考虑在post-training阶段，使用这个verifier来指导模型自动化生产高质量的数据（这是个inference步骤），基于这些数据我们再做对齐。如果这个流程做得好，我们甚至可以直接信任post-training后模型的结果
所以，【优化推理输出】这一部分，你可以把算力全部花在post-training上，也可以花在post-training+inference上，从o1的技术报告上看，它应该选择了后者，同时post-training选择了某种基于强化学习的方法（其实o1在pretrain阶段应该也有变动，具体的分析我们在后文中会通过实验数据给出猜想）。至此，我们就把问题1和问题2都回答清楚了。

![](/public/upload/machine/o1_generate_verifier.jpg)

一个能按照格式，产出中间思考步骤的模型（generator），但中间思考步骤质量得不到保证。一个能对中间思考步骤进行评估的奖励模型PRM（verifier）。而现在我们想做的事情是：如何在不对generator继续做任何训练的情况下，使用verfier，来引导generator搜索出最佳的“steps + answer”？
1. 使用PRM指导搜索过程。
2. 直接改变模型的输出分布

## test-time Compute

[可视化角度具象化理解DeepSeek-R1类推理大模型的习得进程](https://mp.weixin.qq.com/s/ytKTGTgU2T7jSNrBghX1cA)

train-time compute ==> test-time Compute; Scaling Laws ==> Inference/test-time scaling

test-time Compute不是不断增加预训练预算，而是允许模式在推理过程中“思考更长时间” 。

![](/public/upload/machine/test_time_compute.jpg)

test-time Compute 可以是很多不同的东西，包括思路链、修改答案、回溯、采样等等。这些可以大致分为两类：
1. 一个是针对验证者进行搜索Search against Verifiers（抽样生成并选择最佳答案），以输出为中心。
2. 一个是修改提议分布Modifying Proposal Distribution（训练“思考”过程），以输入为中心。

![](/public/upload/machine/verifier_vs_modify_proposal_distribution.jpg)

但这几种都是需要打分奖励的，有两种类型的验证器，一个是结果奖励模型（ORM），一个是流程奖励模型（PRM），ORM只判断结果，并不关心底层过程：相比之下，PRM还会判断导致结果的过程（“推理”）。

![](/public/upload/machine/orm_vs_prm.png)

### 各种 Search against Verifiers

大概思路：首先创建多个推理过程和答案的样本，验证者（奖励模型）对生成的输出进行评分，使用验证器的一个主要优点是不需要重新训练或微调用于回答问题的LLM。

![](/public/upload/machine/search_against_verifiers.jpg)

可以细分为以下几种子类别：
1. 多数表决，也就是投票Majority Voting，最直接的方法其实不是使用奖励模型或验证器，而是进行多数投票。让模型生成多个答案，生成次数最多的答案将作为最终答案。这种方法也称为自洽，以强调生成多个答案和推理步骤的必要性。
    ![](/public/upload/machine/majority_voting.jpg)
2. 最佳N样本Best-of-N samples。LLM（通常称为提议者）使用高温或变化的温度生成多个答案。每个答案都会经过输出奖励模型 (ORM)，并根据答案的质量进行评分。得分最高的答案将被选中。除了判断答案之外，推理过程还可以通过过程奖励模型(PRM) 来判断，该模型会判断每个推理步骤的质量。会选择总权重最高的候选答案。还可以通过 RM 对每个答案候选者进行加权，并选出总权重最高的答案。这称为加权 Best-of-N 样本。
3. 使用过程奖励模型进行集束搜索Beam search with process reward models。生成答案和中间步骤的过程可以通过定向搜索进一步扩展。使用定向搜索，可以抽取多个推理步骤，每个步骤都由PRM进行判断，整个过程都会跟踪排名前3位的“beams”（得分最高的路径），快速停止那些没有结果的“推理”路径。PS: 用prm替代llm输出的logits softmax
    ![](/public/upload/machine/beam_search_with_prm.jpg)
4. 蒙特卡洛树搜索Monte Carlo Tree Search。包括四个步骤：选择（根据预先确定的公式选择给定的叶子）->扩展（创建更多节点）->推出（随机创建新节点，直到到达终点）->反向传播（根据输出更新父节点分数），这些步骤的主要目标是不断扩展最佳推理步骤，同时探索其他路径。因此，这是探索与利用之间的平衡。节点评分和选择方式的示例如下：
    ![](/public/upload/machine/mcts_selection_score.jpg)
    当选择一个新的推理步骤进行探索时，它不一定是迄今为止表现最佳的路径。使用这种类型的公式，首先选择一个节点（推理步骤），然后通过生成新的推理步骤来扩展它。和以前一样，这可以通过合理高且变化的温度值来完成：
    ![](/public/upload/machine/mcts_rm.png)
    选择其中一个扩展推理步骤，并进行多次，直到得出多个答案。这些举措可以根据推理步骤（PRM）、奖励（ORM）或两者的结合来判断。

234 都可以视为搜索算法， 核心是从LLM输出的解空间采样，进而得到最优的解决方法。

### 修改提议分布Modifying Proposal Distribution

这种方式的模型不再使用验证器（以输出为中心）搜索正确的推理步骤，而是经过训练以创建改进的推理步骤（以输入为中心），换句话说，对完成/想法/标记进行采样的分布被修改了。假设有一个问题和一个分布，我们可以从中抽取 token。一个常见的策略是获取得分最高的 token：

![](/public/upload/machine/choose_not_highest_score.jpg)

但是，请注意上图中的某些标记被标记为红色。这些标记更有可能引发推理过程：

![](/public/upload/machine/choose_more_reasoning_token.jpg)

虽然选择贪婪token不一定是错误的，但选择一个引发推理过程的令牌往往会得到更好的答案。**当修改提议分布（标记概率分布）时，本质上是让模型对分布进行重新排序，以便更频繁地选择“推理”标记**：

![](/public/upload/machine/choose_more_reasoning_token2.jpg)

修改提案分布的方法有很多种，但一般可以分为两类，通过提示工程更新提示或者训练模型以关注推理标记/过程。
1. 提示Prompting。通过提示工程，尝试通过更新提示来改进输出。为了通过提示改变提议分布，可以向模型提供示例（上下文学习），这个过程可以进一步简化，只需说“让我们一步一步思考”( “Let’s think step-by-step”)。同样，这也改变了提案的分布，使得LLM 倾向于在回答之前分解整个过程。然而，模型本身并没有学会遵循这个过程。此外，这是一个静态的线性过程，会阻碍自我完善。如果模型以错误的推理过程开始，它往往会保留它，而不是修改它。
    ![](/public/upload/machine/choose_reasoning_by_cot.jpg)
2. STaR。可以通过训练让模型学会“推理”，这样模型在生成这些推理步骤时就会得到奖励。这通常需要大量推理数据和强化学习来奖励某些行为。一种备受争议的技术被称为STaR，即自学推理机。STaR是一种利用LLM生成自身推理数据作为微调模型的输入的方法。它会生成推理步骤和答案。如果答案正确，则将推理和答案添加到三元组训练数据集`<question,reasoning,answer>`，此数据用于对模型进行监督微调。如果模型给出了错误的答案，那么我们会提供一个“提示”（正确答案），并要求模型推理为什么这个答案是正确的，最后的推理步骤是添加相同的三元组训练数据，用于对模型进行监督微调。
    ![](/public/upload/machine/star.jpg)
    
## 如何增强大语言模型的推理能力/推理LLM

[理解推理 LLM：构建和改进推理模型的方法与策略](https://mp.weixin.qq.com/s/qCLs7EbiAKcG8tafOrU4iQ)在本文中，我将"推理"定义为回答需要复杂的、多步骤生成且包含中间步骤的问题的过程。例如，"法国的首都是什么?"这样的事实性问答并不涉及推理。相比之下，"如果一列火车以 60 英里/小时的速度行驶 3 小时，它会行驶多远?"这样的问题需要一些简单的推理。推理模型通常会在回答中包含中间步骤，以部分展现思维过程。大多数现代 LLM 都具备基础推理能力，能够回答诸如"如果火车以每小时 60 英里的速度行驶 3 小时，它能走多远？"这类问题。因此，如今当我们提及推理模型时，通常指的是那些擅长处理更复杂推理任务（如解谜题、破解智力题和完成数学证明）的 LLM。

![](/public/upload/machine/llm_vs_reasoning_llm.jpg)

什么时候我们需要推理模型？ 推理模型被设计用来擅长解决复杂任务，比如解谜题、高等数学问题和具有挑战性的编程任务。然而，对于更简单的任务，如总结、翻译或基于知识的问答，并不需要推理模型。事实上，在所有任务中都使用推理模型可能会效率低下且成本高昂。例如，推理模型通常使用成本更高，输出更冗长，有时由于"过度思考"而更容易出错。

构建和改进推理模型的 4 种主要方法，以DeepSeek R1训练过程来说
1. 推理时扩展。指的是在推理过程中增加计算资源以提高输出质量。
    1. 一个直接方法是巧妙的提示工程。一个经典的例子是思维链(CoT)提示，即在输入提示中包含"一步步思考"这样的短语。这鼓励模型生成中间推理步骤，而不是直接跳到最终答案，这在更复杂的问题上通常(但不总是)能带来更准确的结果。
    2. 另一种推理时扩展的方法是采用投票和搜索策略。一个简单的例子是多数投票法，即让 LLM 生成多个答案，并通过多数表决选择正确答案。类似地，我们可以使用束搜索（beam search）和其他搜索算法来生成更优的响应。
2. 纯强化学习（RL）/DeepSeek-R1-Zero。DeepSeek R1 论文中最让人印象深刻的发现之一是，推理能力是从纯强化学习（RL）中自然产生的行为。在奖励方面，他们没有使用基于人类偏好训练的奖励模型，而是采用了两种类型的奖励：准确度奖励和格式奖励。PS：deepseek 论文用了一个词 reasoning-oriented RL 
    1. 准确度奖励使用 LeetCode 编译器验证编程答案，并使用确定性系统评估数学答案。
    2. 格式奖励依靠 LLM 判断器来确保回答遵循预期格式，比如将推理步骤放在 `<think>` 标签内。
3. 监督式微调和强化学习（SFT + RL）
    1. 使用 DeepSeek-R1-Zero 生成了SFT 数据通过指令微调训练了模型，接着又进行了一轮强化学习（RL）阶段。这个 RL 阶段保留了 DeepSeek-R1-Zero 的 RL 过程中使用的相同准确度和格式奖励。不过，他们增加了一个一致性奖励，以防止语言混合现象，即模型在回答中混用多种语言的情况。
    2. RL 阶段之后是另一轮 SFT 数据收集。在这个阶段，他们使用最新的模型检查点生成了 60 万个思维链（CoT）SFT 样本，同时使用 DeepSeek-V3 基础模型创建了额外的 20 万个基于知识的 SFT 样本。这些 60 万 + 20 万个 SFT 样本随后被用于另一轮 RL。在这个阶段，他们再次对数学和编程问题使用基于规则的方法进行准确度奖励，而对其他类型的问题则使用人类偏好标签。最终的模型 DeepSeek-R1 相比 DeepSeek-R1-Zero 有显著的性能提升
4. 纯监督式微调（SFT）和蒸馏。这里的蒸馏指的是在由更大的 LLM 生成的 SFT 数据集上对较小的 LLM（如 Llama 8B 和 70B 以及 Qwen 2.5 模型（0.5B 到 32B））进行指令微调。结果表明对于较小的模型来说，蒸馏远比纯 RL 更有效。PS：对小模型来说，方法4比方法2有效。


## 其它

[作为开发者，我如何提高任务型大模型应用的响应性能](https://mp.weixin.qq.com/s/_4s8HiRASW59V9S0YMRRww) 减少输出token、选择合适尺寸的模型以及采用流式输出。


