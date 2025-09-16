---

layout: post
title: rl微调
category: 技术
tags: MachineLearning
keywords: rl finetune

---

<script>
  MathJax = {
    tex: {
      inlineMath: [['$', '$'], ['$$', '$$']], // 支持 $和$$ 作为行内公式分隔符
      displayMath: [['$$', '$$']], // 块级公式分隔符
    },
    svg: {
      fontCache: 'global'
    }
  };
</script>
<script async src="/public/js/mathjax/es5/tex-mml-chtml.js"></script>

* TOC
{:toc}

## 简介(未完成)

精心设计的奖励函数对于有效的强化学习训练至关重要，因为它提供了优化信号，引导策略朝着理想的行为发展。在 GRPO 中，基于结果的奖励已被证明既高效又有效，无需密集的中间监督即可支持稳健的策略改进。


## 实践

### 与文档解析

[强化学习用于端到端文档解析Infinity Parser](https://github.com/infly-ai/INF-MLLM/tree/main/Infinity-Parser)核心是训练数据的处理以及强化奖励函数的设计，通过优化多个方面的奖励函数来训练模型，使其对文档布局更加敏感。
1. 奖励函数的设计，在强化阶段，使用Qwen2.5-VL-7B模型进行微调，采用GRPO进行强化学习优化，通过生成一组候选Markdown输出，并使用多方面奖励函数进行评估，从而优化模型。分别是编辑距离奖励（Edit Distance Reward）、计数奖励（Count Reward）以及顺序奖励（Order Reward）。最后总合成一个奖励函数：$R_{multi-aspect} = R_{dist} + R_{count} + R_{order}$
    1. 编辑距离奖励（Edit Distance Reward）：基于预测输出和参考输出之间的归一化Levenshtein距离，衡量语义和格式的差异；
    2. 计数奖励（Count Reward）：鼓励准确的段落分割，通过惩罚缺失或多余的段落来实现。
    3. 顺序奖励（Order Reward）：通过计算参考和预测段落之间的成对逆序数来衡量序列级别的保真度。
2. 训练数据集，包括Infinity-Doc-55K数据集，包含55,066个样本。合成方式是通过HTML模板和浏览器渲染生成，具体使用方式是：从维基百科、网络爬虫和在线语料库等来源收集文本和图像，并使用Jinja模板将采样的内容注入预定义的单列、双列或三列 HTML布局中。这些页面通过浏览器引擎渲染成扫描文档，随后进行自动过滤以去除低质量或重叠的图像。通过解析原始HTML 提取真实标注，生成对齐的Markdown表示。最终形成的数据集涵盖七个不同的文档领域，包括财务报告、医疗报告、学术论文、书籍、杂志、网页等。

### Query生成

[GRPO强化学习增强Query生成](https://zhuanlan.zhihu.com/p/1929855356072362492)查询生成既需要理解用户的意图，合理生成多样性的查询，又需要合理的扩展查询，不能生成一些干扰信息。而这些要求一般都写在prompt里面，但是模型指令遵循比较弱的时候，经常出现query生成的结果非常不好。为了生成语义上与原始查询相似但在字面表达上不同query, 设计了三个奖励函数，分别从语义相似性、文本多样性和输出格式有效性多个维度对模型输出进行评估。
1. calculate_similarity_reward：语义相似 & 表达多样性的核心奖励函数。衡量生成的 rewritten_query 是否在语义上贴近原始查询（query），但又在字面表达上具有差异性。`reward = cosine_similarity - jaccard_similarity`
  1. cosine_similarity: 使用 m3e-small 编码器将原始响应和模型生成的
  2. rewritten_query 编码为向量，通过余弦相似度衡量语义接近度。
  3. jaccard_similarity: 用于衡量重写查询和原始查询在词汇上的重合程度（即字面相似度）。
  目标是 提高语义相似度 但 降低字面相似度，因此用差值作为奖励。
2. json_format_reward_func：格式合规性检查（硬约束）。确保模型输出严格符合 JSON 格式，便于结构化解析和后续使用。只要能解析出一个 JSON 对象，就给 0.2 的奖励，否则为 0.0。
3. soft_json_format_reward_func：格式合规性检查（软约束）。当模型还没有学会完全生成正确 JSON 时，给予部分奖励，鼓励其逐步向正确格式靠拢。 如果文本中包含 `{` +0.2, 如果包含 `}` 且在 `{` 之后 +0.2, 最多奖励 0.4
提到了 unsloth框架。

### 与agent 融合/multi-agent plan/route


[Multi-Agent 的灵活编排之路](https://mp.weixin.qq.com/s/0c8hTMdIALjYdGZkmwLFDg) 案例，multiagent 背景下，训练plannning 模块生成plan（每一个step 是一个选中agent及其要解决的问题）

[无干预，短思考，多行动：新的Multi-step LLM+RL范式](https://zhuanlan.zhihu.com/p/49397670697)在R1提出后我一直在想，这种在post-train阶段reasoning trace一直变长的现象是否是个好事。由于single-step RL任务往往是完全信息的bandit问题，模型的reasoning trace越来越长我觉得是很好理解的，因为更长的reasoning可以反复重构问题中的信息达到与pretrain阶段最匹配的token分布。但是世界上的大部分现实问题都是multi-step的，也就是说需要很多步decision的sequential impact才会拿到最后的reward，这明显用multi-step MDP去model更加合理。我坚信**真正的智能必须能够解决multi-step的问题**。做出一个decision后agent其实获得了新的信息，而这些新的信息对于最后的成败至关重要。在获得能够决定最后成败的新的信息前，agent不应该给出答案。而找这些信息往往并不需要过多的reasoning，都是非常简单的事情。这就是我们近期工作的核心思想。通过一种新的post-train算法，我们希望得到的model具有三个我们所期待的性质：无干预，短思考，多行动。

[从「会说」迈向「会做」，LLM下半场：Agentic强化学习范式综述](https://mp.weixin.qq.com/s/c1LQFS4v79pF_kWfuDCthA)早期 RL 研究多基于 PBRFT 范式（输入提示、输出文本、获得一个偏好分数），可被视为退化的单步 MDP（单 prompt、一次性文本输出、立即终止），而 Agentic RL 则将 LLM 置于部分可观测马尔可夫决策过程（POMDP）下进行多步交互，其中关键变化在于动作空间从单一文本扩展为「文本 + 操作」（A_text => A_text + A_action）；同时奖励从「单步评分」扩展为「时序反馈」，优化整条决策轨迹，把 LLM 从「文本生成器」推进为**可交互的决策体**。要让 LLM 真正成为智能体，仅有动作空间还不够，它必须发展出一套完整的能力体系。
1. 规划（Planning）：为复杂任务设定子目标与多步行动序列。通过外部引导（外部打分生成奖励）或内部驱动（自主规划并修正）实现。
2. 工具使用（Tool Use）：调用外部工具完成任务。从 ReAct 等静态提示模仿演进到 Tool-integrated RL (TIR)，让智能体学会自主选择组合工具。
3. 记忆（Memory）：保持上下文连贯并积累知识，包括基于外部数据库检索记忆、Token 级别记忆和结构化记忆。中，值得关注的工作包括来自字节跳动的 MemAgent 和麻省理工大学的 MEM1，他们都通过强化学习让 LLM Agent 拥有自行管理记忆窗口的能力。
4. 自我改进（Self-Improvement）同样是目前 Agent 最热门的发展方向。
5. 推理（Reasoning）：解决复杂问题的推导能力，分为快速直觉推理（凭经验直觉迅速答题）和慢速缜密推理（多步演绎得出严谨结论）。
6. 感知（Perception）：理解多模态输入的信息获取能力。
借助强化学习，这些能力由人工启发式转变为可学习的策略，规划不再依赖硬编码流程、工具使用也可由模型自主决定、端到端训练。

### tool-use rl

1. multi turn tool-use的prompt template
2. 设计rule based reward(correctness reward, format reward, tool execution rewad等)，

[大模型Agent RL训练多轮planning技术](https://mp.weixin.qq.com/s/tRkeTwaNNEXl7tgq2qyEjw) 
1. agents rl的优点：
  1. 可以直接通过tool交互获取外部知识，进一步提升模型的准确率。
  2. DPO是一个数据驱动的方法，需要大量的数据进行训练，DPO吸收了对比学习的一些思想，所以训练好很不容易。PPO系列的方法是一个online-rl的方式，每次通过sampling的方式生成样本，然后进行训练提升，需要的数据量比DPO要小很多。
2. agents rl的不足。
  1. 真正复杂的任务可能需要几十个步骤才能完成，受限于LLM处理长序列效果下降，长序列后计算效率低等原因，**现有的rl框架还是集中在10个step左右就能完成的任务**，真实的任务往往需要30-100个step才能解决，所以距离真正能解决复杂的问题还有一段的距离。
  2. grpo虽然是rule based的方法，简化了流程，但还是需要标注数据，加上精心设计reward，最后还要调参，调数据才能得到一个不错的效果。
  3. rl需要依赖环境进行训练，一般是一些仿真环境，它的速度肯定不如gpu的计算速度快，能够加速env，跟得上rl训练的步伐也是一个需要值得考虑的问题。
  4. agent rl研究的单一的工具居多，比如code interpreter-only, web search-only等等，多个工具混合多轮调用研究的少一点。

在 ARTIST 中，rollout 的结构设计为在内部推理和与外部工具或环境的交互之间交替进行。与仅由模型生成的 token 组成的标准 RL rollout 不同，ARTIST 采用迭代框架，其中 LLM 将文本生成与工具和环境查询交织在一起。 Prompt Template: A RTIST 使用结构化的提示模板，将输出分为四个部分：

• 内部推理 (<think>...</think>)
• 工具或环境查询 (...</tool_name>)
• 工具输出 (<output>...</output>)
• 最终答案 (<answer>...</answer>)

ARTIST 为奖励设计带来了新的挑战：除了得出正确的最终答案之外，模型还必须以连贯可靠的方式构建其推理、工具使用和环境交互。为了解决这个问题，ARTIST使用了一种复合奖励机制，可以为每次部署提供细粒度的反馈。ARTIST 中的奖励函数由三个关键部分组成：

1. Answer Reward : 当模型生成正确的最终答案（如 <answer>...</answer> 标签中所示）时，该组件会分配正向奖励。答案奖励直接激励模型正确解决任务，确保推理过程的最终目标得以实现。
2. Format Reward : 为了促进结构化和可解释的推理，ARTIST引入了格式奖励，鼓励遵守规定的提示模板。该奖励检查两个主要标准：
  1. 在整个部署过程中，执行顺序——推理 (<think>)、工具调用 () 和工具输出 (<output>) 是否保持正确；
  2. 最终答案是否正确地包含在 <answer> 标签中。格式奖励有助于模型学习以一致且易于解析的方式组织其输出，这对于可靠的工具调用和下游评估至关重要。
3. Tool Execution Reward : 在每次工具交互过程中，模型的查询可能格式正确或可执行，也可能不正确。为了鼓励稳健有效的工具使用，ARTIST引入了工具执行奖励，定义为成功工具调用的比例: Tool Exection Reward = Tool success / Tool total。其中 Tool success 和 Tool total 分别表示成功调用工具的次数和总调用次数。此奖励确保模型学习生成语法正确且可在目标环境中执行的工具查询。

## 优化Planner

planner 将复杂问题拆解为多个子任务并构建 JSON 格式的 DAG，采用思维链→结构化模式，即 LLM 先在内部推理，再一键生成结构化 DAG。