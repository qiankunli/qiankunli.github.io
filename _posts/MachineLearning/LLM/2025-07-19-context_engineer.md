---

layout: post
title: 提升Agent能力——上下文工程
category: 技术
tags: MachineLearning
keywords: context engineer

---

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

* TOC
{:toc}

## 简介(未完成)

对于大多数人来说，处理大量的提示词非常令人头疼，几乎没人愿意花费大量的时间去精心设计这些词。用户更倾向于用一句话、一个简单指令，来表达需求。我们做的事情就是基于用户的上下文来补充信息。这里的上下文与大模型的上下文维度有所不同。大模型的上下文通常是基于单次输入的上下文，而**我们所指的上下文则是用户的全部历史数据**，甚至可能追溯到用户最初使用平台时的数据。在上下文的处理上，大家通常采用的方式大体相似，包括检索召回、知识库、向量库、知识图谱等，但关键问题在于如何高效地利用和整合这些数据。问题可能不是数据不够，而是没有高效利用数据。数据并不是越多就越有效，我们需要对数据进行有规则的筛选和保留，**以保证数据的正交性**。

## 未命名

[从Prompt Engineering到Context Engineering](https://mp.weixin.qq.com/s/nyD5Vc59FYO_ZUD8fSquJw)**跟AI开发相关的大部分工作，都是围绕着如何把上下文窗口填充正确来进行的**。随着LLM性能的进步，人们不再需要为了想出一个像咒语一样的prompt而绞尽脑汁了。但是，随着agent系统的动态性、复杂性逐步增加，保持每一次都能把context组装正确和完整，已经不是一件简单的事情了。这就需要Context Engineering这样一个专业的词汇来指代一整套系统化的方案。Context Engineering包含了所有对组装正确的上下文起到关键作用的技术组件。为了从大量文档内容中选出跟当前任务更相关的数据，就需要retrieve技术（RAG）；为了向模型传达长期记忆和短期记忆，就需要memory工程；为了更好地决策未来，就需要把当前状态以及历史信息传达给模型；另外，还需要一系列的错误处理、恢复、以及guardrails机制。所有这些，都属于Context Engineering的范畴。**至少包括**：
1. 静态的prompt及instruction。
2. RAG返回的片段。
3. web搜索返回的页面内容。
4. 对于工具候选集合的描述。
5. 工具调用的历史结果。
6. 长期记忆及短期记忆。
7. 程序运行的其他历史轨迹信息。
8. 出错信息。
9. 系统执行过程中通过human-in-the-loop获取到的用户反馈。
Context Engineering并不是某一种具体的技术，而更像是一种思想或观念。它也暗含了AI技术圈（尤其是深入一线的工程师们）对于未来技术趋势的一种判断。**AI应用开发在本质上可以看成是，从海量信息中找到恰当的有效信息，最终适配到LLM的上下文窗口上。为了让这个漏斗工作得更高效，你需要检索、过滤、排序**。你需要一套完整的Context Engineering工程架构。PS: **其实主要就是指令、记忆、知识。优化context（构建一个好用的单agent）也是从优化这几个方面着手：记忆召回+工具选择+知识检索**。

![](/public/upload/machine/context_engineering.png)

[别再构建多智能体了](https://mp.weixin.qq.com/s/IPaUMtZDS8ws3FpihfKZnw)来自全球首位AI程序员Devin，热门AI应用DeepWiki的开发团队，Cognition AI认为在2025年的技术水平下，追求让多个AI智能体并行协作的架构，是一种脆弱且极易失败的歧途。为什么？关键在于“上下文灾难”：
1. 信息孤岛： 并行工作的子智能体无法看到彼此的进展和决策，就像蒙着眼睛的工匠，最终做出的“零件”风格迥异、无法组装。
2. 决策冲突： 智能体的每一个行动都包含着“隐性决策”。当多个智能体独立决策时，这些决策极有可能相互冲突，导致整个项目走向混乱。

出路何在？拥抱“上下文工程（Context Engineering）”：Cognition AI 团队提出，构建可靠AI智能体的关键，不是增加智能体的数量，而是精细化地管理和传递信息。他们主张采用单线程线性架构，确保信息流的完整和连续，让每一步行动都基于完整的历史背景。对于超长任务，他们则提出用一个专门的模型来智能“压缩上下文”，而非粗暴地将任务分包。

HTML于1993年问世。2013年，Facebook向世界发布了React。如今已是2025年，React（及其衍生技术）主导了开发者构建网站和应用的方式。为什么？因为React不仅仅是一个编写代码的脚手架，它是一种哲学。通过使用React，你欣然接受了一种以响应式和模块化模式构建应用的方式——人们现在认为这是一种标准要求，但在早期Web开发者看来，这并非理所当然。在LLM和构建AI智能体的时代，**感觉我们仍像是在玩弄原始的HTML和CSS**，试图弄清楚如何将它们组合起来以创造良好的体验。**除了某些最基础的概念外，还没有哪一种构建智能体的方法成为标准**。

在2025年，市面上的模型已经极其智能。但即使是最聪明的人，如果缺乏对任务上下文的理解，也无法有效地完成工作。“提示工程（Prompt engineering）”这个词被创造出来，指的是为LLM聊天机器人以理想格式编写任务所需的努力。而“上下文工程”则是它的下一个层次。它关乎在一个动态系统中自动完成这件事。这需要更精细的把握，并且实际上是构建AI智能体的工程师们的首要工作。以一种常见的智能体类型为例。这种智能体：
1. 将工作分解成多个部分
2. 启动子智能体来处理这些部分
3. 最后（一个总结智能体）将结果合并
这是一个诱人的架构，特别是当你的任务领域包含多个并行组件时。然而，它非常脆弱。关键的失败点在于：假设你的任务是“构建一个Flappy Bird的克隆版”。它被分解为子任务1“构建一个带有绿色管道和碰撞区的移动游戏背景”和子任务2“构建一个可以上下移动的小鸟”。结果，子智能体1实际上误解了你的子任务，开始构建一个看起来像《超级马里奥》的背景。子智能体2为你构建了一只鸟，但它看起来不像游戏素材，其移动方式也与Flappy Bird中的完全不同。现在，最终的智能体只能面对一个棘手的任务：将这两个沟通失误的产物组合起来。

[从Prompt Engineering到Context Engineering](https://mp.weixin.qq.com/s/nyD5Vc59FYO_ZUD8fSquJw)具备高度自主性的Agent，一般来说是由agent loop驱动的运行模式。在每一个循环迭代中，它借助LLM动态决策，自动调用适当的工具，存取恰当的记忆，向着任务目标不断前进，最终完成原始任务。然而，这种agent loop的运行模式，直接拿到企业生产环境中却很难长时间稳定运行。这种所谓的「tool calling loop」在连续运行10~20轮次之后一般就会进入非常混乱的状态，导致LLM再也无法从中恢复。Dex Horthy质疑道，即使你通过努力调试让你的Agent在90%的情况下都运行正确，这还是远远达不到“足以交付给客户使用”的标准。想象一下，应用程序在10%的情况下会崩溃掉，没有人能够接受这个。可以说，**Agent无法长时间稳定运行的原因，大部分都能归结到系统送给LLM的上下文 (Context) 不够准确**。
1. 所以说，Context Engineering产生的第一个背景就是，AI技术落地已经进入了一个非常专业化的时代。这就好比，对于流行歌曲，很多人都能哼上两句。你不管是自娱自乐，还是朋友聚会唱K，这当然没问题。但是，如果你真的要去参加“中国好声音”并拿个名次回来，那就不是一回事了。类似地，Context Engineering这一概念的提出，对于Agent开发的交付质量提升到了专业工程学的高度，它要求你的系统要尽最大可能确保LLM上下文准确无误。
2. Context Engineering产生的第二个背景，来源于LLM的技术本质，它具有天然的不确定性。LLM的底层运行原理，基于概率统计的 predictnexttoken。概率是充满不确定性的，模型本身的行为就不能被精确控制。在模型训练完成之后的生产运行环境中，**你只能通过精细地调整Context来「间接地」引导它的行为**。在很多现实场景中，都采取了较为保守的做法，在现有的业务流程代码中，穿插着调用一两次LLM，对于这种简单的情形，只要在调用的局部把LLM所需的prompt提前设计好、调试好，系统就可以上生产环境了。但是，在更复杂、更高自主性的Agent系统中，对于prompt的管理就没有这么简单了。资深的AI从业者Nate Jones把Context Engineering大体分成两部分。
    1. 第一部分 (the smaller part)，称为deterministic context。这部分指的是我们直接发送给LLM的上下文，包括指令、规则、上传的文档等等，总之它们是可以确定性地进行控制的 (deterministically control）。
    2. 第二部分 (the larger part) ，称为probabilistic context。这部分指的是，当LLM需要访问web以及外部工具的时候，会不可避免地将大量不确定的信息引入到LLM的上下文窗口。典型地，Deep Research就是属于这一类的技术。在这种情况下，我们能直接控制的上下文内容，只占整个上下文窗口的很小一部分（相反，来自web搜索和工具返回的内容，占据了上下文窗口的大部分）。因此，针对probabilistic context这一部分的上下文，你就很难像针对deterministic context那样，对prompt进行精细地微控制 (micro control) 。
    总之，LLM本身的不确定性，加上访问web和外部工具带来的context的不确定性，与企业对于系统稳定运行的要求存在天然的矛盾。这样的难题解决起来，就需要更多、更系统的工程智慧。这成为Context Engineering产生的第二个背景。
3. 至于Agent执行会失败的具体技术原因，更进一步拆解的话，可以归结为两个方面：
    1. 第一，模型本身不够好或者参数不够，即使有了正确的context还是生成了错误结果。
    2. 第二，模型没有被传递恰当的上下文。在实际中，占大多数。这第二个原因，又可以细分成两类：
        1. 上下文不充分，缺失必要的信息 (missing context) 。
        2. 上下文的格式不够好 (formatted poorly) 。类比人类，如果说话没有条例，颠三倒四，即使所有信息都提到了，仍然可能无法传达核心信息。
        3. 上下文污染，幻觉信息混入决策链
        3. 上下文混淆，冗余信息导致推理错误
        4. 上下文冲突，不同轮之间信息自相矛盾
        5. 上下文干扰，重点内容被淹没，性能下降

## 实践

[Agent时代上下文工程的6大技巧](https://mp.weixin.qq.com/s/LdAMqqn54rRReYXS4iuU0w)
1. KV-Cache 命中率，是直接决定Agent的延迟和成本的关键指标。先来看AI Agent的运行方式：用户输入 → 模型按当前上下文挑动作  → 动作在沙箱执行 → 结果写回上下文 → 进入下一次迭代重新按当前上下文挑动作 →  ... → 任务完成。可以看出，上下文在每一步都会增长，而输出的Function Call结果通常相对较短，以 Manus 为例，平均输入与输出 token 的比例约为 100:1。幸运的是，拥有相同前缀的上下文可以利用 KV 缓存（KV-cache）机制，极大降低首个 token 生成时间（TTFT）和推理成本。以 Claude Sonnet 为例，缓存的输入 token 价格为 0.30 美元/百万 token，而未缓存的则高达 3 美元/百万 token，相差 10 倍，很夸张的节省了。从上下文工程的角度看，提升 KV 缓存命中率的关键要点如下：
    1. 让 prompt 前缀绝对稳定。由于LLM的自回归属性，只要有一个 token 存在变动，就会让缓存从该 token 之后开始失效。一个常见的错误是在系统提示词的开始加入时间戳，尤其是精确到秒级，会直接让缓存命中率归0.
    2. 上下文只能追加。避免修改之前的动作以及观测结果，确保你的序列化过程是确定性的。很多编程语言和库在序列化 JSON 对象时并不保证键的顺序稳定，这会在悄无声息地破坏掉缓存。
    3. 需要时明确标记缓存断点。一些模型提供方或推理框架并不支持自动增量前缀缓存，需要手动在上下文中插入缓存断点。注意在设置断点时，要考虑潜缓存可能过期的时间，至少确保断点包含在系统提示词的结尾。
    如果是使用vLLM等框架时，请记得打开 prefix caching，并用 session ID 把请求路由到同一worker。
2. 利用Mask，而非删除。Agent系统中，能力越多，那么工具就需要越多。尤其是MCP大火，如果允许用户自定义配置工具，会有人塞上百个来历不明的工具到你构建的动作空间里。显而易见，模型会更容易选错行动，或者采取低效路径，就是**工具越多的Agent，可能越笨**。一般的做法就是动态加载/卸载工具，类似RAG一样，但Manus尝试过之后，都是血的教训
  1. 工具定义通常在上下文最前面，任何增删都会炸掉 KV-Cache。
  2. 在history里提到的工具一旦消失，模型会困惑甚至幻觉。
  结论就是：除非绝对必要，否则避免在迭代中途动态增删工具。Manus 的解法就是，不动工具定义，利用上下文感知的状态机（state machine）来管理工具，在解码阶段用 **logits mask 阻止或强制选择某些动作**。在实践中，大多数模型提供商和推理框架都支持某种形式的响应预填充，以 NousResearch 的 Hermes 格式为例，
  1. Auto：模型可以选择是否调用函数，通过仅预填充回复前缀（`<|im_start|>assistant`）可实现。
  2. Required：模型必须调用函数，但具体调用哪个函数不受限制，通过预填充到工具调用标记（`<|im_start|>assistant<tool_call>`）可实现。
3. 将文件系统作为上下文。即使大多数模型上下文长度以及支持到128K，但在真实的智能体场景中任然不够，甚至是一种负担：
  1. 观测结果可能极其庞大，当与网页、PDF 这类非结构化数据交互时，上下文长度限制很容易就爆表
  2. 即使模型支持很长上下文，但一般在超过一定长度之后模型性能会有一定的下降。实验显示，在32k tokens的上下文中，**模型对开头部分信息的回忆准确率比中间部分高出40%以上**。
  3. 长上下文输入即使有前缀缓存，依然很贵。
  常见做法是截断或压缩，但不可逆压缩必丢信息，你永远不知道第 10 步会用到哪条观测值。Manus的做法，是把文件系统视为终极上下文，无限大、持久、可由模型直接操作读写，把文件系统不仅当作存储，更当作外部结构化的记忆。具体的压缩策略是保证内容可复原，例如，网页内容可暂时从上下文删掉，但保留原有的URL ；文档内容只要在沙盒中的路径可用于（查找？），那么内容可以也可以被省略。让Manus在不永久丢失信息的前提下，缩减上下文长度。甚至幻想：如果 State Space Model（SSM）能把长期状态外化到文件，而非留在上下文，它们可能成为下一代智能体。
  ![](/public/upload/machine/context_file.png)
4. 通过复述操纵注意力。用过 Manus 的人会注意到它爱创建 todo.md，做完一条勾一条。这不是看起来可爱，而是精心设计的注意力操控机制。Manus中**一个任务一般需要50次工具调用**，在50步的长链中，LLM很容易出行跑题现象，偏离原有主题或忘记早期目标。通过不断重写待办清单，将任务目标不断复述到上下文末尾，相当于用自然语言把目标塞进最近注意力，避免中间遗忘（lost in the middle）。
  ![](/public/upload/machine/context_todo.png)
5. 保留错误内容在上下文中。智能体一定会犯错，LLM的幻觉、环境的报错、工具的抽风，这不是BUG，而是现实。在多步任务中，失败不是例外，而是循环的一部分。常见做法是隐藏这些错误：清理痕迹、重试动作，或者重置模型状态，然后把它交给神奇的“温度”。看起来似乎更安全、更可控。但这会抹掉证据，模型学不到教训。Manus发现：把错误留在上下文里，模型看到失败动作后，会隐式地更新其内部认知，降低重复犯错的概率。认为错误恢复能力是真正具备智能体行为的最明确的指标之一。但现在大多数学术研究和公开基准测试中，往往忽略了这一点，只关注了在理想条件下的任务成功率。
  ![](/public/upload/machine/context_wrong_stuff.png)
6. 不要被few-shot误导。少样本提示（Few-shot Prompting）是提升LLM输出的常用手段，但在Agent系统中可能会适得其反。LLM是出色的模仿者，若上下文里都是大量相似的动作-观测对，模型会倾向遵循这种形式，哪怕者并不是最优解。例如，当使用 Manus 协助审阅20 份简历时，Agent往往会因为上下文里出现了类似操作，就不断重复，陷入一种循环，最终导致行为漂移、过度泛化，有时产生幻觉。Manus的做法：增加多样性。在动作和观察中引入少量结构化变化，例如采用不同序列化模板、措辞、在顺序或格式上加入噪音等，打破惯性。总之，上下文越单一，智能体越脆弱。
  ![](/public/upload/machine/context_do_not_few_shot.png)

PS：这种表达方式很有意思，以context 视角来看很多工程化手段。

## 领域

### AI代码

[AI写代码的“上下文陷阱”：为什么AI总是写错？如何系统性解决？](https://mp.weixin.qq.com/s/dAknYxHhGd0xDNqn9cB73Q)核心是系统性的分层分模块管理上下文，最大程度让AI辅助维护上下文，每次变更的知识负担就可以大幅度下降(输入具体需求即可)，配合渐进式的更新维护，使用AI编程工具的效率正循环就形成了。

