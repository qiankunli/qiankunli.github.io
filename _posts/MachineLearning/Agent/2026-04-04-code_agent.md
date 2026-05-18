---

layout: post
title: Code Agent
category: 技术
tags: MachineLearning
keywords: agent software

---


## 简介（未完成）

姚顺雨：十个前沿大模型在「从上下文中学习新知识并正确应用」这件事上，平均解决率只有 17.2%。本质上是因为信息的单向传递：我们把知识塞进上下文，模型处理，输出结果，结束 —— 整个过程没有任何校验。模型哪怕完全忽略了我们给的信息，也不会有人告诉它。Coding 场景不同。代码写完可以跑，跑完有结果，结果对不对有明确判据：编译报错、测试失败、运行崩溃。这不是谁设计的，是软件工程自带的基础设施。用控制工程的话说，这叫闭环控制（Closed-loop Control）：系统不只产生输出，还能感知输出与目标的偏差，并用偏差驱动修正。

AI 是一个发散引擎。 你给它一句话，它还你五百行。你再给它一句话，它又还你五百行。但人，天然应该是那个收敛引擎。判断什么该要、什么该砍、哪里需要约束、哪里可以自由。把足够多的、边界清晰的收敛动作交给系统，人的收敛能力才能聚焦到机器做不了的事上——架构的取舍、设计的品味、需求的优先级。河岸不限制河——河岸定义了河。没有岸的河，是洪水。洪水没有方向，没有深度，哪里都去，哪里都到不了。河有了岸，才有流向，才有流量，才有力量。约束不是禁止，是指向。如果约束是"禁止"，你的目标就是堵。堵 console.log、堵裸 fetch、堵硬编码密钥。堵完一个再堵下一个，无穷无尽。如果约束是"指向"，你的目标是定义一个边界清晰的行动空间。在这个空间里，速度可以最大化。碰到空间边缘，系统给出信号、给出理由、给出替代方案。你接受的是教育，不是惩罚。**可形式化的走执行层，不可形式化的走认知层。高频的走近实时拦截，低频的走近延迟检查**。

## AI Coding

用AI生成代码，本质上和生成其他非结构化的序列，是一样的。但代码实在是太特殊了，它是数字世界的源头。理论上说，只要我们能可控地生成代码，就能可控地执行任意流程，意味着可以在数字世界做任何事情。所以现在很多人在说，Coding Agent就是通用智能体，这有相当的道理。这一论断在现实中碰到的阻碍，目前看只有两个：
1. 可控地生成代码，仍然是一件专业的事情，而非普通大众能够完成。
2. 并非所有软件都有编程接口，可以使用代码来进行控制。当然，第二个障碍是相对的。只要进入到某个系统层面上，拿到相应的系统权限，也没有什么软件是无法用代码控制的。难易程度的区别。

构建一个 Agent 需要做一系列架构决策：上下文怎么管理？工具怎么加载？工具怎么查找？Agent 的主循环围绕什么来设计？
1. agent loop（vs  DAG、状态机或图执行引擎），围绕循环建立足够完善的错误恢复、状态管理和成本控制机制。将用户的高层目标转化为可执行的步骤序列
  1. prompt context = Base instructions + Tools Defintion + Input(developr + user config + history + query)  ==》 如果模型响应中包含工具调用请求就执行工具并将结果追加到上下文，开始下一次迭代。当模型不再请求工具调用时，循环终止。
    1. Prompt Assembly Architecture. claude getSystemPrompt 先拼静态前缀：身份定位、系统规范、任务哲学、风险动作规范、工具使用规范等，再注入动态后缀：session guidance、memory、MCP instructions、scratchpad、token budget、output style.
  2. 有的一层loop，有的两层loop。一般内层loop 是已知的：llm call + tool call
  3. plan mode. 会给模型注入一份专门的instructions（plan.md) 
2. tools. agent 能做什么完全由工具集决定。工具治理。
  1. 不等llm 流式输出tool call list完成即调用tool
  1. 工具并行执行，冲突处理（多个tool 操作rw 同一个文件），如何处理工具执行中的错误？
  3. 少量通用 vs 大量专用 ==> 信任现实workflow vs 信任模型推理能力提高？
  4. 延迟加载。比如系统里有 50 个工具，首轮请求只发 5 个核心工具 + “还有其他延迟加载工具可检索”的提示，模型这时只真正知道那 5 个的详细参数，如果模型后来搜索到 “github pr review” 相关工具，运行时再把那几个 GitHub 工具的完整 schema 发给模型，从这一刻开始，这几个工具就算“已发现”，后续请求只需要继续带这几类已发现工具，不必带全部 50 个。
  5. 工具执行链：不是直调，是 Pipeline. claude  toolExecution.ts 里的工具执行链路有大概14个步骤。
  5. 工具的行为属性由输入决定，而非由工具类型决定。并发安全性、只读性、是否不可逆都是以输入为参数的方法，而不是静态属性。这意味着同一个 BashTool 在执行 ls 时是并发安全的，但在执行 rm -rf 时显然不是。每个工具还声明了结果大小上限，超出的结果会被持久化到磁盘并替换为预览。
3. 上下文管理。
  1. compact，何时触发，compact prompt，很多时候是一个compact pipeline. 当对话历史和工具结果不断累积，逼近 token 上限时，系统如何决定保留什么、丢弃什么、压缩什么？
  1. memory 解决压缩后的信息衰减
4. 安全。一个可以在用户机器上执行任意 Bash 命令的 Agent，如何在自主性和安全性之间取得平衡？
  2. 权限系统的一个关键抽象是将权限检查封装为一个可插拔的异步函数接口：给定工具、输入和上下文，返回"允许/拒绝/询问"三种决定之一。工具执行管线完全不需要关心权限的具体呈现方式（cli/web/app），都由同一个函数签名统一处理。在交互式权限确认的实现中，有一个精巧的细节：系统会在用户交互的同时异步运行后台安全分类器，两者进行一次"竞赛"。为了防止用户的意外按键（比如在分类器返回结果之前按下回车）取消分类器的检查，系统设置了 200 毫秒的"首次交互宽限期"。这种在用户体验的即时性和安全检查的完整性之间寻找平衡的处理方式，在安全系统设计中是一个值得关注的模式。
5. 扩展
  1. hook,  runtime governance layer. PreToolUse 可以改写输入、直接给 allow/ask/deny、甚至 preventContinuation；PostToolUse 还能追加 message、注入 additional context。
  1. gate-first。工具调用前的权限 gate:这个工具能不能调; 输出后的验证 gate：结果是否达标、是否合规;高风险动作的审批 gate：要不要人工确认; 流程推进的 merge/release gate：测试、评审、检查没过就不能往下走.

Claude Code 还提供了一个显式的任务列表工具作为规划手段。它的状态管理按会话或 Agent 实例隔离，当所有任务项都标记为 completed 时列表会被自动清空。

### 消息类型系统

消息类型/模型 ==> loop state 模型 ==> agent loop
```
// src/types/message.ts — 消息类型层次
type Message =
  | UserMessage           // 人类输入（或工具结果）
  | AssistantMessage      // 模型响应（文本 + 工具调用）
  | AttachmentMessage     // 记忆/资源附件
  | SystemMessage         // 系统消息
  | SystemLocalCommandMessage  // 本地工具结果（bash, read 等）
  | ToolUseSummaryMessage // 压缩后的工具历史
  | TombstoneMessage      // 已删除消息标记
  | ProgressMessage       // 流式进度更新
```

loop state 模型
1. Claude Code 的做法是将消息列表、上下文压缩追踪、输出 token 恢复计数、中断钩子状态等十个字段封装进一个显式的状态类型。每一次循环迭代在开头解构这个对象，在结尾整体赋值来"提交"变更。

## codex

codex Plan mode只在提示词里要求不要用update_plan, 积极使用request_user_input提问。


## claude code
PS：粗看（人去梳理的时候，都是只能先从粗的看），一些设计不算让人意想不到，但细看，都是细节
1. 每个 harness 组件都编码了一个假设。每一个 harness 组件都编码了一个关于”模型自身做不到什么”的假设，这些假设会随着模型进步而过时。不是说 harness 是临时的，而是旧的假设会过时，新的假设会取代它们。
1. Anthropic 对上下文的珍惜程度，远超大多数人的想象。
  2. system prompt 的动静边界 + cache boundary. SYSTEM_PROMPT_DYNAMIC_BOUNDARY 分割静态和动态上下文， 边界前尽量可 cache，边界后是会话特定内容，不能乱改，否则破坏 cache 逻辑。把 prompt 当作可编排的运行时资源来管理。
  2. fork path 共享父线程的 prompt cache.
  3. skill 按需注入，不塞进全局 prompt
  4. MCP instructions 按连接状态动态注入
  5. function result clearing、summarize tool results
  6. compact / transcript / resume 机制
3. 把好行为制度化。getSimpleDoingTasksSection 非常明确地约束模型：不要加用户没要求的功能，先读代码再改代码等等，这一段的意义，远不止“prompt 写得细”。它揭示了一个本质：Claude Code 不把“好习惯”交给模型即兴发挥，而是写进规则里，强制执行。把行为方差压到了最低。同样制度化的，还有风险动作规范：destructive operations、hard-to-reverse operations、修改共享状态、上传第三方工具……全部被标记为“需要确认”。源码里明确要求：发现陌生状态先调查，merge conflict/lockfile 不要粗暴删。
4.  Agent 分工.General Purpose Agent不是一个万能 worker，源码里至少定义了这几个内建 Agent：
  1. Explore Agent：纯读模式，专门做代码探索。它的 system prompt 明确规定——不能创建文件、不能修改、不能删除、不能移动、不能用重定向写文件，Bash 只允许 ls、git status、git log、git diff、find、grep、cat、head、tail 这些读操作。
  2. Plan Agent：纯规划，不做编辑。它的职责是先理解需求、探索代码库、输出 step-by-step implementation plan，最后必须列出 Critical Files for Implementation。
  3. Verification Agent：这是我最惊艳的一个。它的 system prompt 开头就点明——你的工作是 try to break it。不是“确认实现看起来没问题”，而是主动找茬。它强制要求：跑 build、跑 test suite、跑 linter/type-check、做专项验证、跑浏览器自动化或 curl 实测、输出每个 check 的 command 和 observed，最后必须输出 VERDICT: PASS / FAIL / PARTIAL。





### 并行
解决强单体agent无法并发硬伤：当一个编程任务的复杂度超出单次推理循环的处理能力时，Claude Code 通过子 Agent 机制实现多层规划。主 Agent 可以生成子 Agent 来处理独立的子任务，每个子 Agent 拥有独立的 200K 上下文窗口，只将摘要结果返回给父 Agent。一个重要的架构约束是递归深度限制为 1，子 Agent 不能再生成自己的子 Agent。

**Claude Code 提供三种并行工作的方式，从轻到重：Subagent ==> Background ==>  Agent Team**
1.  Subagent 在你的主会话内部生成一个独立的 Claude 实例。它有自己的上下文窗口，干完活后把结果返回给你的主会话。使用方式
  1. 临时口头调用（Ad-hoc），直接用自然语言告诉 Claude “派一个子 Agent 去做什么”。
  2. 预定义自定义 Agent（Custom Agent），在项目中创建一个 Markdown 文件(`.claude/agents/xxx.md   `)，定义好 Agent 的名字、角色、工具权限。之后 Claude 会根据任务自动匹配调用，或者你也可以手动指名调用。
2. Background。你让 Claude 做一件耗时的事（跑测试、编译、分析），但不想干等着。后台任务让它在后台执行，你继续和 Claude 聊别的。使用方式
  1. 用 & 前缀（类似 Unix shell） `& npm run test:all`
  2. 自然语言 `> 在后台跑一下完整的测试套件`
3. Agent Team. 它不是「一个 Claude 带几个子进程」，而是多个完全独立的 Claude 实例，各自有自己的终端面板（如果你用 tmux），可以互相通信、互相发现对方的工作成果、互相挑战观点。有一个 Lead Agent（团队领导）负责协调，其他 Teammate 并行工作。使用方式： 自然语言
  ```
  > 创建一个 Agent Team 来审查我们的认证系统。
  > 启动三个 Teammate：
  > - 安全审查员：审计漏洞，检查 token 处理方式
  > - 性能分析师：分析响应时间，找出瓶颈
  > - 测试覆盖检查员：验证边界条件，找出未覆盖的路径
  > 让他们共享发现并通过任务清单协调。
  ```

启动subagent方式和调用一个普通tool一样，只不过调用的是Agent这个tool。main-agent的每次调用都附带着 3-5个词的简短描述；并且提供足够清晰详细的prompt，说明 sub-agent 要做什么，因为在一般情况下（只要不指定CAN_FORK_CONTEXT=True），sub-agent就不会继承主agent的上下文，因此它对整个问题的背景是完全不了解的，它能看到的只是主agent给它委派的prompt。Subagent 完成后，它的最后一条 assistant message 会作为字符串返回给主 agent，这个消息对于用户是不可见的。主agent需要自己把这个结果整理一下，然后用文字告诉用户。同时返回的还有 agent ID，可供后续 resume 使用（sub-agent的resume和主agent一样，可以恢复上一次调用时候完整的上下文继续工作）。
```
{
    "role": "assistant",
    "content": [
      {
        "name": "Agent",  
        "type": "tool_use",
        "input": {
          "prompt": "探索xxxx 代码库的整体结构。请找出：\n1. 项目根目录下的所有文件和目录\n2. 主要的配置文件（package.json, composer.json, readme.txt, README.md 等）\n3. 主要入口文件\n4. 项目使用的主要编程语言\n\n请读取 readme.txt 或 README.md、package.json、composer.json（如果存在）的内容，并详细报告。",
          "description": "探索项目整体结构和配置",
          "subagent_type": "Explore",
          "run_in_background": true
        }
      }
    ]
}
```

## agent team

"多 Agent + 软件工程团队角色"是不是研发的全能解？我的判断是：它是全能解，但不是最优解。说它是"全能解"，因为它确实能 cover 软件工程的完整生命周期。需求分析、方案设计、代码实现、测试验证、文档维护——每个环节都可以由对应角色的 Agent 来执行，而且已经有人在这么做了。说它"不是最优解"，因为不是所有项目都需要完整的软件工程团队角色支持。每个项目有自己的特性——从个人开发者的 side project 到小团队的快速迭代，很多优秀的软件从来不需要 PM 写 PRD、Architect 画架构图、SM 管 Sprint。即使对于确实需要多角色协作的大型项目，角色化本身也引入了具体的成本：
1. 角色切换的上下文损耗。每次角色切换都有一次完整的上下文重建。而一个人 + 单 Agent 走完全流程，上下文是连续积累的，没有切换损耗。
2. 文档交接的信息衰减。 角色之间通过文档传递信息——PM 写 PRD 给 Architect，Architect 写架构给 DEV。每一次从"理解"到"文档"再到"理解"的转换都有信息损失。真实团队中这个损耗靠会议和即时沟通来弥补，Agent 团队靠什么？反观个人开发者，需求、设计、实现都在一个人的脑子里，信息衰减天然最小——Agent 只需要辅助执行，不需要跨角色"翻译"。
3. 过程正确不等于结果正确。 有了完整的工作流（头脑风暴→PRD→架构→Story→开发→Review），流程上无懈可击。流程完备不能弥补单步质量的不足——每一步的输出质量取决于模型能力，而不是流程设计。 与其在流程上做到完美，不如把精力投入到每一步的上下文质量上——这才是决定 Agent 输出质量的关键变量。

大多数项目，单 Agent + 结构化上下文就够了。关键环节按需引入角色化——需求分析、代码审查这些确实需要不同视角的环节，再引入专门的 Agent。并行用在执行层，不是决策层——代码审查 6 个 Checker 并行、批量测试并行，这些执行性工作的并行收益最直接。但方案设计、架构决策这些需要深度思考的环节，并行反而带来冲突。

## 其它

赋予智能体计算机访问权限（文件系统+Shell环境）是能力跃迁的关键——文件系统提供持久化上下文存储，Shell使其调用工具链、CLI或自主生成代码。

AI Coding 解决了"怎么写"的问题。你不需要苦学语法、不需要背 API、不需要记住配置文件的格式。但有一个问题它永远解决不了。"什么算好？"这不是技术问题。这是价值观问题。价值观的建立，只能靠人。新人不需要练一万个小时就能写出能跑的代码。一万个小时被压缩成了十个小时。但品味没有压缩——品味不服从速度。你不能加速一个人的判断力形成。那怎么办？只有一条路：把品味编码成结构。把默会知识变成显性约束。把"老王觉得"变成"系统执行"。这不是用机器替代人的判断。这是让系统处理"对与错"——那些边界清晰、可精确描述的东西。让人的判断力聚焦在"好与更好"——那些需要品味、上下文、整体感的东西。PS：品味这个东西，训练数据不够多，不同人也是随机的。





