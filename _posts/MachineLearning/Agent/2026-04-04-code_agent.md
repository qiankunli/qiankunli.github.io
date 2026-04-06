---

layout: post
title: Code Agent
category: 技术
tags: MachineLearning
keywords: agent software

---


## 简介（未完成）

姚顺雨：十个前沿大模型在「从上下文中学习新知识并正确应用」这件事上，平均解决率只有 17.2%。本质上是因为信息的单向传递：我们把知识塞进上下文，模型处理，输出结果，结束 —— 整个过程没有任何校验。模型哪怕完全忽略了我们给的信息，也不会有人告诉它。Coding 场景不同。代码写完可以跑，跑完有结果，结果对不对有明确判据：编译报错、测试失败、运行崩溃。这不是谁设计的，是软件工程自带的基础设施。用控制工程的话说，这叫闭环控制（Closed-loop Control）：系统不只产生输出，还能感知输出与目标的偏差，并用偏差驱动修正。

## AI Coding

用AI生成代码，本质上和生成其他非结构化的序列，是一样的。但代码实在是太特殊了，它是数字世界的源头。理论上说，只要我们能可控地生成代码，就能可控地执行任意流程，意味着可以在数字世界做任何事情。所以现在很多人在说，Coding Agent就是通用智能体，这有相当的道理。这一论断在现实中碰到的阻碍，目前看只有两个：
1. 可控地生成代码，仍然是一件专业的事情，而非普通大众能够完成。
2. 并非所有软件都有编程接口，可以使用代码来进行控制。当然，第二个障碍是相对的。只要进入到某个系统层面上，拿到相应的系统权限，也没有什么软件是无法用代码控制的。难易程度的区别。


有什么
1. agent loop
  1. prompt context = Base instructions + Tools Defintion + Input(developr + user config + history + query) 
  2. 有的一层loop，有的两层loop。一般内层loop 是已知的：llm call + tool call
2. tools. agent 能做什么完全由工具集决定
3. compact，何时触发，compact prompt.
4. plan mode. 会给模型注入一份专门的instructions（plan.md) 



## codex

codex Plan mode只在提示词里要求不要用update_plan, 积极使用request_user_input提问。



## claude code

### 消息类型系统

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

### 并行

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



