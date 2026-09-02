---

layout: post
title: 群聊
category: 技术
tags: MachineLearning
keywords: agent team

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

## 简介

主流多 Agent 框架（CrewAI、AutoGen、MetaGPT）大多采用 Leader-Worker 架构。有基于 capability 的分配和结果验证，但本质仍是调度器，没有方案讨论、共识达成、动态重规划的交互。核心缺陷：Worker 之间完全隔离，缺少 peer-to-peer 的横向信息流。

[从 ReAct 到 Agent Teams：一个工程师视角的 Agent 协作机制思考](https://mp.weixin.qq.com/s/T_sYOS11KrOijp_aCEcgnQ)AgentScope 团队开源的 AgentTeams把「多 Agent 协作平台」当作一个云原生系统来做：Kubernetes 控制面（agentteams-controller）通过 CRD 声明式管理 Worker/Team/Manager，Higress AI Gateway 统一托管模型密钥与 MCP（Worker 只拿消费令牌，从架构上根除凭据外泄），Matrix 协议（Tuwunel）作为 Agent 与人共用的通信总线，MinIO 提供跨 Worker 的共享文件系统降低 token 消耗，Skills 通过 skills.sh 按需拉取。OpenClaw / QwenPaw / Hermes 多种运行时可以在同一个 Matrix Room 里共存，人类通过 Element Web 或任意 Matrix 客户端旁听、介入，一切通信「可见、可干预、无隐藏调用」。这套设计把 Leader-Worker 的**工程侧问题**解决得相当彻底：凭据安全、通信基础设施、共享存储、可观测性、人在环、多运行时兼容。但结构上它仍然是 Manager-Workers——Manager 负责编排和心跳汇报，Worker 通过 Matrix 的 m.mentions 互相点名，本质上是「把每个 Agent 拉进同一个群」。通信通道有了，但协作语义没有：没有方案共同讨论、没有 OKR 协商、没有分歧仲裁、没有任务完成后的集体复盘、没有基于历史的团队演化。Agent Teams 现阶段的瓶颈不在基础设施（Matrix / A2A / ANP / 共享文件系统都已成熟），而在协作机制——如何让一群 Agent 像一支真实团队那样讨论、对齐、跟进、复盘。基础设施解决了"能不能说话"，还没解决"该说什么、怎么达成共识、说完之后如何沉淀"。

## 场景

不是所有 Agent 场景都需要群聊。引入群聊的代价是显著的，无论是用户的使用习惯，还是背后的 Agent Infra，包括技术架构、上下文管理、权限治理、并发调度、成本归因，每一项的复杂度都会迅速拉高。
1. 跨领域协作和长链路工作流。
  1. 跨领域协作需要多智能体，本质原因是 Agent 的上下文和记忆等是有限的，和人一样，一个 Agent 塞太多职责，注意力一分散，每件事都做不好。跨领域是注意力在空间上的分散，比如需求分析、编码、测试本身就是不同领域的工作，拆成多个 Agent 按阶段接力，每段上下文干净、注意力集中，中间状态可持久化，断了能从断点续跑，更健壮，并且要发生在所有人都能看到、都能介入、都能纠偏的频道里。
  2. 长链路则是注意力在时间上的衰减。比如从需求到发布的软件研发流程，跑几小时甚至几天，上下文越积越长，Agent 对早期信息的注意力不断被稀释，推理质量持续下降。
2. 多智能体治理
3. 沉淀组织级知识

## 协作

要组织一支团队，第一步得知道「团队」可以长成什么样。有哪些组织形态可选，分为4个维度和五种基本组织形态
1. 协作类型（合作／竞争／竞合）
2. 协调策略（规则／角色／模型驱动）
3. 通信结构（中心化／去中心化／层级）
4. 动态性（静态／运行时可变）

五种基本组织形态，没有哪一种结构能普适所有场景，越是需要灵活交互、频繁互相求助的场景，全连接的 Flat 结构反而比层级结构更合适。
1. 全连接的 Flat/P2P
2. 指挥链清晰的 Hierarchical
3. 群组内合作对外统一的 Team
4. 大规模去中心的 Society
5. 把前几种拼起来的 Hybrid

另一个视角
1. 如何分工（分治）
  1. 分工的前提是有一个计划 ==> 谁做计划
  1. Leader 要具备四个特征：深度参与方案制定，不管细节，但知道"这事大方向该往哪走"；与 Worker 讨论后再执行，方案不是单方灌输，Leader 提大方向，Worker 补执行侧的约束；具备兜底能力——Worker 做不了、卡住了，Leader 能亲自接手完成；负责方案变更审查。
2. 如何协同
  1. 协同的前提是有一个目标 ==> 对齐目标
  2. 主动广播（Worker 有阶段性成果或发现共同风险时，主动 push 给相关方）
  3. 被动查询（Worker 需要另一个 Worker 的中间结果或专业判断时，pull 式请求）
  4. 求助升级（Worker 卡住时先横向找同伴，同伴也解决不了再向 Leader 升级）
3. 经验积累 ==> 团队进化
  1. 单agent进化，分析自己的执行轨迹、提取捷径经验
  2.  Team 层面的进化。Team 复盘应该由 Leader 主持、所有相关 Worker 参与，结构化输出三类资产：方法论（这次成功/失败的关键因素，抽象为下次可复用的原则）；协作模式（哪些 Worker 组合、哪种通信模式在这类任务里效果好，形成 team playbook）；反模式（这次踩的坑、走的弯路，作为下次的 anti-pattern 提前规避）。这些资产存两个层次——团队共享知识库（所有 Worker 都能读）+ 角色专属经验（只对当前岗位有意义的技巧）。

```
playbook = xx
while xx:  # 注意积累trajectory
  plan_and_tasks = plan(question,check,playbook)
  agents = []
  for task in tasks:
    agent = agent(task)
    agents.add(agent)
  progress = wait(agents)
  check = verifier(progress) 
```

## claude方案

"多 Agent + 软件工程团队角色"是不是研发的全能解？我的判断是：它是全能解，但不是最优解。说它是"全能解"，因为它确实能 cover 软件工程的完整生命周期。需求分析、方案设计、代码实现、测试验证、文档维护——每个环节都可以由对应角色的 Agent 来执行，而且已经有人在这么做了。说它"不是最优解"，因为不是所有项目都需要完整的软件工程团队角色支持。每个项目有自己的特性——从个人开发者的 side project 到小团队的快速迭代，很多优秀的软件从来不需要 PM 写 PRD、Architect 画架构图、SM 管 Sprint。即使对于确实需要多角色协作的大型项目，角色化本身也引入了具体的成本：
1. 角色切换的上下文损耗。每次角色切换都有一次完整的上下文重建。而一个人 + 单 Agent 走完全流程，上下文是连续积累的，没有切换损耗。
2. 文档交接的信息衰减。 角色之间通过文档传递信息——PM 写 PRD 给 Architect，Architect 写架构给 DEV。每一次从"理解"到"文档"再到"理解"的转换都有信息损失。真实团队中这个损耗靠会议和即时沟通来弥补，Agent 团队靠什么？反观个人开发者，需求、设计、实现都在一个人的脑子里，信息衰减天然最小——Agent 只需要辅助执行，不需要跨角色"翻译"。
3. 过程正确不等于结果正确。 有了完整的工作流（头脑风暴→PRD→架构→Story→开发→Review），流程上无懈可击。流程完备不能弥补单步质量的不足——每一步的输出质量取决于模型能力，而不是流程设计。 与其在流程上做到完美，不如把精力投入到每一步的上下文质量上——这才是决定 Agent 输出质量的关键变量。

大多数项目，单 Agent + 结构化上下文就够了。关键环节按需引入角色化——需求分析、代码审查这些确实需要不同视角的环节，再引入专门的 Agent。并行用在执行层，不是决策层——代码审查 6 个 Checker 并行、批量测试并行，这些执行性工作的并行收益最直接。但方案设计、架构决策这些需要深度思考的环节，并行反而带来冲突。

看几个典型研发场景。
1. 代码审查：需要多视角交叉检查。一次 Code Review，如果只让一个人看，很容易受到知识盲区和个人经验的限制。比如一个同事可能更擅长性能优化，但对安全漏洞不够敏感；另一个同事可能熟悉代码规范，但不一定能发现复杂的架构隐患。
2. 复杂 Bug 修复：需要并行验证多个假设。是不是数据库连接池打满了？是不是下游服务超时了？是不是缓存逻辑有并发问题？是不是某个边界条件没处理？是不是最近上线的代码引入了副作用？ 传统方式是串行排查，效率比较低。而 Agent Teams 可以把这些假设拆成多个调查方向，让不同 Agent 同时去验证。这样就能把“一个个试”的过程，变成“多个方向并行跑”的过程。
3. 新功能开发：需要多个角色配合。一个新功能通常不是单点任务，而是涉及多个环节：前端页面实现； 后端接口开发； 数据结构设计； 测试用例编写；文档补充； 联调验证。 如果只让一个 Agent 处理，它大概率还是串行完成。但如果用 Agent Teams，就可以让不同 Agent 分别扮演前端、后端、测试、架构审查等角色，各自认领不同模块，并行推进。
这些场景有一个共同点：任务可以被拆解，并且并行处理能带来明显收益。单个 Agent 更像一个能力很强的全栈工程师，但它本质上还是在串行工作。面对复杂问题时，很容易出现两个问题：
1.  一次只能沿着一个思路推进； 
2.  分析视角容易受到当前上下文和推理路径的限制。 

Agent Teams 的核心价值，就是通过模拟人类团队的并行协作方式，让多个 Agent 围绕同一个目标分工、沟通、协作，**从而提升复杂任务的处理效率和覆盖面**。Subagent 更像临时叫来帮忙的工具人。- Agent Team 更像一个可以协作推进任务的项目组。它的核心不是“多开几个 Agent 一起干活”，而是让多个 Agent 围绕一套共享状态进行协作。
- Team Lead：队长，负责理解目标、拆解任务、协调队员和汇总结果。
- Teammates：队员，负责具体执行任务。Team Lead把任务写入共享任务列表(不是把任务“口头告诉”某个 Agent)，Teammates 定期读取任务列表发现新任务；拆解任务时，最重要的是保证每个任务边界清晰，这是任务之间可以相对独立地执行的前提。
- Shared Task List：共享任务列表，用来管理任务、状态和依赖。
  1. Team Lead 创建任务后，任务会被写入共享任务目录 `~/.claude/tasks/{team-name}/`
  2. Teammate 更新任务状态，不是被“主动推送”通知，通过刷新任务列表感知状态变化。也就是说，任务发现通常是通过 轮询 / 刷新共享状态 完成的。
    ```
    Teammate 启动
      ↓
    读取共享任务列表
      ↓
    发现 open 状态任务
      ↓
    判断任务是否适合自己
      ↓
    认领任务或执行被分配的任务
      ↓
    更新任务状态为 in_progress
    ```
  3. 一个任务通常会经历这样的状态流转：`open -> in_progress -> completed`, 也可能存在一些异常状态，比如：blocked/failed/cancelled。
  4. Teammate A 完成了 task-001，并把状态更新为 completed。Teammate B 如果依赖 task-001 的结果，那么它在下一次刷新任务列表时，就能发现：task-001.status=completed, 然后 Teammate B 才能继续执行自己的任务。
- Mailbox：信箱机制，用来支持 Agent 之间的异步通信。Teammate 把消息写入对方 Mailbox，对方定期检查自己的 Mailbox 发现新消息；
  1. Shared Task List 解决的是“任务怎么分、状态怎么管”， Mailbox 解决的就是“信息怎么传”。在真实研发团队里，协作不只是看任务列表，还需要不断沟通：这个接口字段你确认了吗？我这里发现一个风险，你那边注意一下； 我这边排除了一个方向，其他人不用重复看了。 
  1. 本地文件系统上，它对应的目录类似下面，每个 Agent 都会有自己的 inbox 文件。假设 agent1 给 agent2发了一条消息，并不是直接把消息塞进对方上下文里，而是写入对方的 inbox 文件。agent2在自己的执行循环中，会定期检查自己的 inbox。也就是说，Mailbox 的消息发现机制，本质上也是一种 轮询 / 刷新机制。
    ```
    ~/.claude/teams/{team-name}/inboxes/
      Teammate Agent1.json
      Teammate Agent2.json

    {
      "id": "msg-001",
      "from": "agent1",
      "to": "agent2",
      "type": "notice",
      "content": "I found that the token refresh endpoint is called very frequently. Please check whether this could introduce a security risk.",
      "timestamp": "2026-05-20T10:30:00Z",
      "read": false
    }
    ```
- Local File System：本地文件系统，用来持久化任务、消息和协作状态。
从工程视角看，Agent Teams 的本质其实是：基于本地文件系统的轻量级多 Agent 编排机制。它通过文件系统实现了任务分发、状态同步、异步通信和结果汇总。

文件锁：如何避免多个 Agent 同时写坏文件？因为 Agent Teams 的共享任务列表和 Mailbox 都是基于本地文件系统实现的，所以会遇到一个经典问题：多个 Agent 同时读写同一个文件怎么办？系统会引入 `.lock` 文件锁机制。可以简单理解为：某个 Agent 写文件之前，先拿锁；写完之后，再释放锁。比如某个 Agent 要更新任务文件：`~/.claude/tasks/code-review-team/task-001.json`, 它可能会先创建一个锁文件：`~/.claude/tasks/code-review-team/task-001.json.lock` , 写入完成后，再删除这个锁文件。这样其他 Agent 看到锁存在时，就知道这个文件正在被写入，暂时不要修改。

把上面的机制串起来，一个 Agent Team 的完整运行流程大致是这样的。
1. 用户提出目标。比如帮我对这个登录模块做一次全面审查，重点关注安全、性能和代码可维护性。
2. Team Lead 理解目标并拆任务。比如：审查认证流程是否存在安全问题； 检查接口是否存在性能瓶颈； 检查代码结构是否清晰； 汇总所有审查结论。 这些任务会被写入共享任务列表。
3. Teammates 发现任务。Teammates 启动后，会读取共享任务列表。比如 Security Specialist 看到一个安全审查任务，就会认领它。认领时，会把任务状态从open 更新为in_progress，并把 owner 设置为自己。
4. 执行任务并检查 Mailbox。Teammate 在执行任务时，不是完全闭门造车。它会周期性检查自己的 Mailbox，看看有没有新消息。比如 Performance Guru 给 Security Specialist 发了一条消息：我发现 token refresh 接口调用频率异常，你可以重点检查一下这里是否存在安全风险。Security Specialist 下一次检查 inbox 时，会发现这条未读消息，然后把它纳入自己的安全审查上下文。这时，两个 Agent 就完成了一次异步协作。
5. 更新任务状态和结果。当某个 Teammate 完成任务后，会更新任务文件。比如：
  ```
  {
    "id": "task-001",
    "status": "completed",
    "owner": "security-specialist",
    "result": "Found a potential SQL injection risk in user_controller.go line 58.",
    "updated_at": "2026-05-20T10:45:00Z"
  }
  ```
  其他 Agent 下一次读取任务列表时，就能发现这个任务已经完成。如果它们的任务依赖 task-001，就可以继续执行。
6. 重要信息通过 Mailbox 通知。有些信息只写在任务结果里还不够。比如某个队员发现了一个会影响其他方向的重要结论，就可以通过 Mailbox 发消息。例如：下游服务超时已经成功复现，其他排查方向可以降低优先级。这条消息可以发给 Team Lead，也可以广播给所有 Teammates。这样其他 Agent 不需要等最终汇总阶段，执行过程中就能及时调整方向。
7. Team Lead 汇总最终结果。当所有关键任务完成后，Team Lead 会读取：任务列表； 每个任务的执行结果； Mailbox 中的重要讨论； 被阻塞或失败的任务记录。 然后把不同队员的发现整合成一份完整报告。最终，用户看到的是一份结构化的综合输出，而不是多个零散 Agent 的结果拼接。

这种机制的本质：不是“多个 Agent 同时工作”，是共享状态驱动的多 Agent 协作。


## 对 Agent Infra提出了哪些新的挑战？


