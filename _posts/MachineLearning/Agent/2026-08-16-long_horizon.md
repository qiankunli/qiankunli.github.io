---

layout: post
title: Long-horizon Agent
category: 技术
tags: MachineLearning
keywords: agent software

---


## 简介（未完成）

2026年，模型的推理能力、tool calling 能力、长上下文处理能力都已经达到了一个临界点。Long-horizon Agent 不再是"酷炫的 demo"，而是真正能够产出价值的工具。

1. 不变的：在正确的时机，以正确的格式，向 LLM 提供正确的信息。
2. 变化的：构建 Long-horizon Agent 涉及很多微妙的工程细节（compaction 策略、subagent 通信、error handling 等）
  1. 上下文管理
    - Compaction 策略：当上下文窗口不够时如何压缩
    - Memory 管理：如何在跨会话中保持关键信息。像“经验库”，增强稳定性、抗遗忘与长期一致性。
    - Subagent 通信：如何让主 Agent 和子 Agent 高效协作
  - Tool 选择：什么时候使用什么工具。pi的agent loop 只提供了4个工具：read/write/edit/bash。有bash 就可以安装世界上所有软件，有read/write/edit就有了读写、做事儿、记忆的能力。**我们可以去掉一些工具，更多依赖 Skill**。PS：文件系统权限


## 状态外部化

单个 context window 内的任务，可以用对话历史勉强维持连续性。一旦任务跨越多个 session，这种连续性就会失效。要让任务持续推进，状态必须通过 task file、progress file、git commits 等外部 artifact 存活下来。状态外部化在这里是架构基础，而不只是 memory feature 的一次升级。

一个长任务系统必须把目标、计划、已完成事项、失败尝试、阻塞点、证据、diff、测试结果和下一步动作写进可恢复事实源。这类事实源在多智能体研究里一般叫做ledger（账本）：一份 append-only、只追加不覆盖、可逐条回放和审计的记录。


## 其它



[iii](https://github.com/iii-hq/iii )提出用“worker、trigger、function”三个原语重新定义后端，让 agent 成为与服务、队列等同等的 worker，实现实时发现、可扩展性和统一可观测性，消除 harness 与后端的界限。
1. Function, 一个带稳定标识符的工作单元, 接收输入，并且可以选择返回输出, 它可以存在于任何进程里，也可以用任何语言编写。
2. Trigger 是让 function 运行的东西。它可以是对 function 的直接调用，可以是一个 HTTP endpoint，可以是 cron 调度、队列订阅、状态变化、流事件，或者任何其他东西。
3. Worker 是任何连接到 engine 并注册 functions 和 triggers 的进程。一个 TypeScript API 服务是 worker。一个 Python ML pipeline 是 worker。一个 Rust microservice 是 worker。一个 agent 也是 worker。

