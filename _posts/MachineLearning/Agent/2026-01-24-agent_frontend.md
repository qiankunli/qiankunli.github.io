---

layout: post
title: Agent前端
category: 技术
tags: MachineLearning
keywords: agent software

---


## 简介（未完成）

## A2UI

[从Agent视角看A2UI：智能体终于学会了用界面"说话"](https://mp.weixin.qq.com/s/-KZ_LmQ_s29HeEZAi24yAw)


A2UI（Agent-to-User Interface，智能体到用户界面）是一个开放协议，旨在让 AI 智能体能够以安全、原生且高性能的方式驱动用户界面。简而言之：让远程AI智能体在不向客户端发送危险的可执行代码的情况下，生成复杂的交互式界面。它的工作流程如下：
1. 用户与AI智能体应用交流
2. 智能体生成A2UI Message（描述了UI）
3. 流式Message响应到客户端（JsonLine）
4. 客户端通过预定义本地组件渲染（Angular, Flutter, React, etc.）
5. 用户与UI交互，发送操作给Agent
6. AI智能体生成下一个A2UI Message（循环）
它的核心作用是作为智能体循环（Agent Loop）中的 “输出层” —— 将智能体的决策转化为用户可以交互的界面。

一个有效的 A2UI 消息包含以下几个核心部分：

1. version: A2UI 协议版本。
2. component: 要渲染的组件名称（例如 Button, Container, LineChart）。
3. props: 传递给组件的参数（如 label, color, value）。
4. children (可选): 嵌套在其中的其他组件数组。
5. action (可选): 定义当用户与组件交互时（如点击按钮）应触发的操作。

```

{
  "a2ui": "0.9",
  "type": "updateComponents",
  "payload": {
    "components": [
      {
        "id": "btn_1",
        "name": "Button",
        "props": {
          "label": "确认预订",
          "variant": "primary"
        }
      }
    ]
  }
}
```
A2UI使用邻接表来定义组件关系，而不是用嵌套的Tree。这样能够更好进行流式渲染（生成JsonLine）。这意味着组件是作为扁平列表发送的，通过 children 属性中的 ID 相互引用。为什么使用这种结构？
1. 流式友好： 智能体可以先发送父组件，随后再发送子组件，而无需重新发送整个树。
2. 局部更新： 如果一个深层嵌套的按钮需要改变颜色，智能体只需发送该按钮的 ID 和新属性，而不需要发送整个页面布局。
3. 大语言模型 (LLM) 效率： 对于 LLM 来说，生成扁平的 JSON 结构比生成深度嵌套的结构更不容易出错。
A2UI 不仅仅是一个 JSON 格式，它是一个通信协议。虽然 A2UI 提供了一套标准的跨平台组件（如容器、文本、按钮等），但该协议真正的强大之处在于其可扩展性。你可以定义特定于你业务领域的自定义组件，并让智能体（Agent）像使用标准组件一样驱动它们。

如何才能集成A2UI到你的Agent应用中？主要有两种模式：工具调用 (Tool Calling) 和 结构化输出 (Structured Output)。
1. 工具调用，你为智能体提供一个名为 render_ui 的工具。提示词示例："你可以使用 render_ui 工具来显示交互式组件。当你需要展示数据分析、复杂的列表或需要用户确认的操作时，请使用此工具。"
2. 结构化输出。强制模型始终（或在特定条件下）根据 JSON Schema 输出 A2UI 格式的数据。在系统提示词（System Prompt）中加入以下内容：
    1. 提供组件库清单： 告诉模型客户端支持哪些组件（如：DataTable, SummaryStat, Calendar）。
    2. 强调简洁性： 提醒模型只发送必要的数据，不要生成冗余的样式代码，因为样式由客户端处理。
    3. 上下文关联： 鼓励模型在 UI 旁边提供简短的文本解释。


