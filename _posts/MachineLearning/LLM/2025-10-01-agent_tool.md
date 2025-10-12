---

layout: post
title: Agent工具
category: 技术
tags: MachineLearning
keywords: agent software

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

## Browser Use

https://github.com/browser-use/browser-use

[如何让AI“看懂”网页？拆解 Browser-Use 的三大核心技术模块](https://mp.weixin.qq.com/s/KLk-m_E2zx_q-v4ZLobetw) Browser Use 是什么
1. Browser Use 是一种基于 AI 模型的**浏览器自动化工具**，基于 Playwright（浏览器自动化框架）
1. Browser Use 是是一个开源的Python库，基于 LangChain 生态构建

Playwright 是一个底层浏览器自动化驱动层，负责：
1. 启动、控制浏览器进程；
2. 执行 DOM 操作、点击、输入、截图；
3. 模拟网络请求、地理位置、时间；
4. 拦截 / 注入脚本；
5. 提供 API 接口给上层应用调用。
它相当于“浏览器的远程控制器”，用来跑测试、爬取数据、或者做自动化交互。

```python
# 初始化AI模型
llm = ChatOpenAI(model="gpt-4o")
# 定义任务
task = "打开google，搜索'AI编程助手'，告诉我第一条结果的标题"
# 创建AI代理
agent = Agent(task=task, llm=llm)
# 运行任务
await agent.run()
```

[现代Python项目最佳实践：以 agentic tool browser-use为例](https://mp.weixin.qq.com/s/rfdYPW2GRFybVd9DEbYOZg)
```python
class Agent(Generic[Context, AgentStructuredOutput]):
  async def run(self, max_steps: int = 100) -> AgentHistoryList:
    for step in range(max_steps):
      await self.step(step_info)
  async def step(self, step_info: Optional[AgentStepInfo] = None) -> None:
    try:
			# Phase 1: Prepare context and timing
			browser_state_summary = await self._prepare_context(step_info)
			# Phase 2: Get model output and execute actions
			await self._get_next_action(browser_state_summary)
			await self._execute_actions()
			# Phase 3: Post-processing
			await self._post_process()
		except Exception as e:
			# Handle ALL exceptions in one place
			await self._handle_step_error(e)
		finally:
			await self._finalize(browser_state_summary)
```

## computer use

https://github.com/e2b-dev/open-computer-use

## code sandbox

https://github.com/vndee/llm-sandbox