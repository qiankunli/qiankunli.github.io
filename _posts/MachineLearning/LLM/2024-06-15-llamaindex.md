---

layout: post
title: LLamaIndex入门
category: 技术
tags: MachineLearning
keywords: LLamaIndex

---

* TOC
{:toc}

## 简介

构建在LLM模型之上的应用程序通常需要使用私有或特定领域数据来增强这些模型。不幸的是，这些数据可能分布在不同的应用程序和数据存储中。它们可能存在于API之后、SQL数据库中，或者存在在PDF文件以及幻灯片中。LlamaIndex应运而生，为我们建立 LLM 与数据之间的桥梁。 

LlamaIndex 提供了5大核心工具：
1. Data connectorsj。将不同格式的数据源注入llamaindex
2. Data indexes。将数据转为 llm 可以非常容易理解且高效处理、消费的数据。
3. Engines。
4. Data agents。对接生态的其它工具，比如Langchain、Flask等
5. Application integrations

![](/public/upload/machine/llamaindex.jpg)


## 索引阶段

![](/public/upload/machine/llamaindex_index.jpg)

在 LlamaIndex 中，Document 和 Node 是最核心的数据抽象。
1. Document 是任何数据源的容器：
    1. 文本数据
    2. 属性数据：元数据 (metadata)；关系数据 (relationships)
2. Node 即一段文本（Chunk of Text），是 LlamaIndex 中的一等公民。基于 Document 衍生出来的 Node 也继承了 Document 上的属性，同时 Node 保留与其他 Node 和 索引结构 的关系。
    ```python
    # 利用节点解析器生成节点
    from llama_index import Document
    from llama_index.node_parser import SimpleNodeParser

    text_list = ["hello", "world"]
    documents = [Document(text=t) for t in text_list]

    parser = SimpleNodeParser.from_defaults()
    nodes = parser.get_nodes_from_documents(documents)
    ```
首先将数据能读取进来，这样才能挖掘。之后构建可以查询的索引，llamdaIndex将数据存储在Node中，并基于Node构建索引。索引类型包括向量索引、列表索引、树形索引等；
1. List Index：Node顺序存储，可用关键字过滤Node
2. **Vector Store Index**：每个Node一个向量，查询的时候取top-k相似
3. **Tree Index**：树形Node，从树根向叶子查询，可单边查询，或者双边查询合并。
4. Keyword Table Index：每个Node有很多个Keywords链接，通过查Keyword能查询对应Node。


## 查询阶段

有了索引，就必须提供查询索引的接口。通过这些接口用户可以与不同的 大模型进行对话，也能自定义需要的Prompt组合方式。

![](/public/upload/machine/llamaindex_query.jpg)
所有查询的基础是查询引擎（QueryEngine）。获取查询引擎的最简单方式是通过索引创建一个，如下所示：

```
query_engine = index.as_query_engine() 
response = query_engine.query("Write an email to the user given their background information.") 
print(response)
```
查询的三个阶段检索（Retrieval），允许自定义每个组件。
1. 在索引中查找并返回与查询最相关的文档。常见的检索类型是“top-k”语义检索，但也存在许多其他检索策略。
2. 后处理（Postprocessing）：检索到的节点（Nodes）可以选择性地进行排名、转换或过滤，例如要求节点具有特定的元数据，如关键词。
3. 响应合成（Response Synthesis）：将查询、最相关数据和提示结合起来，发送给LLM以返回响应。

```python
class BaseQueryEngine(ChainableMixin, PromptMixin, DispatcherSpanMixin):
    @dispatcher.span
    def query(self, str_or_query_bundle: QueryType) -> RESPONSE_TYPE:
        ...
        query_result = self._query(str_or_query_bundle)
        ...
    @abstractmethod
    def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        pass
class RetrieverQueryEngine(BaseQueryEngine):
    def __init__(
        self,retriever: BaseRetriever,response_synthesizer: Optional[BaseSynthesizer] = None,node_postprocessors: Optional[List[BaseNodePostprocessor]] = None,callback_manager: Optional[CallbackManager] = None,    ) -> None:
        self._retriever = retriever
        self._response_synthesizer = response_synthesizer or get_response_synthesizer(...)
        self._node_postprocessors = node_postprocessors or []
    @dispatcher.span
    def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        nodes = self.retrieve(query_bundle)
        response = self._response_synthesizer.synthesize(query=query_bundle,nodes=nodes,)
    def retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        nodes = self._retriever.retrieve(query_bundle)
        return self._apply_node_postprocessors(nodes, query_bundle=query_bundle)
```

query_engine 可以被封装为  QueryEngineTool 进而作为 初始化一个更复杂 query_engine 的query_engine.query_engine_tools。这是一种简化的工具调用形式，稍微复杂一点的还要推断工具调用的参数，比如在rag检索中，可以让llm 根据用户表述，推断查询时对元数据的过滤参数。再进一步，如果用户提出的是由多个步骤组成的复杂问题或需要澄清的含糊问题呢？这需要用到代理循环。

### 带历史记录

BaseChatEngine 提供 chat 方法，支持传入历史记录
```python
class BaseChatEngine(ABC):
    @abstractmethod
    def chat(self, message: str, chat_history: Optional[List[ChatMessage]] = None) -> AGENT_CHAT_RESPONSE_TYPE:          
```
### 带路由

RouterQueryEngine提出了 BaseSelector（可以EmbeddingSingleSelector） summarizer（BaseSynthesizer子类） 抽象，有分有合。

```
class RouterQueryEngine(BaseQueryEngine):
    def __init__( self,
        selector: BaseSelector,
        query_engine_tools: Sequence[QueryEngineTool],
        llm: Optional[LLM] = None,
        summarizer: Optional[TreeSummarize] = None,
    ) -> None:
        ...
```
### 多步推理

```
class MultiStepQueryEngine(BaseQueryEngine):
    def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        nodes, source_nodes, metadata = self._query_multistep(query_bundle)
        final_response = self._response_synthesizer.synthesize(
            query=query_bundle,
            nodes=nodes,
            additional_source_nodes=source_nodes,
        )
        return final_response
    def _query_multistep(self, query_bundle: QueryBundle) -> Tuple[List[NodeWithScore], List[NodeWithScore], Dict[str, Any]]:
        while not should_stop:
    
```
### 路由 + 工具 + 多步推理。

最复杂场景下的RAG文档：路由 + 工具 + 多步推理。
```
class BaseAgent(BaseChatEngine, BaseQueryEngine):
    def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        agent_response = self.chat( query_bundle.query_str,chat_history=[],)
        return Response(response=str(agent_response), source_nodes=agent_response.source_nodes)

class BaseAgentRunner(BaseAgent):
    @abstractmethod
    def run_step(self,task_id: str,input: Optional[str] = None,step: Optional[TaskStep] = None,**kwargs: Any,) -> TaskStepOutput:
        
class AgentRunner(BaseAgentRunner):
    def _run_step(self,task_id: str,step: Optional[TaskStep] = None,...)-> TaskStepOutput:
        task = self.state.get_task(task_id)
        step_queue = self.state.get_step_queue(task_id)
        cur_step_output = self.agent_worker.run_step(step, task, **kwargs)
        next_steps = cur_step_output.next_steps
        step_queue.extend(next_steps)
        return cur_step_output   
    def _chat(self,message: str,chat_history: Optional[List[ChatMessage]] = None,)-> AGENT_CHAT_RESPONSE_TYPE:
        task = self.create_task(message)
        while True:
            cur_step_output = self._run_step(task.task_id, mode=mode, tool_choice=tool_choice)
            if cur_step_output.is_last:
                result_output = cur_step_output
                break
            result = self.finalize_response(task.task_id,result_output,)
        return result    
```
Step-Wise Agent Architecture: The step-wise agent is constituted by two objects, the AgentRunner and the AgentWorker.
1. Our "agents" are composed of AgentRunner objects that interact with AgentWorkers. AgentRunners are orchestrators that store state (including conversational memory), create and maintain tasks, run steps through each task, and offer the user-facing, high-level interface for users to interact with.
2. AgentWorkers control the step-wise execution of a Task. Given an input step, an agent worker is responsible for generating the next step. They can be initialized with parameters and act upon state passed down from the Task/TaskStep objects, but do not inherently store state themselves. The outer AgentRunner is responsible for calling an AgentWorker and collecting/aggregating the results.

chat 方法由一个while true 循环组成，循环内执行 _run_step， AgentRunner维护了一个step_queue，AgentRunner._run_step 主要是维护 step 与step_queue 的关系、记录step的输入输出，step的具体执行委托给agent_worker 负责。 

## 个性化配置

个性化配置主要通过 LlamaIndex 提供的 ServiceContext 类实现。PS：后来是Settings？

1. 自定义文档分块
2. 自定义向量存储
3. 自定义检索
    ```
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine(similarity_top_k=5)
    ```
4. 指定 LLM `service_context = ServiceContext.from_defaults(llm=OpenAI())`
5. 指定响应模式 `query_engine = index.as_query_engine(response_mode='tree_summarize')`
6. 指定流式响应 `query_engine = index.as_query_engine(streaming=True)`

## 与LangChain 对比

LlamaIndex的重点放在了Index上，也就是通过各种方式为文本建立索引，有通过LLM的，也有很多并非和LLM相关的。LangChain的重点在 Agent 和 Chain 上，也就是流程组合上。可以根据你的应用组合两个，如果你觉得问答效果不好，可以多研究一下LlamaIndex。如果你希望有更多外部工具或者复杂流程可以用，可以多研究一下LangChain。

