---

layout: post
title: LLamaIndex入门
category: 技术
tags: MachineLearning
keywords: LLamaIndex

---

* TOC
{:toc}



## 简介（未完成）

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

## 个性化配置

个性化配置主要通过 LlamaIndex 提供的 ServiceContext 类实现。

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