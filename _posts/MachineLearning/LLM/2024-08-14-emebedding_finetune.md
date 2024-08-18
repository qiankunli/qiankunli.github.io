---

layout: post
title: RAG向量检索与微调
category: 架构
tags: MachineLearning
keywords: llm emebedding

---

<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

* TOC
{:toc}

## 简介（未完成）


在专业数据领域上，嵌入模型的表现不如 BM25，但是微调可以大大提升效果。embedding 模型 可能从未见过你文档的内容，也许你的文档的相似词也没有经过训练。在一些专业领域，通用的向量模型可能无法很好的理解一些专有词汇，所以不能保证召回的内容就非常准确，不准确则导致LLM回答容易产生幻觉（简而言之就是胡说八道）。可以通过 Prompt 暗示 LLM 可能没有相关信息，则会大大减少 LLM 幻觉的问题，实现更好的拒答。
1. [大模型应用中大部分人真正需要去关心的核心——Embedding](https://mp.weixin.qq.com/s/Uqt3H2CfD0sr4P5u169yng) 
2. [分享Embedding 模型微调的实现](https://mp.weixin.qq.com/s/1AzDW9Ubk9sWup2XJWbvlA) ，此外，原则上：embedding 所得向量长度越长越好，过长的向量也会造成 embedding 模型在训练中越难收敛。 [手工微调embedding模型，让RAG应用检索能力更强](https://mp.weixin.qq.com/s/DuxqXcpW5EyLI3lj4ZJXdQ) 未细读
3. embedding训练过程本质上偏向于训练数据的特点，这使得它在为数据集中（训练时）未见过的文本片段生成有意义的 embeddings 时表现不佳，特别是在含有丰富特定领域术语的数据集中，这一限制尤为突出。微调有时行不通或者成本较高。微调需要访问一个中到大型的标注数据集，包括与目标领域相关的查询、正面和负面文档。此外，创建这样的数据集需要领域专家的专业知识，以确保数据质量，这一过程十分耗时且成本高昂。而bm25等往往是精确匹配的，信息检索时面临词汇不匹配问题（查询与相关文档之间通常缺乏术语重叠）。幸运的是，出现了新的解决方法：学习得到的稀疏 embedding。通过优先处理关键文本元素，同时舍弃不必要的细节，学习得到的稀疏 embedding 完美平衡了捕获相关信息与避免过拟合两个方面，从而增强了它们在各种检索任务中的应用价值。支持稀疏向量后，一个chunk 在vdb中最少包含: id、text、稠密向量、稀疏向量等4个字段。

## 稀疏向量
稀疏向量 也是对chunk 算一个稀疏向量存起来，对query 算一个稀疏向量，然后对稀疏向量计算一个向量距离来评价相似度。
1. 关键词检索。BM25是产生稀疏向量的一种方式。其稀疏性体现在，假设使用 jieba（中文分词库）分词，每个词用一个浮点数来表示其统计意义，那么，对于一份文档，就可以用 349047 长度的向量来表示。这个长度是词典的大小，对于某文档，该向量绝大部分位置是0，因此是稀疏向量。基于统计的BM25检索有用，但其对上下文不关心，这也阻碍了其检索准确性。假设有份文档，介绍的是 阿司匹林 的作用，但这篇文档仅标题用了 阿司匹林 这个词，但后续都用 A药 来代指 阿司匹林，这样的表述会导致 阿司匹林 这个词在该文档中的重要程度降低。

    $$
    \text{score}(D, Q) = \sum_{n=1}^{N} \text{IDF}(q_n) \cdot \left(\frac{f(q_n, D)}{k_1 + 1} + \frac{k_1 (1 - b + b \cdot \frac{\text{length}(D)}{\text{avgDL}})}{f(q_n, D) + k_1}\right)
    $$
    其中，f函数则是词在文档中的频率。
2. 稠密检索。核心思想是通过神经网络获得一段语料的、语意化的、潜层向量表示。不同模型产生的向量有长有短，进行相似度检索时，必须由相同的模型产生相同长度的向量进行检索。
3. BM25可以产生稀疏向量用于检索，另一种方式便是使用神经网络。与BM25不同的是
    1. 神经网络赋予单词权重时，参考了更多上下文信息。
    2. 另一个不同点是分词的方式。神经网络不同的模型有不同分词器，但总体上，神经网络的词比传统检索系统分的更小。比如，清华大学 在神经网络中可能被分为4个令牌，而使用jieba之类的，会分为1个词。神经网络的词典大约在几万到十几万不等，但文本检索中的词典大小通常几十万。分词方式不同，使得神经网络产生的稀疏向量，比BM25的向量更稠密，检索执行效率更高。这类的代表有：splade、colbert、cocondenser当使用这类模型编码文档时，其输出一般是：
    ```
    {
        "indices": [19400,724,12,243687,17799,6,1635,71073,89595,32,97079,33731,35645,55088,38],
        "values": [0.2203369140625,0.259765625,0.0587158203125,0.24072265625,0.287841796875,0.039581298828125,0.085693359375,0.19970703125,0.30029296875,0.0345458984375,0.29638671875,0.1082763671875,0.2442626953125,0.1480712890625,0.04840087890625]
    }
    ```
    该向量编码的内容是 RAG："稠密 或 稀疏？混合才是版本答案！"，由 BGE-M3 模型产生。这个稀疏向量中，indices是令牌的ID，values是该令牌的权重。若文档存在重复令牌，则取最大的权重。产生稀疏向量后，无需将所有的都存入数据库中，可以只存权重较高的前几百个即可。具体数值可以根据自身的业务特性实验。


PS： 也就是稀疏向量 可以用一个词表长度的[]来表示，也有其对应的稀疏表示形式。但是计算的时候，还是用[] 去计算的。

## 微调emebedding

1. 微调样本构建
2. 微调脚本
3. 训练过程监控：W&B监控
4. 模型效果评估

### 微调样本构建

微调样本的构建过程，其实就是找出跟一个query相似的句子——正样本，以及不相似的句子——负样本，Embedding在微调时，会使用对比学习loss来让模型提高辨别正负样本的能力。，query自然是用户问题（下面“-”前面的Q），根据正负样本的来源（下面“-”后面的部分），通常可以分为如下几种：
1. Q-Q（question-question）：这种方式适合已经积累了比较多FAQ的企业，希望对用户问题检索FAQ库中的Q，这种情况下，使用Q-Q方式构建的样本，优化的模型检索效果会比较好
2. Q-A（question-answer）：这种方式比较有误导性，看起来感觉最应该用这种方式构建，但实际上线后，要检索的，是一堆documents，而不是answer，如果你真的用这个方式构建过样本，看一些case就会发现，answer跟实际的文档相差非常远，导致模型微调后，性能反而出现下降
3. Q-D（question-document）：这种方式，在几个项目中实践下来，基本上是最适合的构建方式，因为**实际检索时，就是拿问题去检索文档，确保训练、推理时任务的一致性，也是减少模型性能损失最主要的一个方法**。

### 微调脚本

此处原始参考文档来自BGE官方仓库：
https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune

构造好微调样本后，就可以开始微调模型了。代码仓库中包含了4个版本的微调脚本，总体大同小异，此处以finetune_bge_embedding_v4.sh为例

```sh
#!/bin/bash

SCRIP_DIR=$(echo `cd $(dirname $0); pwd`)
export PATH=/work/cache/env/miniconda3/bin:$PATH

# 此处替换为“微调样本构建”步骤产出文件的路径，代码仓库中也有此文件
export TRAIN_DATASET=outputs/v1_20240713/emb_samples_qd_v2.jsonl

# model_name_or_path替换为自己的本机路径，或者BAAI/bge-large-zh-v1.5
torchrun --nproc_per_node ${N_NODES} \
-m FlagEmbedding.baai_general_embedding.finetune.run \
--output_dir ${OUTPUT_DIR} \
--model_name_or_path /DataScience/HuggingFace/Models/BAAI/bge-large-zh-v1.5 \
...
```

### 模型使用

模型的使用方式，与使用RAG技术构建企业级文档问答系统之基础流程中介绍的完全一致，只是替换模型路径model_path即可

```python
from langchain.embeddings import HuggingFaceBgeEmbeddings
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_path = 'stevenluo/bge-large-zh-v1.5-ft-v4'
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_path,
    model_kwargs={'device': device},
    encode_kwargs={'normalize_embeddings': True},
    query_instruction='为这个句子生成表示以用于检索相关文章：'
)
```


## 向量数据库

当我们把通过模型或者 AI 应用处理好的数据喂给它之后（“一堆特征向量”），它会根据一些固定的套路，例如像传统数据库进行查询优化加速那样，为这些数据建立索引。避免我们进行数据查询的时候，需要笨拙的在海量数据中进行。

### 本地

faiss 原生使用
```python
# 准备数据
model = SentenceTransformer('uer/sbert-base-chinese-nli')
sentences = ["住在四号普里怀特街的杜斯利先生及夫人非常骄傲地宣称自己是十分正常的人",
             "杜斯利先生是一家叫作格朗宁斯的钻机工厂的老板", "哈利看着她茫然地低下头摸了摸额头上闪电形的伤疤",
             "十九年来哈利的伤疤再也没有疼过"]
sentence_embeddings = model.encode(sentences)
# 建立索引
dimension = sentence_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(sentence_embeddings)

# 检索
topK = 2
search = model.encode(["哈利波特猛然睡醒"])  # 将要搜索的内容“哈利波特猛然睡醒”编码为向量
D, I = index.search(search, topK)         # D指的是“数据置信度/可信度” I 指的是我们之前数据准备时灌入的文本数据的具体行数。
print(I)
print([x for x in sentences if sentences.index(x) in I[0]])
```
faiss 与LangChain 集合，主要是与  LangChain 的 document和 Embeddings 结合。 faiss 本身只存储 文本向量化后的向量（index.faiss文件），但是vector db对外使用，一定是文本查文本，所以要记录 文本块与向量关系（index.pkl文件）。此外，需支持新增和删除文件（包含多个文本块），所以也要支持按文件删除 文本块对应的向量。 

```python
from langchain.document_loaders import TextLoader
# 录入documents 到faiss
loader = TextLoader("xx.txt")  # 加载文件夹中的所有txt类型的文件
documents = loader.load() # 将数据转成 document 对象，每个文件会作为一个 document
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0) 
docs = text_splitter.split_documents(documents)  # 切割加载的 document

embeddings = OpenAIEmbeddings() # 初始化 openai 的 embeddings 对象
db = FAISS.from_documents(docs, embeddings) # 将 document 通过 openai 的 embeddings 对象计算 embedding 向量信息并临时存入 faiss 向量数据库，用于后续匹配查询

query = "What did the president say about Ketanji Brown Jackson"
docs = db.similarity_search(query)
print(docs[0].page_content)
```

简单的源码分析

```python
# 根据文档内容构建 langchain.vectorstores.Faiss
vectorstore.base.from_documents(cls: Type[VST],documents: List[Document], embedding: Embeddings,    **kwargs: Any,) -> VST:
    """Return VectorStore initialized from documents and embeddings."""
    texts = [d.page_content for d in documents]
    metadatas = [d.metadata for d in documents]
    return cls.from_texts(texts, embedding, metadatas=metadatas, **kwargs)
        # Embeds documents.
        embeddings = embedding.embed_documents(texts)
        cls.__from(texts,embeddings,embedding, metadatas=metadatas,ids=ids,**kwargs,)
            # Initializes the FAISS database
            faiss = dependable_faiss_import()
            index = faiss.IndexFlatL2(len(embeddings[0]))
            vector = np.array(embeddings, dtype=np.float32)
            index.add(vector)
            # 建立id 与text 的关联
            documents = []
            if ids is None:
                ids = [str(uuid.uuid4()) for _ in texts]
            for i, text in enumerate(texts):
                metadata = metadatas[i] if metadatas else {}
                documents.append(Document(page_content=text, metadata=metadata))
            index_to_id = dict(enumerate(ids))
            # Creates an in memory docstore
            docstore = InMemoryDocstore(dict(zip(index_to_id.values(), documents)))
            return cls(embedding.embed_query,index,docstore,index_to_id,normalize_L2=normalize_L2,**kwargs,) 
save_local:
    faiss = dependable_faiss_import()
    faiss.write_index(self.index, str(path / "{index_name}.faiss".format(index_name=index_name)))
    with open(path / "{index_name}.pkl".format(index_name=index_name), "wb") as f:
        pickle.dump((self.docstore, self.index_to_docstore_id), f)   
```



### 在线

Pinecone 是一个在线的向量数据库。所以，我可以第一步依旧是注册，然后拿到对应的 api key。

```python
from langchain.vectorstores import Pinecone
# 从远程服务加载数据
docsearch = Pinecone.from_existing_index(index_name, embeddings)

# 录入documents 持久化数据到pinecone
# 初始化 pinecone
pinecone.init(api_key="你的api key",environment="你的Environment")
loader = DirectoryLoader('/content/sample_data/data/', glob='**/*.txt')
documents = loader.load() # 将数据转成 document 对象，每个文件会作为一个 document
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
split_docs = text_splitter.split_documents(documents) # 切割加载的 document
docsearch = Pinecone.from_texts([t.page_content for t in split_docs], embeddings, index_name=index_name) # 持久化数据到pinecone
```

[ LangChain + GPTCache =兼具低成本与高性能的 LLM](https://mp.weixin.qq.com/s/kC6GB9JaT-WApxU2o3QfdA) 未读。