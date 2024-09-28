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

## 简介


在专业数据领域上，嵌入模型的表现不如 BM25，但是微调可以大大提升效果。embedding 模型 可能从未见过你文档的内容，也许你的文档的相似词也没有经过训练。在一些专业领域，通用的向量模型可能无法很好的理解一些专有词汇，所以不能保证召回的内容就非常准确，不准确则导致LLM回答容易产生幻觉（简而言之就是胡说八道）。可以通过 Prompt 暗示 LLM 可能没有相关信息，则会大大减少 LLM 幻觉的问题，实现更好的拒答。
1. [大模型应用中大部分人真正需要去关心的核心——Embedding](https://mp.weixin.qq.com/s/Uqt3H2CfD0sr4P5u169yng) 
2. [分享Embedding 模型微调的实现](https://mp.weixin.qq.com/s/1AzDW9Ubk9sWup2XJWbvlA) ，此外，原则上：embedding 所得向量长度越长越好，过长的向量也会造成 embedding 模型在训练中越难收敛。 [手工微调embedding模型，让RAG应用检索能力更强](https://mp.weixin.qq.com/s/DuxqXcpW5EyLI3lj4ZJXdQ) 未细读
3. embedding训练过程本质上偏向于训练数据的特点，这使得它在为数据集中（训练时）未见过的文本片段生成有意义的 embeddings 时表现不佳，特别是在含有丰富特定领域术语的数据集中，这一限制尤为突出。微调有时行不通或者成本较高。微调需要访问一个中到大型的标注数据集，包括与目标领域相关的查询、正面和负面文档。此外，创建这样的数据集需要领域专家的专业知识，以确保数据质量，这一过程十分耗时且成本高昂。而bm25等往往是精确匹配的，信息检索时面临词汇不匹配问题（查询与相关文档之间通常缺乏术语重叠）。幸运的是，出现了新的解决方法：学习得到的稀疏 embedding。通过优先处理关键文本元素，同时舍弃不必要的细节，学习得到的稀疏 embedding 完美平衡了捕获相关信息与避免过拟合两个方面，从而增强了它们在各种检索任务中的应用价值。支持稀疏向量后，一个chunk 在vdb中最少包含: id、text、稠密向量、稀疏向量等4个字段。

## 向量匹配的几个问题    
1. **问题在语义上与其答案并不相同**。因此直接将问题与原始知识库进行比较不会有成果。假设一位律师需要搜索数千份文件以寻找投资者欺诈的证据。问题“什么证据表明鲍勃犯了金融欺诈行为？ ”与“鲍勃于 3 月 14 日购买了 XYZ 股票”在语义上基本上没有任何重叠（其中隐含 XYZ 是竞争对手，3 月 14 日是收益公告发布前一周）。PS：构建数据以进行同类比较。与问题→支持文档相比，问题→问题比较将显著提高性能。从实用角度来说，您可以要求 ChatGPT 为每个支持文档生成示例问题，并让人类专家对其进行整理。本质上，您将预先填充自己的 Stack Overflow。此外，用户反馈的问答对也可以用于问题重写或检索。
2. 向量嵌入和余弦相似度是模糊的。向量在完全捕捉任何给定语句的语义内容方面存在固有缺陷。另一个微妙的缺陷是，余弦相似度不一定能产生精确的排名，因为它隐含地假设每个维度都是平等的。在实践中，使用余弦相似度的语义搜索往往在方向上是正确的，但本质上是模糊的。它可以很好地预测前 20 个结果，但通常要求它单独可靠地将最佳答案排在第一位是过分的要求。
3. 在互联网上训练的嵌入模型不了解你的业务和领域。在专业的垂直领域，待检索的文档往往都是非常专业的表述，而用户的问题往往是非常不专业的白话表达。所以直接拿用户的query去检索，召回的效果就会比较差。Keyword LLM就是解决这其中GAP的。例如在ChatDoctor中会先让大模型基于用户的query生成一系列的关键词，然后再用关键词去知识库中做检索。ChatDoctor是直接用In-Context Learning的方式进行关键词的生成。我们也可以对大模型在这个任务上进行微调，训练一个专门根据用户问题生成关键词的大模型。这就是ChatLaw中的方案。

    ![](/public/upload/machine/keyword_recall.jpg)

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

通常，**Embedding模型是通过对比学习来训练的**，而负样本的质量对模型性能至关重要。Language Embedding模型训练通常采用多阶段方案，分为弱监督的预训练以及有监督的精调训练。
1. 微调样本构建
2. 微调脚本
3. 训练过程监控：W&B监控
4. 模型效果评估

### 微调样本构建

微调样本的构建过程，其实就是找出跟一个query相似的句子——正样本，以及不相似的句子——负样本，Embedding在微调时，会使用对比学习loss来让模型提高辨别正负样本的能力。，query自然是用户问题（下面“-”前面的Q），根据正负样本的来源（下面“-”后面的部分），通常可以分为如下几种：
1. Q-Q（question-question）：这种方式适合已经积累了比较多FAQ的企业，希望对用户问题检索FAQ库中的Q，这种情况下，使用Q-Q方式构建的样本，优化的模型检索效果会比较好
2. Q-A（question-answer）：这种方式比较有误导性，看起来感觉最应该用这种方式构建，但实际上线后，要检索的，是一堆documents，而不是answer，如果你真的用这个方式构建过样本，看一些case就会发现，answer跟实际的文档相差非常远，导致模型微调后，性能反而出现下降
3. Q-D（question-document）：这种方式，在几个项目中实践下来，基本上是最适合的构建方式，因为**实际检索时，就是拿问题去检索文档，确保训练、推理时任务的一致性，也是减少模型性能损失最主要的一个方法**。

### 训练过程

[探索更强中文Embedding模型：Conan-Embedding](https://mp.weixin.qq.com/s/5upU8Yf-6Bcn0kfxk7-V2Q) 不是特别直白。

预训练
1. 通过文档提取和语言识别进行格式化处理；
2. 在基于规则的阶段，文本会经过规范化和启发式过滤；
    1. 在安全过滤阶段，执行域名阻止、毒性分类和色情内容分类；
    2. 在质量过滤阶段，文本会经过广告分类和流畅度分类，以确保输出文本的高质量。
3. 通过MinHash方法进行去重

在预训练阶段，为了高效且充分地利用数据，我们使用InfoNCE Loss with In-Batch Negative：

![](/public/upload/machine/InfoNCE_Loss+with_In-Batch_Negative.jpg)

其中是title，input，question，prompt等，$y_i^{+}$是对应的 content，output，answer，response等，认为是正样本；$y_i$是同 batch 其他样本的content，output，answer，response，认为是负样本。In-Batch Negative InfoNCE Loss 是一种用于对比学习的损失函数，它利用了 mini-batch 中的其他样本作为负样本来优化模型。具体来说，在每个 mini-batch 中，除了目标样本的正样本对外，其余样本都被视为负样本。通过最大化正样本对的相似度并最小化负样本对的相似度，In-Batch Negative InfoNCE Loss 能够有效地提高模型的判别能力和表征学习效果。这种方法通过充分利用 mini-batch 中的样本，提升了训练效率并减少了对额外负样本生成的需求。

有监督精调，我们针对不同的下游任务进行特定任务的微调。
1. 检索（Retrieval）。检索任务包括查询、正样本和负样本，经典的损失函数是InfoNCE Loss。
2. 语义文本相似性（STS）。STS任务涉及区分两段文本之间的相似性，经典的损失函数是交叉熵损失。

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

另外，如果你使用的是开源嵌入模型，对其进行微调是提高检索准确性的有效方法。LlamaIndex 提供了 一份详尽的指南，指导如何一步步微调开源嵌入模型，并证明了微调可以在各项评估指标上持续改进性能。（https://docs.llamaindex.ai/en/stable/examples/finetuning/embeddings/finetune_embedding.html ） 下面是一个示例代码片段，展示了如何创建微调引擎、执行微调以及获取微调后的模型：

```
finetune_engine = SentenceTransformersFinetuneEngine(
 train_dataset,
 model_id="BAAI/bge-small-en",
 model_output_path="test_model",
 val_dataset=val_dataset,
)
finetune_engine.finetune()
embed_model = finetune_engine.get_finetuned_model()
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

## 其它embedding 优化

[告别传统的文档切块！JinaAI提出Late Chunking技巧](https://mp.weixin.qq.com/s/BxJNcsfbwZpbG4boU6vjiw)对于传统的分块，类似于固定长度的分块。带来的一个比较大的问题是，上下文缺失。比如一个句子的主语在段落开头，后面的段落/句子中，有一些代词比如 It's， The city等等来表示主语。这种情况下确实主语的句子基本上就变得比较断章取义了~与先分块后向量化不同，JinaAI最新提出的“Late Chunking”方法是一个相反的步骤，首先将整个文本或尽可能多的文本输入到嵌入模型中。在输出层会为每个token生成一个向量表示，其中包含整个文本的文本信息。然后我们可以按照需要的块大小对对向量进行聚合得到每个chunk的embedding。这样的优势是，充分利用长上下文模型的优势，同时又不会让每个块的信息过多，干扰向量表征。

