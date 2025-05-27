---

layout: post
title: Transformers源码学习
category: 技术
tags: MachineLearning
keywords: llm Transformers 

---

* TOC
{:toc}

## 前言

大模型由以下两个关键部分构成：一个是 参数集，另一个是 执行代码。
1. 参数集：这是模型的"大脑"，包含了通过训练学习到的神经网络权重。
2. 执行代码：这是模型的"引擎"，包含用于运行参数集的软件代码，可以采用任何编程语言实现。

[NLP Course](https://huggingface.co/learn/nlp-course) 官方教程，建议从头到尾细看一下。

[Transformers快速入门](https://transformers.run/)Hugging Face 专门为使用 Transformer 模型编写了一个 Transformers 库，建立在 Pytorch 框架之上（Tensorflow 的版本功能并不完善），所有 Transformer 模型都可以在 Hugging Face Hub 中找到并且加载使用，包括训练、推理、量化等。

![](/public/upload/machine/transformers_overview.png)

## HuggingFace 使用

Hugging Face 自然语言处理（NLP）的开源平台和社区，主要提供了以下几个产品和服务：
1. Hub：这是一个机器学习的中心，让你可以创建、发现和协作ML项目。可以从排行榜开始，了解社区中表现较好的模型。如果你没有 GPU，你必须使用小的模型。转到文件目录并查看 .bin 文件的大小。有的项目在型号卡中也会提到所需的最低规格。PS：就像github 包含代码文件一样，这里包含代码的模型文件，git clone 时要安装Git LFS（Git Large File Storage）
2. Transformers：这是一个自然语言处理的库，支持多种编程语言（如Python、JavaScript、Swift等）和框架（如PyTorch、TensorFlow等），并提供了简单易用的API，让你可以快速地加载、训练和部署模型。帮我们跟踪流行的新模型，并且**提供统一的代码风格来使用BERT、XLNet和GPT等等各种不同的模型**。只有configuration，models和tokenizer三个主要类，基于上面的三个类，提供更上层的pipeline和Trainer/TFTrainer，从而用更少的代码实现模型的预测和微调。
    1. pipeline 在底层是由 AutoModel 和 AutoTokenizer 类来实现的。AutoClass（即像 AutoModel 和 AutoTokenizer 这样的通用类）是加载模型的快捷方式，它可以从其名称或路径中自动检索预训练模型。
    2.  
    ```python
    from transformers import MODEL_NAME # 导入模型
    model = MODEL_NAME.from_pretrained('MODEL_NAME') # 实例化模型，其中 MODEL_NAME 是模型的名称或路径。
    inputs = xx             # 准备输入数据，转换为模型支持的格式。（如 tokenizer 后的文本、图像等）
    outputs = model(inputs) # 调用模型并获得输出

    model.save_pretrained('PATH')   # 将模型保存到指定路径
    ```
    3. 在Linux下，模型默认会缓存到`~/.cache/huggingface/transformers/`。所有的模型都可以通过统一的from_pretrained()函数来实现加载，transformers会处理下载、缓存和其它所有加载模型相关的细节。而所有这些模型都统一在Hugging Face Models管理。
    4. 最初我认为您需要为每个模型系列使用特定的Transformers和Tokenizer（例如，如果您使用T5模型系列，则对应T5Tokenizer和 T5ForConditionalGeneration），对于所有预训练模型，您可以声明一个简单的语句：
        ```python
        from transformers import AutoTokenizer, AutoModelForCausalLM
        tokenizer = AutoTokenizer.from_pretrained("databricks/dolly-v2-3b")
        model = AutoModelForCausalLM.from_pretrained("databricks/dolly-v2-3b")
        ```
3. Inference API：这是一个服务，让你可以直接从Hugging Face的基础设施上运行大规模的NLP模型，并在毫秒级别得到响应。
4. Datasets：这是一个数据集的库，让你可以获取、加载和处理超过1400个公开可用的数据集。Datasets支持多种数据类型（如文本、图像、音频等）和格式（如JSON、CSV等），并提供了高效且统一的API，让你可以快速地加载、缓存和转换数据。PS： 一开始看代码的时候，总以为是pytorch dataset

下载模型文件（一般有几个G）有多种方式
1. 到huggingface 官网手动通过文件链接下载
    1. 可以使用 `export HF_ENDPOINT=https://hf-mirror.com` 国内加速一下
2. Git LFS 下载。`git clone https://huggingface.co/THUDM/chatglm-6b` PS：注意不是github 地址
3. Hugging Face Hub 下载。
    ```python
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id="bert-base-chinese")
    ```
4. `huggingface-cli  download  --resume-download --cache-dir ./cache/ --local-dir ./starcoder  bigcode/starcoder`
4. 使用transformers 库，但这种方式速度慢，且经常中断。
    ```
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True,mirror="tuna")
    model = AutoModel.from_pretrained("THUDM/chatglm2-6b",trust_remote_code=True, mirror="tuna")
    ```
    下载后文件会出现在 `~/.cache`目录下
    ```
    ~/.cache
        /torch/sentence_transformers
            /moka-ai_m3e-base
                /config.json
                /pytorch_model.bin
        /huggingface/hub
            /models--THUDM--chatglm2-6b
                /blobs
                /snapshots
                    /b1502f4f75c71499a3d566b14463edd62620ce9f   # 某个版本的文件内容
                        /config.json
                        /pytorch_model.xx.bin
    ```

### pipelines

开箱即用的 pipelines，它封装了预训练模型和对应的前处理和后处理环节。
1. 预处理 (preprocessing)，**Transformer模型无法直接处理原始文本字符串**；具体地，我们会使用每个模型对应的分词器 (tokenizer) 来进行。 PS： 有点类似于tf的各种FeatureColumn
  1. 将输入切分为词语、子词或者符号（例如标点符号），统称为 tokens；
  2. 根据模型的词表将每个 token 映射到对应的 token 编号/id；映射通常是通过创建文本语料库中标记的词汇表，并根据每个标记在语料库中的出现频率为其分配一个整数值来执行的。最常见的tokens被分配较低的整数值，而不太常见的标记被分配较高的值。
  3. 根据模型的需要，添加一些额外的输入。比如tokens补齐和截断操作，填充是指在短序列的末尾添加额外的令牌(通常是一个特殊的令牌，如[PAD])以使它们具有相同的长度的过程。这样做是为了使模型可以同时处理批处理中的所有序列。另一方面，截断是指切断较长序列的末端，如果输入序列比最大长度长，标记器将从右侧截断输入序列。
2. 将处理好的输入送入模型；
3. 对模型的输出进行后处理 (postprocessing)，将其转换为人类方便阅读的格式。

![](/public/upload/machine/transformers_pipeline.jpg)

```python
from transformers import pipeline
classifier = pipeline("sentiment-analysis") # 情感分析
result = classifier("I've been waiting for a HuggingFace course my whole life.")
print(result)

from transformers import pipeline
generator = pipeline("text-generation")     # 文本生成
results = generator("In this course, we will teach you how to")
print(results)
```

### transformers基础组件

transformers开源库的核心组件包括3个：Conﬁguration、Tokenizer、Model
1. 「Conﬁguration」：配置类，通常继承自「PretrainedConﬁg」，保存model或tokenizer的超参数，例如词典大小，隐层维度数，dropout rate等。配置类主要可用于复现模型。
2. 「Tokenizer」：Model只能处理数字，因此Tokenizer需要将我们的文本输入转换为数字。总体上做三件事情：分词；扩展词汇表；识别并处理特殊token。
    1. 通常继承自「PreTrainedTokenizer」，主要存储词典（也就是`from_pretrained()` 的部分），token到index映射关系等。
    2. 此外，还会有一些model-specific的特性，如特殊token，`[SEP]`, `[CLS]`等的处理，token的type类型处理，语句最大长度等，**因此tokenizer通常和模型是一对一适配的**。
3. 「Model」: 模型类。封装了预训练模型的计算图过程，遵循着相同的范式，如根据token ids进行embedding matrix映射（将token的one-hot编码转换成更dense的embedding编码），紧接着多个self-attention层做编码，最后一层task-specific做预测。
针对上述三大类，transformer还额外封装了AutoConfig, AutoTokenizer,AutoModel，可通过模型的命名来定位其所属的具体类，比如’bert-base-cased’，就可以知道要加载BERT模型相关的配置、切词器和模型。

```python
# Building the config
config = BertConfig()
# Building the model from the config,使用随机值对其进行初始化
model = BertModel(config)
print(config)
BertConfig {
  [...]
  "hidden_size": 768,               # 定义了hidden状态向量的大小
  "intermediate_size": 3072,
  "max_position_embeddings": 512,
  "num_attention_heads": 12,
  "num_hidden_layers": 12,          # 定义了Transformer模型的层数
  [...]
}
# 加载已经训练过的Transformers模型
model = BertModel.from_pretrained("bert-base-cased")
```

```python
# 加载与保存分词器
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
tokenizer.save_pretrained("./models/bert-base-cased/")
# 加载与保存模型
from transformers import AutoModel
# 所有存储在 HuggingFace Model Hub 上的模型都可以通过 Model.from_pretrained() 来加载权重，参数可以是 checkpoint 的名称，也可以是本地路径（预先下载的模型目录）
model = AutoModel.from_pretrained("bert-base-cased")
model.save_pretrained("./models/bert-base-cased/") # 保存模型

inputs = tokenizer(["来到美丽的大自然，我们发现"], return_tensors="pt")
# {'input_ids': tensor([[    1, 68846, 68881, 67701, 67668, 98899, 91935]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1]])}

gen_kwargs = {"max_length": 128, "top_p": 0.8, "temperature": 0.8, "do_sample": True, "repetition_penalty": 1.1}
output = model.generate(**inputs, **gen_kwargs)
# decode the new tokens
output = tokenizer.decode(output[0].tolist(), skip_special_tokens=True)
print(output)
```

针对每个模型输入，模型会分别得到对应的高维向量特征（动态词向量，而word2vec是静态词向量）。Transformer的基础架构（无论是编码器、解码器还是seq2seq），将输入张量转换为隐状态向量。这些基础架构是整个模型的一部分。对于不同任务，还有不同的head架构，他们将隐状态输出为任务需要的输出变量。不同任务有相同的Transformer基础架构，但是它们的head架构往往是不同的，具体架构与任务相关。

在Transformers（Transformers库多了一个s，而transformer模型没有s）库中，有许多模型架构，他们一般有基础Transformer架构加上不同的head模块组成，部分例子如下：
1. *Model (retrieve the hidden states)：只输出隐状态
2. *ForCausalLM：常规语言模型，典型的有GPT系列
3. *ForMaskedLM：掩码语言模型，典型的有BERT、RoBERTa、DeBERTa
4. *ForMultipleChoice：多项选择模型
5. *ForQuestionAnswering：问答模型，一般是抽取式问答
6. *ForSequenceClassification：序列分类模型
7. *ForTokenClassification：token分类模型，如命名实体识别和关系抽取

![](/public/upload/machine/transformers_pipelines.png)

### Dataset

```python
from datasets import load_dataset
raw_datasets = load_dataset("glue", "mrpc")
print(raw_datasets)
# 包含训练集、验证集和测试集。每一个集合都包含几个列(sentence1, sentence2, label, and idx)以及一个代表行数的变量
#DatasetDict({
#    train: Dataset({
#        features: ['sentence1', 'sentence2', 'label', 'idx'],
#        num_rows: 3668
#    })
#    validation: Dataset({
#        features: ['sentence1', 'sentence2', 'label', 'idx'],
#        num_rows: 408
#    })
#    test: Dataset({
#        features: ['sentence1', 'sentence2', 'label', 'idx'],
#        num_rows: 1725
#    })
#})
```
load_dataset函数返回的是DatasetDict对象，它类似Python的Dict。DatasetDict里的不同key代表了数据集的不同split（默认的split是train），比如glue数据集包含”train”、”validation”和”test”三个split。DatasetDict的value是Dataset对象，它包含了一个split的全部数据。

为了预处理数据集，我们需要将文本转换为模型能够理解的数字，使用Dataset.map()方法
```python
# example 是一个dict，对应数据集的每个元素，并返回一个包含input_ids、attention_mask 和token_type_ids为key的新dict
# 在机器学习任务中，一个example通常定义为模型的输入（也成为特征集合）
def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
print(raw_datasets)
#DatasetDict({
#    train: Dataset({
#        features: ['attention_mask', 'idx', 'input_ids', 'label', 'sentence1', 'sentence2', 'token_type_ids'],
#        num_rows: 3668
#    })
#    validation: Dataset({
#        features: ['attention_mask', 'idx', 'input_ids', 'label', 'sentence1', 'sentence2', 'token_type_ids'],
#        num_rows: 408
#    })
#    test: Dataset({
#        features: ['attention_mask', 'idx', 'input_ids', 'label', 'sentence1', 'sentence2', 'token_type_ids'],
#        num_rows: 1725
#    })
#})
```
为了解决句子长度统一的问题，我们必须定义一个collate函数，该函数会将每个batch句子填充到正确的长度。

```python
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

raw_datasets = load_dataset("glue", "mrpc")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
```
加载本地数据集
```python
from datasets import load_dataset
#{
#    "data": [
#        {
#            "title": "Terremoto del Sichuan del 2008",
#            "paragraphs": [{...}]
#        }
#    ],
#     "version": "1.1"
#}
squad_it_dataset = load_dataset("json", data_files="SQuAD_it-train.json", field="data")
print(squad_it_dataset) # 加载本地文件会创建一个带有train的DatasetDict 对象
# DatasetDict({
#    train: Dataset({
#        features: ['title', 'paragraphs'],
#        num_rows: 442
#    })
#})
data_files = {"train": "SQuAD_it-train.json", "test": "SQuAD_it-test.json"}
squad_it_dataset = load_dataset("json", data_files=data_files, field="data")
print(squad_it_dataset) # 包括 train 和 test 的 DatasetDict 对象
#DatasetDict({
#    train: Dataset({
#        features: ['title', 'paragraphs'],
#        num_rows: 442
#    })
#    test: Dataset({
#        features: ['title', 'paragraphs'],
#        num_rows: 48
#    })
#})
```

与 Pandas 类似，transformers Datasets 提供了几个函数来操作 Dataset 和 DatasetDict 对象
1. rename_column 重命名DatasetDict中的列
2. filter 过滤一些行
3. 为数据集中的所有行创建新的数据列
    ```python
    def compute_review_length(example):
        return {"review_length": len(example["review"].split())}
    drug_dataset = drug_dataset.map(compute_review_length)
    ```
4. 用于预训练 GPT-2 的 WebText 语料库包含超过 800 万个文档和 40 GB 的文本，全加载到计算机内存吃不消，`pubmed_dataset_streamed = load_dataset("json", data_files=data_files, split="train", streaming=True)`，streaming=True 返回的对象是一个 IterableDataset


## 源码分析

[Deepseek大模型推理算法其实很简单](https://mp.weixin.qq.com/s/kVkw73XhcOE6t4HZyKRUAA)我们听说Deepseek是开源的，这具体是什么意思？训练代码没有开源，但是在论文与技术报告中介绍了关键的Deepseek V3和R1大模型训练的的一些细节，开源的是两个东西，一个是权重，一个是推理代码。这两者是什么关系？可以理解为：权重是人类知识的加密压缩，推理是解码检索知识的工具。**推理过程也不需要太多矩阵知识，就是矩阵乘法/gemm和加法**。

![](/public/upload/machine/transformer_run.jpg)

大模型的矩阵计算，是分层进行的，一个layer接一个layer。每个layer的“结构”都是一样的，所以代码里就是循环。但是，每个layer都有自己固定的参数矩阵（训练出来的），这是不一样的。大模型的威力，就在这些参数矩阵中，但是推理代码里看不出，直接从权重文件中读取。

![](/public/upload/machine/transformer_layer.jpg)

几十层的layer结构中，KQV矩阵不断被计算出来，又传输到FF网络，再到下一个layer，中间隔着一些Norm和Add操作。在最后一个Transformer Layer输出后，要计算logits（概率向量）。最后一层输出是`seq_len*hidden_size`的矩阵，将它乘以一个固定的output矩阵（`hidden_size*vocab_size`的），得到一个`seq_len*vocab_size`的矩阵。虽然最终得到了一个大矩阵，但我们只关心它最后一行的那个32000维的向量。它就代表最终需要的logits概率向量，说明下一个token可以是什么。PS：通过一个线性层把特征向量升维到词表维度（Linear层），并且通过softmax进行归一化（Softmax层），最终输出一个概率分布。该分布表示对词表中每个词匹配这个特征向量的概率，依据这些概率，按照一定的采样规则来采样下一个token，不断重复上述过程。直到LLM输出结束流（EOS）标记表示解码结束或者已经生成所需数量的token。

```python
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]
tokens = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")
print(tokens)
#{
#    'input_ids': tensor([
#        [  101,  1045,  1005,  2310,  2042,  3403,  2005,  1037, 17662, 12172,2607,  2026,  2878,  2166,  1012,   102],
#        [  101,  2061,  2031,  1045,   999,   102,     0,     0,     0,     0, 0,     0,     0,     0,     0,     0]
#    ]),     
#    'attention_mask': tensor([
#        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#        [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#    ])
#}
output = model(**tokens)
print(output)
# SequenceClassifierOutput(
#    loss=None, 
#    logits=tensor([[-1.5607,  1.6123],[-3.6183,  3.9137]], grad_fn=<AddmmBackward0>), 
#    hidden_states=None, 
#    attentions=None
#)
```

`output = model(**tokens)` 对于文本分类来说，整段文本输入返回一个标签，也就是SequenceClassifierOutput.logits，那么对于文本生成来说，整段文本输入返回CausalLMOutputWithPast， 对于 input_ids = [v1,v2,v3] 输出为 [v20,v30,v4]。 v20 是根据v1 生成的下一个token（大概率跟真实的v2 不一样），v4 是根据v1,v2,v3 生成的。
1. 生成式模型的训练是并行的，靠的attention mask操作。训练的时候只需要调用一次即可，因为attention mask的机制可以保证当前token的loss不包含后面的token。inputs 则是一个dict，包含input_ids  和  attention_mask。
2. 推理的时候，只能一个字一个字的推理，所以会调用多次。 output = model(inputs)  或者 output = model.forward(inputs)的时候，inputs 其实也不需要包含 attention_mask，inputs 仅仅是token 就可以

model(xx) ==> `Module.__call__` ==> Module.forward/model.forward，几乎每一个llm 都会自定义forward 方法，如果向forward 方法传入 labels，还会自动计算loss。

```python
class Module:
	__call__ : Callable[..., Any] = _call_impl
	def _call_impl(self, *args, **kwargs):
        forward_call = (self._slow_forward if torch._C._get_tracing_state() else self.forward)
       	...
     	result = forward_call(*args, **kwargs)  # dict 可以作为kwargs参数传入
     	... # 涉及到动态图的构建
     	return result

class PreTrainedModel(nn.Module, ModuleUtilsMixin, GenerationMixin, PushToHubMixin, PeftAdapterMixin):
	def save_pretrained(...)
	@classmethod
    def from_pretrained(...)
class GenerationMixin:
   def generate(inputs,...):> Union[GenerateOutput, torch.LongTensor]:
        ...
        outputs = self(model_input_ids,attention_mask=model_attn,...)
        ...             
class ChatGLMPreTrainedModel(TorchModel, PreTrainedModel):
    ...
class ChatGLMModel(ChatGLMPreTrainedModel):
	def forward(self,input_ids,attention_mask,...):
        ...
class GPT2PreTrainedModel(PreTrainedModel):
    ...
class GPT2LMHeadModel(GPT2PreTrainedModel):
    def __init__(self, config):
        ...
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        ...
    def forward(self,input_ids,attention_mask,labels,...):
       transformer_outputs = self.transformer(input_ids,attention_mask,...)
       hidden_states = transformer_outputs[0]
       lm_logits = self.lm_head(hidden_states)
       loss = None
       if labels is not None:
           labels = labels.to(lm_logits.device) # move labels to correct device to enable model parallelism
           shift_logits = lm_logits[..., :-1, :].contiguous() # Shift so that tokens < n predict n
           shift_labels = labels[..., 1:].contiguous()
           loss_fct = CrossEntropyLoss()
           loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return ((loss,) + output) if loss is not None else output     
```

一些细节

1. Models expect a batch of inputs。当你试图将两个（或更多）句子组合在一起时，它们的长度可能不同。为了解决这个问题，我们将使用填充使张量具有矩形。Padding通过在值较少的句子中添加一个名为Padding token的特殊单词来确保我们所有的句子长度相同。当要求处理更长的序列时
    ```python
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    batched_ids = [
        [200, 200, 200],
        [200, 200, tokenizer.pad_token_id],
    ]
    attention_mask = [
        [1, 1, 1],
        [1, 1, 0],
    ]
    outputs = model(torch.tensor(batched_ids), attention_mask=torch.tensor(attention_mask))
    print(outputs.logits)
    ```
2. attention_mask ，self-attention使用，shape和input_ids一致。
    1. Encoder中的Mask。模型训练和预测基本都是批量化处理的（**训练时不是自递归的**），处理多个序列时，在attention的时候，不去attend被mask掉的部分。 作用是告诉模型一个batch的数据里哪些是padding的，从而可以忽略掉那些padding的部分，也叫padding mask ，比如下图的例子：
        ![](/public/upload/machine/attention_mask.jpg)
    2. decoder中的mask。用于在训练过程中解码的时候掩盖掉当前时刻之后的信息；也叫sequence mask
        ![](/public/upload/machine/sequence_mask.jpg)
    3. 为什么Attention Mask不是0和1构成的矩阵，而是0和负无穷构成的？在 Transformer 模型中通常用于指示模型哪些位置是有效的输入，哪些位置是填充的。它的主要目的是确保模型在计算注意力分数时不会考虑到填充的位置。在大多数实现中，当我们说“mask”时，我们通常是指一个由0和1组成的矩阵，其中1表示“考虑这个位置”而0表示“不考虑这个位置”。但在实际的注意力机制计算中，这种简单的0和1的表示方法并不直接适用。Transformer中的注意力机制涉及到softmax函数，该函数会将输入的原始分数转换为概率分布。为了确保某些位置在softmax之后的概率为0，我们需要在softmax之前为这些位置赋予一个非常小的分数，通常是负无穷。这样，经过softmax转换后，这些位置的概率会接近于0。


### 关于chat template（不是langchain 的ChatTemplate）

对于chat api来说，server 收到的是一个message list，但是，不管是Base Model还是Instruct Model，模型最终接受的输入，应该是一段free text（再转成token）。那么问题来了，这个有结构的对话历史列表，是如何转成free text的呢？显然，这里需要一个模板（template），这就是所谓的chat template，表达各种message（包括system message，user message，assistant message以及其它类型的message）的渲染方式。

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(
  openai_api_key="EMPTY",
  openai_api_base="http://127.0.0.1:8000/v1",
  model_name="llama3.2-1B-instruct"
)
prompt = ChatPromptTemplate.from_messages([
    ("system", "Your are a helpful assistant."),
    ("user", "Hello, how are you?"),
    ("assistant", "I'm doing well, thank you for asking."),
    ("user", "Can you tell me a joke?")
  ]
)
chain = prompt | llm
reponse = chain.invoke({})
```
以llama3.2为例， 最终输入到LLM时会转化成如下的free text（也就是prompt）：

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: 28 Dec 2024

Your are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

Hello, how are you?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

I'm doing well, thank you for asking.<|eot_id|><|start_header_id|>user<|end_header_id|>

Can you tell me a joke?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
```
那么，这个chat template是从哪里来的呢？对于vLLM来说，它启动的时候，有两种方式可以获取到chat template：
1. 一种方式是从模型文件夹中加载。具体地说，chat template的内容存在于tokenizer_config.json文件中。需要注意的是，tokenizer_config.json文件中并不一定包含chat_template字段。
2. 另一种方式是vLLM通过启动参数--chat-template来指定一个chat template模板文件。
如果都没有，vLLM会以未指定chat template模板的方式启动起来。会出现意想不到的结果。
1. 如果Transformers的版本小于4.44，vLLM会自动使用一个默认的chat template。这时候从调用结果上很可能看不出什么大问题，但实际上模型回答的准确度已经大打折扣，这个错误非常不易察觉。
2. 如果Transformers的版本大于等于4.44，vLLM会抛一个异常，如下：`openai.BadRequestError: Error code: 400 - {'object': 'error', 'message': 'As of transformers v4.44, default chat template is no longer allowed, so you must provide a chat template if the tokenizer does not define one.', 'type': 'BadRequestError', 'param': None, 'code': 400}`


### 推理/generate实现

[浅谈LLAMA2核心函数generate源码](https://mp.weixin.qq.com/s/vnke00f7kzlA16Pw_FJFjQ)

```python
def generate(self,
        prompt_tokens: List[List[int]],  # 输入的提示
        max_gen_len: int,  # 最大生成长度
        temperature: float = 0.6,  # 影响生成文本的随机性
        top_p: float = 0.9,  # 用于决定采样过程中保留的 token 集合的概率阈值
        logprobs: bool = False,  # 是否返回每个 token 的对数概率
        echo: bool = False,  # 是否返回输入的提示
) -> Tuple[List[List[int]], Optional[List[List[float]]]]:
    # ---------------------------初始化长度为 total_len tokens张量，并填充 pad_id----------------------------------
    params = self.model.params
    bsz = len(prompt_tokens)
    assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

    min_prompt_len = min(len(t) for t in prompt_tokens)
    max_prompt_len = max(len(t) for t in prompt_tokens)
    assert max_prompt_len <= params.max_seq_len
    total_len = min(params.max_seq_len, max_gen_len + max_prompt_len)

    pad_id = self.tokenizer.pad_id
    tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device="cuda")
    # 将prompt_tokens中的token复制到tokens张量中。
    for k, t in enumerate(prompt_tokens):
        tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")
    if logprobs: # 是否返回每个 token 的对数概率
        # 创建一个与tokens相同形状的token_logprobs张量，并用0填充
        token_logprobs = torch.zeros_like(tokens, dtype=torch.float)

    prev_pos = 0
    eos_reached = torch.tensor([False] * bsz, device="cuda")
    input_text_mask = tokens != pad_id
    # -------------------------------------------------------------

    for cur_pos in range(min_prompt_len, total_len):
        # 调用模型的forward方法获取logits
        logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
        if logprobs: # 是否返回每个 token 的对数概率
            # 计算token level的logprobs
            token_logprobs[:, prev_pos + 1: cur_pos + 1] = -F.cross_entropy(
                input=logits.transpose(1, 2),
                target=tokens[:, prev_pos + 1: cur_pos + 1],
                reduction="none",
                ignore_index=pad_id,
            )
        # 根据温度参数和top_p参数对logits进行softmax和采样，得到下一个token
        if temperature > 0:
            # sample_top_p函数对probs进行采样
            probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
            next_token = sample_top_p(probs, top_p)
        else:
            # 将logits中概率最大的token作为下一个token。
            next_token = torch.argmax(logits[:, -1], dim=-1)

        next_token = next_token.reshape(-1)
        # only replace token if prompt has already been generated
        next_token = torch.where(
            input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
        )
        # tokens张量更新
        tokens[:, cur_pos] = next_token
        eos_reached |= (~input_text_mask[:, cur_pos]) & (
                next_token == self.tokenizer.eos_id
        )
        prev_pos = cur_pos
        # 检查是否已经生成了所有的eos token，如果是则停止生成
        if all(eos_reached):
            break

    if logprobs:
        # token_logprobs列表化
        token_logprobs = token_logprobs.tolist()
    out_tokens, out_logprobs = [], []
    for i, toks in enumerate(tokens.tolist()):
        # cut to max gen len
        # 对于 tokens 张量中的每一行（即每一个生成的序列），如果 echo 参数为假，则去掉提示部分
        start = 0 if echo else len(prompt_tokens[i])
        toks = toks[start: len(prompt_tokens[i]) + max_gen_len]
        probs = None
        if logprobs:
            probs = token_logprobs[i][start: len(prompt_tokens[i]) + max_gen_len]
        # cut to eos tok if any
        # 存在结束标记，则去掉结束标记之后的部分
        if self.tokenizer.eos_id in toks:
            eos_idx = toks.index(self.tokenizer.eos_id)
            toks = toks[:eos_idx]
            probs = probs[:eos_idx] if logprobs else None
        out_tokens.append(toks)
        out_logprobs.append(probs)
    # 返回生成的tokens和对数概率（如果logprobs参数为真）
    return (out_tokens, out_logprobs if logprobs else None)
```
推理的时候不使用mask，要串行执行，只有得到前一个单词，重新进入decoder模块，通过模型推理才能得到下一个输出。

[以LLAMA为例，快速入门LLM的推理过程](https://mp.weixin.qq.com/s/5lbrqbqiHPZIARsVW6l6tA) 建议细读。

![](/public/upload/machine/llama_machine.jpg)

### 解码策略

大模型在运算之后，不是只给一个输出选择，而是会给多种都说得过去的词。可以把解码器文本Transformer模型理解为一个函数，它以词元作为输入并生成一个概率数组，用于表示词汇表中所有词元的概率，然后，程序根据这些概率从所有词元中进行采样，以指导采样过程，并生成下一个词元，这一过程会重复进行。在GPT模型最后一层之前，推理的对象是以向量形式表征的语义，输出的是代表语义的一个“模糊”的向量。此处“模糊”指的是，**这一向量或许并不对应任何一个已知的词**。因此，整个模型最后需要再做一个推测，基于这个“模糊”的向量所包含的语义信息，在词表中寻找最符合这些特征的词，来作为真正的输出。在 transformer 中，最后的输出是一个概率分布，表示每一个词匹配这一“模糊”向量的概率。

[一网打尽文本生成策略（beam search/top-k/top-p/温度系数）](https://zhuanlan.zhihu.com/p/676398366)假设我们输入 “I am a”，GPT会对这个序列进行编码得到embedding，并预测下一个单词的概率。“I am a”作为输入，编码后也会得到三个token对应的embedding。GPT会使用输入序列最后一个token对应的embedding做预测，也就是说，”a”经过编码后的embedding会被映射到整个词表上，得到下一个单词的概率。因为GPT是使用masked attention的，每个token在编码时都只会和他前面的token交互。**那么最后一个token自然就包括了当前序列的所有信息，因此用于预测下一个token是最合理的**。

![](/public/upload/machine/llm_last_token.jpg)

[Transformer中的解码策略](https://mp.weixin.qq.com/s/x5o4BGHLScriYM36AKhesA)解码（Decoding）就是在词表中抓阄。具体而言：模型根据下一个词的概率分布，从词表中选择下一个词。这个选择的策略就是解码策略，也称解码算法、解码方法等。
1. 贪心搜索（Greedy Search）
1. Beam search，保留num_beams个概率较大的句子，最后从num_beams个句子中输出最大概率的句子。当输入为“I am a”, 设置num_beams=2，beam search的过程可表达为：假设A,B,C对应的概率分别为0.5,0.2,0.3。那么此时选择每个token并组成序列的概率分别为：
    ```
    “I am a A”: 0.5
    “I am a B”: 0.3
    “I am a C”: 0.2
    ```
    beam search会选择概率最大的num_beams=2个序列，参与后续的生成。因此“I am a A”和“I am a B”会参与下一个token的预测。假设得到六个可能序列对应的概率。此时再保留num_beams=2个最大概率的序列，比如是”I am a A C”=0.25, ”I am a B C”=0.27，再把这两个序列送到下一个token的预测中。上述步骤会一直重复，直到遇到结束符或指定的长度。最终，会有num_beams个序列被预测出来。此时可以对这几个概率最大的序列做进一步的处理，例如选一个概率最大的作为最终的输出，或者根据概率做个采样作为输出。beam search有什么好处呢？相比于每次都选最大的贪心，beam search显然增大了搜索空间。而且更重要的是，beam search会防止潜在概率很大的序列被过早的抛弃。例如”I am a B C”，在预测到B的时候序列的概率还是不大的，但是到C就变得更大了。当num_beams=1时，beam search等价于贪心。
2. Top-k sampling，就是每一步只考虑概率最大的k个token，并且把这k个token的概率做重归一化，并随机采样得到预测的token。假设在一步中，ABC对应的概率ligits是[5,3,2]，k设置为2。那么会选出字母A,B，并把其对应的概率logits[5,3]进行重新归一化softmax([5,3]) = [0.88,0.12]。随后基于归一化后的概率随机选A或B，拼接到“I am a”后面，并进行下一个token的预测，如此反复。top-k和beam search有什么区别呢？top-k自始至终只有一个序列进行预测，k只用于规定采样的范围，每步只采样一个token作为结果。而beam search会保留num_beams个序列进行预测。
3. top-p sampling，Top-p采样中的p表示累积概率。这种策略会把token的概率按照递减的次序累加，直到累加的概率值超过了阈值p，在这些token中做采样得到预测。假设p=0.7，ABC在第一步预测的概率分布为[0.5,0.3,0.2]。那么A和B的概率值加起来超过了0.7，第一步就会在A,B中采样得到预测。假设第二步概率分布为[0.3,0.3,0.4]，那么ABC三个加起来才会超过0.7，此时第二步就会在这三个里面采样，如此反复。可以看出top-p的每一步实际上会在一个动态长度的范围里做采样。这样的优点是可以排除一些概率不高的单词，例如分布为[0.9,0.05,0.05]时，只考虑第一个就足够了，而top-k还是会考虑前k个。并且在分布相对均衡时，top-p会增加输出的多样性。
4. 温度系数，在上面提到的归一化中，我们还可以引入温度系数调整概率分布

    $$
    Softmax(x_i) = \frac{e^{x_i}/T}{\sum_j e^{x_j}/T}
    $$

    T越大归一化后的概率分布越均匀，而T越小概率分布越陡峭。因此大的T会增加模型输出的随机性，而小的T则会让模型的输出更加固定。**低温输出序列稳定，高温输出序列灵活、变化大**（PS：这有点熵/热力学的感觉了）。这也是“温度系数影响模型创造性”说法的原因。
5. Constrained Decoding。[一文弄懂 LLM 结构化数据生成原理](https://mp.weixin.qq.com/s/M5b-e1eLZyy1SBRiUw1z4Q)，结构化数据生成的原理用一句话概括就是：每个 step 拿到当前 model 给出的 logits 之后，在采样下一个 token 之前，通过人工设定的规则可以得到当前 step 只允许采样的 token 集合，接着通过加 bias 的方式压制其他不允许采样的 token，从而实现指定的结构化数据生成。
    1. [探索LLM推理全阶段的JSON格式输出限制方法](https://mp.weixin.qq.com/s/MNj_EQ62UZC9SlOTtfYy4g)
    2. [我把SQL编辑器装进了大模型？](https://mp.weixin.qq.com/s/7i7gOs8xtSb90PuxCIvfUw) 让llm的输出符合sql 规则。

以上策略体现在代码上 就是：经过多个AttentionLayer  hidden_states，Normalization之后得到 outputs.logits，将 logits 传递给 logits_processor 和 logits_warper（包含TopKLogitsWarper/TopPLogitsWarper等），最后，使用 softmax 函数将经过预处理的 logits 转换为概率分布，并利用 multinomial 方法从中采样得到下一个 token。

![](/public/upload/machine/llm_logprobs.jpg)

在transformers库中，其【Generationmixin.generate()】实现了各种解码策略。generate与forword的区别主要有：
1. generate只用于推理场景，forward既用于推理场景，也用于训练场景。
2. generate是基于forward实现的。
3. generate利用各种解码策略，生成文本；而forward最主要的作用就是提供logits（未归一化的分布）。
4. 解码策略对于文本的生成是非常重要的。我们可以不使用generate，而直接根据forward的logits进行解码，只要自己实现各类解码策略即可。
    1. Hugging Face Logits Processor: 一个标准的接口，用于在 LLM 的每个解码步骤中修改 token 的概率分布 (logits)。

### forward实现

[Qwen整体介绍](https://github.com/datawhalechina/tiny-universe/tree/main/content/Qwen-blog) 以代码组织的视角来看下 transformer的设计图。我们看到很多模型的模块图，层次结构跟代码也是一致的。 

[Qwen2源码阅读——核心代码跟读](https://mp.weixin.qq.com/s/PVSPNfv0I8_cxgPTmOes5w) 建议细读。

![](/public/upload/machine/qwen_code_view.jpg)

```python
def run_qwen2():
    qwen2config = Qwen2Config(vocab_size=151936,...)
    qwen2model = Qwen2Model(config=qwen2config)
    input_ids = torch.randint(0,qwen2config.vocab_size,(4,30))
    res = qwen2model(input_ids)
    print(type(res))
```
Qwen2Model 干活的三个主力（3个成员）：嵌入层（embed_tokens）、解码器层（layers）、归一化层（norm）。Qwen2Model 及其成员都是一个 nn.Module。**最上层是Qwen2Model.forward  最底层是pytorch 运算，中间是一系列 nn.Module 用于抽象和复用**，上一层的nn.Module.forward 通过驱动其成员的forward 干活儿。
```python
class Qwen2Model(Qwen2PreTrainedModel):
    def __init__(self, config: Qwen2Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()
    def forward(input_ids,attention_mask,position_ids,...) -> Union[...]:
        inputs_embeds = self.embed_tokens(input_ids)
        # embed positions
        hidden_states = inputs_embeds
        for idx, decoder_layer in enumerate(self.layers):
            # 将所有的hidden_states保存成tuple
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            # 将hs送入每一层decoder_layer
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
            # 取出上一层decoder_输出的hs,再传入下一个layer
            # 只要第一个,第二个是cache的一个类，然后进入下一个layer
            hidden_states = layer_outputs[0]
        # 将最后layers输出后的hidden_states进行标准化  
        hidden_states = self.norm(hidden_states)   
        # 加上最后一层的hidden_states
        if output_hidden_states:
            all_hidden_states += (hidden_states,) 
```
重点转到了 Qwen2DecoderLayer，三件套对应3个成员:attn+MLP+norm。forward逻辑：首先复制一份hidden_states为residual,然后将hidden_states送入Norm,再送入attn模块。得到attn的输出后，再复制一份residual，再将hidden_states送入Norm，mlp，再与residual进行相加。最后输出的就是这个hidden_states啦。

```python
# 为了提高性能，ATTENTION layer 有一些优化实现，比如flash attention，sdpa
QWEN2_ATTENTION_CLASSES = {
    "eager": Qwen2Attention,  # 一般情况下是这个
    "flash_attention_2": Qwen2FlashAttention2,
    "sdpa": Qwen2SdpaAttention,
}
class Qwen2DecoderLayer(nn.Module):
    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = QWEN2_ATTENTION_CLASSES[config._attn_implementation](config, layer_idx)

        self.mlp = Qwen2MLP(config)
        # input_layernorm和post_attention_layernorm内容是一样的，只是应用的顺序不一样。
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    def forward(hidden_states,attention_mask,position_ids,...) -> Union[...]:
        residual = hidden_states 
        #  标准化后送入attn
        hidden_states = self.input_layernorm(hidden_states)  # RMSNorm标准化
        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(  
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )
        # 残差与新的hidden_states相加
        hidden_states = residual + hidden_states
        # Fully Connected
        residual = hidden_states
        # 同样的RMSNorm标准化
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        outputs = (hidden_states,)
        return outputs
```
重点转到 Qwen2Attention，forward逻辑：首先将hidden_states送入Linear中得到query、key与value。使用旋转位置嵌入操作rotary_emb，使用了旋转位置嵌入的余弦和正弦部分，将他们与query和key相乘，并将结果相加，从而实现旋转位置嵌入的效果。将key_states和value_states重复group次，再执行dot attn操作。在dot attn操作后得到attn_weights,加上attention_mask从而实现读取掩盖操作，在经过softmax与value_states相乘。得到attn_output。再将上述的attn_output进行reshape操作，送入o_proj，得到最终的输出。
```python
class Qwen2Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads # 键值对的头数
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads # 键值对的组数
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.attention_dropout = config.attention_dropout

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        # 后续LoRa也基本都对q_proj/k_proj/v_proj/o_proj 四个Linear操作动的刀子
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        
        self.rotary_emb = Qwen2RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )
    def forward(hidden_states,attention_mask,position_ids,...) -> Union[...]:
        # 获取形状信息,hidden_states输入的为(bs,T,hd)
        bsz, q_len, _ = hidden_states.size()
        # 对hidden_states进行Linear生成query、key、value
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        # reshape多头处理--分块--(bs,T,heads,hd_d)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        # 将旋转位置嵌入应用于查询和键张量。使用了旋转位置嵌入的余弦和正弦部分，将它们与查询和键张量相乘，并将结果相加，从而实现旋转位置嵌入的效果
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        # 先将key_states和value_states重复了num_key_value_groups次
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        # 使用dot attn实现q*kT/hd_d^0.5
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim) # # 多维矩阵乘法
        # 然后 attn_weights 加上 attention_mask，实现读取顺序
        attn_weights = attn_weights + attention_mask
        # softmax + dropout + values_states相乘
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)
        # 转置，修改形状等reshape操作
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        # 最后在进行一次o_proj
        attn_output = self.o_proj(attn_output)
        # 返回结果
        return attn_output, attn_weights, past_key_value
```

Qwen2 MLP 输入hidden_state并行送入两个Linear层，其中一个激活一下，再与另一个相乘，最终再经过一个Linear，输出最终结果。
```python
class Qwen2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 这俩不必多说
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        # 三个全连接层
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj
```

Qwen2RMSNorm 
```python
class Qwen2RMSNorm(nn.Module):  # 标准化层
    def __init__(self, hidden_size, eps=1e-6):
        """
        Qwen2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)
```
## 模型加载

通常情况下，使用Pytorch来加载模型步骤如下：
1. 创建模型（这里一般先有一个读取模型配置的过程， 这样知道有几层，hidden_dim 是多少，然后才是读取每层的权重。权重按层顺序存储在模型文件中，其实也可以认为是按算子顺序存在模型文件中）
2. 将权重加载到内存（一个叫做state_dict的字典对象）
3. 在已创建模型基础上加载权重值（将权重、参数等信息填入到resnet网络结构中）
4. 将模型加载到相应设备上（如：GPU）进行模型推理

```python
import torch

my_model = ModelClass(...)
state_dict = torch.load(checkpoint_file)
my_model.load_state_dict(state_dict)
```

对于超大模型进行推理，这种常规的加载方式存在一些明显的局限性。第一步，在 RAM 中加载模型的完整版本，并花一些时间随机初始化权重，第二步将模型的另一个完整预训练权重加载到 RAM 中。因此，对于一个6B的模型，使用FP16半精度加载，也需要24G内存。因此，这种常规的加载方式已经无法满足要求。在 Transformers 库中利用Accelerate库来完成这些超大模型的加载。那么 Accelerate 如何利用 PyTorch 来加载和运行大模型进行推理呢？具体步骤如下：
1. 创建一个空模型（没有权重）
2. 当有多个计算设备可用时，决定每一层权重的去向。Accelerate库提供了一个函数用来自动检测一个空模型使用的设备类型。它会最大化利用所有的GPU资源，然后再使用CPU资源，并且给不能容纳的权重打上标记，并 offload 到硬盘。
3. 加载部分权重到内存
4. 将内存中的这些权重加载到空模型中
5. 将权重移动到计算设备上完成推理
6. 对剩下的权重，重复步骤 3，直到加载完所有权重，并完成最终推理。

```python
import torch
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

with init_empty_weights():
    model = MyModel(...)

model = load_checkpoint_and_dispatch(
    model, checkpoint=checkpoint_file, device_map="auto" # device_map=auto会根据可用资源自动确定模型每一层的放置位置
)

input = torch.randn(2,3)
input = input.to("cuda")
output = model(input)
```

在 Transformers 库中，也是通过在from_pretrained()或者pipeline()函数中设置device_map=auto完成同样的操作。此外，device_map有如下选项可供设置：
1. "auto" 或 "balanced": Accelerate将会根据所有GPU均衡切分权重，尽量均匀的切分到各个GPU上；
2. "balanced_low_0": Accelerate均匀分割权重到各个GPU上，除了第一个GPU（序号为0）。在第一个GPU上会尽量节省显存（这种模式可以有效节省第一个GPU的显存，以便使用generate函数用于模型生成）；
3. "sequential": Accelerate按照GPU的顺序占用显存（后面的GPU可能根本不会使用）。
当然也可以根据需求设置device_map参数，以决定各个部分权重应该放置的设备。`device_map = {"block1": 0, "block2.linear1": 0, "block2.linear2": 1, "block2.linear3": 1}`
此外，还可以通过max_memory来控制各个设备使用的内存的大小。`max_memory={0: "10GiB", 1: "20GiB", 2: "20GiB", "cpu": "60GiB"}`

在模型的加载过程中，通常包含两个核心部分：模型的权重和配置文件。

每个模型的配置文件的变量的名字都不尽相同，比如表示transformer模型多头注意力机制的head num的变量名，有的模型可能叫num_attention_heads(opt)，有的模型可能叫n_head(starcoder)，所以需要对应到框架统一的变量体系下。

```python
class Opt(GPT):
    """Opt"""
    @classmethod
    def _create_config(cls, ckpt_path: str):
        offset = 2
        config_dict = get_config_from_path(ckpt_path)
        config = GptInitModelParameters(
            head_num=config_dict['num_attention_heads'],
            size_per_head=config_dict['hidden_size'] // config_dict['num_attention_heads'], # 每个注意力头的维度大小
            layer_num=config_dict.get('num_hidden_layers', 12),
            vocab_size=config_dict['vocab_size'],   # 词表的大小
            max_seq_len=config_dict['max_position_embeddings'] + offset # 模型能够处理的最长序列长度
        )
        config.layernorm_type = 'pre_layernorm'
        config.norm_type = "layernorm"
        config.has_post_decoder_layernorm = True
        config.hidden_size = config_dict['hidden_size']
        config.inter_size = config_dict["ffn_dim"]
        config.has_positional_encoding = True
        config.activation_type = 'relu'
        config.add_special_tokens = True
        config.special_tokens.eos_token_id = config_dict.get('eos_token_id', 2)
        config.special_tokens.pad_token_id = config_dict.get('pad_token_id', 1)
        config.special_tokens.bos_token_id = config_dict.get('bos_token_id', 2)
        config.head_num_kv = config.head_num
        return config
```

在加载配置文件后，框架会根据这些参数去加载权重等。例如，如果has_positional_encoding为True，框架会初始化nn.embedding(config.max_seq_len, config.hidden_dim)。

模型权重加载就是将模型的权重参数名对应到框架的权重参数名上, 因为每个模型的权重参数名定义每个都不一样，所以需要对应到框架统一的变量体系下；

```python
class OptWeightInfo(ModelDeployWeightInfo):
    def _get_weight_info(self):
        layer_weights = [
            # * Attention之前 layer_norm部分
            WeightInfo(W.pre_ln_gamma, [CkptWeightInfo('model.decoder.layers.{i}.self_attn_layer_norm.weight', identity)], identity),
            WeightInfo(W.pre_ln_beta, [CkptWeightInfo('model.decoder.layers.{i}.self_attn_layer_norm.bias', identity)], identity),

            # *  Attention
            WeightInfo(W.attn_qkv_w, [
                CkptWeightInfo('model.decoder.layers.{i}.self_attn.q_proj.weight', identity),
                CkptWeightInfo('model.decoder.layers.{i}.self_attn.k_proj.weight', identity),
                CkptWeightInfo('model.decoder.layers.{i}.self_attn.v_proj.weight', identity),
                ], functools.partial(merge_qkv_hf)),
            ...

```

![](/public/upload/machine/model_load_weight.jpg)

```
model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
``` 
AutoModel.from_pretrained 到最后实际是 pytorch.nn.module.load_state_dict。

此外，Transformers 为了减少 GPU VRAM 使用，实现了Offload KV Cache的功能，通过将大多数层的 KV 缓存移至 CPU 来实现这一点。当模型的forward()方法迭代各层时，该策略会在GPU上维护当前层的缓存。同时，它异步预取下一层缓存，并将上一层缓存发送回 CPU。与 KV 缓存量化（可能损失模型精度）不同，此策略始终与默认 KV Cache 实现产生相同的结果。