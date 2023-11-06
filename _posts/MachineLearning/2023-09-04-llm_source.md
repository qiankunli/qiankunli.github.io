---

layout: post
title: LLM部分技术源码学习
category: 技术
tags: MachineLearning
keywords: llm source

---

* TOC
{:toc}

## 前言

## OPENAI API

Completions API 主要用于补全问题，用户输入一段提示文字，模型按照文字的提示给出对应的输出。

||||
|---|---|---|
|model|必选参数|调用的Completions模型名称，如text-davinci-003、text-curie-001等，不同模型参数规模不 同；在大模型领域，（就OpenAI提供的A、B、C、D四大模型来看）参数规模越大、越新版本的模型效果更好（费用也更高）|
|prompt|必选参数|提示词|
|suffix|可选参数|默认为空，具体指模型返回结果的后缀|
|max_tokens|可选参数|默认为16，代表返回结果的token数量|
|temperature|可选参数|取值范围为0—2，默认值为1。参数代表采样温度，数值越小，则模型会倾向于选择概率较高的词汇，生成的文本会更加保守；而当temperature值较高时，模型会更多地选择概率较低的词汇，生成的文本会更加多样|
|top_p|可选参数|取值范围为0—1，默认值为1，和temperature作用类似，用于控制输出文本的随机性，数值越趋近与1，输出文本随机性越强，越趋近于0文本随机性越弱；通常来说若要调节文本随机性，top＿p和temperature两个参数选择一个进行调整即可；更推荐使用temperature参数进行文本随机性调整|
|n|可选参数|默认值为1，表示一个提示返回几个Completion|
|stream|可选参数|默认值为False，表示回复响应的方式，当为False时，模型会等待返回结果全部生成后一次性返回全部结果，而为True时，则会逐个字进行返回|
|logprobs|可选参数|默认为null，该参数用于指定模型返回前N个概率最高的token及其对数概率。例如，如果logprobs设为10，那么对于生成的每个token，API会返回模型预测的前10个token及其对数概率；|
|echo|可选参数|默认为False，该参数用于控制模型是否应该简单地复述用户的输入。如果设为True，模型的响应会尽可能地复述用户的输入|
|stop|可选参数|该参数接受一个或多个字符串，用于指定生成文本的停止信号。当模型生成的文本遇到这些字符串中的任何一个时，会立即停止生成。这可以用来控制模型的输出长度或格式；|
|presence_penalty|可选参数|默认为0，取值范围为［—2，2］，该参数用于调整模型生成新内容（例如新的概念或主题）的倾向性。较高的值会使模型更倾向于生成新内容，而较低的值则会使模型更倾向于坚持已有的内容，当返回结果篇幅较大并且存在前后主题重复时，可以提高该参数的取值；|
|frequency_penalty|可选参数|默认为0，取值范围为［—2，2］，该参数用于调整模型重复自身的倾向性。较高的值会使模型更倾向于避免重复，而较低的值则会使模型更可能重复自身；当返回结果篇幅较大并且存在前后语言重复时，可以提高该参数的取值；|
|best_of||该参数用于控制模型的生成过程。它会让模型进行多次尝试（例如，生成5个不同的响应），然后选择这些响应中得分最高的一个；|
|logit_bias||该参数接受一个字典，用于调整特定token的概率。字典的键是token的ID，值是应用于该token的对数概率的偏置；在GPT中可以使用tokenizer tool查看文本Token的标记。一般不建议修改；|
|user|可选参数|使用用户的身份标记，可以通过人为设置标记，来注明当前使用者身份。|

Chat模型升级的核心功能是对话， 它基于大量高质量对话文本进行微调，能够更好的理解用户对话意图，所以它能更顺利的完成与用户的对话（大语言模型本质上都是概率模型，根据前文提示进行补全是⼤语⾔模型的原始功能，而对话类的功能则是加⼊额外数据集之后训练的结果）。

ChatCompletion.create函数的详细参数和Completion.create函数相比发生了以下变化：
1. 用messages参数代替了prompt参数，使之更适合能够执行对话类任务
2. 新增functions和function_call参数，使之能够在函数内部调用其他工具的API
3. 其他核心参数完全一致，例如temperature、top_p、max_tokens、n、presence_penalty等参数的解释和使用方法都完全一致，且这些参数具体的调整策略也完全一致
4. 剔除了best_of参数，即Chat模型不再支持从多个答案中选择一个最好的答案这一功能

## Transformers

[NLP Course](https://huggingface.co/learn/nlp-course) 官方教程，建议从头到尾细看一下。

[Transformers快速入门](https://transformers.run/)Hugging Face 专门为使用 Transformer 模型编写了一个 Transformers 库，建立在 Pytorch 框架之上（Tensorflow 的版本功能并不完善），所有 Transformer 模型都可以在 Hugging Face Hub 中找到并且加载使用，包括训练、推理、量化等。

![](/public/upload/machine/transformers_overview.png)

### pipelines

开箱即用的 pipelines，它封装了预训练模型和对应的前处理和后处理环节。
1. 预处理 (preprocessing)，**Transformer模型无法直接处理原始文本字符串**；具体地，我们会使用每个模型对应的分词器 (tokenizer) 来进行。 PS： 有点类似于tf的各种FeatureColumn
  1. 将输入切分为词语、子词或者符号（例如标点符号），统称为 tokens；
  2. 根据模型的词表将每个 token 映射到对应的 token 编号/id；
  3. 根据模型的需要，添加一些额外的输入。比如tokens补齐和截断操作
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

### 基础

transformers开源库的核心组件包括3个：Conﬁguration、Tokenizer、Model
1. 「Conﬁguration」：配置类，通常继承自「PretrainedConﬁg」，保存model或tokenizer的超参数，例如词典大小，隐层维度数，dropout rate等。配置类主要可用于复现模型。
2. 「Tokenizer」：Model只能处理数字，因此Tokenizer需要将我们的文本输入转换为数字。
    1. 通常继承自「PreTrainedTokenizer」，主要存储词典（也就是`from_pretrained()` 的部分），token到index映射关系等。
    2. 此外，还会有一些model-specific的特性，如特殊token，`[SEP]`, `[CLS]`等的处理，token的type类型处理，语句最大长度等，**因此tokenizer通常和模型是一对一适配的**。
3. 「Model」: 模型类。封装了预训练模型的计算图过程，遵循着相同的范式，如根据token ids进行embedding matrix映射，紧接着多个self-attention层做编码，最后一层task-specific做预测。
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

### 源码分析

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

model(xx) ==> `Module.__call__` ==> Module.forward/model.forward

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

class ChatGLMPreTrainedModel(TorchModel, PreTrainedModel):
    ...
class ChatGLMModel(ChatGLMPreTrainedModel):
	def forward(self,input_ids,attention_mask,...)
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

## Trainer

在我们定义 Trainer 之前首先要定义一个 TrainingArguments 类，它将包含 Trainer用于训练和评估的所有超参数。唯一必须提供的参数是保存训练模型的目录，以及训练过程中的检查点。对于其余的参数，可以保留默认值。

```python
from transformers import TrainingArguments
raw_datasets = load_dataset("glue", "mrpc")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

training_args = TrainingArguments("test-trainer")
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)
trainer.train()
```
不用trainer，纯手工实现训练过程，也是trainer帮我们自动化的部分
```python
raw_datasets = load_dataset("glue", "mrpc")

# data preprocessing
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")
## tokenized_datasets column_names: ["attention_mask", "input_ids", "labels", "token_type_ids"]
train_dataloader = DataLoader(
    tokenized_datasets["train"], shuffle=True, batch_size=8, collate_fn=data_collator
)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"], batch_size=8, collate_fn=data_collator
)

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
optimizer = AdamW(model.parameters(), lr=3e-5)
## put our model and our batches on GPU
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(   #  the learning rate scheduler 
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)
progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
```



## GPT-2养成记 

[Training and Fine-Tuning GPT-2 and GPT-3 Models Using Hugging Face Transformers and OpenAI API](https://www.it-jim.com/blog/training-and-fine-tuning-gpt-2-and-gpt-3-models-using-hugging-face-transformers-and-openai-api/)  非常经典，入门必读。
1.  it does not implement neural networks from scratch(从头开始) but relies on lower-level frameworks PyTorch, TensorFlow, and FLAX. 
2. it heavily uses Hugging Face Hub, another Hugging Face project, a hub for downloadable neural networks for various frameworks. 
3. Model is a valid PyTorch model with some additional restrictions and naming conventions introduced by the transformers framework. 
4. Neural networks are not able to work with raw text; they only understand numbers. We need a tokenizer to convert a text string into a list of numbers. But first, it breaks the string up into individual tokens, which most often means “words”, although some models can use word parts or even individual characters. Tokenization is a classical natural language processing task. Once the text is broken into tokens, each token is replaced by an integer number called encoding from a fixed dictionary. Note that a tokenizer, and especially its dictionary, is model-dependent: you cannot use Bert tokenizer with GPT-2, at least not unless you train the model from scratch. Some models, especially of the Bert family, like to use special tokens, such as `[PAD]`,`[CLS]`, `[SEP]`, etc. GPT-2, in contrast, uses them very sparingly.

以GPT-2 为例 
1. The transformer itself works with a D-dimensional vector at every position, for GPT-2 D=768. 
2. V=50257 is the GPT-2 dictionary size. 

![](/public/upload/machine/gpt_2_tensor_dimensions.jpg)

### GPT-2 model使用

```python
config = transformers.GPT2Config.from_pretrained(MODEL_NAME)
config.do_sample = config.task_specific_params['text-generation']['do_sample']
config.max_length = config.task_specific_params['text-generation']['max_length']
# print(config)
model = transformers.GPT2LMHeadModel.from_pretrained(MODEL_NAME, config=config)
# Tokenizer
tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
# Tokenize the input
enc = tokenizer(['The elf queen'], return_tensors='pt')
print('enc =', enc)
print(tokenizer.batch_decode(enc['input_ids']))

input_ids = enc['input_ids']
attention_mask = torch.ones(input_ids.shape, dtype=torch.int64)
# predicts the next token at each position. 也就是 input_ids = [v1,v2,v3] 输出为 [v20,v30,v4]]。 v20 是根据v1 生成的下一个token，大概率跟v2 不一样，v4 是根据v1,v2,v3 生成的。
out = model(input_ids=input_ids, attention_mask=attention_mask)
logits = out['logits']
# -1 在python list 里表示最后一个元素
new_id = logits[:, -1, :].argmax(dim=1)
print(new_id)
print(tokenizer.batch_decode(new_id))
```

一次想预测一个字符太慢了 可以直接生成一段话，这也是model.generate 的原理。 

```python
input_ids = enc['input_ids']
for i in range(20):
    attention_mask = torch.ones(input_ids.shape, dtype=torch.int64)
    logits = model(input_ids=input_ids,attention_mask=attention_mask)['logits']                    
    new_id = logits[:, -1, :].argmax(dim=1)    # Generate new ID
    input_ids = torch.cat([input_ids, new_id.unsqueeze(0)], dim=1)
```

|i|input_ids|decoded text|next token|
|---|---|---|---|
|0|[464,23878,16599]|the elf queen|11|
|1|[464,23878,16599,11]|the elf queen,|508|
|2|[464,23878,16599,11,508]|the elf queen,who|550|

### 微调GPT-2 model

GPT models are trained in an unsupervised way on a large amount of text (or text corpus). The corpus is broken into sequences, usually of uniform size (e.g., 1024 tokens each). The model is trained to predict the next token (word) at each step of the sequence. For example (here, we write words instead of integer encodings for clarity) :

|position|1|2|3|4|5|6|7|8|9|
|---|---|---|---|---|---|---|---|---|---|
|input_ids|The|elf|queen|was|wearing|a|cloak|.|[END]|
|labels|elf|queen|was|wearing|a|cloak|.|[END]|[-1]

**The labels are identical to input_ids, but shifted to one position to the left**. Note that for GPT-2 in Hugging Face transformers this shift happens automatically when the loss is calculated, so from the user perspective, the tensor labels should be identical to input_ids.  PS：常规深度模型的训练输入是 `feature1,feature2,...,label`，LLM也是，不过label 有时是由input_id得到的。

There are two ways to train Hugging Face transformers models: with the Trainer class or with a standard PyTorch training loop. We start with Trainer. PS: 下面代码基于GPT-2 已有的参数微调GPT-2，**感觉模型微调 跟model = load_checkpoint(xx) 断点重训没啥区别**，侧重点在于讲Transformers库原理。

```python
class MyDset(torch.utils.data.Dataset):
    """A custom dataset that serves 1024-token blocks as input_ids == labels"""
    def __init__(self, data: list[list[int]]):
        self.data = []
        for d in data:
            input_ids = torch.tensor(d, dtype=torch.int64)
            attention_mask = torch.ones(len(d), dtype=torch.int64)
            self.data.append({'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': input_ids})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]
def break_text_to_pieces(text_path: str, tokenizer: transformers.PreTrainedTokenizer, block_len: int = 512) -> list[str]:
    """Read a file and convert it to tokenized blocks, edding <|endoftext|> to each block"""
    with open(text_path) as f:
        text = f.read()
    chunk_len0 = block_len - 1  # Leave space for a TOKEN_ENDOFTEXT
    tokens = tokenizer.encode(text) # 原文本直接弄，够粗暴
    blocks = []
    pos = 0
    while pos < len(tokens):
        chunk = tokens[pos: pos + chunk_len0]
        chunk.append(TOKEN_ENDOFTEXT)
        blocks.append(chunk)
        pos += chunk_len0

    if len(blocks[-1]) < block_len:
        del blocks[-1]

    return blocks
def train_val_split(data: list[str], ratio: float):
    n = len(data)
    assert n >= 2
    n_val = max(1, int(n * ratio))
    return data[n_val:], data[:n_val]
def prepare_dsets(text_path: str, tokenizer: transformers.PreTrainedTokenizer, block_len: int):
    """Read the text, prepare the datasets """
    data = break_text_to_pieces(text_path, tokenizer, block_len)
    data_train, data_val = train_val_split(data, 0.2)
    return MyDset(data_train), MyDset(data_val)

# Load model and tokenizer
model = transformers.GPT2LMHeadModel.from_pretrained(MODEL_NAME)
tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
training_args = transformers.TrainingArguments(output_dir="idiot_save/", learning_rate=1e-3,...)
# 传给trainer 的必须是预处理好的dataset（包含input_ids 等column）
trainer = transformers.Trainer(model=model,args=training_args,train_dataset=dset_train,eval_dataset=dset_val)
trainer.train()
# Save the model if needed
model.save_pretrained('./trained_model/')
tokenizer.save_pretrained('./trained_model/')
# Now our model is trained, try the generation
text = 'Natural language understanding comprises a wide range of diverse tasks'
batch = tokenizer([text], return_tensors='pt')
for k, v in batch.items():
    batch[k] = v.to(DEVICE)
out = model.generate(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], max_length=20)
print('GENERATION=', tokenizer.batch_decode(out.cpu()))
```

一把情况下 you are not allowed to train a model from scratch.  Neither are you allowed to fine-tune on a text corpus or fine-tune with additional heads. The only type of fine-tuning allowed is fine-tuning on prompt+completion pairs, represented in JSONL format, for example:

```
{"prompt":"banana is ","completion":"yellow"}
{"prompt":"orange is ","completion":"orange"}
{"prompt":"sky is ","completion":"blue"}
```
How exactly is GPT-3 trained on such examples? We are not exactly sure (OpenAI is very secretive), but perhaps the two sequences of tokens are concatenated together, then GPT-3 is trained on such examples, **but the loss is only calculated in the “completion” part**. PS: 终于知道为何要分成两段，而不是喂一个文本就算了。


