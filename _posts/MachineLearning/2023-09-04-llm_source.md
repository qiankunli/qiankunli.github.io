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
# output = model(**tokens) 对于文本分类来说，整段文本输入返回一个标签，也就是SequenceClassifierOutput.logits，那么对于文本生成来说，整段文本输入返回CausalLMOutputWithPast，它的logits 是自动补全的是 input_ids 的下一个token呢，还是直到eos的多个token呢？
```

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
2. attention_mask ，self-attention使用，shape和input_ids一致。模型训练和预测基本都是批量化处理的，处理多个序列时，在attention的时候，不去attend被mask掉的部分。 作用是告诉模型一个batch的数据里哪些是padding的，从而可以忽略掉那些padding的部分，比如下图的例子：
    ![](/public/upload/machine/attention_mask.jpg)

PS: 深度学习都得指定features/labels。在llm 场景下，features 和labels 有几个特点
1. llm 有base model、sft model 等，不同的model 数据集格式不同，一般分为几个部分，比如sft 的`{"question:":"xx","answer":"xx"}`，各家模型都不太一样，很多数据集是不公开的。但不管如何，这几部分都会拼为一个sentence（中间可能有一些特殊字符起到连接作用），然后把sentence通过tokenizer转换成input_ids，之后再走embedding 模块等等就是Transformer系列模型内的事儿了，最后得到output_ids.
2. 模型输入格式，模型输入dict 一般包含3个key： input_ids,attention_mask,labels
    1. 有些模型内置从input ids 提取attention mask的操作
    2. 预训练场景 labels 一般由input_ids copy而来，然后做一些处理，比如labels 全部左移一位（预训练）
    3. 明确指定labels 的话，一般是要微调，比如sft时，sentence部分中question 的位置都置为-100，-100表示在计算loss的时候会被忽略，这个由任务性质决定。
2. 预处理（将dataset 转为模型输入）过程由​ Dataset.map() + tokennizer 来办。
    ```python
        def tokenize_function(example):
            # example 表示数据集中的一行数据
            return tokenizer(example["sentence1"], example["sentence2"], truncation=True)
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
    ```
3. 之后就是对output_ids 和 labels 计算loss。
3. 上述过程也是Transformers 库抽象的基础，指定input_ids,labels，则计算output_ids 和 loss 可以自动进行。对于一个base llm，可以基于finetune做很多task specific llm模型，主要体现在 input 数据集格式 和labels 的不同。




### generate实现

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
    input_ids = torch.cat([input_ids, new_id.unsqueeze(0)], dim=1)  # input_ids 加入新生成的字符
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

