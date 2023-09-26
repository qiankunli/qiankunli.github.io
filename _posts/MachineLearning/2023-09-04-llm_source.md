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

[Transformers快速入门](https://transformers.run/)Hugging Face 专门为使用 Transformer 模型编写了一个 Transformers 库，建立在 Pytorch 框架之上（Tensorflow 的版本功能并不完善），所有 Transformer 模型都可以在 Hugging Face Hub 中找到并且加载使用，包括训练、推理、量化等。

transformers开源库的核心组件包括3个：Conﬁguration、Tokenizer、Model
1. 「Conﬁguration」：配置类，通常继承自「PretrainedConﬁg」，保存model或tokenizer的超参数，例如词典大小，隐层维度数，dropout rate等。配置类主要可用于复现模型。
2. 「Tokenizer」：切词类，通常继承自「PreTrainedTokenizer」，主要存储词典，token到index映射关系等。此外，还会有一些model-specific的特性，如特殊token，[SEP], [CLS]等的处理，token的type类型处理，语句最大长度等，因此tokenizer通常和模型是一对一适配的。
3. 「Model」: 模型类。封装了预训练模型的计算图过程，遵循着相同的范式，如根据token ids进行embedding matrix映射，紧接着多个self-attention层做编码，最后一层task-specific做预测。
针对上述三大类，transformer还额外封装了AutoConfig, AutoTokenizer,AutoModel，可通过模型的命名来定位其所属的具体类，比如’bert-base-cased’，就可以知道要加载BERT模型相关的配置、切词器和模型。

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

开箱即用的 pipelines，Transformers 库最基础的对象就是 `pipeline()` 函数，它封装了预训练模型和对应的前处理和后处理环节。
1. 预处理 (preprocessing)，将原始文本转换为模型可以接受的输入格式；具体地，我们会使用每个模型对应的分词器 (tokenizer) 来进行：
  1. 将输入切分为词语、子词或者符号（例如标点符号），统称为 tokens；
  2. 根据模型的词表将每个 token 映射到对应的 token 编号（就是一个数字）；
  3. 根据模型的需要，添加一些额外的输入。
2. 将处理好的输入送入模型；
3. 对模型的输出进行后处理 (postprocessing)，将其转换为人类方便阅读的格式。

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
```

PS: 深度学习都得指定features/labels。在llm 场景下，features 和labels 有几个特点
1. llm 有base model、sft model 等，不同的model 数据集格式不同，一般分为几个部分，比如`{"question:":"xx","answer":"xx"}`，但不管如何，这几部分都会拼为一个sentence（中间可能有一些特殊字符起到连接作用），然后把sentence通过tokenizer转换成input_ids，之后再走embedding 模块等等就是Transformer系列模型内的事儿了，最后得到output_ids.
2. labels 一般由input_ids copy而来，然后做一些处理，比如labels 全部左移一位（预训练），也或者将sentence部分中question 的位置都置为-100，-100表示在计算loss的时候会被忽略（sft），这个由任务性质决定。
3. 之后就是对output_ids 和 labels 计算loss。
3. 上述过程也是Transformers 库抽象的基础，指定input_ids,labels，则计算output_ids 和 loss 可以自动进行。对于一个base llm，可以基于finetune做很多task specific llm模型，主要体现在 input 数据集格式 和labels 的不同。

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

**The labels are identical to input_ids, but shifted to one position to the left**. Note that for GPT-2 in Hugging Face transformers this shift happens automatically when the loss is calculated, so from the user perspective, the tensor labels should be identical to input_ids.  PS：常规深度模型的训练输入是 `feature1,feature2,...,label`，LLM也是，不过关系更隐式一些。

There are two ways to train Hugging Face transformers models: with the Trainer class or with a standard PyTorch training loop. We start with Trainer. PS: 下面代码基于GPT-2 已有的参数微调GPT-2，侧重点在于讲Transformers库原理。

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

## LoRA

[从0到1！得物如何打造通用大模型训练和推理平台](https://mp.weixin.qq.com/s/5EE1VXxq7k_VoC9gRPvyMg)

对于大语音模型来说，其参数量非常多。GPT3有1750亿参数，而且LLAMA系列模型包括 7B,13B,33B,65B，而其中最小的7B都有70亿参数。要让这些模型去适应特定的业务场景，需要对他们进行微调。如果直接对这些模型进行微调，由于参数量巨大，需要的GPU成本就会非常高。LoRA的做法是对这些预训练好的大模型参数进行冻结，也就是在微调训练的时候，这些模型的参数设置为不可训练。然后往模型中加入额外的网络层，并只训练这些新增的网络层参数。这样可训练的参数就会变的非常少，可以以低成本的GPU微调大语言模型。LoRA在Transformer架构的每一层注入可训练的秩分解矩阵，与使用Adam微调的GPT-3 175B相比，LoRA可以将可训练参数数量减少10000倍，GPU内存需求减少3倍，并且在效果上相比于传统微调技术表现的相当或更好。当前已经得到HuggingFace 的 PEFT库 https://github.com/huggingface/peft 的支持。


### 原理

LoRA，LoRA背后有一个假设：我们现在看到的这些大语言模型，它们都是被过度参数化的。而过度参数化的大模型背后，都有一个低维的本质模型。通俗讲人话：大模型参数很多，但并不是所有的参数都是发挥同样作用的；大模型中有其中一部分参数，是非常重要的，是影响大模型生成结果的关键参数，这部分关键参数就是上面提到的低维的本质模型。LoRA的基本思路，在原始预训练模型旁边增加一个旁路，先用一个 Linear 层 A，将数据从 d 维降到 r 维，在用第二个 Linear 层 B，将数据从 r 维变回 d 维。LoRA 训练的时候固定预训练模型的参数，只训练降维矩阵 A 和升维矩阵 B。包括以下几步：
1. 要适配特定的下游任务，要训练一个特定的模型，将Y=WX变成$Y=(W+∆W)X$，这里面$∆W$主是我们要微调得到的结果；
2. 将∆W进行低维分解`∆W=AB` (`∆W`为m * n维，A为m * r维，B为r * n维，r就是上述假设中的低维)；
3. 接下来，用特定的训练数据，训练出A和B即可得到∆W，在推理的过程中直接将∆W加到W上去，再没有额外的成本。另外，如果要用LoRA适配不同的场景，切换也非常方便，做简单的矩阵加法即可：$(W + ∆W) - ∆W + ∆W`$。

![](/public/upload/machine/lora.jpg)

[深入浅出剖析 LoRA 技术原理](https://mp.weixin.qq.com/s/jk1qBRjiq80nK0e04LQqiw) 未细读。

### 实现

许多朋友在使用LoRA的过程中，都会用到HuggingFace Peft库封装好的LoRA接口，这个接口是对微软版LoRA代码的改写和封装，目的是减少大家在使用LoRA过程中的手工活（例如徒手更改模型架构，为模型添加LoRA adapter结构等），除此外核心处理逻辑不变。

[LoRA](https://huggingface.co/docs/peft/conceptual_guides/lora)As with other methods supported by PEFT, to fine-tune a model using LoRA, you need to:
1. Instantiate a base model.
2. Create a configuration (LoraConfig) where you define LoRA-specific parameters.
3. Wrap the base model with get_peft_model() to get a trainable PeftModel.
4. Train the PeftModel as you normally would train the base model.


```python
from peft import LoraConfig, get_peft_model
# # 创建基础transformer模型
model = xx.from_pretrained(args.model_dir)
config = LoraConfig(r=args.lora_r,lora_alpha=32,...)
model = get_peft_model(model, config)     # 初始化Lora模型
model = model.half().cuda()   

# 设置DeepSpeed配置参数，并进行DeepSpeed初始化
conf = {"train_micro_batch_size_per_gpu": args.train_batch_size,...)
model_engine, optimizer, _, _ = deepspeed.initialize(config=conf,model=model,model_parameters=model.parameters())
model_engine.train()                                                     
train_dataset = Seq2SeqDataSet(args.train_path, tokenizer,...)
train_dataloader = DataLoader(train_dataset,batch_size=...)
# 开始模型训练   
for i_epoch in range(args.num_train_epochs):         
    train_iter = iter(train_dataloader)    
    for step, batch in enumerate(train_iter):
        # 获取训练结果
        input_ids = batch["input_ids"].cuda()
        labels = batch["labels"].cuda()     
        outputs = model_engine.forward(input_ids=input_ids, labels=labels)     
        loss = outputs[0]    
        # 损失进行回传 
        model_engine.backward(loss)  
        model_engine.step() # 进行参数优化
    # 每一轮模型训练结束，进行模型保存  
    save_dir = os.path.join(args.output_dir, f"global_step-{global_step}")  
    model_engine.save_pretrained(save_dir)                           
```

```python
def get_peft_model(model: PreTrainedModel, peft_config: PeftConfig, adapter_name: str = "default") -> PeftModel:
    model_config = getattr(model, "config", {"model_type": "custom"})
    peft_config.base_model_name_or_path = model.__dict__.get("name_or_path", None)
    if peft_config.task_type not in MODEL_TYPE_TO_PEFT_MODEL_MAPPING.keys() and not peft_config.is_prompt_learning:
        return PeftModel(model, peft_config, adapter_name=adapter_name)
    if peft_config.is_prompt_learning:
        peft_config = _prepare_prompt_learning_config(peft_config, model_config)
    return MODEL_TYPE_TO_PEFT_MODEL_MAPPING[peft_config.task_type](model, peft_config, adapter_name=adapter_name)
```

（微软的）LoraModel是LoRA模块的核心类，冻结base model的参数，旁路低秩矩阵的创建，替换，合并等逻辑都在这个类中。

```python
# LoraModel也是继承torch.nn.Module，相当于pytorch的一个网络模块
class LoraModel(torch.nn.Module):
    def __init__(self, model, config, adapter_name) -> None:
        super().__init__()
        self.model = model  # # model 被用来微调的基础大模型
        self.forward = self.model.forward   # oraModel把自己的前向传播函数forword设置为大模型的forward方法
        self.peft_config = config
        self.add_adapter(adapter_name,self.peft_config[adapter_name])
    def _find_and_replace(self,adapter_name):
        # 使用新的LoraLayer替换target_modules中配置的Layer，实现添加旁路低秩矩阵的功能。
        ...
    def mark_only_lora_as_trainable(model: nn.Module, bias:str="none") -> None:
        for n,p in model.named_parameters():
            if "lora_" not in n:
                p.requires_grad = False # 除了新增的LoraLayer的模块外，其他所有参数都被冻结。
        ...
    def forward(self, x:torch.Tensor):
        ...
        result = F.linear(x,...,bias=self.bias) # 使用大模型target_module中线性层进行计算，得出结果result。
        result += ( # 使用lora_A与lora_B的低秩矩阵进行计算并把计算结果加到result上。
            self.lora_B[self.active_adapter](
                self.lora_A[self.active_adapter](self.lora_dropout[self.active_adapter](x))
            )
            * self.scaling(self.active_adapter)
        )
```
可以看到
1. Freeze参数的本质：requires_grad = False 

[图解大模型微调系列之：大模型低秩适配器LoRA（源码解读与实操篇）](https://mp.weixin.qq.com/s/RuV_4lCQentq4kBr3fv08A) 未读。

[深入浅出剖析 LoRA 源码及实践](https://mp.weixin.qq.com/s/QdX1R0GPe6dvEDXjdJj5lA) 未读。