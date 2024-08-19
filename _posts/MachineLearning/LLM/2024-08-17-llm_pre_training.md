---

layout: post
title: LLM微调理论
category: 技术
tags: MachineLearning
keywords: llm finetune

---

* TOC
{:toc}


## 简介


[LLama3 405B 技术解读](https://mp.weixin.qq.com/s/51h70Zg-bfbvnQWr6UH1ZQ)大模型之所以能力仍在快速提升，主要驱动力有三个：
1. 首先就是不断扩大模型和数据规模（Scaling Law）。
2. 一个是越来越强调数据质量的作用，各种数据筛选方法和工具越来越多，保证质量是第一位的
3. 不断增加数学、逻辑、代码这种能够提升大模型理性能力的数据配比比例，包括在预训练阶段（增加预训练数据此类数据比例，且在预训练后面阶段来上采样此类数据，就是说**同样数据多执行几遍，以增加其对模型参数影响的权重**）和Post-Training阶段（增加此类数据占比，Llama3的经过instruct的模型比仅做预训练模型相比，各种尺寸的效果提升都很大）皆是如此。
目前看，在通用数据快被用完情况下，第三个因素会成为之后大模型进步的主导力量，包括使用数学、逻辑、代码合成数据在Post-Training阶段的应用，目前技术也越来越成熟，其质量和数量会是决定未来大模型效果差异的最关键因素。PS：合成数据其实是模型蒸馏的一种变体，合成数据是更大的模型输出数据作为Teacher，小点的模型作为Student从中学习知识，所以其实本质上是一种模型蒸馏。


## 继续预训练

1. 混合数据，如果想要领域的模型还具备一定的通用能力，即通用的能力不会退化（或者灾难性遗忘）这就需要在语言模型训练的时候混杂通用的数据。
2. 要不要从零训。回顾人对知识的理解：小学中学都在学习通用领域的知识，然后大学阶段继续进一步学习特定领域的知识。所以在通用模型的基础上继续二次预训练注入领域知识是合理的。但是如果想通过二次预训练进行语言层面的迁移就会比较难，没有从零开始训练好。回顾人对语言的学习，如果刚“出生”时候就在学习一门语言，进行听说读写的训练，这就是母语了。会比长大以后再去学习一门外语要容易的多，效果也要好很多。所以基于llama做的中文适配 不如 纯中文训练的baichuan 在中文任务上效果好。

[浅谈-领域模型训练](https://mp.weixin.qq.com/s/qZ97QM0qV-vfWYQ0KGG6UQ) 提到了很多pre-training 和post-training 的why/trick。
pretrain 最重要的几个东西：数据，学习率，优化器！
1. 数据就不多说了，质量为王，记得去重！
2. 学习率：模型的更新幅度，size越大的模型，特征空间越大、表达能力和学习能力越强，因此学习率也应该小一点（做个假设，模型 size 无限大，有无数的神经元，那么它完全可以启用没用到的神经元来学习新知识，这样就避免了遗忘旧知识这个现象的发生）。
3. 优化器：Adam 的基础知识我就不谈了，这里只强调一点，模型的优化方向是“历史动量”和“当前数据 grad”共同决定的。也就是说，不管当前数据多 bad，优化器都会限制你做出太大幅度的更新，梯度裁剪/梯度正则类似。因此，基本可以认为我们的模型具有一定的抗噪能力。

目前，大家基本都默认使用如下三个步骤进行 pretrain：
1. warmup：在训练过程中，将学习率慢慢提高。（可以这么理解，你的模型还没有积攒足够的动量去抗噪，太大的学习率容易造成不可逆的影响）
2. linear / constant / cosine decay：维持稳定的学习率，或者缓慢衰减的学习率。
3. Anneal：用小学习率去学高精数据，IFT数据，逻辑数据，去提高通用逻辑能力能力和打榜能力。

### 同源小模型是大模型的实验场

[大模型 VS 小模型](https://mp.weixin.qq.com/s/QLq64i3VSWTO6vzeVnr3mQ)scaling law 告诉我们：小模型的性能表现能用来预测大模型的性能表现。这也就是说，大部分情况下，我们是可以通过在同源小模型上做实验，去预测大模型的效果的。

在 pretrain / post_pretrain 阶段有很多需要做实验才能知道答案的问题。怎么样的数据配比最合理，课程学习中哪种学习顺序效果最好，数据的质量是否过关，数据的去重程度是否过关，先训4k、再扩到 32k 和直接训 32k 的效果差异，post_pretrain 的时候怎样调整学习率和数据分布来防止模型断崖式的能力遗忘？

直接启动大模型的成本实在是在太高昂了，可能训练两三周，loss 曲线才会表现出一点点差异。但我们完全可以在小模型上大胆的训，每天训 100B token，两天就能出一版实验结果。观察 tensorbord 的 loss 曲线，刷 benchmark 打榜，或是做 sft 看效果，总之小模型可以帮助我们快速地敲定 pretrain 阶段使用的数据配置。

在 alignment 阶段，我们也可以去借助小模型和 scaling law 来指导工作。我要强化模型的某个能力，准备了 N 条训练数据，能让模型达到多大的提升呢？可以看看这份数据在小模型上能有大提升，绘制一条曲线，去预估大模型的性能表现。说的再通俗一点，100B token 能让 0.5B 模型下降 0.2 loss，能让 72B 模型下降 0.1 loss， alignment 数据能让 0.5B 模型提高 x% 的 task 能力，那么大概率这份数据也只能让 72B 模型提升 0.5x % 的 task 能力。

### 大模型背后的无数小模型

一个优秀的大模型，无论是在训练阶段，还是线上部署阶段，其背后默默付出的小模型都数不胜数。
1. 数据质量分类器：llama3 和 qwen2 都提到了，他们的 pretrain 训练数据是有得分的，然后通过阈值来找出最高质量的训练数据，开源 pretrain 数据集 fineweb 也提到了他们给数据打分的工作。Good data makes good model performance！李沐大佬在他的视频里说到，llama3 的数据打分器是 RoBERTa，这很合理，效果又好、推理又快的分类模型确实还要看 BERT 家族。
2. 数据 domain 分类器：垂直领域模型的 post_pretrain 工作，往往需要非常精准的数据配比，domain 数据的数据质量也需要非常优质。这也就是说，我们需要一个分类器，去提取海量数据中的 domain 数据，这个分类器最好还能把低质量的 domain 数据也视为非 domain 数据，通常承担这个工作的模型也是 BERT 家族。

## GPT-2养成记 

[Training and Fine-Tuning GPT-2 and GPT-3 Models Using Hugging Face Transformers and OpenAI API](https://www.it-jim.com/blog/training-and-fine-tuning-gpt-2-and-gpt-3-models-using-hugging-face-transformers-and-openai-api/)  非常经典，入门必读。
1.  it does not implement neural networks from scratch(从头开始) but relies on lower-level frameworks PyTorch, TensorFlow, and FLAX. 
2. it heavily uses Hugging Face Hub, another Hugging Face project, a hub for downloadable neural networks for various frameworks. 
3. Model is a valid PyTorch model with some additional restrictions and naming conventions introduced by the transformers framework. 
4. Neural networks are not able to work with raw text; they only understand numbers. We need a tokenizer to convert a text string into a list of numbers. But first, it breaks the string up into individual tokens, which most often means “words”, although some models can use word parts or even individual characters. Tokenization is a classical natural language processing task. Once the text is broken into tokens, each token is replaced by an integer number called encoding from a fixed dictionary. Note that a tokenizer, and especially its dictionary, is model-dependent: you cannot use Bert tokenizer with GPT-2, at least not unless you train the model from scratch. Some models, especially of the Bert family, like to use special tokens, such as `[PAD]`,`[CLS]`, `[SEP]`, etc. GPT-2, in contrast, uses them very sparingly.

![](/public/upload/machine/gpt_architecture.jpg)

different GPT versions differ pretty much only in size, minor details, and the dataset+training regime. If you understand how GPT-2 or even GPT-1 works, you can, to a large extent, understand GPT-4 also. PS： 不同的gpt从模型结构上差别不大。

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
# -1 在python list 里表示最后一个元素。
new_id = logits[:, -1, :].argmax(dim=1)
print(new_id)
print(tokenizer.batch_decode(new_id))
```

[GPT2 源码解析](https://zhuanlan.zhihu.com/p/630970209) 建议细读

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

GPT models are trained in an unsupervised way on a large amount of text (or text corpus). The corpus is broken into sequences, usually of uniform size (e.g., 1024 tokens each). PS： 预训练素材通常被切成特定长度的句子。The model is trained to predict the next token (word) at each step of the sequence. For example (here, we write words instead of integer encodings for clarity) :

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
How exactly is GPT-3 trained on such examples? We are not exactly sure (OpenAI is very secretive), but perhaps the two sequences of tokens are concatenated together, then GPT-3 is trained on such examples, **but the loss is only calculated in the “completion” part**. PS: 终于知道为何要分成两段，而不是喂一个文本就算了。labels 中prompt部分的位置都置为-100，-100表示在计算loss的时候会被忽略，这个由任务性质决定。

