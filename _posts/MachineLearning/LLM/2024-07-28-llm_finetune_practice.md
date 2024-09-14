---

layout: post
title: LLM微调实践
category: 技术
tags: MachineLearning
keywords: llm finetune

---

* TOC
{:toc}

## 简介

微调模型的主要原因在于，当模型的推理结果未能达到预期时，简单地通过单样本或少样本推理来调整模型输出效果并不总是奏效，特别是对于较小的LLM（大语言模型）。通过在提示中添加一个或多个已完成的示例来优化输出，但这种方法有其局限性。首先，提示中包含的示例会占用上下文窗口的空间，导致可用来包含其他有用信息的空间减少。其次，这些策略在实际应用中并不总是能够解决问题。与预训练阶段不同，微调是一个监督学习过程，它使用标记好的示例数据集来更新模型的权重，从而使模型更好地完成特定任务。微调的核心价值所在：在优化模型输出的同时，最大化利用有限的上下文窗口空间，提升模型的实际应用效果。

用一个更牛逼的模型准备数据
1. identity tasks by prompt-engineering a large llm
2. find tasks that you see an llm doing ~ok at 
3. pick one task
4. Get ~1000 inputs and outpus for the task Better than the ~ok from the llm
5. finetune a small llm on this data

steps to prepare your data:
1. collect instruction-response pairs
2. concatenate pairs(add prompt template,if applicable)
3. tokenize: pad, truncate
    1. Tokenizer.encode 文本转数字，Tokenizer.decode 数字转文本。
    2. batch input时，`Tokenizer.encode(texts,padding=True)` 文本转数字，不同text通过padding 对齐。
    3. 如果text 超过了max_length还可以使用截断`Tokenizer.encode(texts,max_length=xx,truncation=True)`。
4. split into train/test
    1. datasets.train_test_split

## 多轮对话怎么转化为模型接受的input和用于计算loss的label

### 预训练

通常，我们把经过预训练（pretrain）阶段得到的模型称为base模型。这个阶段主流的数据组织方式叫packing。在不采用packing的时候，为了将不同长度的句子组成一个batch tensor，我们需要进行填充（pad），这个填充过程既可以按照batch内最长句子填充，也可以按照模型最长输入长度填充。为了防止一个batch内存在许多的`<pad>token`，浪费计算资源，packing直接采取多条示例的拼接方法。下图是传统方法和packing的对比：

![](/public/upload/machine/pretrain_padding.jpg)

左侧是传统的padding做法，右侧是packing，其中红色部分代表pad token，黄色部分代表sep token。为了区分不同的训练示例，我们在不同示例之间加上一个分割标记sep token，注意力窗口不会跨示例。这个注意力模式叫块对角矩阵（BlockDiagonalMask）【本质上是在示例内的下三角矩阵】，而不是传统的全局下三角矩阵。由此，就消除了对pad token的需要，所以开源大模型刚问世的时候（2023-3那阵子），存在很多base model放出来的tokenizer并没有pad token，比如llama-base。需要注意，packing时示例3可能会被截断，这个行为在预训练时是可以接受的。注意，这个时候的**学习模式**非常的简单，就是next token prediction。

数据组成形式：
```
输入input： <bos> X1 X2 X3
标签labels：X1 X2 X3 </s>
```
典型的Decoder架构的数据训练方式；

### 指令微调

指令微调不仅仅考虑了对人类指令和多任务的适应性，更是希望能将角色系统融入大模型中，从而让大模型变成chat模型，指令微调并不直接产生chat model，只是其中必不可少的一步。其中比较特殊的数据形式就是多轮对话。对话里必不可少的存在“角色”这个概念，因为和大模型的对话仅限于用户和模型，所以极大多数的对话模板（template）里都只考虑了两个角色——user和assistant。注意，对话模板只有非base模型才需要，所以很多的base模型的tokenizer里并不携带chat_template。

举个例子，比如：LLAMA2-chat的对话模板中user标识 是 `[INST]` ， assistant 标识是`[/INST]`，下面是一个单轮的例子
```
chat_dict = [
                {"role": "user", "content": 你好},
                {"role": "assistant", "content": 你也好},
            ]
```
因为模型输入只能是非结构化的，我们`利用模板将其非结构化`。得到的字符串就是`[INST]你好[/INST]你也好`。那么假如我们现在获得了一个现成的训练数据
```
chat_dict = [
                {"role": "user", "content": U1},
                {"role": "assistant", "content": A1},
                {"role": "user", "content": U2},
                {"role": "assistant", "content": A2},
            ]
```
模型的input_ids和对应的labels应该是什么呢（input_ids 对应模型输入，labels 则是为了和模型oupput标量计算loss）？最常规的做法应该是在每一轮首尾用`[BOS]`和`[EOS]`包裹，轮次内部正常用模板非结构化就行。上例可以转换为input_ids= `[BOS][INST]U1[\INST]A1[EOS][BOS][INST]U2[\INST]A2[EOS]`，难点在于LABELS应该是什么呢？ 我们可以根据学习模式来确定LABELS。
1. 在推理场景下，假如是第一轮对话开始，我们会输入给模型[BOS][INST]U1[\INST]，那么我们希望模型吐出的是什么呢？是A1和[EOS]，A1是模型自己的回答，EOS是为了告诉解码系统生成结束了，否则模型将一直生成到最大长度才会停止。我们获得了一个初步的学习模式需求，就是根据`[BOS][INST]U[\INST] → A[EOS]`。
    |input|`[BOS]`|`[INST]`|U|`[/INST]`|A|
    |---|---|---|---|---|---|
    |label|-100|-100|-100|A|`[EOS]`|

    在多轮的场景下，也只是复制这个过程。

    |input|`[BOS]`|`[INST]`|U|`[/INST]`|A|`[EOS]`|`[BOS]`|`[INST]`|U2|`[/INST]`|A2|
    |---|---|---|---|---|---|---|---|---|---|---|---|
    |label|-100|-100|-100|A|`[EOS]`| X| X| X| X| A2| `[EOS]`|

2. 在llama factory里有一个参数叫 Efficient EOS，所谓efficient eos并不代表是一个新的token，而是一个特殊的input和label的设计方式。 [多轮对话的训练过程详解](https://zhuanlan.zhihu.com/p/695202364)

不管是PreTraining阶段还是SFT阶段，loss函数都是一样的，只是计算的方式存在差异，PreTraining阶段计算的是整段输入文本的loss，而SFT阶段计算的是response部分的loss。此外，LLM在推理也就是generate的时候，是要不断调用forward的。**训练的时候一句话abcd变成a->b, ab->c,abc->d三个样本，forward只需要调用一次**（一句话的多个样本构成一个batch）。且训练的时候不生成新的token（自然也就没有解码策略那一堆事），因为输入已经是prompt+response了，是根据整个序列的各个位置的最后的logits（推理时需要根据logits再sample 得到token）和response真实值做loss计算。

## low-level 微调代码

### 手写

自己写正向传播、反向传播、更新权重。
1. 加载数据集，pytorch Dataset/DataLoader
2. 构建模型，在实际操作中，除了使用预训练模型编码文本外，我们通常还会进行许多自定义操作，因此在大部分情况下我们都需要自己编写模型，不过不用从0写，更为常见的写法是继承 Transformers 库中的预训练模型来创建自己的模型。
    ```python
    class BertForPairwiseCLS(BertPreTrainedModel):     # 继承 BERT 模型（BertPreTrainedModel 类）
        def __init__(self, config):
            super().__init__(config)
            self.bert = BertModel(config, add_pooling_layer=False)
            self.dropout = nn.Dropout(config.hidden_dropout_prob)
            self.classifier = nn.Linear(768, 2)
            self.post_init()
        
        def forward(self, x):
            bert_output = self.bert(**x)
            cls_vectors = bert_output.last_hidden_state[:, 0, :]
            cls_vectors = self.dropout(cls_vectors)
            logits = self.classifier(cls_vectors)
            return logits
    config = AutoConfig.from_pretrained(checkpoint) # 通过预置的 from_pretrained 函数来加载模型参数
    model = BertForPairwiseCLS.from_pretrained(checkpoint, config=config).to(device) # 加载 预置模型
    print(model)
    # Transformers 库同样实现了很多的优化器，相比 Pytorch 固定学习率，Transformers 库的优化器会随着训练过程逐步减小学习率（通常会产生更好的效果）
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    def train_loop(dataloader, model, loss_fn, optimizer,...): ...
    def test_loop(dataloader, model, mode='Test'): ...
    for t in range(epoch_num):
        total_loss = train_loop(train_dataloader, model, loss_fn, optimizer, lr_scheduler, t+1, total_loss)
        valid_acc = test_loop(valid_dataloader, model, mode='Valid')
        if valid_acc > best_acc:
            best_acc = valid_acc
            print('saving new weights...\n')
            torch.save(model.state_dict(), ...) #     # 保存模型
    ```

### 使用huggingface的Trainer API进行模型微调

transformer 支持自定义dataset，自定义model实现forward（forward 支持的参数均可以作为dataset的column），forward 过程中还计算loss，模型的差异性基本已经兜住了，这也是为何 只要提供包含特定column的dataset，剩下的训练代码都可以交给trainer封装掉。

在我们定义 Trainer 之前首先要定义一个 TrainingArguments 类，它将包含 Trainer用于训练和评估的所有超参数，也内置了Accelerate和deepspeed等支持。唯一必须提供的参数是保存训练模型的目录，以及训练过程中的检查点。对于其余的参数，可以保留默认值。

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


[Fine-tuning a model with the Trainer API](https://huggingface.co/learn/nlp-course/chapter3/3)Transformers provides a Trainer class to help you fine-tune any of the pretrained models it provides on your dataset. Once you’ve done all the data preprocessing work in the last section, you have just a few steps left to define the Trainer. The hardest part is likely to be preparing the environment to run `Trainer.train()`

```python
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

from transformers import TrainingArguments
training_args = TrainingArguments("test-trainer")

from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
from transformers import Trainer
trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)
# To fine-tune the model on our dataset, we just have to call the train() method of our Trainer
trainer.train()
```

[A full training](https://huggingface.co/learn/nlp-course/chapter3/4)  手写train loop。

## high-level 框架

huggingface transformer库虽然已经支持的很全了，但代码量还是很大，所以出现一批框架比如Llama。
```python
from llama import BasicModelRunner
model = BasicModelRunner("aaa/bbb")
model.load_data_from_jsonlines("xx.jsonl")
model.train()
```
随着时间的推移，采用了越来越高级的接口，训练的代码已经大大简化。

[四个大模型轻量级微调训练框架：兼看PPT转Markdown工具](https://mp.weixin.qq.com/s/1Wjap8kiNGXkCQQ35pJE7g) 建议把这几个框架Firefly/LLaMA-Factory/unsloth/SWIFT都看下，找找共性。 

### LLaMA-Factory

[使用医患对话数据训练新冠诊疗模型的例子](https://github.com/hiyouga/ChatGLM-Efficient-Tuning/blob/main/examples/covid_doctor.md)

```
LLaMA-Factory
    /src
        /llmtuner
            /train
                /data 
                    /loader.py     # get_dataset
                    /preprocess.py # preprocess_dataset
                /model
                    /loader.py     # load_model_and_tokenizer
                /dpo
                    /trainer.py     # 一些trainer 用到的函数
                    /workflow.py    # run_dpo
                /ppo
                    /trainer.py     # 一些trainer 用到的函数
                    /workflow.py    # run_ppo
                /pt
                    /trainer.py     # 一些trainer 用到的函数
                    /workflow.py    # run_pt
                /rm
                    /trainer.py     # 一些trainer 用到的函数
                    /workflow.py    # run_rm
                /sft
                    /trainer.py     # 一些trainer 用到的函数
                    /workflow.py    # run_sft
```

workflow.py 的逻辑言简意赅，就是拼凑运行 Trainer的dataset、model、tokenizer、data_collator等参数
1. 对于dataset 有一个load_dataset 和preprocess_dataset 的过程，preprocess_dataset 会根据任务目标不同，处理逻辑不同，也就是将数据转为input_ids 的方式不同。 最终转为trainer 也就是transformer model 可以接受的dataset，包含列 input_ids/attention_task/labels（或其它model.forward 可以支持的参数）。 
2. Trainer 对训练逻辑已经封的很好了，内部也支持了accelerate 和 deepspeed，只要合适的配置 training_args 即可。

以pt对应的workflow.py 为例

```python
def run_pt(model_args: "ModelArguments",data_args: "DataArguments",training_args: "Seq2SeqTrainingArguments",finetuning_args: "FinetuningArguments",callbacks: Optional[List["TrainerCallback"]] = None):
    dataset = get_dataset(model_args, data_args)
    model, tokenizer = load_model_and_tokenizer(model_args, finetuning_args, training_args.do_train, stage="pt")
    dataset = preprocess_dataset(dataset, tokenizer, data_args, training_args, stage="pt")
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(model,training_args,tokenizer,data_collator,callbacks,**split_dataset(dataset, data_args, training_args)
    # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        trainer.save_model()
        if trainer.is_world_process_zero() and model_args.plot_loss:
            plot_loss(training_args.output_dir, keys=["loss", "eval_loss"])
     # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval")
        perplexity = math.exp(metrics["eval_loss"])
        metrics["perplexity"] = perplexity
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
```

不管是PreTraining阶段还是SFT阶段，loss函数都是一样的，只是计算的方式存在差异，PreTraining阶段计算的是整段输入文本的loss，而SFT阶段计算的是response部分的loss。
1. preprocess_pretrain_dataset处理PreTraining阶段的数据，数据组成形式：
    1. 输入input： `<bos> X1 X2 X3`
    2. 标签labels：`X1 X2 X3 </s>`
    典型的Decoder架构的数据训练方式；
2. preprocess_supervised_dataset处理SFT阶段的数据，数据组成形式：
    1. 输入input：`<bos> prompt response`
    2. 标签labels： `-100 ... -100 response </s>`
对于prompt部分的labels被-100所填充，这样在计算loss的时候模型只计算response部分的loss，-100的部分被忽略了。这个机制得益于torch的CrossEntropyLossignore_index参数，ignore_index参数定义为如果labels中包含了指定了需要忽略的类别号（默认是-100），那么在计算loss的时候就不会计算该部分的loss也就对梯度的更新不起作用。

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

### 页面操作 

[基于LLaMA-Factory框架对Qwen2-7B模型进行微调实践](https://mp.weixin.qq.com/s/EIwmK5tHw3n7EGXkKhg4YQ)

## 微调实践

1. 合适的损失曲线：训练过程中可能会有波动，但整体是呈现一个下降的趋势，一般模型微调后，最终的训练集损失降低到1以下是合适的，特别低可能会过拟合，但一般使用特定领域知识做微调的话，建议往先过拟合靠拢，优先保证模型能精准回答知识库中的内容。
2. 无规律波动且不收敛：一般是learningRate太大导致，无法呈现稳定的下降趋势，如果训练中出现这个趋势持续到3~5轮，可以提前关闭训练任务，尝试将学习率learningRate降低一个数量级。
[LLM is not all you need（大模型领域知识微调实践）](https://zhuanlan.zhihu.com/p/689800667)
3. 为什么要用SFT （指令微调）而不用RLHF？RLHF的loss主要在优化LLM输出特定分布（风格）的内容，而笔者想要的是让LLM看到某个问题输出非常精确的答案。那能不能用RL来做基于标签反馈的优化呢？这里边有一个很大的问题是目标token序列非常少，这涉及到稀疏空间中的RL优化问题，比较棘手，不如用SFT来得直接有效，而且SFT对于格式输出的学习几乎是无可挑剔的。
4. 指令微调是全参数微调还是PEFT？首先说PEFT吧，有Lora，P-tuning等很多种。笔者一直用Lora，虽然这些方法各有千秋，但当任务很难时性能差异不是很大。其次说全参数微调，笔者用PEFT微调发现效果比较差之后就采用了全参数微调，但是提升不是很显著（需要的是20+或者30+个点的提升，而不是个位数的提升）。
5. 能不能将领域知识以continue training的形式注入LLM？尝试过，几乎没有提升。主要是用来continue training的数据不是非常多。所以对SFT没有加成。
6. 在效率和资源都达标和到位的情况上，优先用大 size 的模型进行实验和微调，因为大 size 的模型在容错性上比小 size 的好太多。尽管大尺寸模型也可能存在多任务不稳定、标签不平衡等问题，但其表现通常会比小尺寸模型更为稳定。因此，选用大尺寸模型其实是节省了人力成本，避免了很多之后可能会遇到的各种坑。
8. [浅谈大模型 SFT 的实践落地： 10 问 10 答](https://zhuanlan.zhihu.com/p/692892489)SFT真的不能学到知识？很遗憾的说，经过一年的实践和普遍的认知。常识和世界知识难以通过 SFT 灌输给模型。SFT更应该关注激发模型在预训练中已学到的知识、让模型学习业务所需要的特定规则、以及输出格式稳定。那么，何为常识和世界知识？例如，“2023年NBA总冠军是掘金”便属于世界知识。如果 LLM 的训练数据仅更新至2022年，那么它自然无法得知这一信息。即便你的SFT数据中包含 “谁是2023年NBA总冠军？答案是掘金” 这样的问答对，训练后的模型可能只能回答这个语序的问题，而无法举一反三。比如，当你问“掘金在哪一年获得了NBA总冠军？”时，它无法回答“2023年”。这种举一反三的能力需要模型在预训练阶段就接触过“2023年NBA总冠军是掘金”这类知识的多种不同文本表达，如这条知识在预训练文本中出现在不同的表述中（主动句、被动句、出现在新闻语料、出现在聊天对话语料等等等等）。因此，从这个角度看，SFT并不能学得常识、世界知识。但这并不意味着我们应该放弃SFT。相反，我们应当关注SFT在以下方面：
    1. 激发预训练知识：虽然SFT不能直接学的新知识，但需要靠它激发模型在预训练中已学到的知识。
    2. 稳定格式输出：通过SFT，我们可以训练模型以稳定的格式输出结果，便于线上的稳定。
    3. 更遵循具体任务：如多标签多分类时，模型老输出一些不在标签体系的任务。
    4. 学习业务逻辑：SFT能够教导模型特定的业务规则，如让他习得“买了 20 万以上的车算有钱人”。
[大模型生产环境下部署微调的10条戒律](https://mp.weixin.qq.com/s/eRakAQP7FP51Gj6lL-_bNg)汝应当编写提示语，并创建一个基准，证明任务是可行的。如果提示语有效，微调有90%的可能性会改善模型表现；如果无效，微调只有25%的可能性有效；

一些体会
1. pre-training，通常使用自监督算法进行训练。对last token 计算loss，目的是学习语言能力/一句话的统计概率/世界知识。
    1. 继续预训练（也称为第二阶段预训练）将使用全新的、未见过的领域数据进一步训练基础模型。同样使用与初始预训练相同的自监督算法。通常会涉及所有模型权重，并将一部分原始数据与新数据混合。
2. post-training 比如sft 微调， 在包含正确标签/答案/偏好的注释数据集上进行监督训练，而不是自监督训练，主要目的是提高能力，如指令遵循、人类对齐、任务执行等。仅对response 部分计算loss，目的是让response 对prompt 做出反应。prompt 本身并不值得太多学习。

### 数据准备

4. 一开始不需要急着构造大量 SFT 数据集，可以先用少量数据（50条~100条）对模型做 SFT 后观察真实评估是否有收益。如果有收益，可以尝试以部分数据为种子数据集继续扩充，找到 scaling law。如果没有收益，那么再重新检查 SFT 数据集的质量。如何判断 SFT 数据质量好坏？
    1. badcase 的覆盖度：做 SFT 之前，我们肯定有一批 badcase。那么，我们做 SFT 的目的就是去解决这些badcase。所以，要确保 SFT 数据集中对这些 badcase 有一定的覆盖度。这样，做完 SFT 才可能会有收益。
5. 高质量，**数据质量重要性大于数据数量**。大模型中 SFT 的过程中，会学习 prompt 到 response 到映射关系，如果我们 SFT 的数据存在噪声（如错别字、错误格式、不符合预期输出的样本等），那么会对模型的训练过程造成比较严重的影响。因此，不可以一味去堆叠 SFT 的样本数量，样本的质量比数量更重要。举一个例子，往往我们会遇到模型在一些任务上不遵循输出格式的问题。这时，我们会通过 SFT 来做对齐。假如说这里的格式是指 json 格式的话，那么我们可以通过代码 check 的方式，对 SFT 数据集做一遍检查。其他格式问题都可以通过这类代码校验的方式进行检查。
    1. 尽量人工生成：语言模型生成的文本，有一种隐含的“模式”。在看一些文字的时候，经常能识别出来“这是语言模型生成的”。
    2. 数量不少太少：通过LoRA论文看，100条开始有明显的改善，1000条左右，有不错的效果。
1. 多样性。SFT 数据集中 response 的多样性，当我们希望同样的 prompt 生成出来的内容更多样时，可以尝试在 SFT 数据集中，构造同样的输入 prompt，多样性的 response 数据。经测试后，模型输出多样性指标得到提升。
6. 是否混入预置数据，即在自定义 SFT 数据集的基础上，额外增加一部分比例的模型基座训练时的 SFT 数据（看base 模型提供方是否开放）。一定程度上缓解大模型在某个领域上 SFT 后，通用能力下降的问题。混入预置数据后，由于模型在 SFT 时，拟合的是全部数据集 prompt 到 response 之间的映射关系，所以混入比例过大会导致模型在特定领域上的能力下降，过小会导致缓解通用能力下降不明显。按照经验，通常设置为 15%～30%左右，依据自己的需求和实验迭代来决定。
7. 建议先使用小参数量大模型+ LoRA 验证实验设置及数据是否有效，如果有效的话，再迁移到大参数量大模型上验证scaling law（即随着模型参数量的增加，模型的效果越好）是否成立。这样做可以显著加快迭代效率，避免在大参数模型上反复做无意义的迭代。

多任务训练时怎么确保每个任务都优秀？目前实践下来，任务的相互影响是一个普遍现象，例如训练集中包含四个任务，现在针对任务1补充了大量 bad cases 后重新训练，这种调整很可能会对其他任务产生或正或负的影响。训练集本身存在的任务数据不平衡也是一个不可忽视的问题，某个任务占比大，那其它占比小的任务大概率效果也是不稳定的。有两种方法应对这种挑战：
1. 不同任务独立训练模型：针对每个任务单独训练一个模型。当某个任务至关重要，且要求性能指标高度稳定时，这是不得不采用的方法。
2. 任务取舍与额外训练：例如，在四个任务中，若其中两个任务尤为重要，可以在全部任务训练完毕后，对这两个关键任务额外训练多一个 epoch。这种做法能最大程度地确保重要任务的效果。
### 训练参数选择

在微调大语言模型时，几个关键参数会直接影响模型的表现和训练效果。下面是对常见微调参数的解释：
1. 学习率 (Learning Rate)：这是控制模型权重更新步幅的参数。学习率决定了每次权重更新的幅度，设置较大时会加速模型迭代，但是模型可能无法收敛到最优点；设置过小时会使得模型迭代较慢，可能陷入局部最优。按照经验来讲， LORA 训练选择 learning rate 在 1e-5 ~ 2e-5。
2. 训练轮次 (Epochs)：指整个训练数据集被输入模型的次数。更多的训练轮次可以让模型更好地学习数据，但也可能导致过拟合，特别是在小数据集上。
    1. 在指令微调阶段，不建议进行过多轮次的训练。针对少量数据进行多个epoch的训练，可能会导致模型的关键区域发生变化，从而影响整体性能。为了保证模型语言能力关键区不被大幅度调整，需要在指令微调过程中添加通用指令数据或者预训练数据。
    2. epoch 越多，就是对特定训练集死记硬背，适合硬知识。如果是为了学习软知识，比如逻辑知识 类似于“如果碰到xx，就xx”，则应该靠多样化的数据集，而不是扩大epoch，很容易过耦合。
    1. 模型训练轮数，通常选择2～5，可以根据验证集 loss 曲线来判断：如果训练集 loss 曲线下降，验证集loss 曲线上升，则说明模型已经过拟合，此刻应该停止训练；如果训练集和验证集 loss 曲线均在缓慢下降，则说明模型还未收敛，可以继续训练。此外，对于文案生成，小说创作等生成类任务，由于某些上下文逻辑及风格不能通过 loss 体现，所以不应仅看 loss 来决定模型何时停止训练，按经验来说，通常生成类任务 epoch 数可以略微设置大一点，如5～10范围内。
    2. 刚开始训练时，模型的权重是随机初始化的，如果选择一个较大的学习率，可能会带来模型的不稳定（震荡）。warmup预热学习率的方式，在训练开始的时候先选择使用一个较小的学习率，训练一些epoch或者step之后，等模型相对稳定后，再修改为预先设置的学习率开进行训练。对于复杂的模型，可能需要更长的warm-up时间来适应训练过程，因为这些模型可能更容易受到初始学习率的影响。如果数据集很大，可能不需要太长的warm-up时间，因为模型有足够的数据来快速适应。num warmup steps 和 warmup step rate：二者设置一个即可，通常不需要调整。当遇到模型在 SFT 训练开始时，loss 始终不降的问题时，可尝试调大 warmup step rate，再观察 loss 是否按预期下降。
    3. warmup_ratio很重要。通常LLM训练的warmup_ratio是epoch * 1%左右。例如pre-train阶段一般只训一个epoch，则ratio是0.01；SFT通常3个epoch，ratio对应为0.03。如果你的数据集很大，有几百b，那warmup其实不影响最终的模型效果。但通常我们的数据集不会有那么大，所以更小的ratio可以让模型“过渡”得更平滑。我甚至试过3个epoch的训练(SFT)，第一个epoch全部用来warmup，结果是work的。所以学习率和warmup_ratio是两个相辅相成的概念，二者通常是成正比的关系。或者说如果你正在用一个较大的学习率，那你或许可以同时尝试增加warmup来防止模型“烂掉”。请勿迷信3个epoch的训练，实测1个epoch就能对话。当然，更多的epoch确实会让模型的评测效果更佳。如果数据量比较小，如只有1k，可以尝试更多的epoch。无他，人为过拟合而已。
3. 批处理大小 (Batch Size)：这是**每次更新模型权重时使用的数据量**。较大的批处理大小能更稳定地更新权重，但需要更多的内存资源；较小的批处理大小则训练速度较慢，但内存需求更少。
4. lora alpha和 lora rank：alpha 参数将模型权重进行放缩，rank 决定了 lora 训练参数量的大小，我们推荐二者参数设置为同一值。通常来讲，对于较简单的任务，推荐设置 lora alpha 和 lora rank 64或者128；对于较复杂的任务，推荐设置 lora alpha 和 lora rank 为256或者512。


### 效果评估

有两类评估方式：1）训练过程中观察loss指标；2）发布服务后在真实评估集上评估。
1. loss 指标：一般观察在训练集和验证集上的损失来评估模型精调的效果。
2. 真实评估集：发布服务后评估业务指标。
针对这两种评估方式，在不同的任务有不同的侧重点。我们把任务分成两类：确定性任务和生成式任务。我们把 label 准确的任务称为确定性任务，如分类任务、提槽类等任务。相反地，比如参考问答、文案生成或角色扮演对话类我们称为生成式任务。
1. 确定性任务：在这类任务上，我们主要关注训练集和验证集上的损失是否同步在下降，以及训练多个 epoch 后loss 是否收敛。下图应该算是一个比较良好的训练过程，训练集和校验集损失同步下降，且训练集损失也接近收敛。
2. 生成式任务：在这类任务上，我们还是主要依赖观察训练集上的损失以及真实评估集上的业务指标。因为某些时候，尽管校验集的 loss 损失没有明显的下降，但其实真实评估集上也可能有收益。这是因为，验证集上 loss 计算依赖 token 级别的KL散度损失，而真实评估集上并不需要输出分布和 label 完全一致。所以，对于这类任务，我们建议将训练集损失作为参考，以真实数据集上评估后的结论为准。

## 其它

[sft 的局限性](https://zhuanlan.zhihu.com/p/717275921) 非常经典
1. sft 的训练过程，是一个让模型学习条件概率的过程，Prob( E | ABCD )。这也就是说，模型在训练和学习过程中，只知道 next_token 出什么是正确的，而不知道 next_token 出什么是错误的。无论你的 sft 语料如何构造，都无济于事，模型不知道“什么 token 是不能生成的”。这也间接解释了另外一个现象：为什么 sft 的数据多样性很重要。因为没办法， 我们无法直接让模型知道错误的 token 是什么，但只要我们把正确的 token 都喂给它学习，孤立那个错误的 token，似乎也能起到类似的效果。sft 缺乏负反馈机制引发的糟糕后果，还远不止此。你越是在 sft 阶段告诉它什么是错误的，它越是容易提高错误 token 的概率。站在模型的角度来思考，这个现象非常合理：“训练者不断让我提高 Prob( E | ABCD ) 的概率，那我举一反三，顺带提高一下 Prob( E | ACD ) 的概率是不是也合理？训练者是不是应该表扬我？”可问题是，好巧不巧，B 这个 token，恰好是“not”，恰好是“不”。这里问一个我曾经被问过的问题，“一句绝对正确的话，是不是可以放进 sft 训练语料中？”我的观点是：不应该，因为一句绝对正确的话，它可能有局部是不正确的，这些局部错误的知识内容也会在 sft 的过程中被模型学到。
2. sft 没有负反馈，但 rlhf 有啊。reward_model 就像是一个教官，你敢续写出某个不能出的 token，我就抽你，抽到你不敢出这个 token 为止。这可能也是为什么 rlhf 的最大应用方向是安全场景吧，毕竟 sft 真的做不好安全。
3. sft 不具有“向后看”的能力。sft 的另一个不足，就是它放大了 transformer 单向注意力结构的缺陷。在 sft 的训练过程中，每一个 token 都只看得见前面的 token。还是那个经典例子，“台湾不是中国的，这个观点是严重错误的”。无论你用什么炼丹技巧来做 sft，Prob(中国 | 台湾不是) 的概率都是在增加的，模型无法利用“后半个句子在否定前半句子”这个重要信息。那 rlhf 是怎么学习这句话呢？首先这句话是正确的，他会得到一个正向的 reward_model，但这句话中的每个 token 又不是同等正确的。如果对 critic_model 进行可视化，它大概率会在 reward 反向衰减传递的时候，把最大的奖励赏赐给“错误 ”这个 token，而“中国 ”这个 token 可能并不会得到很多的 reward。所以，sft 在更新某个 token 的概率的时候，是只参考前面信息的，是一种局部的有偏的训练方法。但 rlhf 或者 dpo 并不是这样，每一个 token 在更新概率的时候，都是观察到了整个 sentence 的，因而理论上，rlhf 的训练方法能带来更高的训练上限。换一个角度来说，sft 的 loss 是平均 loss， rlhf 的 loss 是加权 loss。至于怎么加权，去问 reward_model 和 critic_model。
综上所述，我个人认为，除非 sft 的训练方式发生改变（比如每个 token 的 loss，不再是算术平均），否则 rlhf 还是一个不可取代的环节。

