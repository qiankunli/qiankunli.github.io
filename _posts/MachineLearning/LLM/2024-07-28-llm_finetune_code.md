---

layout: post
title: LLM微调代码
category: 技术
tags: MachineLearning
keywords: llm finetune

---

* TOC
{:toc}

## 简介

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
模型的input_ids和对应的labels应该是什么呢？最常规的做法应该是在每一轮首尾用`[BOS]`和`[EOS]`包裹，轮次内部正常用模板非结构化就行。上例可以转换为input_ids= `[BOS][INST]U1[\INST]A1[EOS][BOS][INST]U2[\INST]A2[EOS]`，难点在于LABELS应该是什么呢？ 我们可以根据学习模式来确定LABELS。
1. 在推理场景下，假如是第一轮对话开始，我们会输入给模型[BOS][INST]U1[\INST]，那么我们希望模型吐出的是什么呢？是A1和[EOS]，A1是模型自己的回答，EOS是为了告诉解码系统生成结束了，否则模型将一直生成到最大长度才会停止。我们获得了一个初步的学习模式需求，就是根据`[BOS][INST]U[\INST] → A[EOS]`。
    |input|`[BOS]`|`[INST]`|U|`[/INST]`|A|
    |---|---|---|---|---|---|
    |label|-100|-100|-100|A|`[EOS]`|

    在多轮的场景下，也只是复制这个过程。

    |input|`[BOS]`|`[INST]`|U|`[/INST]`|A|`[EOS]`|`[BOS]`|`[INST]`|U2|`[/INST]`|A2|
    |---|---|---|---|---|---|---|---|---|---|---|---|
    |label|-100|-100|-100|A|`[EOS]`| X| X| X| X| A2| `[EOS]`|

2. 在llama factory里有一个参数叫 Efficient EOS，所谓efficient eos并不代表是一个新的token，而是一个特殊的input和label的设计方式。 [多轮对话的训练过程详解](https://zhuanlan.zhihu.com/p/695202364)

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

[四个大模型轻量级微调训练框架：兼看PPT转Markdown工具](https://mp.weixin.qq.com/s/1Wjap8kiNGXkCQQ35pJE7g) 建议把这几个框架都看下，找找共性。 

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



