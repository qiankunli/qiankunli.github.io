---

layout: post
title: 微调理论及实践
category: 技术
tags: MachineLearning
keywords: llm finetune

---

* TOC
{:toc}

## 为什么需要微调

[通俗解读大模型微调(Fine Tuning)](https://mp.weixin.qq.com/s/PXTAhvUGzvOPLdBYNWb3xw)Prompt Engineering的方式是一种相对来说容易上手的使用大模型的方式，但是它的缺点也非常明显。
1. 因为通常大模型的实现原理，都会对输入序列的长度有限制，Prompt Engineering 的方式会把Prompt搞得很长。**越长的Prompt，大模型的推理成本越高**，因为推理成本是跟Prompt长度的平方正向相关的。
2. Prompt太长会因超过限制而被截断，进而导致大模型的输出质量打折口，这也是一个非常严重的问题。可能会胡乱地给你一些东西，它更适合一些泛化的任务。而fine-tuning则不一样，只要你的微调训练数据是高质量的，它可以很稳定地在你自己的数据集或任务上进行稳定的输出。
3. Prompt Engineering的效果达不到要求时，企业又有比较好的自有数据，能够通过自有数据，更好的提升大模型在特定领域的能力。这时候微调就非常适用，**微调可以使你为LLM提供比prompt多得多的数据，你的模型可以从该数据中进行学习，而不仅仅是访问该数据**。
4. 要在个性化的服务中使用大模型的能力，这时候针对每个用户的数据，训练一个轻量级的微调模型，就是一个不错的方案。
5. 数据安全的问题。如果数据是不能传递给第三方大模型服务的，那么搭建自己的大模型就非常必要。通常这些开源的大模型都是需要用自有数据进行微调，才能够满足业务的需求，这时候也需要对大模型进行微调。

外挂知识库也不是全能的
1. 向量化的匹配能力是有上限的，搜索引擎实现语义搜索已经是好几年的事情了，为什么一直无法上线，自然有他的匹配精确度瓶颈问题，只是很多以前没有接触过AI 的朋友对之不熟悉罢了。
2. 在引入外部知识这个事情上，如果是特别专业领域，纯粹依赖向量、NLP、策略/规则在某些场景仍然不奏效。对于某个具体业务而言，不要管是不是90%会被解决，只有需要被解决或不需要被解决。

## 如何对大模型进行微调

1. 对全量的参数，进行全量的训练，这条路径叫全量微调FFT(Full Fine Tuning)。但FFT也会带来一些问题，影响比较大的问题，主要有以下两个：一个是训练的成本会比较高，因为微调的参数量跟预训练的是一样的多的；一个是叫灾难性遗忘(Catastrophic Forgetting)，用特定训练数据去微调可能会把这个领域的表现变好，但也可能会把原来表现好的别的领域的能力变差。
2. 只对部分的参数进行训练，这条路径叫PEFT(Parameter-Efficient Fine Tuning)。有以下几条技术路线：
    1. 一个是监督式微调SFT(Supervised Fine Tuning) ，这个方案主要是用人工标注的数据，用传统机器学习中监督学习的方法，对大模型进行微调；
    2. 一个是基于人类反馈的强化学习微调RLHF(Reinforcement Learning with Human Feedback) ，这个方案的主要特点是把人类的反馈，通过强化学习的方式，引入到对大模型的微调中去，让大模型生成的结果，更加符合人类的一些期望；
    3. 还有一个是基于AI反馈的强化学习微调RLAIF(Reinforcement Learning with AI Feedback) ，这个原理大致跟RLHF类似，但是反馈的来源是AI。这里是想解决反馈系统的效率问题，因为收集人类反馈，相对来说成本会比较高、效率比较低。

一些比较流行的PEFT方案
1. Prompt Tuning，Prompt Tuning的出发点，是基座模型(Foundation Model)的参数不变，为每个特定任务，训练一个少量参数的小模型，在具体执行特定任务的时候按需调用。Prompt Tuning的基本原理是在输入序列X之前，增加一些特定长度的特殊Token，以增大生成期望序列的概率。具体来说，就是将$X = [x1, x2, ..., xm]变成，X` = [x`1, x`2, ..., x`k; x1, x2, ..., xm], Y = WX`$。如果将大模型比做一个函数：Y=f(X)，那么Prompt Tuning就是在保证函数本身不变的前提下，在X前面加上了一些特定的内容，而这些内容可以影响X生成期望中Y的概率。
2. Prefix Tuning，Prefix Tuning的出发点，跟Prompt Tuning的是类似的，只不过它们的具体实现上有一些差异。Prompt Tuning是在Embedding环节，往输入序列X前面加特定的Token。而Prefix Tuning是在Transformer的Encoder和Decoder的网络中都加了一些特定的前缀。具体来说，就是将Y=WX中的W，变成$W` = [Wp; W]，Y=W`X$。Prefix Tuning也保证了基座模型本身是没有变的，只是在推理的过程中，按需要在W前面拼接一些参数。
3. LoRA，LoRA背后有一个假设：我们现在看到的这些大语言模型，它们都是被过度参数化的。而过度参数化的大模型背后，都有一个低维的本质模型。通俗讲人话：大模型参数很多，但并不是所有的参数都是发挥同样作用的；大模型中有其中一部分参数，是非常重要的，是影响大模型生成结果的关键参数，这部分关键参数就是上面提到的低维的本质模型。LoRA的基本思路，在原始预训练模型旁边增加一个旁路，先用一个 Linear 层 A，将数据从 d 维降到 r 维，在用第二个 Linear 层 B，将数据从 r 维变回 d 维。LoRA 训练的时候固定预训练模型的参数，只训练降维矩阵 A 和升维矩阵 B。
4. QLoRA，QLoRA就是量化版的LoRA（Quantize+LoRA），量化，是一种在保证模型效果基本不降低的前提下，通过降低参数的精度，来减少模型对于计算资源的需求的方法。量化的核心目标是降成本，降训练成本，特别是降后期的推理成本。

[使用 DPO 微调 Llama 2](https://mp.weixin.qq.com/s/u-GqdifZy8ArKgZaQWmh8Q) 没看懂。

难点：
1. 计算量太大
2. 不好评估，因为大模型生成的海量内容暂无标准的答案，所以我们无法全部依赖人工去评判内容的质量。让模型做选择题不能准确的评估性能，一些垂类领域也很难搞到相关测试集，如果用GPT4评估又涉及到数据隐私问题，GPT4更倾向于给句子长的、回答更多样性的答案更高的分数，有时候也是不准的。
3. 微调数据量到底要多少？
4. LoRA VS 全量参数微调？微调基础模型LoRA还是比全量微调差一些的，微调对话模型差距不大,个人任务有时候全量微调是不如LoRA的，全量微调灾难遗忘现象会更加严重

## PEFT/LoRA

假如LLM的原始权重视为矩阵“X”。对于任何给定的任务，经过优化微调的 LLM 的权重由矩阵“Y”表示。微调的目标是发现一个 delta 矩阵“Z”，使得 X+Z=Y。然而，在 LoRA 的情况下，这个增量矩阵“Z”是通过低秩分解来近似的。因此，对于某些类型的任务来说， 一些数据集可能更容易对齐，而另一些数据集可能会有效果损失。相比之下，全参数微调则没有这个约束， 学习到的权重保留了原始模型的表达能力，可能简化了拟合不同数据的任务。

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
        self.forward = self.model.forward   # LoraModel把自己的前向传播函数forword设置为大模型的forward方法
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
1. Freeze 预训练模型权重/参数的本质：requires_grad = False ，不接受梯度更新，只微调参数A和B。，此时该步骤模型微调的参数量由$d×k$变成$d×r+r×k$，而$r≪min(d,k)$，因此微调参数量大量减少了。

[图解大模型微调系列之：大模型低秩适配器LoRA（源码解读与实操篇）](https://mp.weixin.qq.com/s/RuV_4lCQentq4kBr3fv08A) 未读。

[深入浅出剖析 LoRA 源码及实践](https://mp.weixin.qq.com/s/QdX1R0GPe6dvEDXjdJj5lA) 未读。

## P-Tuning 系列/可学习的提示

P-Tuning，简称PT，是一种针对于大模型的soft-prompt方法，包括两个版本；P-Tuning，仅对大模型的Embedding加入新的参数； P-Tuning-V2，将大模型的Embedding和每一层前都加上新的参数，这个也叫深度prompt。

很多任务我们只需要在输入端输入合适的prompt，大语言模型就能给出一个比较合理的结果。但对于人工设计的prompt，prompt的变化对模型最终的性能特别敏感，加一个词、少一个词或者变动位置都会造成比较大的变化。那干脆让大模型学习一个合理的prompt，然后直接输入到模型中，这样是不是就可以直接得到一个比较合理的结果了？放弃之前人工或半自动离散空间的hard prompt设计，采用连续可微空间soft prompt设计，通过端到端优化学习不同任务对应的prompt参数。

Prefix Tuning（面向文本生成领域（NLG）） 方法在输入 token 之前构造一段**任务相关的 virtual tokens** 作为 Prefix，然后训练的时候只更新 Prefix 部分的参数，而 PLM 中的其他部分参数固定。

![](/public/upload/machine/prefix_tuning.jpg)

Prompt Tuning 可以看作是 Prefix Tuning 的简化版本，它给每个任务定义了自己的Prompt，然后拼接到数据上作为输入，但只在输入层加入prompt tokens（在输入embedding层加入一段定长的可训练的向量，在微调的时候只更新这一段prompt的参数），另外，virtual token的位置也不一定是前缀，插入的位置是可选的。 但也有一些缺点 PS： Prompt Tuning 和 P Tuning 分不清了。
1. 针对模型参数量缺少通用性，之前的试验证明了p-tuning针对参数量大于10B的模型有很好的效果，甚至可以达到全量微调的效果。但是针对中等规模的模型，效果就不是很明显了。
2. 针对不同任务的通用性也比较差，之前的实验结果证明了在一些NLU任务上效果比较好，对序列标注类比较难的任务效果较差。
2. prompt只加在了embedding层，这样做就让prompt比较难训练，同时也导致了可训练参数比较少。

基于此，作者提出了P-tuning v2
1. 在每一层都加入了Prompts tokens作为输入，而不是仅仅加在输入层，可学习的参数更多（从P-tuning和Prompt Tuning的0.01%增加到0.1%-3%），prompts与更深层相连，对模型输出产生更多的直接影响。

除Prefix Tuning用于NLG任务外，Prompt Tuning、P-Tuning、P-Tuning V2 均用于NLU，P-Tuning和Prompt Tuning技术本质等同，Prefix Tuning和P-Tuning V2技术本质等同。

**对于领域化的数据定制处理，P-Tune（Parameter Tuning）更加合适**。领域化的数据通常包含一些特定的领域术语、词汇、句法结构等，与通用领域数据不同。对于这种情况，微调模型的参数能够更好地适应新的数据分布，从而提高模型的性能。相比之下，LORA（Layer-wise Relevance Propagation）更注重对模型内部的特征权重进行解释和理解，通过分析模型对输入特征的响应来解释模型预测的结果。虽然LORA也可以应用于领域化的数据定制处理，但它更适合于解释模型和特征选择等任务，而不是针对特定领域的模型微调。

## 实践（未完成）

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

不管是PreTraining阶段还是SFT阶段，loss函数都是一样的
1. preprocess_pretrain_dataset处理PreTraining阶段的数据，数据组成形式：
    1. 输入input： `<bos> X1 X2 X3`
    2. 标签labels：`X1 X2 X3 </s>`
    典型的Decoder架构的数据训练方式；
2. preprocess_supervised_dataset处理SFT阶段的数据，数据组成形式：
    1. 输入input：`<bos> prompt response`
    2. 标签labels： `-100 ... -100 response </s>`
对于prompt部分的labels被-100所填充，这样在计算loss的时候模型只计算response部分的loss，-100的部分被忽略了。这个机制得益于torch的CrossEntropyLossignore_index参数，ignore_index参数定义为如果labels中包含了指定了需要忽略的类别号（默认是-100），那么在计算loss的时候就不会计算该部分的loss也就对梯度的更新不起作用。



## 代码

### 手写

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

