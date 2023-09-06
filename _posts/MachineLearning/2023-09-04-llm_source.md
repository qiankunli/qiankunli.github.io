---

layout: post
title: LLM部分技术源码学习
category: 技术
tags: MachineLearning
keywords: langchain

---

* TOC
{:toc}

## 前言（未完成）

## LoRA

[从0到1！得物如何打造通用大模型训练和推理平台](https://mp.weixin.qq.com/s/5EE1VXxq7k_VoC9gRPvyMg)

对于大语音模型来说，其参数量非常多。GPT3有1750亿参数，而且LLAMA系列模型包括 7B,13B,33B,65B，而其中最小的7B都有70亿参数。要让这些模型去适应特定的业务场景，需要对他们进行微调。如果直接对这些模型进行微调，由于参数量巨大，需要的GPU成本就会非常高。LoRA的做法是对这些预训练好的大模型参数进行冻结，也就是在微调训练的时候，这些模型的参数设置为不可训练。然后往模型中加入额外的网络层，并只训练这些新增的网络层参数。这样可训练的参数就会变的非常少，可以以低成本的GPU微调大语言模型。LoRA在Transformer架构的每一层注入可训练的秩分解矩阵，与使用Adam微调的GPT-3 175B相比，LoRA可以将可训练参数数量减少10000倍，GPU内存需求减少3倍，并且在效果上相比于传统微调技术表现的相当或更好。当前已经得到HuggingFace 的 PEFT库 https://github.com/huggingface/peft 的支持。

LoraModel是LoRA模块的核心类，冻结base model的参数，旁路低秩矩阵的创建，替换，合并等逻辑都在这个类中。

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

[图解大模型微调系列之：大模型低秩适配器LoRA（源码解读与实操篇）](https://mp.weixin.qq.com/s/RuV_4lCQentq4kBr3fv08A) 未读。