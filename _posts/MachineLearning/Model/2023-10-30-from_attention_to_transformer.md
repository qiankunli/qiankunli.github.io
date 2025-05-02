---

layout: post
title: 从Attention到Transformer
category: 架构
tags: MachineLearning
keywords:  rnn attention transformer

---

<script>
  MathJax = {
    tex: {
      inlineMath: [['$', '$']], // 支持 $和$$ 作为行内公式分隔符
      displayMath: [['$$', '$$']], // 块级公式分隔符
    },
    svg: {
      fontCache: 'global'
    }
  };
</script>
<script async src="/public/js/mathjax/es5/tex-mml-chtml.js"></script>

## 简介

* TOC
{:toc}

## Attention is all you need——Transformer

[李沐读 Transformer 视频笔记](https://zhuanlan.zhihu.com/p/619234992) 李沐对于 Transformer论文的讲解，还是看作者的第一手材料+大佬解说最准确。未读。

Transformer模型采用的也是编码器-解码器架构，但是在该模型中，编码器和解码器不再是 RNN结构，取而代之的是编码器栈（encoder stack）和解码器栈（decoder stack）（注：所谓的“栈”就是将同一结构重复多次，“stack”翻译为“堆叠”更为合适）。

![](/public/upload/machine/transformer.jpg)

[Transformer - Attention is all you need](https://zhuanlan.zhihu.com/p/311156298)Encoder层和Decoder层内部结构如下图所示。

![](/public/upload/machine/transformer_internal.jpg)

在经典的 Attention 机制中，Q 往往来自于一个序列，K 与 V 来自于另一个序列，都通过参数矩阵计算得到，从而可以拟合这两个序列之间的关系。
1. 在 Transformer 的 Decoder 结构中，Q 来自于 Decoder 的输入，K 与 V 来自于 Encoder 的输出， 从而拟合了编码信息与历史信息之间的关系，便于综合这两种信息实现未来的预测。
1. 在 Transformer 的 Encoder 结构中，使用的是 Attention 机制的变种 —— self-attention （自注意力）机制。 所谓自注意力，即是计算本身序列中每个元素都其他元素的注意力分布， 即在计算过程中，Q、K、V 都由同一个输入通过不同的参数矩阵计算得到。 从而拟合输入语句中每一个 token 对其他所有 token 的关系，从而建模文本之间的依赖关系。​

Transformer 模型本来是为了翻译任务而设计的。在训练过程中，Encoder 接受源语言的句子作为输入，主要负责理解，而 Decoder 则接受目标语言的翻译作为输入。在 Encoder 中，由于翻译一个词语需要依赖于上下文，因此注意力层可以访问句子中的所有词语；而 Decoder 是顺序地进行解码，在生成每个词语时，注意力层只能访问前面已经生成的单词。例如，假设翻译模型当前已经预测出了三个词语，我们会把这三个词语作为输入送入 Decoder，然后 Decoder 结合 Encoder 端的输出 和 当前block mask attention layer的输出作为输入（多头交叉注意力，Q来自前一个解码器层输出向量，K和V来自编码器输出的注意力向量，编码器并非只传递最后一步的隐状态，而是把所有时刻（对应每个位置）产生的所有隐状态都传给解码器，这就解决了中间语义编码上下文的长度是固定的问题）来预测第四个词语。每一个部分都有公式对应。

想要进一步了解 Transformer 这一架构的原理，强烈推荐阅读 Jay Alammar 的博客。从为了解决翻译问题时 Seq2Seq 模型的提出的 Attention 的基本概念 https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention，到 Transformer 架构中完全抛弃 RNN https://www.youtube.com/watch?v=UNmqTiOnRfg 提出的 Attention is All You Need http://jalammar.github.io/illustrated-transformer，到 GPT-2 和 GPT-3 的架构解读http://jalammar.github.io/illustrated-gpt2 http://jalammar.github.io/how-gpt3-works-visualizations-animations，Jay Alammar 的博客都提供了精彩的可视化配图便于理解模型结构。

![](/public/upload/machine/transformer.png)

![](/public/upload/machine/encoder_decoder_matrix.jpg)


## 位置编码/Positional Encoding

Transformer提出了深度学习领域既MLP、CNN、RNN后的第4大特征提取器。一个好的特征提取器需要自带输入处理模块的前后顺序信息，而Attention机制并没有考虑先后顺序信息（源于注意力机制固有的排列不变性，因此修改tokens的顺序不会改变输出加权值。因此，注意机制本身缺乏对token顺序的意识），但前后顺序信息对语义影响很大，因此需要通过位置嵌入这种方式把前后位置信息加在输入的Embedding上。

对self-attention来说，它跟每一个input vector都做attention，在计算注意力矩阵时每个 token 都与其他所有 token 交互，没有考虑到input sequence的顺序。更通俗来讲，大家可以发现我们前文的计算每一个词向量都与其他词向量计算内积，得到的结果丢失了我们原来文本的顺序信息。打乱词向量的顺序，得到的结果仍然是相同的。但实际上位置信息很有用，比如动词出现在句首的概率就比名词低，所以要把位置信息塞到学习过程中去。 

Positional Embeddings 基于一个简单但有效的想法：使用与位置相关的值模式来增强词向量。为了在序列中编码位置信息，transformer的设计者使用了不同频率的正弦函数。他们还尝试了习得的位置Embeddings，但结果并没有什么不同。PS： 波形相加，不同的波。不同就行。如果不用相加，则位置编码和emebedding 弄出一个新的向量（长度一样），无非线性和非线性变换，线性变换跟相加差不多，非线性其实最后的mlp也可以学。有点像llm出现之前的特征融合。

位置信息编码位于encoder和decoder的embedding之后（**emebedding的长度，即每个token嵌入的大小，这个维度影响模型的表示能力**，有时被称为模型维度$d_model$），每个block之前。

如果预训练数据集足够大，那么最简单的方法就是让模型自动学习位置嵌入。 除此以外，Positional Embeddings 还有一些替代方案：
1. Learned Positional Embedding ，这个是绝对位置编码，即直接对不同的位置随机初始化一个postion embedding，这个postion embedding作为参数进行训练。缺点：
  1. 不同位置对应的positional embedding固然不同，但是位置1和位置2的距离比位置3和位置10的距离更近，这些关于位置的相对含义，模型能够通过绝对位置编码参数学习到吗？
  2. 位置之间没有约束关系，我们只能期待它隐式地学到，是否有更合理的方法能够显示的让模型理解位置的相对关系呢？
2. Sinusoidal Position Embedding ，相对位置编码，即三角函数编码。由于正弦函数能够表达相对位置信息，那么对每个positional embedding进行 sin 或者cos激活，好处是位置 i 处的单词的psotional embedding可以被位置 i+k 处单词的psotional embedding线性表示，反应两处单词的其相对位置关系。此外位置i和i+k的psotional embedding内积会随着相对位置的递增而减小，从而表征位置的相对距离。缺点：由于距离的对称性，Sinusoidal Position Encoding虽然能够反映相对位置的距离关系，但是无法区分i和i+j的方向。

transformer中，模型输入encoder的每个token向量由两部分加和而成：Position Encoding, Input Embedding。Positional Embedding的成分直接叠加于Embedding之上，使得每个token的位置信息和它的语义信息(embedding)充分融合，并被传递到后续所有经过复杂变换的序列表达中去。

![](/public/upload/machine/positional_encoding.jpg)

为什么 RNN 不需要位置编码呢？RNN 是串行处理输入数据的，所以每个 RNN 单元其实都包含了所在位置之前的全部输入信息，所以对当前位置输入的处理是有状态的。和 RNN 的逐步迭代相比，**attention 可以认为是个全连接图（如果是 causal attention 则有些边断开）**，不会考虑位置。比如对于 attention 看来，"八王大"和"大王八" 是一样的。因此，我们需要在最开始就给每个输入的嵌入向量一个位置编号，这样模型才能通过输入判断它在整体中的位置。

[transformer位置编码如何去理解？ - magicwt的回答 - 知乎](https://www.zhihu.com/question/633536226/answer/3319109019) 基于论文解释了3个与原因，建议细读。token位置编码后输出的向量维度也是$d_model$和token通过Embedding层后输出的向量维度相同，这样便于两者直接相加作为编码器和解码器层的输入。论文中采用的方法是正余弦函数， 论文中解释采用这种位置编码的原因
1. 可以将将长短不一的句子中的token位置编码为固定维度的向量
2. 可以将离散的位置值转化为连续值
3. 可以借助正余弦函数的特性，对于某个偏移量k ，可以基于$PE_{pos}$通过线性变换快速计算出相对位置的 $PE_{pos+k}$

## attention层

[大模型结构基础（五）：注意力机制的升级](https://zhuanlan.zhihu.com/p/702890483) 未读

每个attention层包含2个子层，分别为多头注意力（MSA）和前馈网络（FFN），并采用相加（add）和层归一化（LayerNorm）操作连接两个子层。目前公认的说法是，
1. Attention 作为token-mixer做spatial interaction。
2. FFN （又称MLP）在后面作为channel-mixer进一步增强representation。

### self-attention 机制/捕捉上下文依赖

在处理每个单词的表示时，要对你传递给它的句子中某些单词特别关注（并且忽略其他单词）。一个单词本身具有意义，但是该意义深受上下文影响，这可以是正在研究的单词之前或之后的任何其他单词（或多个）。self-attention 机制用于计算句子中当前词与其他词的联系，每个单词的新表示形式是由它自己和其他单词的交互得到的。举个例子：

```
The animal didn’t cross the street because it was too tired
The animal didn’t cross the street because it was too wide
```

两句话中的单词 it 指代不同，第一句话 it 指代 animal 而第二句指代 street。对于我们来说能很简单的判断出来，但是对于机器来说，是很难判断的，尤其是相对于传统seq2seq模型。两句话在单词 it 之前的内容是一样的，传统seq2seq模型encoder的顺序输入导致模型无法区分这种差别。而self-attention机制通过计算单词it与其他词之间的联系得知it的具体指代，最终结果如下图所示。

![](/public/upload/machine/self_attention.jpg)

图中下半部分的颜色深浅，意义就是单词之间的相关度。**引入自注意力机制的原因是它可以让编码器在对特定元素进行编码时使用输入序列中其他元素的信息**。在对‘it’进行编码时，有一部分注意力集中在"animal"上，并将它们的部分信息融入到"it"的编码中，颜色越深的部分表示注意力得分越多。PS： 也就是在考虑一个词的编码时，只考虑embedding 和 position encoding 不够，还需要考虑和其他词的关系（尤其是it这种），通过 W 将词之间的关系学出来，其输出可以用作词的表示。

[超详细图解Self-Attention](https://zhuanlan.zhihu.com/p/410776234)那么具体的计算过程是怎样的呢，我们先上公式，然后一步步拆解：

$$
Attention(Q,K,V) = softmax(\frac{QK^T}{\sqrt{d_k}})V
$$

![](/public/upload/machine/self_attention_qkv.jpg)

所谓的Q K V矩阵、查询向量之类的字眼，其来源是X与矩阵的乘积，本质上都是X的线性变换，经过不同的权重矩阵$W_Q$$W_K$$W_V$投影到不同的空间中。为什么不直接使用X而要对其进行线性变换？当然是为了提升模型的拟合能力，矩阵W都是可以训练的，起到一个缓冲的效果。所以 self-Attention最原始的形态其实是 $softmax(XX^T)X$。

1. $XX^T$代表什么？一个矩阵（一个句子）乘以它自己的转置，会得到什么结果，有什么意义？我们知道，矩阵可以看作由一些向量组成，**一个矩阵乘以它自己转置的运算，其实可以看成这些向量分别与其他向量计算内积**（一个句子的各个词向量的内积）。向量的内积，其几何意义是什么？表征两个向量的夹角，表征一个向量在另一个向量上的投影。**投影的值大，说明两个向量相关度高**（两个词相关度越高）。如果两个向量夹角是九十度，那么这两个向量线性无关，完全没有相关性！更进一步，这个向量是词向量，是词在高维空间的数值映射。词向量之间相关度高表示什么？是不是在一定程度上（不是完全）表示，在关注词A的时候，应当给予词B更多的关注？对应到qkv，$qK^T$基于矩阵乘法的定义，$qK^T$ 即为 q 与每一个 k 值的点积，反映了 Query 和每一个 Key 的相似程度。
  ![](/public/upload/machine/self_attention_xxt.jpg)
2. **Softmax操作的意义是什么呢？归一化**。 Softmax之后，这些数字的和为1了。当我们关注"早"这个字的时候，我们应当分配0.4的注意力给它本身，剩下0.4关注"上"，0.2关注"好"。
  ![](/public/upload/machine/self_attention_softmax_xxt.jpg)
3. 最后一个 X 有什么意义？ $softmax(XX^T)X$表示什么？我们取 $softmax(XX^T)X$
 的一个行向量举例。这一行向量与X的一个列向量相乘，表示什么？在新的向量中，每一个维度的数值都是由三个词向量在这一维度的数值加权求和得来的，**这个新的行向量就是"早"字词向量经过注意力机制加权求和之后的表示**。PS：变换后词向量的长度不变，结合了上下文的动态向量来表示这个词。 
  ![](/public/upload/machine/self_attention_softmax_xxt_x.jpg)
  一张更形象的图是这样的，图中右半部分的颜色深浅，其实就是我们上图中黄色向量中数值的大小，意义就是单词之间的相关度
  ![](/public/upload/machine/self_attention_softmax_xxt_x_matrix.jpg)
4. $\sqrt{d_k}$的意义。假设Q,K里的元素的均值为0，方差为1，那么$A^T=Q^TK$中元素的均值为0，方差为d。当d变得很大时，A中的元素的方差也会变得很大，如果A中的元素方差很大，那么softmax(A)的分布会趋于陡峭(分布的方差大，分布集中在绝对值大的区域)。总结一下就是softmax(A)的分布会和d有关。因此A中每一个元素除以$\sqrt{d_k}$后，方差又变为1。这使得softmax(A)的分布“陡峭”程度与d解耦，从而使得训练过程中梯度值保持稳定。

```python
'''注意力计算函数'''
from torch.nn import functional as F
def attention(q, k, v):
    # 此处我们假设 q、k、v 维度都为 (B, T, n_embed)，分别为 batch_size、序列长度、隐藏层维度
    # 计算 QK^T / sqrt(d_k)，维度为 (B, T, n_embed) x (B, n_embed, T) -> (B, T, T)
    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    # 计算 softmax，维度为 (B, T, T)
    att = F.softmax(att, dim=-1)
    # V * Score，维度为(B, T, T) x (B, T, n_embed) -> (B, T, n_embed)
    y = att @ v 
    return y
```

[小白看得懂的 Transformer (图解)](https://mp.weixin.qq.com/s/VrzkxEVBAO6abJcUsYGr0Q)计算自注意力的步骤
1. 从每个编码器的输入向量（每个单词的embedding向量，512维）与矩阵相乘，这个矩阵是随机初始化的，维度为（64，512），生成三个向量（QKV，64维）。
  ![](/public/upload/machine/self_attention_vector.jpg)
2. 计算得分。假设我们在为这个例子中的第一个词“Thinking”计算自注意力向量，我们需要拿输入句子中的每个单词对“Thinking”打分。这些分数决定了在编码单词“Thinking”的过程中有多重视句子的其它部分。这些分数是通过打分单词（所有输入句子的单词）的K向量与“Thinking”的Q向量相点积来计算的。PS： 点积是一种常用的相似度度量
  ![](/public/upload/machine/self_attention_vector_dot_product.jpg)
3. 将分数除以8(8是论文中使用的键向量的维数64的平方根，这会让梯度更稳定。这里也可以使用其它值，8只是默认值)，然后通过softmax传递结果。**softmax的作用是使所有单词的分数归一化，得到的分数都是正值且和为1**。这个softmax分数决定了每个单词对编码当下位置（“Thinking”）的贡献。显然，已经在这个位置上的单词将获得最高的softmax分数，但有时关注另一个与当前单词相关的单词也会有帮助。
  /![](/public/upload/machine/self_attention_score.jpg)
4. 将每个V 乘以softmax分数(这是为了准备之后将它们求和)。这里的直觉是希望关注语义上相关的单词，并弱化不相关的单词(例如，让它们乘以0.001这样的小数)。对加权值向量求和，然后即得到自注意力层在该位置的输出(在我们的例子中是对于第一个单词)。
  ![](/public/upload/machine/self_attention_convert.jpg)
6. 这样自自注意力的计算就完成了。得到的向量就可以传给前馈神经网络。然而实际中，这些计算是以矩阵形式完成的（一个句子一个矩阵），以便算得更快。

自注意力的另一种解释就是在编码某个单词时，就是将所有单词的表示（值向量）进行加权求和，而权重是通过该词的表示（键向量）与被编码词表示（查询向量）的点积并通过softmax得到。

![](/public/upload/machine/transformer_qkv.gif)

![](/public/upload/machine/transformer_x_to_z.jpg)

Attention机制的目标是输入$x_n$，输出$z_n$。这个过程可以简单概括为 3 步：
1. 注意力矩阵$QK^T$衡量QK 之间的相似性，这种相似性对于向量来说就是内积。$\sqrt{d_k}$ 作为调节因子，使得内积不至于太大，从而使梯度稳定很多。其中 $q_n$
与$k_1,k_2,...k_n$均有运算关系。所以可以通过缓存$k_1,k_2,...k_{n-1}$向量加速推理。这是kvcache中K矩阵需要缓存的原因。不过很意外的发现最右边一列$q_1,q_2,...q_{n-1}$与$k_n$之间存在计算。不是说好的只有KV缓存，没有Q矩阵缓存？推导没有错，也没有Q矩阵缓存。因为在推理阶段，Attention机制有一个非常重要的细节：mask掩码。注意力矩阵在训练推理过程中，为了模拟真实推理场景，当前位置token是看不到下一位置的，且只能看到上一位置以及前面序列的信息，所以在训练推理的时候加了attention mask。
  ![](/public/upload/machine/transformer_qkt.jpg)
2. 查找权重 a。这是通过 SoftMax 完成的。相似性像一个完全连接的层一样连接到权重。SoftMax 之后的注意力分数，其分值大小代表了相关性强弱，这种差异在计算梯度时，可以相对均匀地流入多个token位置。
3. 值的加权组合。将 a 的值和 V 对应的值相乘累加 $z_n=a_1*v_1+a_2*v_2+...+a_n*v_n$，最终输出是所需的结果注意力值。这是kvcache中V矩阵需要缓存的原因。

[Transformer内部原理四部曲3D动画展示](https://mp.weixin.qq.com/s/QEvdSPaf6Ikz15iaEJXjlw)

### MultiHeadAttention

​Attention 机制可以实现并行化与长期依赖关系拟合，但一次注意力计算只能拟合一种相关关系，单一的 Attention 机制很难全面拟合语句序列里的相关关系。因此 Transformer 使用了 Multi-Head attention 机制，即同时对一个语料进行多次注意力计算，每次注意力计算都能拟合不同的关系，将最后的多次结果拼接起来作为最后的输出，即可更全面深入地拟合语言信息。（这些集合中的每一个W都是随机初始化的，在训练之后，每个集合都被用来将输入词嵌入(或来自较低编码器/解码器的向量)投影到不同的表示子空间中）。PS：多头注意力机制借鉴了CNN中multi-kernel 的思想，对不同头使用不同的线性变换。

$$
head_i = Attention(QW{_i}{^Q},KW{_i}{^K},VW{_i}{^V})
$$
$$
MultiHead(Q,K,V) = Contact(head_1,head_2,...head_h)W^O
$$

不仅仅只初始化一组Q、K、V的矩阵，而是初始化多组，tranformer是使用了8组，所以最后得到的结果是8个矩阵。

![](/public/upload/machine/self_attention_multi_headed.jpg)

![](/public/upload/machine/self_attention_multi_z.jpg)

以h=2为例（有几个head 是一个超参数），**根据输入序列X和$W_1^Q,W_1^K,W_1^V$** 我们就计算得到了$Q_1,K_1,V_1$，进一步根据公式1就得到了单个自注意力模块的输出$Z_1$；同理，根据X和$W_2^Q,W_2^K,W_2^V$就得到了另外一个自注意力模块输出$Z_2$。最后，前馈层不需要8个矩阵，它只需要一个矩阵(由每一个单词的表示向量组成)。所以我们需要一种方法**把这八个矩阵压缩成一个矩阵**。那该怎么做？其实可以直接把这些矩阵**拼接**在一起，传入一个Linear层（Linear层，全连接的神经网络层的映射是非线性变换，它的作用是对输入进行维度变换和特征提取。 线性变换只能进行简单的比例缩放和平移操作，而非线性变换可以引入更多的复杂性，例如曲线形状、峰值和谷底等。这样可以使模型更加灵活，能够更好地适应不同类型的数据分布，从而增加模型的表达能力），得到 Multi-Head Attention 最终的输出Z。根据公式2将$Z_1,Z_2$水平堆叠形成Z，然后再用Z乘以$W^O$便得到了整个多头注意力层的输出。PS：计算自注意力需要四个方阵，$W^O$、$W^Q$、$W^K$、$W^V$

![](/public/upload/machine/multi_head_cal.jpg)

把所有的矩阵放到一张图内看一下总体的流程。

![](/public/upload/machine/multi_head_attention.jpg)

既然我们已经摸到了注意力机制的这么多“头”，那么让我们重温之前的例子，看看我们在例句中编码“it”一词时，不同的注意力“头”集中在哪里：

![](/public/upload/machine/multi_head_example.jpg)

当我们编码“it”一词时，一个注意力头集中在“animal”上，而另一个则集中在“tired”上，从某种意义上说，模型对“it”一词的表达在某种程度上是“animal”和“tired”的代表。

（假设$d_model$=512）权重矩阵$W^Q$,$W^K$,$W^V$是与 Multi-head Attention 多头注意力机制相关的，它们的形状并非是`n*512`，而是`n*64`，即把一个进程分为 8 个并列的进程来实施（512÷8=64)。这个进程被原论文称为“头 head”，这 8 个头之间互不干扰，各自运算各自的 Attention 机制。8 个头中的每一个头都只采用初始 Embedding 的向量长度 512 的 8 分之一来运行，即通过把 Embedding 向量与$W^Q$,$W^K$,$W^V$三矩阵分别相乘之后，得到的向量长度为 Embedding 原始向量长度的 1/8，即 64（$d_head$）。这相当于把 Embedding 向量作线性变换的同时，顺便被切成了 8 分来运行。当然，这样的“切”并不是直接在一个长度为 512 的向量上等分 8 份，而是通过与$W^Q$,$W^K$,$W^V$三矩阵分别相乘，线性变换而来的。这样的降维变换明显含有某种特殊的意义，好比龙生九子，各有所好。PS：多个head concat在一起后，还是原来输入向量的长度。也可以存在矩阵$W_0$以可学习的方式来综合不同头的输出，

多头注意力机制类似于赛马机制，每个编码器/解码器使用8个“头”（可以理解为8个互不干扰自的注意力机制运算），每一组的Q/K/V都不相同。然后，得到8个不同的权重矩阵Z，每个权重矩阵被用来将输入向量投射到不同的表示子空间。它有助于减少模型初始化的随机性对模型效果的影响。所以即使只留下一个注意力头也能使用，但这会导致模型的稳定性和多样性无法得到保障，进而造成模型的性能下降。

[图解Transformer多头注意力机制](https://mp.weixin.qq.com/s/Aefek7zCftt4oXLWoKN57A)
![](/public/upload/machine/end_to_end_multi_head.jpg)
PS: head的概念类似卷积中的通道，只不过每个通道的输入都是一样的，类似于把一个通道的数据复制多次。多头注意力的计算过程类似深度可分离卷积，把通道分开计算，再融合到一起。经过注意力层，**输入向量和输出向量的shape是一致的**，可以看做向量相互"交流"并根据彼此信息更新自身的值。Attention模块的作用就是要确定上下文中哪些词对更新其他词的意义有关，以及应该如何准确地更新这些含义。

多头注意力（MHA），由于对键-值缓存的二次计算复杂度和高内存消耗而受到限制。为了解决这些问题，提出了几种变体，如多查询注意力（MQA）、组查询注意力（GQA）和多潜在注意力（MLA）。

## Position-wise Feed Forward（对输出进行非线性变换）

Attention模块的作用就是确定上下文中哪些词之间有语义关系，以及如何准确地理解这些含义（更新相应的向量）。这里说的“含义”meaning指的是编码在向量中的信息。Attention模块让输入向量们彼此充分交换了信息（例如machine learning model和fashion model，单词“model”指的应该是“模特”还是“模型”）， 然后，这些向量会进入第三个处理阶段：Feed-forward / MLPs。针对所有向量做一次性变换。这个阶段，向量不再互相"交流"，而是并行地经历同一处理。**Transformer基本不断重复Attention和Feed-forward这两个基本结构，这两个模块的组合成为神经网络的一层**。输入向量通过attention更新彼此；feed-forward 模块将这些更新之后的向量做统一变换，得到这一层的输出向量；在Attention模块和多层感知机（MLP）模块之间不断切换。

FFN（前馈神经网络）在 Transformer 模型中的作用是将输入向量投射到更高维度的空间中，以便发掘数据中原本隐藏的细微差别。

FFN 设计的初衷，是为模型引入非线性变换，有一个维度为n_ff的隐藏层。主要还是$W_1$和$W_2$，进行了“先把hidden_size的输入向量维度升到n_ff，又降回hidden_size”的操作。
1. FFN的作用就是空间变换。FFN层由两个线性变换组成，线性变换 中间有一个激活函数。
  1. FFN包含了2层linear transformation层，中间的激活函数是ReLu。
2. attention层的output最后会和相乘，为什么这里又要增加一个2层的FFN网络？仔细看一下 Attention 的计算公式，其中确实有一个针对 q 和 k 的 softmax 的非线性运算。但是对于 value 来说，并没有任何的非线性变换。所以每一次 Attention 的计算相当于是对 value 代表的向量进行了加权平均，虽然权重是非线性的权重。**Attention内部就是对特征向量V加权平均的过程。只用self-Attention搭建的网络结构就只有线性表达能力**。FFN的加入引入了非线性(ReLu激活函数)，**变换了attention output的空间**, 从而增加了模型的表现能力。把FFN去掉模型也是可以用的，但是效果差了很多。
  $$
  FFN(x) = max(0,xW_1+b1)W_2+b_2
  $$
2. 前馈神经网络的输入是self-attention的输出，是一个矩阵（即上图Z），矩阵的维度是（序列长度×D词向量），**之后前馈神经网络的输出也是同样的维度**。self-attention + 前馈神经网络 就是一个小编码器的内部构造了，一个大的编码部分就是将这个过程重复了6次，最终得到整个编码部分的输出。为了解决梯度消失的问题，在Encoders和Decoder中都是用了残差神经网络的结构，即每一个前馈神经网络的输入不光包含上述self-attention的输出Z，还包含最原始的输入。

```python
'''全连接模块'''
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Transformer 的全连接模块有两个线性层，中间加了一个 RELU 激活函数
        # 此处我们将隐藏层维度设为输出层的四倍，也可以设置成其他维度
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.relu    = nn.ReLU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
    def forward(self, x):
        x = self.c_fc(x)
        x = self.relu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
```

其它

1. 大语言模型中的参数都用在哪了？可以看到在百亿参数量以上，**差不多三分之二的参数实际上是 FFN 参数**，剩下的基本都是 attention 参数。所以虽然论文名叫 attention is all you need，但实际上 FFN 仍然起到了很重要的作用。
2. transformer 中最重要的是self-attention，self-attention 由三个线性矩阵Q、K、V 决定，如果我们把Q、K矩阵设置为零，那么self-attention 就变成了FFN，$Z_0 =V_0*X$，也就是说，FFN是self-attention 的一个特例，FFN能表达的逻辑，self-attention 也可以，但反过来却不成立。至于transformer 中的FFN部分，当初设计是为了输入输出维度的对齐，毕竟多注意力的输出 $W_0$的维度比输入X维度高很多。但如果一定要用FFN去表达self-attention的逻辑，也是可以的，但需要的参数量却要大很多，感兴趣的可以去试验一下，用FFN去拟合self-attention 的逻辑。就好像乘法能计算的东西，单纯用加法依然可以做到，但效率要低很多，self-attention就是乘法，FFN就是加法。

[大模型结构基础（四）：前馈网络层的升级](https://zhuanlan.zhihu.com/p/702190813) 未读。FFN组件的一个显著进步是混合专家（MoE）架构，它采用稀疏激活的FFN。在MoE中，每个输入只有一部分FFN层（或专家）被激活，显著减少了计算负载，同时保持了高模型容量。

## 辅助架构

除了上述主要模块之外，Transformer模型中还应用了LayerNorm（层归一化）和ResNet（残差连接）等设计方法。对提高模型的整体表示能力非常重要。

###  Layer Normalization/对应Norm

[Batch Normalization原理与实战](https://mp.weixin.qq.com/s/7B-gSLQm0PAKMefKHMb8nw) 是一种常规的模型“构件”，非transformer独有。 
归一化核心是为了让不同层输入的取值范围或者分布能够比较一致。由于深度神经网络中每一层的输入都是上一层的输出，**因此多层传递下，对网络中较高的层，之前的所有神经层的参数变化会导致其输入的分布发生较大的改变**。也就是说，随着神经网络参数的更新，各层的输出分布是不相同的，且差异会随着网络深度的增大而增大。**但是，需要预测的条件分布始终是相同的**，从而也就造成了预测的误差。因此，在深度神经网络中，往往需要归一化操作，将每一层的输入都归一化成标准正态分布。

Normalization有两种方法，Batch Normalization和Layer Normalization。关于两者区别不再详述。

一般情况下，输入是一个矩阵，然后矩阵的 每一行是一个样本，多个行（多个样本）是 一个 batch，每一列是一个特征，多个列是 feature。batchnorm 是说每一次，去把每一个列，就是每一个特征，把它在一个小 mini-batch 里面，每列 的均值变成 0 方差变成 1。 [Batch Norm详解之原理及为什么神经网络需要它](https://zhuanlan.zhihu.com/p/441573901)

**在 Transformer 里面，或者说正常的 RNN 里面，它的输入是一个三维的矩阵**。因为 输入的是一个序列的样本，即每一个样本里面有很多个元素。一个序列，如：一个句子里面有 n 个词，所以每个词表示为一个向量的话，还有一个 batch 维度，那么就是个 3D 的输入。列不再是特征，而是序列的长度，对每一个 sequence 就是 每个词，每个词有自己对应的向量。如果是 layernorm 的话，那么就是对每个样本切一下（横着切一下）。为什么 layer norm 用的多一点？一个原因是：在 时序的序列模型 里面，每个样本的长度可能会发生变化。那些不够 sequence 的长度 n 的样本 ，一般是补 0。 layernorm 是对 每个样本来做，所以不管样本是长还是短，反正算均值是在样本自己算的，这样的话相对来说它稳定一些。

[大模型结构基础（三）：归一化技术的升级](https://zhuanlan.zhihu.com/p/701779323) 未读。

### Residual Network 残差网络/对应Add

由于 Transformer 模型结构较复杂、层数较深，​为了避免模型退化，Transformer 采用了残差连接的思想来连接每一个子层。残差连接，即下一层的输入不仅是上一层的输出，还包括上一层的输入。残差连接允许最底层信息直接传到最高层，让高层专注于残差的学习。​例如，在 Encoder 中，在第一个子层，输入进入多头自注意力层的同时会直接传递到该层的输出，然后该层的输出会与原输入相加，再进行标准化。在第二个子层也是一样。

在神经网络中，每个层通常由一个非线性变换函数和一个线性变换函数组成，非线性变换函数通常由激活函数，例如ReLU、Sigmoid、Tanh等实现，而线性变换函数则通常由矩阵乘法实现。在传统的神经网络中，这些变换函数直接作用于输入数据，然后传递到下一层。而在使用残差连接的神经网络中，每个层都添加了一个跨层连接，可以将输入数据直接连接到输出数据，也可以将输入数据直接传递到后续层次，从而提高信息的传递效率和网络的训练速度。同时，残差连接还可以解决梯度消失和梯度爆炸的问题，从而提供网络的性能和稳定性。

在transformer模型中，encoder和decoder各有6层，为了使当模型中的层数较深时仍然能得到较好的训练效果，模型中引入了残差网络，**将输入和输出直接相加**。

![](/public/upload/machine/residual_network.jpg)

```python
def forward(self, x):
    # 此处通过加x实现了残差连接
    x = self.ln_1(x)
    # Encoder 使用 Self Attention，所以 Q、K、V 都是 x
    x = x + self.attn(x, x, x)
    x = x + self.mlp(self.ln_2(x))
    return x
```

## 概率输出/Linear & Softmax

最后一层feed-forward输出中的最后一个向量the very last vector in the sequence的备选列表及其概率， 产生一个覆盖所有可能Token的概率分布，这些Token代表的是可能接下来出现的任何小段文本，包含句子的核心意义essential meaning of the passage。对这个向量进行 unembedding 操作（也是一次性矩阵运算）， 得到的就是下一个单词的备选列表及其概率。

**softmax用于多分类过程中**，它将多个神经元的输出，映射到（0,1）区间内，可以看成概率来理解，从而来进行多分类！假设我们有一个数组，V，$V_i$表示V中的第i个元素，那么$V_i$的softmax值就是 

$$
S_i = \frac{e^{V_i}}{\sum_j e^{V_j}}
$$

如下图表示：

![](/public/upload/machine/softmax_example.jpg)

softmax直白来说就是将原来输出是$z_1=3,z_2=1,z_3=-3$通过softmax函数一作用，就映射成为(0,1)的值，而**这些值的累和为1**（满足概率的性质），那么我们就可以将它理解成概率，在最后选取输出结点的时候，我们就可以选取概率最大（也就是值对应最大的$z_1$）结点，作为我们的预测目标！

另：Decoder最后是一个线性变换和softmax层。解码组件最后会输出一个实数向量（在推理时使用的并不是decoder的所有输出，而是最后一个token对应的向量即只使用输出序列中最后一个单词的猜测结果，训练时则使用全部输出）。我们如何**把hidden_size长度的浮点数向量变成一个单词**？这便是线性变换层要做的工作，它之后就是Softmax层。词表中每个单词都有可能成为下一个单词， 所以模型需要对所有它知道的单词均按可能性打分，最终选出其中最合适的单词推荐给用户。
1. 线性变换层是一个简单的全连接神经网络，它可以把解码组件产生的向量投射到一个比它大得多的、被称作对数几率（logits）的向量里。PS：hidden_size向量 ==> vocab_size向量，其中词汇表中的每个token都有一个对应的值，称为 logit。该过程类似CNN中，卷积层之后再接一个线性层做分类。
2. Softmax 层。因为是要预测，所以需要根据模型的输出 logits 为词汇表中的每个token分配一个概率。这些概率决定了每个token成为序列中下一个单词的可能性。具体操作是应用 softmax 函数将 logits 转换为总和为 1 的概率分布。

## 其它视角

[逐步理解Transformers的数学原理](https://mp.weixin.qq.com/s/b9YHoCOp5Pu5kfyeTdoBxw) 可以看看从字符串到输出向量 到各层的变换过程。

输入 token 序列的维度：【batch, seqlen】
经过 embedding 之后的维度：【batch，seqlen，D】
后面xx变换，都是在 tensor 之间加权组合，**不会改变这个输入的 shape。不然就没办法残差连接了**。

### 李宏毅

[台大李宏毅自注意力机制和Transformer详解！](https://www.bilibili.com/video/BV1v3411r78R) 视频总结了 Self-Attention 与RNN 和 CNN和 GNN的关系，Self-Attention 可以用来学习原来由  RNN 和 CNN  和 GNN学习的任务。

![](/public/upload/machine/self_attention_from.png)

为向量a1生成一个向量b1，b1是a1考虑了整个句子后的表示。关键是如何计算a1与句子内其他词之间的相关度？

![](/public/upload/machine/self_attention_qkv.png)

$q^i=w^qa^i$
$$
q^1q^2q^3q^4=W^qa^1a^2a^3a^4
$$
用矩阵运算来表示就是 $Q=WX$

![](/public/upload/machine/2_head_attention.png)

一个知乎回答：transformer中的Q,K,V到底是什么？就是查字典。假想你有一个map/dict或者其他名字，一个key对应一个value，在检索的时候，给定query，如果query in map，就是query等于其中一个key，就返回对应的value。这个方法太hard了，有就是有，没有就是没有。对于qkv都是向量的情况，这种方法不可行，只能让它变soft，那就是算一算query和key的关系，按照比例对value加和，这和max变成softmax有异曲同工之妙。PS：有点像外挂知识库的chunk 召回，根据query_embedding(Q) 找到多个 chunk_embedding(K) 对应的chunk(V)，再对chunks 进行加工，得到最后的回答。

### 编码视角

NLP 神经网络模型的本质就是对输入文本进行编码，常规的做法是首先对句子进行分词，因为模型是无法直接处理文本的，只能处理数字，就跟ASCII码表、Unicode码表一样，计算机在处理文字时也是先将文字转成对应的字码(token)，然后为每个字码编写一个对应的数字记录在表中（Tokenizer）。然后将每个数字都转化为对应的词向量 (token embeddings)，这样文本就转换为一个由词语向量组成的矩阵 $X=(x_1,x_2,...,x_n)$，其中$x_i$就表示第i个词语的词向量。

以将$x_t$ 编码为 $y_t$ 的视角来理解（结合了上下文词表示）
1. RNN（例如 LSTM）的方案很简单，每一个词语$x_t$对应的编码结果$y_t$通过递归地计算得到：$y_t=f(y_{t-1},xt)$，RNN 本质是一个马尔科夫决策过程，难以学习到全局的结构信息；
2. CNN 则通过滑动窗口基于局部上下文来编码文本，例如核尺寸为 3 的卷积操作就是使用每一个词自身以及前一个和后一个词来生成嵌入式表示：$y_t=f(x_{t-1},x_{t},x_{t+1})$，由于是通过窗口来进行编码，所以更侧重于捕获局部信息，难以建模长距离的语义依赖。
3. 直接使用 Attention 机制编码整个文本。相比 RNN 要逐步递归才能获得全局信息（因此一般使用双向 RNN），而 CNN 实际只能获取局部信息，需要通过层叠来增大感受野，Attention 机制一步到位获取了全局信息：$y_t=f(x_t,A,B)$，其中 A,B是另外的词语序列（矩阵），如果取A=B=X就称为 Self-Attention，即直接将$x_t$与自身序列中的每个词语进行比较$y_t=f(x_{0},x_{t},x_{...})$，最后算出$y_t$。如对于单词I，它的新表征L(I)就是其它所有单词Value的加权和。所以仅仅只看这一点，**I的新表征中就能用到其它所有单词的信息。因此，没有距离的概念，也就没有长依赖的问题**。PS：在深度学习里，拟合$y_t$ 和$x_{0},x_{t},x_{...}$ 关系，就使用W将x串起来，学习/优化W，直接$y_t=W_0x_{0}+W_tx_{t}+W_{...}x_{...}$太粗暴，整出来QKV。


[如何理解 Transformer 中的自注意力机制？ - 宇文树雪的回答 - 知乎](https://www.zhihu.com/question/560879732/answer/3041597591)注意力机制中的“注意力”其实指的是序列中每两个元素之间的相关性程度，更进一步说，**这种相关性就是指两个元素在自然语句中同时出现的概率**。那么，序列元素之间的相关性是如何衡量的呢？在transformer的输入之前，往往要做词嵌入转换，把输入的文本序列转为降维到固定长度的矢量。在这个矢量空间中，其实就通过跳元模型或连续词包模型把词元之间的共现关系学习到一个高维的空间中了。通过**直接进行矢量乘法就能知道两个嵌入词向量的相似度**。词元矢量内积越大，说明相关性越强，在词嵌入空间中越近，更可能组成语句。也就是说，我们能基于词元矢量内积判断输入序列任意两个元素的共现的概率，也就是组成语句的可能性。（词嵌入原始的计算方式就是统计词元之间共同出现的频次，然后对共现矩阵做了奇异值分解得到降维后的词向量表示。）**既然词嵌入能学习到语义，为什么还要进行更复杂的训练呢？**这是因为嵌入词是在较小的窗口训练成的，不能反映大规模语句的上下文结构特征。因此，需要加一个权重对序列中每个嵌入词的重要性进行调整，这个权重需要要通过大量的语料训练得到。从而学习到自然语言序列固有的结构，并形成语言模型。

[白话科普：Transformer和注意力机制](https://mp.weixin.qq.com/s/jyy7WXtOqJPXJYssPpfiUA)从Transformer整体来看，Encoder负责将输入序列（通常是自然语言的）变换成一个「最佳的」**内部表示**；而Decoder则负责将这个「内部表示」变换成最终想要的目标序列（通常也是自然语言的）。现在，我们先来看一下Encoder，它其实是由多个网络层组成的。输入序列进入Encoder之后，会经过多个Encoder Layer。每经过一层，相当于输入序列中的每个token进行了一次向量变换（非线性的），也就离那个「最佳的」内部表示又接近了一步。但是，每次变换都不改变向量的维度数量。每个Encoder Layer到底做了什么呢？这里面关键的一个机制是自注意力 (self-attention)。为什么需要自注意力呢？在模型内部，每个token都是用一个多维向量来表示的。向量的值决定了这个token在多维空间中的位置，也决定了它所代表的真实含义。一个token的真实含义，不仅仅取决于它自身，还取决于句子中的其它上下文信息（来自其它token的信息）。而借助向量，就可以用数量关系来描述这些现象了：相当于是说，**一个token的向量值，需要从句子上下文中的其他token中「吸收」信息，在数学上可以表达为所有token的向量值的加权平均**。这些权重值，我们可以称之为注意力权重 (attention weights)。在Encoder中，每经过一层Encoder Layer，一个token都会「参考」上一层的所有token，并根据对它们注意力权重的不同，决定「携带」它们中多少量的信息进来。对于这一过程，一个最简化的说法可以表达为：一个token会注意到 (attend to)所有其他token。对于Decoder这里的自注意力来说，生成的过程需要遵循因果关系。也就是说，生成下一个token的时候，它必须只能注意到 (attend to) 之前已经生成的token；所以，对于已经生成的序列来说，Decoder Layer对这个序列进行处理的时候，序列中每个token也都应该保持跟生成时一样的逻辑，即它只能注意到 (attend to) 在它之前的token。在计算上需要构建一个mask矩阵。而Encoder中的自注意力却允许序列中的每个token都可以注意到 (attend to) 所有的token（包括在它之前和它之后的）。

**对于算法这类代码来说，打印输入输出尺寸，往往可以帮助我们从直觉上快速了解这个模块在做一件什么事**。在这个基础上我们再去看细节。【看不懂就动手跑，千万不要原地纠结】

### 与MLP对比

特征矩阵X每一行代表一个 item，以每一列代表一个 feature。通常我们所说的线性层都是 $X^{t+1}=X^tW^t+B^t$，也就是在 feature 这个维度上作线性变换，完全没有考虑 item 与 item 之间的关系。这也是为什么 MLP 会比不过什么 CNN、attention之类的原因，在 MLP 的框架里每个 item 的运算都是独立的，不会与其他 item 产生交互。用代数的话来说，就是在矩阵乘法中，左乘的矩阵行其向量之间是不会产生交互的，相应的右乘的矩阵其列向量之间是不会产生交互的。

那要怎么做到 item 与 item 之间的交互呢？很简单，把线性层里面的右乘一个权重矩阵变成左乘就行了，即$X^{t+1}=W^tX^t+B^t$，由于这也可以看成是把原来的输入取了转置以后再扔进通常的线性层，得到输出后再做一个转置，我们就把这样的层叫做转置线性层。为了方便我们就不要最后的偏差项了，拿每一个 item 来看，$X^{t+1}_{i}=W_i^tX^t=\sum_jW_{ij}^tX_j^t$

这个输出的形式和 attention 可以说是一模一样了。这里拿最常见的 Attention 举例，即

$$
Attention(Q,K,V) = softmax(\frac{QK^T}{\sqrt{d_k}})V
$$

其中V就等价于$X^t$，而$softmax(\frac{QK^T}{\sqrt{d_k}})$就是我们想要学到的$W^t$（所以说self-attention 的权重是根据输入的改变和改变的，而CNN中一旦训练完成，各个通道的卷积核参数就固定了）， attention 多做了一些什么呢？对于要学习的参数矩阵$W^t$增加了低秩的约束。参考 self-attention 的情况，即输出和输入的 size 保持一致，那么理论上没有任何约束的情况下$W^t$可以是一个满秩的n*n方阵。但是，在 self-attention 中，假设$Q,K\in\R^{n*d}$那么$softmax(\frac{QK^T}{\sqrt{d_k}})$就是一个秩至多为d的方阵。秩上的降低也意味着表征能力的下降，但好处在于$\mathcal{O}(n^2) $变成了$\mathcal{O}(nd) $， $d\ll n$ 的时候这样做的价值就体现出来了。比如原来比如GPT2 一个token向量 是768维，context若是4k，input item数量是768*4096，所以一定是预处理一下再接入mlp；cnn类似，一张1024*768的图片 每个pixel 作为一个item，weight就不小了。可以说attention 是一种为了压缩参数量而牺牲一些表征能力，是用类似于 matrix fatorization 的方式来近似转置线性层的做法。所以有人提出什么 **MLP is all you need** 之类的说法，用 MLP 替代 CNN 或者 attention 也能取得不俗的图像分类效果……那可不是必须的嘛？人家那可是加了各种约束，为了压缩参数量牺牲了表征能力来近似你这个线性层的结果啊。PS：大部分模型最后的部分都是mlp，所以可以视为前半部分的处理是在缩小input 规模以便接入mlp。

大可：整个 transformer 的重点在 QKV 结构上。以前的 CNN 试图通过卷积来表达不同位置数值之间的关系，学习卷积值也就是学习矩阵里的数值之间的特征，所以适合用在图像里面。因为图像就是一个个的像素点形成的矩阵。RNN 试图通过加入反馈机制来理解一串数值前后的关系，所以适用于语言模型，因为这些数值之间有前后关系，像我们的句子里有先后逻辑。而 transformer 里的 QKV 给你提供了一个新的思路：只研究问题和答案之间的关系。不去找前后，不去找相邻，就是单纯的问题（Query）和答案(Value)，最多加了一个(Key)来辅助。那为什么要用 QKV 呢？因为这是谷歌搜索等搜索引擎最开始的结构。一个搜索引擎的设计其实就是给一个问题然后找到对应答案。任何一个问题(Query)，会有很多的答案（Value），而之所以能找到这些答案，是因为这些答案里面包包含了有关于这个问题的关键信息（Key）：V=f(Q,K)，这是一个万能形式，任何问题的答案都是通过“问题本身+相关的关键信息”找到的，比如你去谷歌搜索“今天天气怎么样”，这个问题本身就是 Q，而你的语言是“中文”，你的位置是“北京”，你的时间是“今天”，这些就都是 K，那么找到的答案“下雨”就是 V。一般来说肯定是通过方法找到 f( Q, K) 中的一些系数，就可以找到正确的 V 了。我们也可以把 V挪到公式右边，并且把他们存在的关系叫成 attention，那么就是：attention = F(Q, K, V)。这就是整个 transformer 的最基础结构，有了这个万能结构，只需要学习 F里的各个参数，就可以回答你想要的问题。为什么叫 transformer而不是简单的 attention 呢？因为 transformer 它为了提高这个 F的 运算效率，做了一些规定，比如你的 attention 的输入输出维度需要一样，这样矩阵运算就可以加快。而且多个 attention 合在一起来算，也是为了加快运算速度和效率。

### 自己的理解 

[一文图解自注意力机制（深度好文）](https://mp.weixin.qq.com/s/QiYMyOUdtrBm35I7SGpN8g)提到：自注意力机制模块接受n个输入并返回n个输出。且每个输出$y_i$都要考虑下所有$x_i$，看来是找一个函数 $y_i=f(x_1,x_2,...,x_{inputlen})$。

直接上全连接，可问题来了，这么干，每个$x_i$对应一个$d_{len}*d_{emblen}$ W矩阵，那可大了，且跟input_len 有关系/成正比了。因此最好找一个小W矩阵，不要跟$d_{emblen}$ 有关系。

![](/public/upload/machine/self_attetion_guess.png)


既然不能用矩阵乘法，那用加法，直接把所有的 $x_i$ 加一遍也不是个事儿。从另一个角度，两个向量内积表示表示两个向量的相似度，把$x_1$ 跟其它所有$x_i$ 乘一下，让$y_1=x_1*x_1*x_1 + x_1*x_2*x_2 + x_1*x_{inputlen}*x_{inputlen}$。看样子也挺不错，但这么算，就没啥可学习的W了，这不是$y_i$ 每个加法单元是3个向量相乘嘛，那干脆把$x_i$ 跟一个矩阵wi 变换一下再计算吧，于是有了$W_k$/$W_q$/$W_v$和QKV。 $y_1=Q_1*K_1*V_1 + Q_1*K_2*V_2 + Q_1*K_{inputlen}*V_{inputlen}$。

论文“MetaFormer is Actually What You Need for Vision”则描述了一种**通用架构**，在该结构中，输入首先经过embedding，得到 𝑋。然后embedding送入重复的blocks中，第一个block主要包含了token mixer，使得不同的token能够相互信息通信（Y = TokenMixer(Norm(X)) + X,）；第二个block包含两层MLP。该架构通过指定token mixer的具体设计，可以获得不同的模型。如果将token mixer指定为注意力或spatial MLP，则MetaFormer将分别成为一个transformer或类似MLP的模型。PS：transformer 与mlp的不同可以理解为 token mixer的选择不同。

![](/public/upload/machine/meta_former.jpg)


[面试时被问到“Scaling Law”，怎么答？](https://mp.weixin.qq.com/s/Q0fThU-4YP5OwFmfJM_q-Q)对于 Decoder-only 的模型，计算量C(Flops)，模型参数量 N(除去 Embedding 部分)，数据大小 D(token 数)，三者的关系为: C≈6ND。推导如下，记模型的结构为：decoder 层数l，attention隐层维度d，attention feedforward层维度 $d_{ff}$，一般来说 $d_{ff} = 4 * d$。
模型的参数量N（忽略emebedding、norm和bias） 计算如下，transformer 每层包含 self-attention 和mlp 两个部分
1. self-attention 的参数为$W_q$、$W_k$、$W_v$、$W_o$，每个矩阵的维度为$R^{d*d}$，整体参数量：$4 * d^2$
2. mlp 的参数为 $W_1$（维度为$R^{d*d_{ff}}$）和$W_2$（维度为$R^{d_{ff}*d}$），整体参数量：$2 * d * d_ff = 2 * d * 4d = 8d^2$
3. 所以每层参数是$4d^2 + 8d^2 = 12d^2$，全部l 层的参数量为 $12*ld^2$，即$N=12*ld^2$

计算量的单位是FLOPS，float point operations 对于矩阵A (m*n)和B（n*p） AB 相乘的计算量为2mnp，一次加法一次乘法。假设decoder 层的输入X（b*s*d），b为batch size，s为序列长度，d为模型维度。

1. 前向推理的计算量：
    1. self-attention 部分的计算：
        1. 输入线性层，$XW_q$、$XW_k$、$XW_v$，计算量为 $3 * b * s * d * d * 2 = 6bsd^2$
        2. attention 计算 $QK^T$，计算量为 $2 * b * s* s * d = 2bs^2d$
        3. score 与V 的计算，$S_{attention}V$，计算量为 $b*2 * s * s * d = 2bs^2d$
        4. 输出线性层 $X^{\prime} W_O$，计算量为 $b * 2 * s * d * d = 2bsd^2$
    2. MLP 部分的计算：
        1. 升维 $XW_1$，计算量为 $b * 2 * s * d * 4d = 8bsd^2$
        2. 降维 $XW_2$，计算量为 $b * 2 * s * 4d * d = 8bsd^2$
    3. 所以整个decoder层的计算量为 $24bsd^2 + 4bs^2d$，全部l层为 $C_{forward}=24bsd^2 + 4bs^2d$。
2. 反向传播计算量是正向的2倍，所以全部计算量为 $C=3*C_{forward} = 72bsd^2 + 12bs^2d$。
3. 平均每个token的计算量为 $C_{token}=\frac{C}{b s} = 72ld^2 + 12lsd = 6N (1+\frac{s}{6 d}) \approx 6N(s \le 6d)$
4. 所以对于包含全部D个token的数据集 $C = C_{token}D \approx 6ND$

PS：mlp参数量和计算量都不输attention。 $C_{token} \approx 6N$ 可以认为是正反向3倍*加法乘法2倍=6倍。 一般推演的时候，可以先只考虑一个句子的 输入10个token 输出10个token的变换，实际计算时，还要再考虑batch 维度。

### 代码


```python
'''整体模型'''
class Transformer(nn.Module):

    def __init__(self, config):
        super().__init__()
        # 必须输入词表大小和 block size
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = PositionalEncoding(config),
            drop = nn.Dropout(config.dropout),
            encoder = Encoder(config),
            decoder = Decoder(config),
        ))
        # 最后的线性层，输入是 n_embd，输出是词表大小
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # 初始化所有的权重
        self.apply(self._init_weights)

        # 查看所有参数的数量
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    '''统计所有参数的数量'''
    def get_num_params(self, non_embedding=False):
        # non_embedding: 是否统计 embedding 的参数
        n_params = sum(p.numel() for p in self.parameters())
        # 如果不统计 embedding 的参数，就减去
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    '''初始化权重'''
    def _init_weights(self, module):
        # 线性层和 Embedding 层初始化为正则分布
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    '''前向计算函数'''
    def forward(self, idx, targets=None):
        # 输入为 idx，维度为 (batch size, sequence length)；targets 为目标序列，用于计算 loss
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"不能计算该序列，该序列长度为 {t}, 最大序列长度只有 {self.config.block_size}"

        # 通过 self.transformer
        # 首先将输入 idx 通过 Embedding 层，得到维度为 (batch size, sequence length, n_embd)
        print("idx",idx.size())
        # 通过 Embedding 层得到的维度是 (batch size, sequence length, vocab_size, n_embd)，因此我们去掉倒数第二个维度
        tok_emb = self.transformer.wte(idx)
        print("tok_emb",tok_emb.size())
        # 然后通过位置编码
        pos_emb = self.transformer.wpe(tok_emb) 
        # 再进行 Dropout
        x = self.transformer.drop(pos_emb)
        # 然后通过 Encoder
        print("x after wpe:",x.size())
        enc_out = self.transformer.encoder(x)
        print("enc_out:",enc_out.size())
        # 再通过 Decoder
        x = self.transformer.decoder(x, enc_out)
        print("x after decoder:",x.size())

        if targets is not None:
            # 训练阶段，如果我们给了 targets，就计算 loss
            # 先通过最后的 Linear 层，得到维度为 (batch size, sequence length, vocab size)
            logits = self.lm_head(x)
            # 再跟 targets 计算交叉熵
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # 推理阶段，我们只需要 logits，loss 为 None
            # 取 -1 是只取序列中的最后一个作为输出
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    '''配置优化器'''
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # weight_decay: 权重衰减系数，learning_rate: 学习率，betas: AdamW 的 betas，device_type: 设备类型
        # 首先获取所有命名参数
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # 过滤掉不需要更新的参数
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # 参数根据维度分为两组。
        # 维度大于等于2的参数（通常是权重）会应用权重衰减，而维度小于2的参数（通常是偏置和层归一化参数）不会应用权重衰减。
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        # 打印一下参数数量
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"应用权重衰减的层数: {len(decay_params)}； 总参数量为：{num_decay_params:,}")
        print(f"不应用权重衰减的层数: {len(nodecay_params)}, 总参数量为：{num_nodecay_params:,}")
        # 检查 torch.optim.AdamW 是否支持融合版本（fused version），这是针对 CUDA 设备优化的版本。如果可用且 device_type 为 'cuda'，则使用融合版本。
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        # 创建优化器
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"是否使用 fused AdamW: {use_fused}")

        return optimizer

    '''进行推理'''
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        # 推理阶段，输入为 idx，维度为 (batch size, sequence length)，max_new_tokens 为最大生成的 token 数量即按序推理 max_new_tokens 次
        for _ in range(max_new_tokens):
            # 如果输入序列太长，我们需要将它截断到 block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # 前向计算，得到 logits，维度为 (batch size, sequence length, vocab size)
            logits, _ = self(idx_cond)
            # 使用最后一个 token 的 logits 作为当前输出，除以温度系数控制其多样性
            logits = logits[:, -1, :] / temperature
            # 如果使用 Top K 采样，将 logits 中除了 top_k 个元素的概率置为 0
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # 对输出结果进行 Softmax
            probs = F.softmax(logits, dim=-1)
            # 对结果概率进行采样
            idx_next = torch.multinomial(probs, num_samples=1)
            # 将输出结果拼接到输入序列后面，作为下一次的输入
            idx = torch.cat((idx, idx_next), dim=1)
            # print("idx:", idx)

        return idx
```



### mask

[举个例子讲下transformer的输入输出细节及其他](https://zhuanlan.zhihu.com/p/166608727)
1. 对于机器翻译来说，一个样本是由原始句子和翻译后的句子组成的。比如原始句子是： “我爱机器学习”，那么翻译后是 ’i love machine learning‘。 则该一个样本就是由“我爱机器学习”和 "i love machine learning" 组成。这个样本的原始句子的单词长度是length=4,即‘我’ ‘爱’ ‘机器’ ‘学习’。经过embedding后每个词的embedding向量是512。那么“我爱机器学习”这个句子的embedding后的维度是[4，512 ] （若是批量输入，则embedding后的维度是[batch, 4, 512]）。
2. padding。因为每个样本的原始句子的长度是不一样的，那么怎么能统一输入到encoder呢。此时padding操作登场了，假设样本中句子的最大长度是10，那么对于长度不足10的句子，需要补足到10个长度，shape就变为[10, 512], 补全的位置上的embedding数值自然就是0了。
3. Padding Mask。对于输入序列一般我们都要进行padding补齐，也就是说设定一个统一长度N，在较短的序列后面填充0到长度为N。对于那些补零的数据来说，我们的attention机制不应该把注意力放在这些位置上，所以我们需要进行一些处理。具体的做法是，把这些位置的值加上一个非常大的负数(负无穷)，（**KQ矩阵和它相加得到KQ_masked**）。这样经过softmax后，这些位置的权重就会接近0。Transformer的padding mask实际上是一个张量，每个值都是一个Boolean，值为false的地方就是要进行处理的地方。
4. 注意encoder的输出并没直接作为decoder的直接输入。训练的时候，
  1. 初始decoder的time step为1时(也就是第一次接收输入)，其输入为一个特殊的token，可能是目标序列开始的token(如<BOS>)，也可能是源序列结尾的token(如<EOS>)，也可能是其它视任务而定的输入等等，不同源码中可能有微小的差异，其目标则是预测翻译后的第1个单词(token)是什么；
  2. 然后<BOS>和预测出来的第1个单词一起，再次作为decoder的输入，得到第2个预测单词；
  3. 后续依此类推；
5. 样本：“我/爱/机器/学习”和 "i/ love /machine/ learning"
  1. 把“我/爱/机器/学习”embedding后输入到encoder里去，最后一层的encoder最终输出的outputs [10, 512]（假设我们采用的embedding长度为512，而且batch size = 1),此outputs 乘以新的参数矩阵，可以作为decoder里每一层用到的K和V；
  2. 将<bos>作为decoder的初始输入，将decoder的最大概率输出词 A1和‘i’做cross entropy计算error。
  3. 将<bos>，"i" 作为decoder的输入，将decoder的最大概率输出词 A2 和‘love’做cross entropy计算error。
  4. 将<bos>，"i"，"love" 作为decoder的输入，将decoder的最大概率输出词A3和'machine' 做cross entropy计算error。
  5. 将<bos>，"i"，"love "，"machine" 作为decoder的输入，将decoder最大概率输出词A4和‘learning’做cross entropy计算error。
  6. 将<bos>，"i"，"love "，"machine"，"learning" 作为decoder的输入，将decoder最大概率输出词A5和终止符</s>做cross entropy计算error。
6. 推理，比如用 '机器学习很有趣'当作测试样本，得到其英语翻译。这一句经过encoder后得到输出tensor，送入到decoder(并不是当作decoder的直接输入)：
  1. 然后用起始符<bos>当作decoder的 输入，得到输出 machine
  2. 用<bos> + machine 当作输入得到输出 learning
  3. 用 <bos> + machine + learning 当作输入得到is
  4. 用<bos> + machine + learning + is 当作输入得到interesting
  5. 用<bos> + machine + learning + is + interesting 当作输入得到 结束符号<eos>
  我们就得到了完整的翻译 'machine learning is interesting'，可以看到，在测试过程中，只能一个单词一个单词的进行输出，是串行进行的。

Sequence Mask， 上述训练过程是挨个单词串行进行的，那么能不能并行进行呢，当然可以。可以看到上述单个句子训练时候，输入到 decoder的分别是
```
<bos>
<bos>，"i"
<bos>，"i"，"love"
<bos>，"i"，"love "，"machine"
<bos>，"i"，"love "，"machine"，"learning"
```
那么为何不将这些输入组成矩阵，进行输入呢？怎么操作得到这个矩阵呢？将decoder在上述2-6步次的输入补全为一个完整的句子

```
<bos>，"i"，"love "，"machine"，"learning"
<bos>，"i"，"love "，"machine"，"learning"
<bos>，"i"，"love "，"machine"，"learning"
<bos>，"i"，"love "，"machine"，"learning"
<bos>，"i"，"love "，"machine"，"learning"
```
```
1 0 0 0 0
1 1 0 0 0
1 1 1 0 0
1 1 1 1 0
1 1 1 1 1 
```
然后将上述矩阵矩阵乘以一个 mask矩阵就得到了。
```
<bos>，MASK 【MASK】  【MASK】   【MASK】
<bos>，"i"，【MASK】  【MASK】   【MASK】
<bos>，"i"，"love "， 【MASK】   【MASK】
<bos>，"i"，"love "，"machine"， 【MASK】
<bos>，"i"，"love "，"machine"，"learning"
```
这个mask矩阵就是 sequence mask，其实它和encoder中的padding mask 异曲同工。这样将这个矩阵输入到decoder（其实你可以想一下，此时这个矩阵是不是类似于批处理，矩阵的每行是一个样本，只是每行的样本长度不一样，每行输入后最终得到一个输出概率分布，作为矩阵输入的话一下可以得到5个输出概率分布）。这样我们就可以进行并行计算进行训练了。

![](/public/upload/machine/causal_mask.jpg)

![](/public/upload/machine/decoder_mask.jpg)

## 其它

[没有思考过 Embedding，不足以谈 AI](https://mp.weixin.qq.com/s/7kPxUj2TN2pF9sV06Pd13Q)推理的核心是transformer，transformer的核心是attention机制，attention机制是什么？一言以蔽之：计算词义向量之间的“距离”后 ，对距离近的词投向更多注意力，**而收到高注意力的词义则获得更高的激活值**，当预测完成后，通过反向传播算法：当特定的激活帮助了最终的预测，对应词之间关联将被强化，反之则被弱化，**模型便是通过这一方式学到了词之间的关系**。而在“Distribution Hypothesis”（一个词的意义，可以被它所出现的上下文定义。这句话换一种说法又可以表述为：上下文相似的词在词义上也一定存在相似性）这一视角下，“认字”的实质就是认识一个词和其它词之间的关系。于是就形成了认字为了背书，背书帮助认字的结构。这里提炼一个我个人的观点：attention 机制之所以重要和好用，原因之一是可以有效帮助词义向量（embedding）聚类。GPT的例子想想其实很有趣，一般的工程思维是将大的问题拆成多个小的问题然后一个一个解决，正如文中开始说的那句：让计算机理解自然语言，我们需要做什么？计算的基础是数，而自然语言是文字，因此很容易想到要做的第一步是让文字数字化…这个表述隐含了一个解决问题的路径：先将文字数字化后，再考虑理解句子的问题。有趣的地方是：对词进行向量化编码的最好方法，是直接训练一个理解句子的语言模型；这就像为了让婴儿学会走路，我们直接从跑步开始训练。人类会摔跤会受伤，但机器不会 —— 至少在embodied之前不会，因此人类为了降低代价所建立的步骤化学习过程或许并不适合人工智能 —— 也不难发现，深度学习中，许多好的解决方案往往都是一步到位的。

[transformer架构的核心公式其实类似于数学期望，理解起来也不复杂，但为什么这个模型这么强呢？ - SamZhang的回答 - 知乎](https://www.zhihu.com/question/580810624/answer/2977786600)Transformer模型之所以这么强大,主要有以下几个原因:
1. Transformer使用Attention来建模序列之间的依赖关系,这比RNN等其他模型更加灵活和有效。Attention可以自动学习序列元素之间的重要性和依赖,不需要像RNN那样依赖固定的拓扑结构。
2. Transformer的Attention和Feed Forward网络等都是并行计算,所以Transformer在计算速度和效率上都超过RNN。这使其能够实现较长序列的建模。
3. 多头Attention。Transformer使用了Multi-Head Attention,即多个Attention对不同的表示子空间进行注意力计算。这使得Attention能够同时考虑不同的语义关系,如上下义、定语等关系。这进一步提高了模型的表征能力。
4. Layer Normalization。Transformer使用Layer Normalization代替Batch Normalization。这使得模型对输入的依赖性降低,对输入的变化更加鲁棒。这是Transformer能处理较长序列的另一个重要原因。
5. 可重复利用的模块。Transformer由Attention、Layer Normalization和Feed Forward Neural Network等可重复使用的模块构成。这使得Transformer既深又宽,能够实现很强的表征能力。
6. 损失函数。Transformer使用成对的softmax损失函数,这促使模型同时学习两个序列的表征,这比序列续生成等其他技术更加有效。

鲁提辖：要理解Transformer就是搞懂Attention，再就是Positional Encoding。Attention主要负责对长程依赖内容依赖建模，Positional Encoding负责对短程依赖和位置依赖建模，两者合力就有了Transformer强大的拟合能力。

大模型的Embedding层和独立的Embedding模型有什么区别？Embedding层是神经网络中的一层，用于将离散的符号（如单词）映射到连续的向量空间。
1. 独立Embedding模型的特点，独立的Embedding模型通常以特定的损失函数（如Skip-gram或CBOW）单独训练，如最小化上下文中词汇的预测误差，目标是学习到通用的词向量表示。训练完成后，可以在多个任务中复用，适用于各种下游任务，如文本分类、情感分析等。
2. 在大模型中，Embedding层作为模型的第一层，用于将输入的词汇序列映射到向量空间。通常与Transformer Blocks等其他层一起训练。由于Embedding层与整个模型的训练目标（如提高下一个token的预测准确率）一致，与模型的其他部分共享一个统一的损失函数（通常是提高下一个token预测的准确率），这意味着Embedding层并非为通用目的设计，而是为优化特定任务（如语言模型任务）而定制。这种联合训练方式使得Embedding层不仅仅是词汇表示的工具，更是为整个模型的任务优化服务的组件。


[五年时间被引用3.8万次，Transformer宇宙发展成了这样](https://mp.weixin.qq.com/s/cVuBfrrtGBpNlZUekxgnmg) 未读。 





