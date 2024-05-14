---

layout: post
title: 从Attention到Transformer
category: 架构
tags: MachineLearning
keywords:  rnn attention transformer

---

<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

## 简介

* TOC
{:toc}

## Attention is all you need——Transformer

[李沐读 Transformer 视频笔记](https://zhuanlan.zhihu.com/p/619234992) 李沐对于 Transformer论文的讲解，还是看作者的第一手材料+大佬解说最准确。未读。

Transformer模型采用的也是编码器-解码器架构，但是在该模型中，编码器和解码器不再是 RNN结构，取而代之的是编码器栈（encoder stack）和解码器栈（decoder stack）（注：所谓的“栈”就是将同一结构重复多次，“stack”翻译为“堆叠”更为合适）。

![](/public/upload/machine/transformer.jpg)

[Transformer - Attention is all you need](https://zhuanlan.zhihu.com/p/311156298)Encoder层和Decoder层内部结构如下图所示。

![](/public/upload/machine/transformer_internal.jpg)

1. Encoder具有两层结构，self-attention和前馈神经网络。self-attention计算句子中的每个词都和其他词的关联，从而帮助模型更好地理解上下文语义，引入Muti-Head attention后，每个头关注句子的不同位置，增强了Attention机制关注句子内部单词之间作用的表达能力。前馈神经网络为encoder引入非线性变换，增强了模型的拟合能力。本质上，自注意力机制使得模型能理解语句中不同单词间的关系。而且**跟以往按固定顺序处理单词**的传统模型不同，transformers 其实是同时检查所有单词，并根据每个词跟句中其他词之间的相关性，为各词分配所谓“注意力得分”指标。
2. 跟编码器类似，解码器同样由多层自注意力加前馈神经网络组成。解码器的第一个注意力层只计算已翻译输入的注意力，第二个注意力层则使用了编码器的输出，保证解码器在生成输出时考虑到来自输入序列的相关信息。 解码器中第一个注意力层的注意力掩码可用于编码器或解码器，来保证不会对一些特殊token计算注意力。例如计算批量文本的时候，由于文本长度不一致，需要进行补齐操作（那些补齐token是没有意义的），那么使用注意力掩码可以防止计算注意力时也把补齐字符考虑进去。

Transformer 模型本来是为了翻译任务而设计的。在训练过程中，Encoder 接受源语言的句子作为输入，而 Decoder 则接受目标语言的翻译作为输入。在 Encoder 中，由于翻译一个词语需要依赖于上下文，因此注意力层可以访问句子中的所有词语；而 Decoder 是顺序地进行解码，在生成每个词语时，注意力层只能访问前面已经生成的单词。例如，假设翻译模型当前已经预测出了三个词语，我们会把这三个词语作为输入送入 Decoder，然后 Decoder 结合 Encoder 所有的源语言输入来预测第四个词语。每一个部分都有公式对应。

想要进一步了解 Transformer 这一架构的原理，强烈推荐阅读 Jay Alammar 的博客。从为了解决翻译问题时 Seq2Seq 模型的提出的 Attention 的基本概念 https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention，到 Transformer 架构中完全抛弃 RNN https://www.youtube.com/watch?v=UNmqTiOnRfg 提出的 Attention is All You Need http://jalammar.github.io/illustrated-transformer，到 GPT-2 和 GPT-3 的架构解读http://jalammar.github.io/illustrated-gpt2 http://jalammar.github.io/how-gpt3-works-visualizations-animations，Jay Alammar 的博客都提供了精彩的可视化配图便于理解模型结构。

![](/public/upload/machine/transformer.png)

![](/public/upload/machine/encoder_decoder_matrix.jpg)

## self-attention 机制

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

1. $XX^T$代表什么？一个矩阵（一个句子）乘以它自己的转置，会得到什么结果，有什么意义？我们知道，矩阵可以看作由一些向量组成，**一个矩阵乘以它自己转置的运算，其实可以看成这些向量分别与其他向量计算内积**（一个句子的各个词向量的内积）。向量的内积，其几何意义是什么？表征两个向量的夹角，表征一个向量在另一个向量上的投影。**投影的值大，说明两个向量相关度高**（两个词相关度越高）。如果两个向量夹角是九十度，那么这两个向量线性无关，完全没有相关性！更进一步，这个向量是词向量，是词在高维空间的数值映射。词向量之间相关度高表示什么？是不是在一定程度上（不是完全）表示，在关注词A的时候，应当给予词B更多的关注？
  ![](/public/upload/machine/self_attention_xxt.jpg)
2. Softmax操作的意义是什么呢？归一化。 Softmax之后，这些数字的和为1了。当我们关注"早"这个字的时候，我们应当分配0.4的注意力给它本身，剩下0.4关注"上"，0.2关注"好"。
  ![](/public/upload/machine/self_attention_softmax_xxt.jpg)
3. 最后一个 X 有什么意义？ $softmax(XX^T)X$表示什么？我们取 $softmax(XX^T)X$
 的一个行向量举例。这一行向量与X的一个列向量相乘，表示什么？在新的向量中，每一个维度的数值都是由三个词向量在这一维度的数值加权求和得来的，**这个新的行向量就是"早"字词向量经过注意力机制加权求和之后的表示**。PS：变换后词向量的长度不变，结合了上下文的动态向量来表示这个词。 
  ![](/public/upload/machine/self_attention_softmax_xxt_x.jpg)
  一张更形象的图是这样的，图中右半部分的颜色深浅，其实就是我们上图中黄色向量中数值的大小，意义就是单词之间的相关度
  ![](/public/upload/machine/self_attention_softmax_xxt_x_matrix.jpg)
4. $\sqrt{d_k}$的意义。假设Q,K里的元素的均值为0，方差为1，那么$A^T=Q^TK$中元素的均值为0，方差为d。当d变得很大时，A中的元素的方差也会变得很大，如果A中的元素方差很大，那么softmax(A)的分布会趋于陡峭(分布的方差大，分布集中在绝对值大的区域)。总结一下就是softmax(A)的分布会和d有关。因此A中每一个元素除以$\sqrt{d_k}$后，方差又变为1。这使得softmax(A)的分布“陡峭”程度与d解耦，从而使得训练过程中梯度值保持稳定。


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

这个过程可以简单概括为 3 步：
1. 衡量QK 之间的相似性，这种相似性对于向量来说就是内积。$\sqrt{d_k}$ 作为调节因子，使得内积不至于太大，从而使梯度稳定很多。
2. 查找权重 a。这是通过 SoftMax 完成的。相似性像一个完全连接的层一样连接到权重。SoftMax 之后的注意力分数，其分值大小代表了相关性强弱，这种差异在计算梯度时，可以相对均匀地流入多个token位置。
3. 值的加权组合。将 a 的值和 V 对应的值相乘累加，最终输出是所需的结果注意力值。


## MultiHeadAttention

自注意力机制的方式确实解决了“传统序列模型在编码过程中都需顺序进行的弊端”的问题，有了自注意力机制后，仅仅只需要对原始输入进行几次矩阵变换便能够得到最终包含有不同位置注意力信息的编码向量。**模型在对当前位置的信息进行编码时，会过度的将注意力集中于自身的位置**（虽然每个编码都在z1中有或多或少的体现），因此作者提出了通过多头注意力机制来解决这一问题。同时，使用多头注意力机制还能够给予注意力层的输出包含有不同子空间中的编码表示信息，从而增强模型的表达能力（这些集合中的每一个W都是随机初始化的，在训练之后，每个集合都被用来将输入词嵌入(或来自较低编码器/解码器的向量)投影到不同的表示子空间中）。

$$
head_i = Attention(QW{_i}{^Q},KW{_i}{^K},VW{_i}{^V})
$$
$$
MultiHead(Q,K,V) = Contact(head_1,head_2,...head_h)W^O
$$

不仅仅只初始化一组Q、K、V的矩阵，而是初始化多组，tranformer是使用了8组，所以最后得到的结果是8个矩阵。

![](/public/upload/machine/self_attention_multi_headed.jpg)

![](/public/upload/machine/self_attention_multi_z.jpg)

以h=2为例（有几个head 是一个超参数），根据输入序列X和$W_1^Q,W_1^K,W_1^V$ 我们就计算得到了$Q_1,K_1,V_1$，进一步根据公式1就得到了单个自注意力模块的输出$Z_1$；同理，根据X和$W_2^Q,W_2^K,W_2^V$就得到了另外一个自注意力模块输出$Z_2$。最后，前馈层不需要8个矩阵，它只需要一个矩阵(由每一个单词的表示向量组成)。所以我们需要一种方法**把这八个矩阵压缩成一个矩阵**。那该怎么做？其实可以直接把这些矩阵**拼接**在一起，传入一个Linear层（Linear层，全连接的神经网络层的映射是非线性变换，它的作用是对输入进行维度变换和特征提取。 线性变换只能进行简单的比例缩放和平移操作，而非线性变换可以引入更多的复杂性，例如曲线形状、峰值和谷底等。这样可以使模型更加灵活，能够更好地适应不同类型的数据分布，从而增加模型的表达能力），得到 Multi-Head Attention 最终的输出Z。根据公式2将$Z_1,Z_2$水平堆叠形成Z，然后再用Z乘以$W^O$便得到了整个多头注意力层的输出。

![](/public/upload/machine/multi_head_cal.jpg)

把所有的矩阵放到一张图内看一下总体的流程。

![](/public/upload/machine/multi_head_attention.jpg)

既然我们已经摸到了注意力机制的这么多“头”，那么让我们重温之前的例子，看看我们在例句中编码“it”一词时，不同的注意力“头”集中在哪里：

![](/public/upload/machine/multi_head_example.jpg)

当我们编码“it”一词时，一个注意力头集中在“animal”上，而另一个则集中在“tired”上，从某种意义上说，模型对“it”一词的表达在某种程度上是“animal”和“tired”的代表。

权重矩阵$W^Q$,$W^K$,$W^V$是与 Multi-head Attention 多头注意力机制相关的，它们的形状并非是`n*512`，而是`n*64`，即把一个进程分为 8 个并列的进程来实施（512÷8=64)。这个进程被原论文称为“头 head”，这 8 个头之间互不干扰，各自运算各自的 Attention 机制。8 个头中的每一个头都只采用初始 Embedding 的向量长度 512 的 8 分之一来运行，即通过把 Embedding 向量与$W^Q$,$W^K$,$W^V$三矩阵分别相乘之后，得到的向量长度为 Embedding 原始向量长度的 1/8，即 64。这相当于把 Embedding 向量作线性变换的同时，顺便被切成了 8 分来运行。当然，这样的“切”并不是直接在一个长度为 512 的向量上等分 8 份，而是通过与$W^Q$,$W^K$,$W^V$三矩阵分别相乘，线性变换而来的。这样的降维变换明显含有某种特殊的意义，好比龙生九子，各有所好。PS：多个head加在一起后，还是原来输入向量的长度。

多头注意力机制类似于赛马机制，每个编码器/解码器使用8个“头”（可以理解为8个互不干扰自的注意力机制运算），每一组的Q/K/V都不相同。然后，得到8个不同的权重矩阵Z，每个权重矩阵被用来将输入向量投射到不同的表示子空间。它有助于减少模型初始化的随机性对模型效果的影响。所以即使只留下一个注意力头也能使用，但这会导致模型的稳定性和多样性无法得到保障，进而造成模型的性能下降。

[图解Transformer多头注意力机制](https://mp.weixin.qq.com/s/Aefek7zCftt4oXLWoKN57A)
![](/public/upload/machine/end_to_end_multi_head.jpg)
PS: 经过注意力层，输入向量和输出向量的shape是一致的。head的概念类似卷积中的通道，只不过每个通道的输入都是一样的，类似于把一个通道的数据复制多次。多头注意力的计算过程类似深度可分离卷积，把通道分开计算，再融合到一起

## 位置编码/Positional Encoding

Transformer提出了深度学习领域既MLP、CNN、RNN后的第4大特征提取器。一个好的特征提取器需要自带输入处理模块的前后顺序信息，而Attention机制并没有考虑先后顺序信息（源于注意力机制固有的排列不变性，因此修改tokens的顺序不会改变输出加权值。因此，注意机制本身缺乏对token顺序的意识），但前后顺序信息对语义影响很大，因此需要通过位置嵌入这种方式把前后位置信息加在输入的Embedding上。

对self-attention来说，它跟每一个input vector都做attention，在计算注意力矩阵时每个 token 都与其他所有 token 交互，没有考虑到input sequence的顺序。更通俗来讲，大家可以发现我们前文的计算每一个词向量都与其他词向量计算内积，得到的结果丢失了我们原来文本的顺序信息。打乱词向量的顺序，得到的结果仍然是相同的。但实际上位置信息很有用，比如动词出现在句首的概率就比名词低，所以要把位置信息塞到学习过程中去。 

Positional Embeddings 基于一个简单但有效的想法：使用与位置相关的值模式来增强词向量。为了在序列中编码位置信息，transformer的设计者使用了不同频率的正弦函数。他们还尝试了习得的位置Embeddings，但结果并没有什么不同。PS： 波形相加，不同的波。

位置信息编码位于encoder和decoder的embedding之后，每个block之前。

如果预训练数据集足够大，那么最简单的方法就是让模型自动学习位置嵌入。 除此以外，Positional Embeddings 还有一些替代方案：
1. Learned Positional Embedding ，这个是绝对位置编码，即直接对不同的位置随机初始化一个postion embedding，这个postion embedding作为参数进行训练。缺点：
  1. 不同位置对应的positional embedding固然不同，但是位置1和位置2的距离比位置3和位置10的距离更近，这些关于位置的相对含义，模型能够通过绝对位置编码参数学习到吗？
  2. 位置之间没有约束关系，我们只能期待它隐式地学到，是否有更合理的方法能够显示的让模型理解位置的相对关系呢？
2. Sinusoidal Position Embedding ，相对位置编码，即三角函数编码。由于正弦函数能够表达相对位置信息，那么对每个positional embedding进行 sin 或者cos激活，好处是位置 i 处的单词的psotional embedding可以被位置 i+k 处单词的psotional embedding线性表示，反应两处单词的其相对位置关系。此外位置i和i+k的psotional embedding内积会随着相对位置的递增而减小，从而表征位置的相对距离。缺点：由于距离的对称性，Sinusoidal Position Encoding虽然能够反映相对位置的距离关系，但是无法区分i和i+j的方向。

transformer中，模型输入encoder的每个token向量由两部分加和而成：Position Encoding, Input Embedding。Positional Embedding的成分直接叠加于Embedding之上，使得每个token的位置信息和它的语义信息(embedding)充分融合，并被传递到后续所有经过复杂变换的序列表达中去。

![](/public/upload/machine/positional_encoding.jpg)

为什么 RNN 不需要位置编码呢？RNN 是串行处理输入数据的，所以每个 RNN 单元其实都包含了所在位置之前的全部输入信息，所以对当前位置输入的处理是有状态的。但是，Transformer 会并行地处理所有输入的内容，所以各个并行单元会无状态地处理每个输入。因此，我们需要在最开始就给每个输入的嵌入向量一个位置编号，这样模型才能通过输入判断它在整体中的位置。

[transformer位置编码如何去理解？ - magicwt的回答 - 知乎](https://www.zhihu.com/question/633536226/answer/3319109019) 基于论文解释了3个与原因，建议细读。token位置编码后输出的向量维度也是$d_model$和token通过Embedding层后输出的向量维度相同，这样便于两者直接相加作为编码器和解码器层的输入。论文中采用的方法是正余弦函数， 论文中解释采用这种位置编码的原因
1. 可以将将长短不一的句子中的token位置编码为固定维度的向量
2. 可以将离散的位置值转化为连续值
3. 可以借助正余弦函数的特性，对于某个偏移量k ，可以基于$PE_{pos}$通过线性变换快速计算出相对位置的 $PE_{pos+k}$

##  Layer Normalization

这些都是用于Normalization激活的技术，可以加速学习，提高模型的性能。

在每个block中，最后出现的是Layer Normalization，其作用是规范优化空间，加速收敛。当我们使用梯度下降算法做优化时，我们可能会对输入数据进行归一化，但是经过网络层作用后，我们的数据已经不是归一化的了。随着网络层数的增加，数据分布不断发生变化，偏差越来越大，导致我们不得不使用更小的学习率来稳定梯度。Layer Normalization 的作用就是保证数据特征分布的稳定性，将数据标准化到ReLU激活函数的作用区域，可以使得激活函数更好的发挥作用

Normalization有两种方法，Batch Normalization和Layer Normalization。关于两者区别不再详述。

一般情况下，输入是一个矩阵，然后矩阵的 每一行是一个样本，多个行（多个样本）是 一个 batch，每一列是一个特征，多个列是 feature。batchnorm 是说每一次，去把每一个列，就是每一个特征，把它在一个小 mini-batch 里面，每列 的均值变成 0 方差变成 1。 [Batch Norm详解之原理及为什么神经网络需要它](https://zhuanlan.zhihu.com/p/441573901)

**在 Transformer 里面，或者说正常的 RNN 里面，它的输入是一个三维的矩阵**。因为 输入的是一个序列的样本，即每一个样本里面有很多个元素。一个序列，如：一个句子里面有 n 个词，所以每个词表示为一个向量的话，还有一个 batch 维度，那么就是个 3D 的输入。列不再是特征，而是序列的长度，对每一个 sequence 就是 每个词，每个词有自己对应的向量。如果是 layernorm 的话，那么就是对每个样本切一下（横着切一下）。为什么 layer norm 用的多一点？一个原因是：在 时序的序列模型 里面，每个样本的长度可能会发生变化。那些不够 sequence 的长度 n 的样本 ，一般是补 0。 layernorm 是对 每个样本来做，所以不管样本是长还是短，反正算均值是在样本自己算的，这样的话相对来说它稳定一些。

## Position-wise Feed Forward

1. 每一层经过attention之后，还会有一个FFN，这个FFN的作用就是空间变换。FFN包含了2层linear transformation层，中间的激活函数是ReLu。
2. attention层的output最后会和相乘，为什么这里又要增加一个2层的FFN网络？**Attention内部就是对特征向量V加权平均的过程。只用self-Attention搭建的网络结构就只有线性表达能力**。FFN的加入引入了非线性(ReLu激活函数)，**变换了attention output的空间**, 从而增加了模型的表现能力。把FFN去掉模型也是可以用的，但是效果差了很多。
$$
FFN(x) = max(0,xW_1+b1)W_2+b_2
$$
2. 前馈神经网络的输入是self-attention的输出，是一个矩阵（即上图Z），矩阵的维度是（序列长度×D词向量），之后前馈神经网络的输出也是同样的维度。self-attention + 前馈神经网络 就是一个小编码器的内部构造了，一个大的编码部分就是将这个过程重复了6次，最终得到整个编码部分的输出。为了解决梯度消失的问题，在Encoders和Decoder中都是用了残差神经网络的结构，即每一个前馈神经网络的输入不光包含上述self-attention的输出Z，还包含最原始的输入。

其它

1. 大语言模型中的参数都用在哪了？可以看到在百亿参数量以上，差不多三分之二的参数实际上是 FFN 参数，剩下的基本都是 attention 参数。所以虽然论文名叫 attention is all you need，但实际上 FFN 仍然起到了很重要的作用。
2. transformer 中最重要的是self-attention，self-attention 由三个线性矩阵Q、K、V 决定，如果我们把Q、K矩阵设置为零，那么self-attention 就变成了FFN，$Z_0 =V_0*X$，也就是说，FFN是self-attention 的一个特例，FFN能表达的逻辑，self-attention 也可以，但反过来却不成立。至于transformer 中的FFN部分，当初设计是为了输入输出维度的对齐，毕竟多注意力的输出 $W_0$的维度比输入X维度高很多。但如果一定要用FFN去表达self-attention的逻辑，也是可以的，但需要的参数量却要大很多，感兴趣的可以去试验一下，用FFN去拟合self-attention 的逻辑。就好像乘法能计算的东西，单纯用加法依然可以做到，但效率要低很多，self-attention就是乘法，FFN就是加法。


## Residual Network 残差网络

残差网络是深度学习中一个重要概念。在神经网络可以收敛的前提下，随着网络深度的增加，网络表现先是逐渐增加至饱和，然后迅速下降，这就是我们经常讨论的网络退化问题。事实上，现在几乎不可能看到一个不使用残差连接的神经网络模型。残差连接缓解了梯度不稳定问题，有助于模型更快收敛。

在神经网络中，每个层通常由一个非线性变换函数和一个线性变换函数组成，非线性变换函数通常由激活函数，例如ReLU、Sigmoid、Tanh等实现，而线性变换函数则通常由矩阵乘法实现。在传统的神经网络中，这些变换函数直接作用于输入数据，然后传递到下一层。而在使用残差连接的神经网络中，每个层都添加了一个跨层连接，可以将输入数据直接连接到输出数据，也可以将输入数据直接传递到后续层次，从而提高信息的传递效率和网络的训练速度。同时，残差连接还可以解决梯度消失和梯度爆炸的问题，从而提供网络的性能和稳定性。

在transformer模型中，encoder和decoder各有6层，为了使当模型中的层数较深时仍然能得到较好的训练效果，模型中引入了残差网络，**将输入和输出直接相加**。

![](/public/upload/machine/residual_network.jpg)

## Linear & Softmax

**softmax用于多分类过程中**，它将多个神经元的输出，映射到（0,1）区间内，可以看成概率来理解，从而来进行多分类！假设我们有一个数组，V，$V_i$表示V中的第i个元素，那么$V_i$的softmax值就是 

$$
S_i = \frac{e^{V_i}}{\sum_j e^{V_j}}
$$

如下图表示：

![](/public/upload/machine/softmax_example.jpg)

softmax直白来说就是将原来输出是$z_1=3,z_2=1,z_3=-3$通过softmax函数一作用，就映射成为(0,1)的值，而**这些值的累和为1**（满足概率的性质），那么我们就可以将它理解成概率，在最后选取输出结点的时候，我们就可以选取概率最大（也就是值对应最大的$z_1$）结点，作为我们的预测目标！


Decoder最后是一个线性变换和softmax层。解码组件最后会输出一个实数向量。我们如何把浮点数变成一个单词？这便是线性变换层要做的工作，它之后就是Softmax层。
线性变换层是一个简单的全连接神经网络，它可以把解码组件产生的向量投射到一个比它大得多的、被称作对数几率（logits）的向量里。不妨假设我们的模型从训练集中学习一万个不同的英语单词（我们模型的“输出词表”）。因此对数几率向量为一万个单元格长度的向量——每个单元格对应某一个单词的分数（相当于做vocaburary_size大小的分类）。接下来的Softmax 层便会把那些分数变成概率（都为正数、上限1.0）。概率最高的单元格被选中，并且它对应的单词被作为这个时间步的输出。


## 其它视角

[逐步理解Transformers的数学原理](https://mp.weixin.qq.com/s/b9YHoCOp5Pu5kfyeTdoBxw) 可以看看从字符串到输出向量 到各层的变换过程。

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

NLP 神经网络模型的本质就是对输入文本进行编码，常规的做法是首先对句子进行分词，然后将每个词语 (token) 都转化为对应的词向量 (token embeddings)，这样文本就转换为一个由词语向量组成的矩阵 $X=(x_1,x_2,...,x_n)$，其中$x_i$就表示第i个词语的词向量。

以将$x_t$ 编码为 $y_t$ 的视角来理解（结合了上下文词表示）
1. RNN（例如 LSTM）的方案很简单，每一个词语$x_t$对应的编码结果$y_t$通过递归地计算得到：$y_t=f(y_{t-1},xt)$，RNN 本质是一个马尔科夫决策过程，难以学习到全局的结构信息；
2. CNN 则通过滑动窗口基于局部上下文来编码文本，例如核尺寸为 3 的卷积操作就是使用每一个词自身以及前一个和后一个词来生成嵌入式表示：$y_t=f(x_{t-1},x_{t},x_{t+1})$，由于是通过窗口来进行编码，所以更侧重于捕获局部信息，难以建模长距离的语义依赖。
3. 直接使用 Attention 机制编码整个文本。相比 RNN 要逐步递归才能获得全局信息（因此一般使用双向 RNN），而 CNN 实际只能获取局部信息，需要通过层叠来增大感受野，Attention 机制一步到位获取了全局信息：$y_t=f(x_t,A,B)$，其中 A,B是另外的词语序列（矩阵），如果取A=B=X就称为 Self-Attention，即直接将$x_t$与自身序列中的每个词语进行比较$y_t=f(x_{0},x_{t},x_{...})$，最后算出$y_t$。如对于单词I，它的新表征L(I)就是其它所有单词Value的加权和。所以仅仅只看这一点，**I的新表征中就能用到其它所有单词的信息。因此，没有距离的概念，也就没有长依赖的问题**。PS：在深度学习里，拟合$y_t$ 和$x_{0},x_{t},x_{...}$ 关系，就使用W将x串起来，学习/优化W，直接$y_t=W_0x_{0}+W_tx_{t}+W_{...}x_{...}$太粗暴，整出来QKV。




[如何理解 Transformer 中的自注意力机制？ - 宇文树雪的回答 - 知乎](https://www.zhihu.com/question/560879732/answer/3041597591)注意力机制中的“注意力”其实指的是序列中每两个元素之间的相关性程度，更进一步说，**这种相关性就是指两个元素在自然语句中同时出现的概率**。那么，序列元素之间的相关性是如何衡量的呢？在transformer的输入之前，往往要做词嵌入转换，把输入的文本序列转为降维到固定长度的矢量。在这个矢量空间中，其实就通过跳元模型或连续词包模型把词元之间的共现关系学习到一个高维的空间中了。通过**直接进行矢量乘法就能知道两个嵌入词向量的相似度**。词元矢量内积越大，说明相关性越强，在词嵌入空间中越近，更可能组成语句。也就是说，我们能基于词元矢量内积判断输入序列任意两个元素的共现的概率，也就是组成语句的可能性。（词嵌入原始的计算方式就是统计词元之间共同出现的频次，然后对共现矩阵做了奇异值分解得到降维后的词向量表示。）**既然词嵌入能学习到语义，为什么还要进行更复杂的训练呢？**这是因为嵌入词是在较小的窗口训练成的，不能反映大规模语句的上下文结构特征。因此，需要加一个权重对序列中每个嵌入词的重要性进行调整，这个权重需要要通过大量的语料训练得到。从而学习到自然语言序列固有的结构，并形成语言模型。

[白话科普：Transformer和注意力机制](https://mp.weixin.qq.com/s/jyy7WXtOqJPXJYssPpfiUA)从Transformer整体来看，Encoder负责将输入序列（通常是自然语言的）变换成一个「最佳的」**内部表示**；而Decoder则负责将这个「内部表示」变换成最终想要的目标序列（通常也是自然语言的）。现在，我们先来看一下Encoder，它其实是由多个网络层组成的。输入序列进入Encoder之后，会经过多个Encoder Layer。每经过一层，相当于输入序列中的每个token进行了一次向量变换（非线性的），也就离那个「最佳的」内部表示又接近了一步。但是，每次变换都不改变向量的维度数量。每个Encoder Layer到底做了什么呢？这里面关键的一个机制是自注意力 (self-attention)。为什么需要自注意力呢？在模型内部，每个token都是用一个多维向量来表示的。向量的值决定了这个token在多维空间中的位置，也决定了它所代表的真实含义。一个token的真实含义，不仅仅取决于它自身，还取决于句子中的其它上下文信息（来自其它token的信息）。而借助向量，就可以用数量关系来描述这些现象了：相当于是说，**一个token的向量值，需要从句子上下文中的其他token中「吸收」信息，在数学上可以表达为所有token的向量值的加权平均**。这些权重值，我们可以称之为注意力权重 (attention weights)。在Encoder中，每经过一层Encoder Layer，一个token都会「参考」上一层的所有token，并根据对它们注意力权重的不同，决定「携带」它们中多少量的信息进来。对于这一过程，一个最简化的说法可以表达为：一个token会注意到 (attend to)所有其他token。对于Decoder这里的自注意力来说，生成的过程需要遵循因果关系。也就是说，生成下一个token的时候，它必须只能注意到 (attend to) 之前已经生成的token；所以，对于已经生成的序列来说，Decoder Layer对这个序列进行处理的时候，序列中每个token也都应该保持跟生成时一样的逻辑，即它只能注意到 (attend to) 在它之前的token。在计算上需要构建一个mask矩阵。而Encoder中的自注意力却允许序列中的每个token都可以注意到 (attend to) 所有的token（包括在它之前和它之后的）。

### 与MLP对比

特征矩阵X每一行代表一个 item，以每一列代表一个 feature。通常我们所说的线性层都是 $X^{t+1}=X^tW^t+B^t$，也就是在 feature 这个维度上作线性变换，完全没有考虑 item 与 item 之间的关系。这也是为什么 MLP 会比不过什么 CNN、attention之类的原因，在 MLP 的框架里每个 item 的运算都是独立的，不会与其他 item 产生交互。用代数的话来说，就是在矩阵乘法中，左乘的矩阵行其向量之间是不会产生交互的，相应的右乘的矩阵其列向量之间是不会产生交互的。

那要怎么做到 item 与 item 之间的交互呢？很简单，把线性层里面的右乘一个权重矩阵变成左乘就行了，即$X^{t+1}=W^tX^t+B^t$，由于这也可以看成是把原来的输入取了转置以后再扔进通常的线性层，得到输出后再做一个转置，我们就把这样的层叫做转置线性层。为了方便我们就不要最后的偏差项了，拿每一个 item 来看，$X^{t+1}_{i}=W_i^tX^t=\sum_jW_{ij}^tX_j^t$

这个输出的形式和 attention 可以说是一模一样了。这里拿最常见的 Attention 举例，即

$$
Attention(Q,K,V) = softmax(\frac{QK^T}{\sqrt{d_k}})V
$$

其中V就等价于$X^t$，而$softmax(\frac{QK^T}{\sqrt{d_k}})$就是我们想要学到的$W^t$（所以说self-attention 的权重是根据输入的改变和改变的，而CNN中一旦训练完成，各个通道的卷积核参数就固定了）， attention 多做了一些什么呢？对于要学习的参数矩阵$W^t$增加了低秩的约束。参考 self-attention 的情况，即输出和输入的 size 保持一致，那么理论上没有任何约束的情况下$W^t$可以是一个满秩的n*n方阵。但是，在 self-attention 中，假设$Q,K\in\R^{n*d}$那么$softmax(\frac{QK^T}{\sqrt{d_k}})$就是一个秩至多为d的方阵。秩上的降低也意味着表征能力的下降，但好处在于$\mathcal{O}(n^2) $变成了$\mathcal{O}(nd) $， $d\ll n$ 的时候这样做的价值就体现出来了。比如原来比如GPT2 一个token向量 是768维，context若是4k，input item数量是768*4096，所以一定是预处理一下再接入mlp；cnn类似，一张1024*768的图片 每个pixel 作为一个item，weight就不小了。可以说attention 是一种为了压缩参数量而牺牲一些表征能力，是用类似于 matrix fatorization 的方式来近似转置线性层的做法。所以有人提出什么 **MLP is all you need** 之类的说法，用 MLP 替代 CNN 或者 attention 也能取得不俗的图像分类效果……那可不是必须的嘛？人家那可是加了各种约束，为了压缩参数量牺牲了表征能力来近似你这个线性层的结果啊。PS：大部分模型最后的部分都是mlp，所以可以视为前半部分的处理是在缩小input 规模以便接入mlp。

大可：整个 transformer 的重点在 QKV 结构上。以前的 CNN 试图通过卷积来表达不同位置数值之间的关系，学习卷积值也就是学习矩阵里的数值之间的特征，所以适合用在图像里面。因为图像就是一个个的像素点形成的矩阵。RNN 试图通过加入反馈机制来理解一串数值前后的关系，所以适用于语言模型，因为这些数值之间有前后关系，像我们的句子里有先后逻辑。而 transformer 里的 QKV 给你提供了一个新的思路：只研究问题和答案之间的关系。不去找前后，不去找相邻，就是单纯的问题（Query）和答案(Value)，最多加了一个(Key)来辅助。那为什么要用 QKV 呢？因为这是谷歌搜索等搜索引擎最开始的结构。一个搜索引擎的设计其实就是给一个问题然后找到对应答案。任何一个问题(Query)，会有很多的答案（Value），而之所以能找到这些答案，是因为这些答案里面包包含了有关于这个问题的关键信息（Key）：V=f(Q,K)，这是一个万能形式，任何问题的答案都是通过“问题本身+相关的关键信息”找到的，比如你去谷歌搜索“今天天气怎么样”，这个问题本身就是 Q，而你的语言是“中文”，你的位置是“北京”，你的时间是“今天”，这些就都是 K，那么找到的答案“下雨”就是 V。一般来说肯定是通过方法找到 f( Q, K) 中的一些系数，就可以找到正确的 V 了。我们也可以把 V挪到公式右边，并且把他们存在的关系叫成 attention，那么就是：attention = F(Q, K, V)。这就是整个 transformer 的最基础结构，有了这个万能结构，只需要学习 F里的各个参数，就可以回答你想要的问题。为什么叫 transformer而不是简单的 attention 呢？因为 transformer 它为了提高这个 F的 运算效率，做了一些规定，比如你的 attention 的输入输出维度需要一样，这样矩阵运算就可以加快。而且多个 attention 合在一起来算，也是为了加快运算速度和效率。

### 代码

[以LLAMA为例，快速入门LLM的推理过程](https://mp.weixin.qq.com/s/5lbrqbqiHPZIARsVW6l6tA)

```
model = LlamaForCausalLM.from_pretrained(model, torch_dtype='auto')
print(model)
```
可以通过print看模型结构：
```
LlamaForCausalLM(
  (model): LlamaModel(
    (embed_tokens): Embedding(32000, 4096, padding_idx=31999)
    (layers): ModuleList(
      (0-31): 32 x LlamaDecoderLayer(
        (self_attn): LlamaAttention(
          (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (k_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (v_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (rotary_emb): LlamaRotaryEmbedding()
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(in_features=4096, out_features=11008, bias=False)
          (down_proj): Linear(in_features=11008, out_features=4096, bias=False)
          (up_proj): Linear(in_features=4096, out_features=11008, bias=False)
          (act_fn): SiLUActivation()
        )
        (input_layernorm): LlamaRMSNorm()
        (post_attention_layernorm): LlamaRMSNorm()
      )
    )
    (norm): LlamaRMSNorm()
  )
  (lm_head): Linear(in_features=4096, out_features=32000, bias=False)
)
```

LLAMA属于Decoder-only models，有32个LlamaDecoderLayer，每个Decoder包含一个LlamaAttention和LlamaMLP，然后是LlamaRMSNorm和head部分，核心的结构是LlamaDecoderLayer。30B的话有60个LlamaDecoderLayer，30B和7B的差别也就是decoder的个数和decoder的不同配置。

### mask

[举个例子讲下transformer的输入输出细节及其他](https://zhuanlan.zhihu.com/p/166608727)
1. 对于机器翻译来说，一个样本是由原始句子和翻译后的句子组成的。比如原始句子是： “我爱机器学习”，那么翻译后是 ’i love machine learning‘。 则该一个样本就是由“我爱机器学习”和 "i love machine learning" 组成。这个样本的原始句子的单词长度是length=4,即‘我’ ‘爱’ ‘机器’ ‘学习’。经过embedding后每个词的embedding向量是512。那么“我爱机器学习”这个句子的embedding后的维度是[4，512 ] （若是批量输入，则embedding后的维度是[batch, 4, 512]）。
2. padding。因为每个样本的原始句子的长度是不一样的，那么怎么能统一输入到encoder呢。此时padding操作登场了，假设样本中句子的最大长度是10，那么对于长度不足10的句子，需要补足到10个长度，shape就变为[10, 512], 补全的位置上的embedding数值自然就是0了。
3. Padding Mask。对于输入序列一般我们都要进行padding补齐，也就是说设定一个统一长度N，在较短的序列后面填充0到长度为N。对于那些补零的数据来说，我们的attention机制不应该把注意力放在这些位置上，所以我们需要进行一些处理。具体的做法是，把这些位置的值加上一个非常大的负数(负无穷)，这样经过softmax后，这些位置的权重就会接近0。Transformer的padding mask实际上是一个张量，每个值都是一个Boolean，值为false的地方就是要进行处理的地方。
4. 注意encoder的输出并没直接作为decoder的直接输入。 训练的时候，
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
然后将上述矩阵矩阵乘以一个 mask矩阵就得到了。
```
1 0 0 0 0
1 1 0 0 0
1 1 1 0 0
1 1 1 1 0
1 1 1 1 1 
```
这个mask矩阵就是 sequence mask，其实它和encoder中的padding mask 异曲同工。这样将这个矩阵输入到decoder（其实你可以想一下，此时这个矩阵是不是类似于批处理，矩阵的每行是一个样本，只是每行的样本长度不一样，每行输入后最终得到一个输出概率分布，作为矩阵输入的话一下可以得到5个输出概率分布）。这样我们就可以进行并行计算进行训练了。

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

[五年时间被引用3.8万次，Transformer宇宙发展成了这样](https://mp.weixin.qq.com/s/cVuBfrrtGBpNlZUekxgnmg) 未读。 





