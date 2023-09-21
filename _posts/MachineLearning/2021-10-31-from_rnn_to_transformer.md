---

layout: post
title: 从RNN到Transformer
category: 架构
tags: MachineLearning
keywords:  rnn attention transformer

---

<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

## 简介

* TOC
{:toc}

脉络：传统神经网络或CNN ==> 解决输入输出不固定的问题引入RNN ==> 解决M to N问题引入编码集-解码器（两个RNN+context） ==> 为了避免相同的上下文变量对模型性能的限制，给编码器-解码器模型加入了注意力机制 ==> 如何计算注意力？自注意力——Transformer。

## 为什么要发明循环神经网络

为什么提出RNN？或者传统神经网络或CNN 为什么不行？传统的神经网络，以及CNN，它们存在的一个问题是，**采用固定的大小的输入并产生固定大小的输出**（input是一个固定大小的向量）。而RNN呢？比如处理文本，其输入和输出的长度是可变的（input是一个不固定长度的多个向量，一个向量代表一个词），比如，一对一，一对多，多对一，多对多。

![](/public/upload/machine/rnn_usage.png)

RNN是神经网络中的一种，它的链状结构，擅长对序列数据进行建模处理。序列数据有很多种形式。音频是一种自然的序列，你可以将音频频谱图分成块并将其馈入RNN。文本也是一种形式的序列，你可以将文本分成一系列字符或一系列单词。

1. 循环神经网络，引入状态变量来存储过去的信息，并用其与当期的输入共同决定当前的输出。 多层感知机 + 隐藏状态 = 循环神经网络
2. 应用到语言模型中时 ，循环神经网络根据当前词预测下一时刻词
3. 通常使用困惑度来衡量语言模型的好坏

RNN 输入和输出 根据目的而不同
1. 比如 根据一个字预测下一个字，输入就是一个字的特征向量（后续就是这个字的某个数字编号）
2. 给一个词 标记是名词还是动词
3. 语音处理。输入一个每帧的声音信号 的特征向量


### RNN 结构

[史上最详细循环神经网络讲解（RNN/LSTM/GRU）](https://zhuanlan.zhihu.com/p/123211148)先来看一个NLP很常见的问题，命名实体识别，举个例子，现在有两句话：
1. 第一句话：I like eating apple！（我喜欢吃苹果！）
2. 第二句话：The Apple is a great company！（苹果真是一家很棒的公司！）

现在的任务是要给apple打Label，我们都知道第一个apple是一种水果，第二个apple是苹果公司，假设我们现在有大量的已经标记好的数据以供训练模型，当我们使用全连接的神经网络时，我们做法是把apple这个单词的特征向量输入到我们的模型中，在输出结果时，让我们的label里，正确的label概率最大，来训练模型。

```
// 序列模型的X 和 Y，实际上一般是随机采样，每个样本都是在原始的长序列上任意捕获的子序列。基于所有词会创建一个词表，每个词由其在词表中的位置 对应的向量来表示。
// features ==> labels
I like ==> eating
like eating ==> apple
the apple ==> is
apple is  ==> a 
is a ==> greate
a greate ==> company
```

但我们的语料库中，有的apple的label是水果，有的label是公司，这将导致，模型在训练的过程中，预测的准确程度，取决于训练集中哪个label多一些，这样的模型对于我们来说完全没有作用。问题就出在了我们没有结合上下文去训练模型

![](/public/upload/machine/rnn_nn.jpg)

1. 「输入层」：X是一个向量，它表示「输入层」的值，并且**与隐藏层之间不是全连接**，而是按照时刻进行与隐藏层之间进行对齐连接。
2. 「隐藏层」：h是一个向量，它表示「隐藏层」的值（节点数与向量S的维度相同）；
3. 「输出层」：y是一个向量，它表示「输出层」的值；

RNN 输出$y_i$依赖于上一个状态$h_{i-1}$和当前输入$x_i$所推导出的隐状态$h_i$，这种机制在解决了传统神经网络无法与过去输入建立联系的问题。与多层感知机 (MLP) 等前馈网络不同，RNN 有一个内部反馈回路，负责记住每个时间步的信息状态。

RNN Cell（RNN 就是一个RNN Cell 的不断复制）：a typical vanilla RNN uses only 3 sets of weights to perform its calculations: $W_{xh}$,$W_{hh}$,$W_{hy}$。 We’ll also use two biases for our RNN: $b_h$,$b_y$。 

$$
h_t = tanh(W_{xh}x_t + W_{hh}h_{t-1} + b_h)
$$
$$
y_t = W_{hy}h_t + b_y
$$

rnn 是通过 $W_{hh}$ 存储时序信息的。

### RNN 代码

[如何深度理解RNN？——看图就好！](https://zhuanlan.zhihu.com/p/45289691)

```
rnn = RNN()
ff = FeedForwardNN()
hidden_state = [0.0,0.0,0.0,0.0]
for word in input:
    output , hidden_state = rnn(word, hidden_state)
    ...
prediction = ff(output)
```

[An Introduction to Recurrent Neural Networks for Beginners](https://victorzhou.com/blog/intro-to-rnns/) RNN 的前向 后向传播等，并给出了一个基于文本给出情感的例子，很仔细。PS： forward 和 backward 有点从左到右 和 从右向左的意思。 

```python
class RNN:
  # A Vanilla Recurrent Neural Network.
  def __init__(self, input_size, output_size, hidden_size=64):
    # Weights
    self.Whh = randn(hidden_size, hidden_size) / 1000
    self.Wxh = randn(hidden_size, input_size) / 1000
    self.Why = randn(output_size, hidden_size) / 1000

    # Biases
    self.bh = np.zeros((hidden_size, 1))
    self.by = np.zeros((output_size, 1))
  def forward(self, inputs):
    '''
    Perform a forward pass of the RNN using the given inputs.
    Returns the final output and hidden state.
    - inputs is an array of one-hot vectors with shape (input_size, 1).
    '''
    h = np.zeros((self.Whh.shape[0], 1))
    # Perform each step of the RNN
    for i, x in enumerate(inputs):
      h = np.tanh(self.Wxh @ x + self.Whh @ h + self.bh)
    # Compute the output
    y = self.Why @ h + self.by
    return y, h
  def forward(self, inputs):
    '''
    Perform a forward pass of the RNN using the given inputs.
    Returns the final output and hidden state.
    - inputs is an array of one-hot vectors with shape (input_size, 1).
    '''
    h = np.zeros((self.Whh.shape[0], 1))

    self.last_inputs = inputs
    self.last_hs = { 0: h }
    # Perform each step of the RNN
    for i, x in enumerate(inputs):
      h = np.tanh(self.Wxh @ x + self.Whh @ h + self.bh)
      self.last_hs[i + 1] = h
    # Compute the output
    y = self.Why @ h + self.by
    return y, h
```

在 RNN 里面，给一个序列，它的计算是把这个序列 **从左往右** 一步一步往前做。假设一个序列是一个句子，它就是一个词一个词的看。对第 t 个词，它会计算一个输出叫做$h_t$(该词 的隐藏状态)，然后该词的$h_t$是由 前面一个词的隐藏状态$h_{t-1}$和当前 第 t 个词本身决定的。这样它就可以把前面学到的历史信息通过$h_{t-1}$放到当下，然后和当前的词 做一些计算 得到输出。这也是 RNN 为何能够有效处理时序信息的一个关键所在。**它把之前的信息全部放在隐藏状态里面**，然后一个一个循环下去。但它的问题也来自于此
1. RNN 是一个时序一步一步计算的过程，比较难以并行。就是说在算 第 t 个词的时候，算出 $h_t$的时候，必须要保证 前面那个词的$h_{t-1}$输入完成。假设 句子有 100 个词的话，那么需要 时序的算 100 步，导致说在这个时间上，无法并行。现在GPU 都是成千上万个线程，无法在这个上面并行的话，导致 并行度比较低，使得在计算上性能比较差。
2. 历史信息是一步一步的往后传递的，如果 时序比较长的话，那么 **很早期的那些时序信息，在后面的时候可能会丢掉，如果不想丢掉的话，那可能得要$h_t$比较大**。但是问题是：如果$h_t$比较大，在 每一个时间步 都得把它存下来，导致内存开销是比较大。

###  GRU 和 LSTM

RNN网络对任意时刻的输入都是赋予相同权重计算，这样区分不出重点因素。可以进行一个短期的记忆，长期记忆的实现一般使用LSTM模型，当预测点与依赖的相关信息距离比较远的时候，就难以学到该相关信息。例如在句子“我是一名中国人，.......(省略数十字），我会说中文”，如果我们要预测末尾的“中文”两个字，我们需要上文的“中国人”，或者“中国”。

GRU： 能关注的机制（更新门/Zt），能遗忘的机制（重置门/Rt）。PS：**多了几个要学习的weight**。

$$
R_t = \sigma(X_tW_{xr} + H_{t-1}W_{hr}+b_r)
$$
$$
Z_t = \sigma(X_tW_{xz} + H_{t-1}W_{hz}+b_z)
$$
$$
\tilde{H_t} = tanh(X_tW_{xh}+(R_t\bigodot H_{t-1})W_{hh+b_h})
$$
$$
H_t = Z_t \bigodot H_{t-1} + (1-Z_t) \bigodot \tilde{H_t} 
$$

$\bigodot$ 是按元素乘法的意思，比如$R_t$ 全是0 ，则$H_{t-1}$ 就几乎无效了。

由于 RNN 自身的结构问题，在进行反向传播时，容易出现梯度消失或梯度爆炸。LSTM 网络在 RNN 结构的基础上进行了改进，通过精妙的门控制将短时记忆与长时记忆结合起来，一定程度上解决了梯度消失与梯度爆炸的问题。

## Encoder-Decoder/两个RNN加context

[NLP注意力机制的视觉应用——谈谈看图说话的SAT模型](https://zhuanlan.zhihu.com/p/353350370)由于结构上的受限，RNN只能实现“1 to N”、“N to 1”和“N to N”的形式。那么**对于“M to N”这种形式的句子关系（输入输出不等长问题，如机器翻译、阅读理解等场景），RNN便显得有些乏力**。于是，大佬们提出了Seq2Seq，这是一个拥有编码器Encoder和解码器Decoder的模型，其中，**Encoder和Decoder都是RNN类型的网络**，上下文变量为二者搭建起了信息传递的桥梁，编码器将输入序列的信息编码到上下文变量中，解码器将上下文变量中的信息解码生成输出序列。依靠“意义”这一中介，Seq2Seq成功解决了两端语句单词数量不对等的情况，即与传统RNN模型相比，更好的解决了“M to N”。

编码器-解码器的设计是多种多样的，需要根据具体问题具体分析。**编码器获取输入并将其编码为固定长度的向量**，而解码器获取该向量并将其解码为输出序列。编码器和解码器联合训练以最小化条件对数似然。一旦训练完毕，编码器/解码器就可以在给定输入序列的情况下生成输出，或者可以对成对的输入/输出序列进行处理。比如在中译英任务中，编码器会将输入序列从源空间（例如中文）投影到一个高维语义空间的向量表示中。接着，解码器将这个高维向量从语义空间映射回目标空间（例如英文），生成一个新的序列作为翻译输出。只是在做各个语义空间中的相互投影罢了，只不过这个投影的方法叫做编码器和解码器。

![](/public/upload/machine/encoder_decoder.jpg)

编码器实现将输入的任意长度的输入序列x映射为固定长度的上下文序列 c，该上下文序列为输入序列的一个中间编码表示 $c = Encoder(x_1,...,x_m)$

获得c的具体方法有多种
1. 可以取RNN编码器的最后一个隐状态，即 $c=h_m$
2. 可以是最后一个隐状态的某种变换，即$c=q(h_m)$
3. 可以对针对所有隐状态做的某种变换，即 $c=q(h_1,h_2,...h_m)$
解码器用来将上述固定长度的中间序列c映射为变长度的目标序列作为最终输出 ，其中输出序列中的每一个元素$y_i(i=1,2,...,n)$依赖中间序列c 以及其之前的隐状态，即$y_i=Decoder(c,s_1,s_2,...,s_{i-1})$。$s_i$ 是decoder的隐藏状态。


解码器跟编码器的不一样的是：在解码器里面，词是一个一个生成的。对编码器来讲，很有可能是一次性看全整个句子，比如做机器翻译的时候，可以把整个英语的句子给你。但是在解码的时候，只能一个一个的生成，这个东西叫做一个叫做自回归（auto-regressivet），在自回归里面，模型的输入又是模型的输出。

具体来说，在最开始给定 z，那么要去生成第一个输出$y_1$，拿到$y_1$之后，就可以去生成$y_2$。一般来说要生成$y_t$，可以把之前所有的 
$y_1$到$y_{t-1}$全部拿到。也就是说在机器翻译的时候是一个词一个词地往外蹦，所以在过去时刻的输出，也会作为当前时刻的输入，所以这个叫做自回归。

Seq2Seq模型可以认为是一个序列到序列转换的通用框架，具有广泛的应用场景，可以完成诸如“中文->英文”的翻译任务，也可以完成“文章->关键词”的摘要提取任务，甚至可以完成“图像->文字”的看图说话任务。然而，Seq2Seq模型的编码器-解码器架构也存在着明显的缺陷。
1. Seq2Seq模型理论上可以接受任意长度的序列作为输入，但是机器翻译的实践表明，输入的序列越长，模型的翻译质量越差。产生这一问题的原因在于无论输入序列的长短，编码器都会将其映射为一个具有固定长度的上下文序列c。这就意味着当输入序列的长度过长时，上下文序列将无法表示整个输入序列的信息。试想在一个文本摘要生成的应用中，若c为一个几百维的向量，在针对一段短新闻稿时，也许能够表达新闻稿的全部语义信息，但是面对一篇长篇小说，恐怕其在语义信息表达方面将显得力不从心。
1. 在上述编码器-解码器框架中，**在生成每一个目标元素$y_i$时使用的下文序列c都是相同的，这就意味着输入序列x中的每个元素对输出序列y中的每一个元素都具有相同的影响**。换句话说，**解码阶段不同时间步看到的输入序列的信息都是一样的，这种现象是有悖常理的**。毕竟在一个输入序列中，不同元素所携带的信息量是不同的，受到关注的程度也自然存在差异。例如在英文到中文的机器翻译应用中，英文语句中的不定冠词“a”或“an”在很多场合是不需要显式翻译的，而类似“very”这样的副词在很多语句中却携带着很重的情感信息。

## 注意力机制/抛弃RNN

[从RNN到“只要注意力”——Transformer模型](https://zhuanlan.zhihu.com/p/353423931)基于RNN的架构存在着一个明显弊端，那就是RNN属于序列模型，需要以一个接一个的序列化方式进行信息处理，注意力权重需要等待序列全部输入模型之后才能确定，即需要RNN对序列“从头看到尾”。这种架构无论是在训练环节还是推断环节，都具有大量的时间开销，并且难以实现并行处理。例如面对翻译问题“A magazine is stuck in the gun.”，其中的“magazine”到底应该翻译为“杂志”还是“弹匣”？当看到“gun”一词时，将“magazine”翻译为“弹匣”才确认无疑。在基于RNN的机器翻译模型中，需要一步步的顺序处理从magazine到gun的所有词语，而当它们相距较远时RNN中存储的信息将不断被稀释，翻译效果常常难以尽人意，而且效率非常很低。**我们不禁要问一个问题：RNN 结构是否真的必要？**谷歌大脑、谷歌研究院等团队于 2017 年联合发表文章《Attention Is All You Need》，给出了的答案——“RNN is unnecessary, attention is all you need”。PS：为什么RNN上下文理解能力弱？因为RNN通过一个隐藏层记录当前及之前所见过的词汇，已经将语义信息杂糅在一起，而往往理解 it 这个词的语义时候，通过几个词就行，而不是it 之前的所有词汇。

[NLP注意力机制的视觉应用——谈谈看图说话的SAT模型](https://zhuanlan.zhihu.com/p/353350370)

对于一个由 n 个单词组成的句子来说，不同位置的单词，重要性是不一样的。因此，我们需要让模型“注意”到那些相对更加重要的单词，这种方式我们称之为注意力机制，也称作 Attention 机制。比如“我今天中午跑到了肯德基吃了仨汉堡”。这句话中，你一定对“我”、“肯德基”、“仨”、“汉堡”这几个词比较在意，不过，你是不是没注意到“跑”字？其实 Attention 机制要做的就是这件事：找到最重要的关键内容。它对网络中的输入（或者中间层）的不同位置，给予了不同的注意力或者权重，然后再通过学习，网络就可以逐渐知道哪些是重点，哪些是可以舍弃的内容了。

与标准Seq2Seq模型相比，**注意力模型最大的改进在于其不再要求编码器将输入序列的所有信息都压缩为一个固定长度的上下文序列c中**，取而代之的是**将输入序列映射为多个上下文序列$c_1,c_2,...,c_n$**，其中$c_i$是与输出$y_i$对应的上下文信息。

下图示意了一个注意力模型的基本结构，其中的注意力模块可以视为是一个具有m个输入节点和n个输出节点的全连接神经网络。PS：**Attention 可以看做为解码器阶段的每个单元，单独准备了一个定制、全局的 C**。

![](/public/upload/machine/seq2seq_attention.jpg)

在注意力模型中，每一个上下文序列为编码器所有隐状态向量的加权和

$$
c_i=\sum_{j=1}^m\alpha_{ij}h_j
$$

其中$\alpha_{ij}$为注意力权重系数（也称为注意力得分）。在编码器中，隐变量$h_j$蕴含了输入序列第j个元素的信息，因此对编码器隐变量按照不同权重求和表示在生成预测结果$y_i$时，对输入序列中的各个元素上分配的注意力是不同的——$\alpha_{ij}$越大，表示第i个输出在第j个输入上分配的注意力越多，即生成i个输出时受到第j个输入的影响也就越大，反之亦反。

剩下最后一个问题即**如何得到注意力权重系数$\alpha_{ij}$了**。在注意力模型中，注意力权重系数是通过构造一个全连接网络，然后再对该网络输出向量进行概率化得到的。这个参考原文。 

小结：为模型的每一个输入项（比如语句中的某个单词）分配一个权重，这个权重的大小就代表了我们希望模型对该部分一个关注程度。这样一来，通过权重大小来模拟人在处理信息的注意力的侧重，有效的提高了模型的性能，并且一定程度上降低了计算量。PS：注意力机制本质上就是学习权重，权重大的地方被关注了，权重小的地方被轻视。

[Attention综述](https://zhuanlan.zhihu.com/p/62136754)

![](/public/upload/machine/attention_overview.jpg)

## Attention is all you need——Transformer

[李沐读 Transformer 视频笔记](https://zhuanlan.zhihu.com/p/619234992) 李沐对于 Transformer论文的讲解，还是看作者的第一手材料+大佬解说最准确。未读。

Transformer模型采用的也是编码器-解码器架构，但是在该模型中，编码器和解码器不再是 RNN结构，取而代之的是编码器栈（encoder stack）和解码器栈（decoder stack）（注：所谓的“栈”就是将同一结构重复多次，“stack”翻译为“堆叠”更为合适）。

![](/public/upload/machine/transformer.jpg)

[Transformer - Attention is all you need](https://zhuanlan.zhihu.com/p/311156298)Encoder层和Decoder层内部结构如下图所示。

![](/public/upload/machine/transformer_internal.jpg)

1. Encoder具有两层结构，self-attention和前馈神经网络。self-attention计算句子中的每个词都和其他词的关联，从而帮助模型更好地理解上下文语义，引入Muti-Head attention后，每个头关注句子的不同位置，增强了Attention机制关注句子内部单词之间作用的表达能力。前馈神经网络为encoder引入非线性变换，增强了模型的拟合能力。本质上，自注意力机制使得模型能理解语句中不同单词间的关系。而且**跟以往按固定顺序处理单词**的传统模型不同，transformers 其实是同时检查所有单词，并根据每个词跟句中其他词之间的相关性，为各词分配所谓“注意力得分”指标。
2. Decoder接受encoder输入的同时也接受output输入，帮助当前节点获取到需要重点关注的内容。解码器则从编码器处获取编码，之后产生输出序列。在机器翻译和文本生成等任务中，解码器会根据从编码器处接收到的输入生成经过翻译的序列。跟编码器类似，解码器同样由多层自注意力加前馈神经网络组成。**但解码器还包含额外的注意力机制，用于专注处理编码器的输出，保证解码器在生成输出时考虑到来自输入序列的相关信息**。

Transformer 模型本来是为了翻译任务而设计的。在训练过程中，Encoder 接受源语言的句子作为输入，而 Decoder 则接受目标语言的翻译作为输入。在 Encoder 中，由于翻译一个词语需要依赖于上下文，因此注意力层可以访问句子中的所有词语；而 Decoder 是顺序地进行解码，在生成每个词语时，注意力层只能访问前面已经生成的单词。例如，假设翻译模型当前已经预测出了三个词语，我们会把这三个词语作为输入送入 Decoder，然后 Decoder 结合 Encoder 所有的源语言输入来预测第四个词语。每一个部分都有公式对应。

想要进一步了解 Transformer 这一架构的原理，强烈推荐阅读 Jay Alammar 的博客。从为了解决翻译问题时 Seq2Seq 模型的提出的 Attention 的基本概念 https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention，到 Transformer 架构中完全抛弃 RNN https://www.youtube.com/watch?v=UNmqTiOnRfg 提出的 Attention is All You Need http://jalammar.github.io/illustrated-transformer，到 GPT-2 和 GPT-3 的架构解读http://jalammar.github.io/illustrated-gpt2 http://jalammar.github.io/how-gpt3-works-visualizations-animations，Jay Alammar 的博客都提供了精彩的可视化配图便于理解模型结构。

![](/public/upload/machine/transformer.png)

### self-attention 机制

self-attention 机制用于计算句子中当前词与其他词的联系，举个例子：

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

所谓的Q K V矩阵、查询向量之类的字眼，其来源是X与矩阵的乘积，本质上都是X的线性变换。为什么不直接使用X而要对其进行线性变换？当然是为了提升模型的拟合能力，矩阵W都是可以训练的，起到一个缓冲的效果。所以 self-Attention最原始的形态其实是 $softmax(XX^T)X$。

1. $XX^T$代表什么？一个矩阵（一个句子）乘以它自己的转置，会得到什么结果，有什么意义？我们知道，矩阵可以看作由一些向量组成，**一个矩阵乘以它自己转置的运算，其实可以看成这些向量分别与其他向量计算内积**（一个句子的各个词向量的内积）。向量的内积，其几何意义是什么？表征两个向量的夹角，表征一个向量在另一个向量上的投影。**投影的值大，说明两个向量相关度高**（两个词相关度越高）。如果两个向量夹角是九十度，那么这两个向量线性无关，完全没有相关性！更进一步，这个向量是词向量，是词在高维空间的数值映射。词向量之间相关度高表示什么？是不是在一定程度上（不是完全）表示，在关注词A的时候，应当给予词B更多的关注？
  ![](/public/upload/machine/self_attention_xxt.jpg)
2. Softmax操作的意义是什么呢？归一化。 Softmax之后，这些数字的和为1了。当我们关注"早"这个字的时候，我们应当分配0.4的注意力给它本身，剩下0.4关注"上"，0.2关注"好"。
  ![](/public/upload/machine/self_attention_softmax_xxt.jpg)
3. 最后一个 X 有什么意义？ $softmax(XX^T)X$表示什么？我们取 $softmax(XX^T)X$
 的一个行向量举例。这一行向量与X的一个列向量相乘，表示什么？在新的向量中，每一个维度的数值都是由三个词向量在这一维度的数值加权求和得来的，这个新的行向量就是"早"字词向量经过注意力机制加权求和之后的表示。
  ![](/public/upload/machine/self_attention_softmax_xxt_x.jpg)
  一张更形象的图是这样的，图中右半部分的颜色深浅，其实就是我们上图中黄色向量中数值的大小，意义就是单词之间的相关度
  ![](/public/upload/machine/self_attention_softmax_xxt_x_matrix.jpg)
4. $\sqrt{d_k}$的意义。假设Q,K里的元素的均值为0，方差为1，那么$A^T=Q^TK$中元素的均值为0，方差为d。当d变得很大时，A中的元素的方差也会变得很大，如果A中的元素方差很大，那么softmax(A)的分布会趋于陡峭(分布的方差大，分布集中在绝对值大的区域)。总结一下就是softmax(A)的分布会和d有关。因此A中每一个元素除以$\sqrt{d_k}$后，方差又变为1。这使得softmax(A)的分布“陡峭”程度与d解耦，从而使得训练过程中梯度值保持稳定。


[小白看得懂的 Transformer (图解)](https://mp.weixin.qq.com/s/VrzkxEVBAO6abJcUsYGr0Q)计算自注意力的步骤
1. 从每个编码器的输入向量（每个单词的词向量）中生成三个向量（QKV）。
2. 计算得分。假设我们在为这个例子中的第一个词“Thinking”计算自注意力向量，我们需要拿输入句子中的每个单词对“Thinking”打分。这些分数决定了在编码单词“Thinking”的过程中有多重视句子的其它部分。这些分数是通过打分单词（所有输入句子的单词）的K向量与“Thinking”的Q向量相点积来计算的。
3. 将分数除以8(8是论文中使用的键向量的维数64的平方根，这会让梯度更稳定。这里也可以使用其它值，8只是默认值)，然后通过softmax传递结果。softmax的作用是使所有单词的分数归一化，得到的分数都是正值且和为1。这个softmax分数决定了每个单词对编码当下位置（“Thinking”）的贡献。显然，已经在这个位置上的单词将获得最高的softmax分数，但有时关注另一个与当前单词相关的单词也会有帮助。
  /![](/public/upload/machine/self_attention_score.jpg)
4. 将每个V 乘以softmax分数(这是为了准备之后将它们求和)。这里的直觉是希望关注语义上相关的单词，并弱化不相关的单词(例如，让它们乘以0.001这样的小数)。
5. 对加权值向量求和（译注：自注意力的另一种解释就是在编码某个单词时，就是将所有单词的表示（值向量）进行加权求和，而权重是通过该词的表示（键向量）与被编码词表示（查询向量）的点积并通过softmax得到。），然后即得到自注意力层在该位置的输出(在我们的例子中是对于第一个单词)。
  ![](/public/upload/machine/self_attention_convert.jpg)
6. 这样自自注意力的计算就完成了。得到的向量就可以传给前馈神经网络。然而实际中，这些计算是以矩阵形式完成的（一个句子一个矩阵），以便算得更快。

[如何理解 Transformer 中的自注意力机制？ - 宇文树雪的回答 - 知乎](https://www.zhihu.com/question/560879732/answer/3041597591)注意力机制中的“注意力”其实指的是序列中每两个元素之间的相关性程度，更进一步说，**这种相关性就是指两个元素在自然语句中同时出现的概率**。那么，序列元素之间的相关性是如何衡量的呢？在transformer的输入之前，往往要做词嵌入转换，把输入的文本序列转为降维到固定长度的矢量。在这个矢量空间中，其实就通过跳元模型或连续词包模型把词元之间的共现关系学习到一个高维的空间中了。通过**直接进行矢量乘法就能知道两个嵌入词向量的相似度**。词元矢量内积越大，说明相关性越强，在词嵌入空间中越近，更可能组成语句。也就是说，我们能基于词元矢量内积判断输入序列任意两个元素的共现的概率，也就是组成语句的可能性。（词嵌入原始的计算方式就是统计词元之间共同出现的频次，然后对共现矩阵做了奇异值分解得到降维后的词向量表示。）**既然词嵌入能学习到语义，为什么还要进行更复杂的训练呢？**这是因为嵌入词是在较小的窗口训练成的，不能反映大规模语句的上下文结构特征。因此，需要加一个权重对序列中每个嵌入词的重要性进行调整，这个权重需要要通过大量的语料训练得到。从而学习到自然语言序列固有的结构，并形成语言模型。

### MultiHeadAttention

自注意力机制的方式确实解决了“传统序列模型在编码过程中都需顺序进行的弊端”的问题，有了自注意力机制后，仅仅只需要对原始输入进行几次矩阵变换便能够得到最终包含有不同位置注意力信息的编码向量。**模型在对当前位置的信息进行编码时，会过度的将注意力集中于自身的位置**（虽然每个编码都在z1中有或多或少的体现），因此作者提出了通过多头注意力机制来解决这一问题。同时，使用多头注意力机制还能够给予注意力层的输出包含有不同子空间中的编码表示信息，从而增强模型的表达能力（这些集合中的每一个W都是随机初始化的，在训练之后，每个集合都被用来将输入词嵌入(或来自较低编码器/解码器的向量)投影到不同的表示子空间中）。

$$
head_i = Attention(QW{_i}{^Q},KW{_i}{^K},VW{_i}{^V})
$$
$$
MultiHead(Q,K,V) = Contact(head_1,head_2,...head_h)W^O
$$


![](/public/upload/machine/multi_head_attention.jpg)

以h=2为例（有几个head 是一个超参数），根据输入序列X和$W_1^Q,W_1^K,W_1^V$ 我们就计算得到了$Q_1,K_1,V_1$，进一步根据公式1就得到了单个自注意力模块的输出$Z_1$；同理，根据X和$W_2^Q,W_2^K,W_2^V$就得到了另外一个自注意力模块输出$Z_2$。最后，前馈层不需要8个矩阵，它只需要一个矩阵(由每一个单词的表示向量组成)。所以我们需要一种方法**把这八个矩阵压缩成一个矩阵**。那该怎么做？其实可以直接把这些矩阵拼接在一起，然后用一个附加的权重矩阵WO与它们相乘。根据公式2将$Z_1,Z_2$水平堆叠形成Z，然后再用Z乘以$W^O$便得到了整个多头注意力层的输出。

![](/public/upload/machine/multi_head_cal.jpg)

既然我们已经摸到了注意力机制的这么多“头”，那么让我们重温之前的例子，看看我们在例句中编码“it”一词时，不同的注意力“头”集中在哪里：

![](/public/upload/machine/multi_head_example.jpg)

当我们编码“it”一词时，一个注意力头集中在“animal”上，而另一个则集中在“tired”上，从某种意义上说，模型对“it”一词的表达在某种程度上是“animal”和“tired”的代表。

多头注意力机制类似于赛马机制，它有助于减少模型初始化的随机性对模型效果的影响。所以即使只留下一个注意力头也能使用，但这会导致模型的稳定性和多样性无法得到保障，进而造成模型的性能下降。



### 位置编码/Positional Encoding

Positional Embeddings 基于一个简单但有效的想法：使用与位置相关的值模式来增强词向量。

对self-attention来说，它跟每一个input vector都做attention，所以没有考虑到input sequence的顺序。更通俗来讲，大家可以发现我们前文的计算每一个词向量都与其他词向量计算内积，得到的结果丢失了我们原来文本的顺序信息。打乱词向量的顺序，得到的结果仍然是相同的。但实际上位置信息很有用，比如动词出现在句首的概率就比名词低，所以要把位置信息塞到学习过程中去。 

位置信息编码位于encoder和decoder的embedding之后，每个block之前。

如果预训练数据集足够大，那么最简单的方法就是让模型自动学习位置嵌入。 除此以外，Positional Embeddings 还有一些替代方案：
1. Learned Positional Embedding ，这个是绝对位置编码，即直接对不同的位置随机初始化一个postion embedding，这个postion embedding作为参数进行训练。缺点：
  1. 不同位置对应的positional embedding固然不同，但是位置1和位置2的距离比位置3和位置10的距离更近，这些关于位置的相对含义，模型能够通过绝对位置编码参数学习到吗？
  2. 位置之间没有约束关系，我们只能期待它隐式地学到，是否有更合理的方法能够显示的让模型理解位置的相对关系呢？
2. Sinusoidal Position Embedding ，相对位置编码，即三角函数编码。由于正弦函数能够表达相对位置信息，那么对每个positional embedding进行 sin 或者cos激活，好处是位置 i 处的单词的psotional embedding可以被位置 i+k 处单词的psotional embedding线性表示，反应两处单词的其相对位置关系。此外位置i和i+k的psotional embedding内积会随着相对位置的递增而减小，从而表征位置的相对距离。缺点：由于距离的对称性，Sinusoidal Position Encoding虽然能够反映相对位置的距离关系，但是无法区分i和i+j的方向。

transformer中，模型输入encoder的每个token向量由两部分加和而成：Position Encoding, Input Embedding。Positional Embedding的成分直接叠加于Embedding之上，使得每个token的位置信息和它的语义信息(embedding)充分融合，并被传递到后续所有经过复杂变换的序列表达中去。

![](/public/upload/machine/positional_encoding.jpg)

为什么 RNN 不需要位置编码呢？RNN 是串行处理输入数据的，所以每个 RNN 单元其实都包含了所在位置之前的全部输入信息，所以对当前位置输入的处理是有状态的。但是，Transformer 会并行地处理所有输入的内容，所以各个并行单元会无状态地处理每个输入。因此，我们需要在最开始就给每个输入的嵌入向量一个位置编号，这样模型才能通过输入判断它在整体中的位置。

### Position-wise Feed Forward

大语言模型中的参数都用在哪了？可以看到在百亿参数量以上，差不多三分之二的参数实际上是 FFN 参数，剩下的基本都是 attention 参数。所以虽然论文名叫 attention is all you need，但实际上 FFN 仍然起到了很重要的作用。

每一层经过attention之后，还会有一个FFN，这个FFN的作用就是空间变换。FFN包含了2层linear transformation层，中间的激活函数是ReLu。

attention层的output最后会和相乘，为什么这里又要增加一个2层的FFN网络？FFN的加入引入了非线性(ReLu激活函数)，变换了attention output的空间, 从而增加了模型的表现能力。把FFN去掉模型也是可以用的，但是效果差了很多。
$$
FFN(x) = max(0,xW_1+b1)W_2+b_2
$$

transformer 中最重要的是self-attention，self-attention 由三个线性矩阵Q、K、V 决定，如果我们把Q、K矩阵设置为零，那么self-attention 就变成了FFN，$Z_0 =V_0*X$，也就是说，FFN是self-attention 的一个特例，FFN能表达的逻辑，self-attention 也可以，但反过来却不成立。至于transformer 中的FFN部分，当初设计是为了输入输出维度的对齐，毕竟多注意力的输出 $W_0$的维度比输入X维度高很多。但如果一定要用FFN去表达self-attention的逻辑，也是可以的，但需要的参数量却要大很多，感兴趣的可以去试验一下，用FFN去拟合self-attention 的逻辑。就好像乘法能计算的东西，单纯用加法依然可以做到，但效率要低很多，self-attention就是乘法，FFN就是加法。

###  Layer Normalization

在每个block中，最后出现的是Layer Normalization，其作用是规范优化空间，加速收敛。当我们使用梯度下降算法做优化时，我们可能会对输入数据进行归一化，但是经过网络层作用后，我们的数据已经不是归一化的了。随着网络层数的增加，数据分布不断发生变化，偏差越来越大，导致我们不得不使用更小的学习率来稳定梯度。Layer Normalization 的作用就是保证数据特征分布的稳定性，将数据标准化到ReLU激活函数的作用区域，可以使得激活函数更好的发挥作用

Normalization有两种方法，Batch Normalization和Layer Normalization。关于两者区别不再详述。

一般情况下，输入是一个矩阵，然后矩阵的 每一行是一个样本，多个行（多个样本）是 一个 batch，每一列是一个特征，多个列是 feature。batchnorm 是说每一次，去把每一个列，就是每一个特征，把它在一个小 mini-batch 里面，每列 的均值变成 0 方差变成 1。 [Batch Norm详解之原理及为什么神经网络需要它](https://zhuanlan.zhihu.com/p/441573901)

**在 Transformer 里面，或者说正常的 RNN 里面，它的输入是一个三维的矩阵**。因为 输入的是一个序列的样本，即每一个样本里面有很多个元素。一个序列，如：一个句子里面有 n 个词，所以每个词表示为一个向量的话，还有一个 batch 维度，那么就是个 3D 的输入。列不再是特征，而是序列的长度，对每一个 sequence 就是 每个词，每个词有自己对应的向量。如果是 layernorm 的话，那么就是对每个样本切一下（横着切一下）。为什么 layer norm 用的多一点？一个原因是：在 时序的序列模型 里面，每个样本的长度可能会发生变化。那些不够 sequence 的长度 n 的样本 ，一般是补 0。 layernorm 是对 每个样本来做，所以不管样本是长还是短，反正算均值是在样本自己算的，这样的话相对来说它稳定一些。


### Residual Network 残差网络

残差网络是深度学习中一个重要概念。在神经网络可以收敛的前提下，随着网络深度的增加，网络表现先是逐渐增加至饱和，然后迅速下降，这就是我们经常讨论的网络退化问题。

在transformer模型中，encoder和decoder各有6层，为了使当模型中的层数较深时仍然能得到较好的训练效果，模型中引入了残差网络，**将输入和输出直接相加**。

![](/public/upload/machine/residual_network.jpg)

### Linear & Softmax

Decoder最后是一个线性变换和softmax层。解码组件最后会输出一个实数向量。我们如何把浮点数变成一个单词？这便是线性变换层要做的工作，它之后就是Softmax层。
线性变换层是一个简单的全连接神经网络，它可以把解码组件产生的向量投射到一个比它大得多的、被称作对数几率（logits）的向量里。不妨假设我们的模型从训练集中学习一万个不同的英语单词（我们模型的“输出词表”）。因此对数几率向量为一万个单元格长度的向量——每个单元格对应某一个单词的分数（相当于做vocaburary_size大小的分类）。接下来的Softmax 层便会把那些分数变成概率（都为正数、上限1.0）。概率最高的单元格被选中，并且它对应的单词被作为这个时间步的输出。

### 其它

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


[transformer架构的核心公式其实类似于数学期望，理解起来也不复杂，但为什么这个模型这么强呢？ - SamZhang的回答 - 知乎](https://www.zhihu.com/question/580810624/answer/2977786600)Transformer模型之所以这么强大,主要有以下几个原因:
1. Transformer使用Attention来建模序列之间的依赖关系,这比RNN等其他模型更加灵活和有效。Attention可以自动学习序列元素之间的重要性和依赖,不需要像RNN那样依赖固定的拓扑结构。
2. Transformer的Attention和Feed Forward网络等都是并行计算,所以Transformer在计算速度和效率上都超过RNN。这使其能够实现较长序列的建模。
3. 多头Attention。Transformer使用了Multi-Head Attention,即多个Attention对不同的表示子空间进行注意力计算。这使得Attention能够同时考虑不同的语义关系,如上下义、定语等关系。这进一步提高了模型的表征能力。
4. Layer Normalization。Transformer使用Layer Normalization代替Batch Normalization。这使得模型对输入的依赖性降低,对输入的变化更加鲁棒。这是Transformer能处理较长序列的另一个重要原因。
5. 可重复利用的模块。Transformer由Attention、Layer Normalization和Feed Forward Neural Network等可重复使用的模块构成。这使得Transformer既深又宽,能够实现很强的表征能力。
6. 损失函数。Transformer使用成对的softmax损失函数,这促使模型同时学习两个序列的表征,这比序列续生成等其他技术更加有效。

鲁提辖：要理解Transformer就是搞懂Attention，再就是Positional Encoding。Attention主要负责对长程依赖内容依赖建模，Positional Encoding负责对短程依赖和位置依赖建模，两者合力就有了Transformer强大的拟合能力。

## 其它视角

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

一个知乎回答：transformer中的Q,K,V到底是什么？就是查字典。假想你有一个map/dict或者其他名字，一个key对应一个value，在检索的时候，给定query，如果query in map，就是query等于其中一个key，就返回对应的value。这个方法太hard了，有就是有，没有就是没有。对于qkv都是向量的情况，这种方法不可行，只能让它变soft，那就是算一算query和key的关系，按照比例对value加和，这和max变成softmax有异曲同工之妙。

### 编码视角

NLP 神经网络模型的本质就是对输入文本进行编码，常规的做法是首先对句子进行分词，然后将每个词语 (token) 都转化为对应的词向量 (token embeddings)，这样文本就转换为一个由词语向量组成的矩阵 $X=(x_1,x_2,...,x_n)$，其中$x_i$就表示第i个词语的词向量。
1. RNN（例如 LSTM）的方案很简单，每一个词语$x_t$对应的编码结果$y_t$通过递归地计算得到：$y_t=f(y_{t-1},xt)$，RNN 本质是一个马尔科夫决策过程，难以学习到全局的结构信息；
2. CNN 则通过滑动窗口基于局部上下文来编码文本，例如核尺寸为 3 的卷积操作就是使用每一个词自身以及前一个和后一个词来生成嵌入式表示：$y_t=f(x_{t-1},x_{t},x_{t+1})$，由于是通过窗口来进行编码，所以更侧重于捕获局部信息，难以建模长距离的语义依赖。
3. 直接使用 Attention 机制编码整个文本。相比 RNN 要逐步递归才能获得全局信息（因此一般使用双向 RNN），而 CNN 实际只能获取局部信息，需要通过层叠来增大感受野，Attention 机制一步到位获取了全局信息：$y_t=f(x_t,A,B)$，其中 A,B是另外的词语序列（矩阵），如果取A=B=X就称为 Self-Attention，即直接将$x_t$与自身序列中的每个词语进行比较，最后算出$y_t$。

### 与MLP对比

特征矩阵X每一行代表一个 item，以每一列代表一个 feature。通常我们所说的线性层都是 $X^{t+1}=X^tW^t+B^t$，也就是在 feature 这个维度上作线性变换，完全没有考虑 item 与 item 之间的关系。这也是为什么 MLP 会比不过什么 CNN、attention之类的原因，在 MLP 的框架里每个 item 的运算都是独立的，不会与其他 item 产生交互。用代数的话来说，就是在矩阵乘法中，左乘的矩阵行其向量之间是不会产生交互的，相应的右乘的矩阵其列向量之间是不会产生交互的。

那要怎么做到 item 与 item 之间的交互呢？很简单，把线性层里面的右乘一个权重矩阵变成左乘就行了，即$X^{t+1}=W^tX^t+B^t$，由于这也可以看成是把原来的输入取了转置以后再扔进通常的线性层，得到输出后再做一个转置，我们就把这样的层叫做转置线性层。为了方便我们就不要最后的偏差项了，拿每一个 item 来看，$X^{t+1}_{i}=W_i^tX^t=\sum_jW_{ij}^tX_j^t$

这个输出的形式和 attention 可以说是一模一样了。这里拿最常见的 Attention 举例，即

$$
Attention(Q,K,V) = softmax(\frac{QK^T}{\sqrt{d_k}})V
$$

其中V就等价于$X^t$，而$softmax(\frac{QK^T}{\sqrt{d_k}})$就是我们想要学到的$W^t$（所以说self-attention 的权重是根据输入的改变和改变的，而CNN中一旦训练完成，各个通道的卷积核参数就固定了）， attention 多做了一些什么呢？对于要学习的参数矩阵$W^t$增加了低秩的约束。参考 self-attention 的情况，即输出和输入的 size 保持一致，那么理论上没有任何约束的情况下$W^t$可以是一个满秩的n*n方阵。但是，在 self-attention 中，假设$Q,K\in\R^{n*d}$那么$softmax(\frac{QK^T}{\sqrt{d_k}})$就是一个秩至多为d的方阵。秩上的降低也意味着表征能力的下降，但好处在于$\mathcal{O}(n^2) $变成了$\mathcal{O}(nd) $， $d\ll n$ 的时候这样做的价值就体现出来了。（比如原来比如GPT2 一个token向量 是768维，context若是4000，那mlp就非常deep，所以一定是预处理一下再接入mlp）。可以说attention 是一种为了压缩参数量而牺牲一些表征能力，是用类似于 matrix fatorization 的方式来近似转置线性层的做法。PS：所以有人提出什么 **MLP is all you need** 之类的说法，用 MLP 替代 CNN 或者 attention 也能取得不俗的图像分类效果……那可不是必须的嘛？人家那可是加了各种约束，为了压缩参数量牺牲了表征能力来近似你这个线性层的结果啊。

## 其它

[五年时间被引用3.8万次，Transformer宇宙发展成了这样](https://mp.weixin.qq.com/s/cVuBfrrtGBpNlZUekxgnmg) 未读。 

[通俗解构语言大模型的工作原理](https://mp.weixin.qq.com/s/21V8g_7teuRgHLWUej1NzA)当LLM“阅读”一篇短篇小说时，它似乎会记住关于故事角色的各种信息：性别和年龄、与其他角色的关系、过去和当前的位置、个性和目标等等。研究人员并不完全了解LLM是如何跟踪这些信息的，但从逻辑上讲，模型在各层之间传递时信息时必须通过修改隐藏状态向量来实现。现代LLM中的向量维度极为庞大，这有利于表达更丰富的语义信息。例如，GPT-3最强大的版本使用有12288个维度的词向量，也就是说，每个词由一个包含12288个的数字列表表示。这比Google在2013年提出的word2vec方案要大20倍。你可以把所有这些额外的维度看作是GPT-3可以用来记录每个词的上下文的一种“暂存空间（scratch space）”。较早层所做的信息笔记可以被后来的层读取和修改，使模型逐渐加深对整篇文章的理解。因此，假设我们将上面的图表改为，描述一个96层的语言模型来解读一个1000字的故事。第60层可能包括一个用于约翰（John）的向量，带有一个表示为“（主角，男性，嫁给谢丽尔，唐纳德的表弟，来自明尼苏达州，目前在博伊西，试图找到他丢失的钱包）”的括号注释。同样，所有这些事实（可能还有更多）都会以一个包含12288个数字列表的形式编码，这些数字对应于词John。或者，该故事中的某些信息可能会编码在12288维的向量中，用于谢丽尔、唐纳德、博伊西、钱包或其他词。这样做的目标是，让网络的第96层和最后一层输出一个包含所有必要信息的隐藏状态，以预测下一个单词。Transformer在更新输入段落的每个单词的隐藏状态时有两个处理过程：
1. 在注意力步骤中，词汇会“观察周围”以查找具有相关背景并彼此共享信息的其他词。
2. 在前馈步骤中，每个词会“思考”之前注意力步骤中收集到的信息，并尝试预测下一个单词。在注意力头在词向量之间传输信息后，在这个阶段单词之间没有交换信息，前馈层会独立地分析每个单词。前馈层之所以强大，是因为它有大量的连接。

在论文 BERTnesia: Investigating the capture and forgetting of knowledge in BERT 的结论中也能看出，**不同的知识在 Transformer 的存储是存在分层特点的**，这与我们在视觉预训练模型学到的人脸识别算法的分层特点很像。

语言本身是可预测的。语言的规律性通常（尽管并不总是这样）与物质世界的规律性相联系。因此，当语言模型学习单词之间的关系时，通常也在隐含地学习这个世界存在的关系。预测可能是生物智能以及人工智能的基础。如果我们提供足够的数据和计算能力，语言模型能够通过找出最佳的下一个词的预测来学习人类语言的运作方式。不足之处在于，最终得到的系统内部运作方式人类还并不能完全理解。

Aidan Gomez：说到Transformer以及这一代语言模型，我们能够看到很多功能强大的应用，比如GPT-3、Cohere以及ChatGPT等。它们的基本原则是，**通过扩展模型实现对更复杂数据集的建模**，显然，最复杂的数据集应该是互联网数据之类的数据，这类大型文本语料库已经积累了几十年，目前，互联网使用人数占到全球总人数的百分之六七十，人们在线上进行各类活动，如开办编程课程、语言课程，并谈论各种事件、问题等等。如果要对这个大型、高度多样化的数据集建模，我们需要用到极其复杂的模型，而这正是Transformer的用处所在。Transformer是一种神经网络架构，这种结构非常擅长扩展，并且可以有效地进行并行化处理，这在拥有成千上万个GPU加速器的大型超级计算机上进行训练非常重要。扩展模型和数据集带来了极好的成效，正如OpenAI所说：Transformer模型成为了多任务处理大师。这意味着，**相同的模型、同一组权重能够完成多种任务**，包括翻译、实体抽取、撰写博客和文章等等。现在，我们创建出了能够通过交流让其完成任务的模型（Cohere称之为命令模型，OpenAI称之为指令模型），语言大模型技术已经走进人们生活，变得更加直观可用。如今，在多数人眼中，我们可以向语言大模型下达自然语言指令，然后模型会按照指令生成相应的结果。
1. 如果最初的原始模型拥有来自网页的万亿级单词，那么在微调阶段，需要的数据数量级要小很多，重要的是这一阶段的数据要体现我们对模型行为的期望。以命令模型（与OpenAI的指令模型类似）为例，我们希望给模型下达一些自然语言指令，然后模型能以直观正确的方式做出响应，比如让模型以兴奋的语调编辑一篇博客文章。为了达成上述目标，需要收集语气兴奋的博客文章，将它们放入数据集，用这些数据对模型进行微调，这样就能将知识渊博的大型模型引向可直观控制的模型。
2. 与ChatGPT或对话模型类似，面对知识渊博且功能强大的模型，我们可用小型精细化的数据集对其进行微调，将模型引向我们所希望的发展方向，如果想要一个对话模型，就需要向模型展示大量对话，模型会根据我们展示的数据集进行调整。

汪涛：这次的人工智能爆发一方面是算力的不断提升，另一个是Trasformer这个新算法的进步。**它是CNN（神经网络）带来的深度学习算法之后又一次小的算法革命（本质上还是神经网络）**。只要利用了这种新的算法，只有量的区别，不会有什么“涌现”“不涌现”的本质区别。有些是已经涌现了，而有些还没有涌现。如果只是一些量的差异，只要在量上不断改进就可趋同或超越，而如果是质的差别，就可能很长时间超越不了。PS：Trasformer 的上限决定了这一波“智能”的上限。




