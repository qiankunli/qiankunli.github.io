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

编码器-解码器的设计是多种多样的，需要根据具体问题具体分析。**编码器获取输入并将其编码为固定长度的向量**，而解码器获取该向量并将其解码为输出序列。编码器和解码器联合训练以最小化条件对数似然。一旦训练完毕，编码器/解码器就可以在给定输入序列的情况下生成输出，或者可以对成对的输入/输出序列进行处理。

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

下图示意了一个注意力模型的基本结构，其中的注意力模块可以视为是一个具有m个输入节点和n个输出节点的全连接神经网络。

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



每一个部分都有公式对应。

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

### 位置编码/Positional Encoding

对self-attention来说，它跟每一个input vector都做attention，所以没有考虑到input sequence的顺序。更通俗来讲，大家可以发现我们前文的计算每一个词向量都与其他词向量计算内积，得到的结果丢失了我们原来文本的顺序信息。打乱词向量的顺序，得到的结果仍然是相同的。但实际上位置信息很有用，比如动词出现在句首的概率就比名词低，所以要把位置信息塞到学习过程中去。 

位置信息编码位于encoder和decoder的embedding之后，每个block之前。

关于positional embedding ，有两种方法：
1. Learned Positional Embedding ，这个是绝对位置编码，即直接对不同的位置随机初始化一个postion embedding，这个postion embedding作为参数进行训练。缺点：
  1. 不同位置对应的positional embedding固然不同，但是位置1和位置2的距离比位置3和位置10的距离更近，这些关于位置的相对含义，模型能够通过绝对位置编码参数学习到吗？
  2. 位置之间没有约束关系，我们只能期待它隐式地学到，是否有更合理的方法能够显示的让模型理解位置的相对关系呢？
2. Sinusoidal Position Embedding ，相对位置编码，即三角函数编码。由于正弦函数能够表达相对位置信息，那么对每个positional embedding进行 sin 或者cos激活，好处是位置 i 处的单词的psotional embedding可以被位置 i+k 处单词的psotional embedding线性表示，反应两处单词的其相对位置关系。此外位置i和i+k的psotional embedding内积会随着相对位置的递增而减小，从而表征位置的相对距离。缺点：由于距离的对称性，Sinusoidal Position Encoding虽然能够反映相对位置的距离关系，但是无法区分i和i+j的方向。

transformer中，模型输入encoder的每个token向量由两部分加和而成：Position Encoding, Input Embedding。Positional Embedding的成分直接叠加于Embedding之上，使得每个token的位置信息和它的语义信息(embedding)充分融合，并被传递到后续所有经过复杂变换的序列表达中去。

![](/public/upload/machine/positional_encoding.jpg)

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

## 其它

[为什么Transformer适合做多模态任务？](https://mp.weixin.qq.com/s/zGUwdaS5qlET_PZ6O2amxg)
1. 其实不是“Transformer适合做多模态任务”，而是Transformer中的Attention适合做多模态任务，准确的说，应该是“Transformer中的Dot-product Attention适合做多模态任务”．
2. 之前的多模态任务是怎么做的? 在Transformer，特别是Vision Transformer出来打破CV和NLP的模型壁垒之前，CV的主要模型是CNN，NLP的主要模型是RNN，那个时代的多模态任务，主要就是通过CNN拿到图像的特征，RNN拿到文本的特征，然后做各种各样的Attention与concat过分类器。
3. 为什么Transformer可以做图像也可以做文本，为什么它适合做一个跨模态的任务？说的直白一点，因为Transformer中的Self-Attetion机制很强大，使得Transformer是一个天然强力的一维长序列特征提取器，而所有模态的信息都可以合在一起变成一维长序列被Transformer处理。当你输入纯句子时，模型能学到这个it主要指animal，那么比如当你输入图片猫+这个句子的时候，模型可能就能学到你前面那张猫指的就是这个animal。
4. attention本身就是很强大的，已经热了很多年了，而self-attention更是使得Transformer的大规模pretrain成为可能的重要原因self-attention的序列特征提取功能其实是非常强大的，如果你用CNN，那么一次提取的特征只有一个限定大小的矩阵，如果在句子里做TextCNN，那就是提取一小段文字的特征，最后汇聚到一起；如果做RNN，那么会产生长程依赖问题，当句子 太长最后RNN会把前面的东西都忘掉。**self-attention的本质就是对每个token，计算这个token相对于这个句子其他所有token的特征再concat到一起**，无视长度，输入有多长，特征就提多远。**Transformer也可以认为是一个全连接图**， 缓解了序列数据普遍存在的长距离依赖和梯度消失等缺陷 。
5. 那么如果传入的不是句子，而是普通一维序列(也就是一个数组)呢？那就是对序列的每个点(数组的每个值)，计算这个点与序列里其他点的所有特征，这也是Vision Transformer成功的原因，既然是对序列建模，我就把一张图片做成序列不就完了？一整张图片的像素矩阵直接平铺变成序列复杂度太大，那就切大块一点呗(反正CNN也是这种思想，卷积核获得的是局部特征，换个角度来说也是特定patch的特征)，ViT就把一张图片做成了16个patch然后加上对应的position embedding(就是割成小方块变成token向量塞进去加上patch对应图片原始位置的标号)
    ![](/public/upload/machine/vit.jpg)
6. 所以如果你用Transformer来当backbone的时候，**你需要做的就只是把图片，文本，甚至表格信息等其他的所有模态信息全部flatten再concat或者相加成一维数组送进Transformer**，然后期待强大的Self-Attention开始work就可以了。

[面向统一的AI神经网络架构和预训练方法](https://mp.weixin.qq.com/s/3KWbxBf1hEcgSnt2XBCgkg)近几年，自然语言处理（NLP）和图形图像领域（CV）在神经网络架构和预训练上正在经历一个趋向统一的融合趋势。Transformer 的神经网络架构率先在自然语言处理（NLP）领域取得了很好的效果，并成为该领域的主流神经架构。从 2020 年下半年开始，计算机视觉领域也开始将 Transformer 应用到各种视觉问题中，并取代此前的卷积神经网络，成为新的主流神经架构。关于统一的好处，这里列举三点：
1. 技术和知识共享：技术和知识共享能让不同领域都有更快的进步，对于 NLP 和 CV 领域来说，深度学习革命的早期主要是 CV 技术在影响 NLP 领域，而最近两年 NLP 领域进展迅速，很多很好的技术被 CV 领域所借鉴。
2. 促进多模态应用：技术的统一也在催生全新的多模态研究和应用，例如实现零样本物体识别能力的 CLIP 模型、能给定任意文本生成图片的 DALL-E 模型等等。
3. 实现降本增效：我们举芯片的例子，以前设计芯片时需要优化不同的算子，例如卷积、Transformer 等，而现在芯片设计只需要优化 Transformer 就可以解决 90% 以上问题。Nvidia 最新的 H100 GPU 的一个主要宣传点，就是可以将 Transformer 的训练性能提升 6 倍。

谷歌的 ViT 模型就是把图像切块切成不重叠的块，将每个块当作一个 token，于是就可以将 NLP 里的 Transformer 直接应用于图像分类。这种建模相比 CNN 有两个明显的好处：
1. 相比 CNN 这种基于静态模板的网络，Transformer 的滤波模板是随位置动态的，可以更高效地建模关系。
2. 可以建模长程关系（Long-term relationship），可以建模远距离的 token 之间的关联。

[五年时间被引用3.8万次，Transformer宇宙发展成了这样](https://mp.weixin.qq.com/s/cVuBfrrtGBpNlZUekxgnmg) 未读。 

汪涛：这次的人工智能爆发一方面是算力的不断提升，另一个是Trasformer这个新算法的进步。**它是CNN（神经网络）带来的深度学习算法之后又一次小的算法革命（本质上还是神经网络）**。只要利用了这种新的算法，只有量的区别，不会有什么“涌现”“不涌现”的本质区别。有些是已经涌现了，而有些还没有涌现。如果只是一些量的差异，只要在量上不断改进就可趋同或超越，而如果是质的差别，就可能很长时间超越不了。PS：Trasformer 的上限决定了这一波“智能”的上限。
