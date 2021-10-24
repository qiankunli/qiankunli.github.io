---

layout: post
title: 直觉上理解深度学习
category: 技术
tags: MachineLearning
keywords: 深度学习

---

<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

* TOC
{:toc}

## 发展历史

神经网络技术起源于上世纪五、六十年代，当时叫感知机（perceptron），拥有输入层、输出层和一个隐含层。输入的特征向量通过隐含层变换达到输出层，在输出层得到分类结果。但是，单层感知机有一个严重得不能再严重的问题，即它对稍复杂一些的函数都无能为力（比如最为典型的“异或”操作）。随着数学的发展，一票大牛发明多层感知机，摆脱早期离散传输函数的束缚，使用sigmoid或tanh等连续函数模拟神经元对激励的响应，在训练算法上则使用反向传播BP算法。对，这货就是我们现在所说的神经网络NN。多层感知机给我们带来的启示是，**神经网络的层数直接决定了它对现实的刻画能力**——利用每层更少的神经元拟合更加复杂的函数。但随着神经网络层数的加深带来了很多问题，参数数量膨胀，优化函数越来越容易陷入局部最优解，“梯度消失”现象更加严重。当然有一些通用方法可以解决部分问题， 但在具体的问题领域 人们利用问题域的特点提出了 一些变形来解决 层数加深带来的问题。PS：充分利用问题域的特点 是设计算法的基本思路。

Artificial neural networks use networks of activation units (hidden units) to map inputs to outputs. The concept of deep learning applied to this model allows the model to have multiple layers of hidden units where we feed output from the previous layers. However, **dense connections between the layers is not efficient, so people developed models that perform better for specific tasks**.

The whole "convolution" in convolutional neural networks is essentially based on the fact that we're lazy and want to exploit spatial relationships in images. This is a huge deal because we can then group small patches of pixels and effectively "downsample" the image while training multiple instances of small detectors with those patches. Then a CNN just moves those filters around the entire image in a convolution. The outputs are then collected in a pooling layer. The pooling layer is again a down-sampling of the previous feature map. If we have activity on an output for filter a, we don't necessarily care if the activity is for (x, y) or (x+1, y), (x, y+1) or (x+1, y+1), so long as we have activity. So we often just take the highest value of activity on a small grid in the feature map called max pooling.

If you think about it from an abstract perspective, **the convolution part of a CNN is effectively doing a reasonable way of dimensionality reduction**. After a while you can flatten the image and process it through layers in a dense network. Remember to use dropout layers! (because our guys wrote that paper :P)

Let's talk RNN. Recurrent networks are basically neural networks that evolve through time. Rather than exploiting spatial locality, they exploit sequential, or temporal locality. Each iteration in an RNN takes an input and it's previous hidden state, and produces some new hidden state. The weights are shared in each level, but we can unroll an RNN through time and get your everyday neural net. Theoretically RNN has the capacity to store information from as long ago as possible, but historically people always had problems with the gradients vanishing as we go back further in time, meaning that the model can't be differentiated numerically and thus cannot be trained with backprop. This was later solved in the proposal of the LSTM architecture and subsequent work, and now we train RNNs with BPTT (backpropagation through time). Here's a link that explains LSTMs really well: http://colah.github.io/posts/2015-08-Understanding-LSTMs/Since then RNN has been applied in many areas of AI, and many are now designing RNN with the ability to extract specific information (read: features) from its training examples with attention-based models.

学习路径上，先通过单层神经网络（线性回归、softmax回归）理解基本原理，再通过两层感知机理解正向传播和反向传播，增加层数可以增强表现能力，增加特殊的层来应对特定领域的问题。

## 线性回归

很多人工智能问题可以理解为分类问题，比如判断一个邮件是否为垃圾邮件，根据身高体重预测一个人的性别，当我们将目标特征数值化之后，可以将目标映射为n维度空间上的一个点

从几何意义上来说，分类问题就是 在n维空间上将 一系列点 划分成 不同的集合，以二分为例

1. 对于二维空间，分类器就是一条线，直线or 曲线
2. 对于三维空间，分类器就是一个面，平面or 曲面
3. 对于高维空间，xx

**我们要做的就是根据一系列样本点，找到这条“分界线/面”**。线性回归假设特征和结果满足线性关系，线性关系的表达能力非常强大，通过对所有特征的线性变换加一次非线性变换，我们可以划一条“近似的分界线/面”，虽然并不完全准确，但大部分时候已堪大用。就像数学中的泰特展开式一样，不管多么复杂的函数，泰勒公式可以用这些导数值做系数构建一个多项式来近似函数在这一点的邻域中的值。而不同的函数曲线其实就是这些基础函数的组合，理所当然也可以用多项式去趋近

### 正向传播——计算估计值的过程

[【机器学习】代价函数（cost function）](https://www.cnblogs.com/Belter/p/6653773.html)

给出一个输入数据，我们的算法会通过一系列的过程得到一个估计的函数，这个函数有能力对没有见过的新数据给出一个新的估计，也被称为构建一个模型。假设有训练样本(x, y)，数据有两个特征x1,x2，模型为h，参数为θ，$\theta=(w1,w2,b)$，估计函数$h(x)=g(θ^T x)=g(w_1x_1+w_2x_2+b)$。即对x1,x2 施加一次线性变换 + 非线性变换，（$θ^T$表示θ的转置）。因为x1,x2 都是已知的，所以h(θ) 是一个关于θ的函数

两层神经网络的表示（不同文章的表示用语略微有差异）

![](/public/upload/machine/two_layer_neural_network.jpeg)

### 反向传播——调整θ的过程

从直觉上来说， 如何确定最优的w、b？其他参数都不变，w（或b）的微小变动，记作Δw（或Δb），然后观察输出有什么变化。不断重复这个过程，直至得到对应最精确输出的那组w和b，就是我们要的值。

反向传播的思路：

1. 概况来讲，任何能够衡量模型预测出来的值h(θ)与真实值y之间的差异的函数都可以叫做代价函数C(θ)，如果有多个样本，则可以将所有代价函数的取值求均值，记做J(θ)。因此很容易就可以得出以下关于代价函数的性质：

    * 对于每种算法来说，代价函数不是唯一的；
    * 代价函数是参数θ的函数；
    * 总的代价函数J(θ)可以用来评价模型的好坏，代价函数越小说明模型和参数越符合训练样本(x, y)；
    * J(θ)是一个标量；

2. 当我们确定了模型h，后面做的所有事情就是训练模型的参数θ。那么什么时候模型的训练才能结束呢？这时候也涉及到代价函数，由于代价函数是用来衡量模型好坏的，我们的目标当然是得到最好的模型（也就是最符合训练样本(x, y)的模型）。因此训练参数的过程就是不断改变θ，从而得到更小的J(θ)的过程。理想情况下，当我们取到代价函数J的最小值时，就得到了最优的参数θ.例如，J(θ) = 0，表示我们的模型完美的拟合了观察的数据，没有任何误差。

3. 代价函数衡量的是模型预测值h(θ) 与标准答案y之间的差异，所以**总的代价函数J是h(θ)和y的函数**，即J=f(h(θ), y)。又因为y都是训练样本中给定的，h(θ)由θ决定，所以，最终还是模型参数θ的改变导致了J的改变。

4. 在优化参数θ的过程中，最常用的方法是梯度下降，这里的梯度就是代价函数J(θ)对`θ1, θ2, ..., θn`的偏导数。由于需要求偏导，我们可以得到另一个关于代价函数的性质：选择代价函数时，最好挑选对参数θ可微的函数（全微分存在，偏导数一定存在）


### 梯度下降法 ==> 求使得J极小的(w1,w2,b)

梯度是一个矢量，梯度的方向是**方向导数**中取到最大值的方向，梯度的值是方向导数的最大值。举例来说，对于3变量函数 $f=x^2+3xy+y^2+z^3$ ，它的梯度可以这样求得
$$\frac{d_f}{d_x}=2x+3y$$
$$\frac{d_f}{d_y}=3x+2y$$
$$\frac{d_f}{d_z}=3z^2$$

于是，函数f的梯度可表示为：$grad(f)=(2x+3y,3x+2y,3z^2)$，针对某个特定点，如点`A(1, 2, 3)`，带入对应的值即可得到该点的梯度`(8,7,27)`，朝着向量点`(8,7,27)`方向进发，函数f的值增长得最快。

对于机器学习来说

1. 首先对(w1,w2,b)赋值，这个值可以是随机的，也可以让(w1,w2,b)是一个全零的向量
2. 改变(w1,w2,b)的值，使得J(w1,w2,b)按梯度下降的方向进行减少，梯度方向由J(w1,w2,b)对(w1,w2,b)的偏导数确定
3. $w1=w1+\sigma d_{w1}$ 依次得到 w1,w2,b 的新值，$\sigma$为学习率


### 细节问题

#### 正向传播为什么需要非线性激活函数

![](/public/upload/machine/activation_function.jpg)

如果要用线性激活函数，或者没有激活函数，那么无论你的神经网络有多少层，神经网络只是把输入线性组合再输出，两个线性函数组合本身就是线性函数，所以不如直接去掉所有隐藏层。唯一可以用线性激活函数的通常就是输出层

常见激活函数： sigmoid,tanh,ReLU,softMax [机器学习中常用激活函数对比与总结](https://zhuanlan.zhihu.com/p/40327813)

#### 损失函数如何确定

[机器学习-损失函数](https://www.csuldw.com/2016/03/26/2016-03-26-loss-function/)

**损失函数是h(θ)和y的函数**，本质是关于$\theta$的函数，分为

1. log对数损失函数，PS：概率意义上，假设样本符合伯努利分布

	$$L(Y,P(Y|X))=-logP(Y|X)$$
2. 平方损失函数 ，PS：概率意义上，假设误差符合高斯分布
	
	$$L(Y,h_\theta(X))=(Y-h\_\theta(x))^2$$

   $$MSE(\theta)=MSE(X,h_\theta)=\frac{1}{m}\sum\_{i=1}^m(\theta^T.x^i-y^i)^2$$
    
    

3. 指数损失函数
4. Hinge损失函数
5. 0-1损失函数
6. 绝对值损失函数

	$$L(Y,h_\theta(X))=|Y-h\_\theta(x)|$$



https://zhuanlan.zhihu.com/p/46928319

[对线性回归，logistic回归和一般回归的认识](https://www.cnblogs.com/jerrylead/archive/2011/03/05/1971867.html)

## 从自动微分法来理解线性回归

训练的目的是 得到一个 `<W,b>` 使损失函数值最小，损失函数是所有样本数据 损失值的和，是一个关于`<W,b>` 的函数（只是工程上通常不求所有样本，只取 batch ）。 最优解（极大值/极小值）在 导数为0 的地方，对于`y=f(x)`，手动可以求解 `f'(x)=0` 的解（解析解），大部分深度学习模型并没有解析解，对于计算机 只能梯度下降法来求微分， 已知 $x_i$，计算`dy/dx`，进而得到 $x_{i+1}$，使得$f'(x_{i+1})$ 更接近0。结论：求最优解必须  先求微分。

[一文读懂自动微分（ AutoDiff）原理](https://zhuanlan.zhihu.com/p/60048471)假设我们定义了 $f(x,y)=x^2y+y+2$
，需要计算它的偏微分 `df/dx` 和 `df/dy` 来进行梯度下降，微分求解大致可以分为4种方式：
1. 手动求解法(Manual Differentiation)，直接算出来 $\frac{df}{dx}=2xy$，用代码实现这个公式。然而大部分深度学习模型不好算公式。 
2. 数值微分法(Numerical Differentiation)，强调一开始直接代入数值近似求解. 直接 $\lim_{x \to x_0}\frac{h(x)-h(x_0)}{x-x_0}$

3. 符号微分法(Symbolic Differentiation)，代替第一种手动求解法的过程，强调直接对代数进行求解，最后才代入问题数值；
4. 自动微分法(Automatic Differentiation)

自动微分法是一种介于符号微分和数值微分的方法，自动微分将符号微分法应用于**最基本的算子**（不直接算完），比如常数，幂函数，指数函数，对数函数，三角函数等，然后代入数值，保留中间结果，最后再应用于整个函数。因此它应用相当灵活，可以做到完全向用户隐藏微分求解过程，由于它只对基本函数或常数运用符号微分法则，所以它可以灵活结合编程语言的循环结构，条件结构等，使用自动微分和不使用自动微分对代码总体改动非常小，并且由于它的计算实际是一种图计算，可以对其做很多优化，这也是为什么该方法在现代深度学习系统中得以广泛应用。

![](/public/upload/machine/cal_diagram.png)

要计算 `dy/dx` 也就是 `dn7/dx`，可以先计算 `dn7/dn5`，即 `dn7/dx=dn7/dn5 * dn5/dx`，这么链式推导下去。

如果你在Tensorflow 中加入一个新类型的操作，你希望它和自动微分兼容，你需要提供一个函数来建一个计算它的偏导数（相对于它的输入）的图。例如你实现了一个函数来计算输入的平方，$f(x)=x^2$，你需要提供一个偏导数函数 $f'(x)=2x$。

个人理解，对于`y=f(x)`， `f'(x)` 要么是个常量，要么也是x 的函数。也就是`dn7/dn5` 不是常量 就是关于n5 的函数，在自上到下 求导（反向传播）用到 `dn7/dn5`之前，得先自下而上（正向传播）求出来n5 再说。所以正向 + 负向传播加起来 是为了求微分，反向传播依赖正向传播，下一次迭代正向传播依赖本次反向传播。在机器学习框架里，每个node 会包含一个grad 属性，记录了当前的node微分函数或值。

## 逻辑回归

逻辑回归是线性回归的一种特例，激活函数使用 Sigmoid，损失函数使用 log对数损失函数

正向传播过程

$$f(x)=\theta^Tx=w_1x_1+w_2x_2+b$$
$$g(z)=\frac{1}{1+e^{-z}}$$

## 细节问题

### 激活函数为什么使用Sigmoid

1. 它的输入范围是正负无穷，而值刚好为（0，1），正好满足概率分布为（0，1）的要求。我们**用概率去描述分类器**，自然比单纯的某个阈值要方便很多
2. 它是一个单调上升的函数，具有良好的连续性，不存在不连续点

### 损失函数为什么使用对数损失函数

$h_\theta(x)$与y 只有0和1两个取值

#### 对数损失函数的直接意义

[logistic回归详解(二）：损失函数（cost function）详解](https://blog.csdn.net/bitcarmanlee/article/details/51165444)

$$
cost(h_\theta(x),y)=
\begin{cases}
-logh_\theta(x)      & \text{if y=1} \\
-log(1-h_\theta(x))  & \text{if y=0} 
\end{cases}
$$

当y=1时

1. 如果此时$h_θ(x)=1$，$logh\_θ(x)=0$，则单对这个样本而言的cost=0，表示这个样本的预测完全准确。
2. 如果此时预测的概率$h_θ(x)=0$，$logh\_θ(x)=-\infty$，$-logh\_θ(x)=\infty$，相当于对cost加一个很大的惩罚项。 

当y=0 时类似，所以，**取对数在直觉意义上可以将01二值与 正负无穷映射起来**

汇总一下就是

|$h_\theta(x)$|y|cost|
|---|---|---|
|0|0|0|
|0|1|$\infty$|
|1|0|$\infty$|
|1|1|0|

将以上两个表达式合并为一个，则单个样本的损失函数可以描述为： 

$$
cost(h_\theta(x),y)=
-y\_ilogh\_\theta(x)-
(1-y\_i)log(1-h\_\theta(x))
$$

因为$h_\theta(x)$与y 只有0和1两个取值，该函数与上述表格 或分段表达式等价

全体样本的损失函数可以表示为： 

$$
cost(h_\theta(x),y)=
\sum\_{i=1}^m
-y\_ilogh\_\theta(x)-
(1-y\_i)log(1-h\_\theta(x))
$$


#### 对数损失函数的概率意义

[逻辑回归为什么使用对数损失函数](https://blog.csdn.net/saltriver/article/details/63683092)

对于逻辑回归模型，假定的概率分布是伯努利分布（p未知）

$$
P(X=n)=
\begin{cases}
1-p      & \text{n=0}  \\
p        & \text{n=1}
\end{cases}
$$

概率公式可以表示为（ x只能为0或者1）

$$
f(x)=p^x(1-p)^{1-x}
$$

假设我们做了N次实验，得到的结果集合为 `data={x1,x2,x3}`，对应的最大似然估计函数可以写成

$$
P(data|p)=
\prod_{i=1}^Nf(x\_i)=
\prod\_{i=1}^Np^{x\_i}(1-p)^{1-x\_i}
$$

要使上面的式子最大，等价于使加上ln底的式子值最大，我们加上ln的底就可以将连乘转换为加和的形式

$$
lnP(data|p)=
\sum\_{i=1}^Nx\_ilnp+(1-x\_i)(1-p)
$$

对数损失函数与上面的极大似然估计的对数似然函数本质上是等价的，xi 对应样本实际值yi，p即 $h_\theta(x)$，所以逻辑回归直接采用对数损失函数来求参数，实际上与采用极大似然估计来求参数是一致的

**几何意义和概率意义殊途同归（当然，不是所有的损失函数都可以这么理解）**


## 参数和超参数

parameters: $W^{[1]},b^{[1]},W^{[2]},b^{[2]},...$

超参数hyperparameters: 每一个参数都能够控制w 和 b

1. learning rate
2. 梯度下降算法循环的数量
3. 隐层数
4. 每个隐层的单元数
5. 激活函数

深度学习的应用领域，很多时候是一个凭经验的过程，选取一个超参数的值，试验，然后再调整。

吴恩达：我通常会从逻辑回归开始，再试试一到两个隐层，把隐藏数量当做参数、超参数一样调试，这样去找比较合适的深度。

### 为什么批量是一个超参数

理论上来讲，要减小优化的损失是在整个训练集上的损失，所以每次更新参数时应该用训练集上的所有样本来计算梯度。但是实际中，如果训练集太大，每次更新参数时都用所有样本来计算梯度，训练将非常慢，耗费的计算资源也很大。所以用训练集的一个子集上计算的梯度来近似替代整个训练集上计算的梯度。这个子集的大小即为batch size，相应地称这样的优化算法为batch梯度下降。batch size可以为训练集大小，可以更小，甚至也可以为1，即每次用一个样本来计算梯度更新参数。batch size越小收敛相对越快，但是收敛过程相对不稳定，在极值附近振荡也比较厉害。

## 各种深度模型

虽然深度模型看上去只是一个具有很多层的神经网络，然而获得有效的深度模型并不容易。 

### CNN

卷积神经网络 CNN 是含有卷积层的神经网络。CNN 新增了卷积核大小、卷积核数量、填充、步幅、输出通道数等超参数。
1. 卷积层保留输入形状，使图像的像素在高和宽两个方向上的相关性均可能被有效识别。
2. 卷积层通过滑动窗口将同一卷积核与不同位置的输入重复计算，从而避免参数尺寸过大。

提起卷积神经网络，也许可以避开VGG、GoogleNet，甚至可以忽略AlexNet，但是很难不提及LeNet。LeNet开创性的利用卷积从直接图像中学习特征，在计算性能受限的当时能够节省很多计算量，同时卷积层的使用可以保证图像的空间相关性，最后再使用全连接神经网络进行分类识别。

原始的LeNet是一个5层的卷积神经网络，它主要包括两部分：卷积层块 和 全连接层块，其中卷积层数为2（池化和卷积往往是被算作一层的），全连接层数为3。卷积层块由卷积层加池化层两个这样的基本单位重复堆叠构成。卷积层用来识别图像里的空间模式，如线条和物体局部，之后的最大池化层则用来降低卷积层对位置的敏感性。

![](/public/upload/machine/lenet5_overview.jpeg)

`28*28` 的灰度图片 如果直接用 DNN来训练的话，全连接层输入为28*28=784。LeNet 中卷积层块的输出形状为(通道数 × 高 × 宽)，当卷积层块的输出传入全连接层块时，全连接层块会将每个样本变平（flatten）。原来是形状是：(16 × 5 × 5)，现在直接变成一个长向量，向量长度为通道数 × 高 × 宽。在本例中，展平后的向量长度为：16 × 5 × 5 = 400。全连接层块含3个全连接层。它们的输出个数分别是120、84和10，其中10为输出的类别个数。

![](/public/upload/machine/lenet5_layer.jpeg)

在卷积神经网络，卷积核的数值是未知的，是需要通过“学习”得到的，也就是我们常说的参数。根据不同的卷积核计算出不同的“响应图”，这个“响应图”，就是特征图(feature map)。这就是为什么总是说利用CNN提取图像特征，卷积层的输出就是图像的特征。卷积核的数量关系到特征的丰富性。

### RNN（未完成）

循环神经网络，引入状态变量来存储过去的信息，并用其与当期的输入共同决定当前的输出。

## 小结

首先你要知道 机器学习的两个基本过程

1. 正向传播，线性 + 非线性函数（激活函数） 得到一个估计值
2. 反向传播，定义损失函数，通过（链式）求导 更新权重

关键问题

1. 激活函数的选择
2. 损失函数的选择
3. 对损失函数及$\theta$链式求偏导数，涉及到矩阵的导数，根据偏导 + 学习率 调整$\theta$
4. 向量化上述过程
4. 使用python 中线程的 矩阵/向量计算的库 将上述过程代码化， 涉及到numpy的学习