---

layout: post
title: 《技术的本质》笔记
category: 生活
tags: Life
keywords: 技术的本质

---

## 简介

许式伟：架构就是对业务系统的**正交分解**。自己的一些理解：需求纷繁复杂，要找到需求之前正交的部分，也就是实现领域需求的最小单元。

三个基本原理：
1. 一切技术都是元素的组合；
2. 这些元素本身也是技术；
3. 所有技术都利用现象达到某个目的。

## 为什么要模块化？

表面看，喷气发送机和一台计算机 是很不同的东西， 一个是物质零件， 一个是逻辑指令。但他们的结构却是相同的：都是由集成块结构起来， 集成块之间的相互联系，共同服务于某一个基本原理的核心集成。比如发送机，它的原理很简单：在稳定的压缩空气流中燃烧燃料，产生高速气体向后排出。为完成任务，机器会采用一个主集成，包括五个主要的子系统：进气道、压气机、燃烧室、涡轮和尾喷管。空气由进气道进入，流进由一系列大风扇组成的压气机加压，高压气流进入到燃烧室中， 和燃料混合并被点燃，之后高温气体驱动涡轮运转，涡轮驱动增压机，高温气体从尾喷管高速喷出，进而产生推力。这些元器件形成了核心集成，围绕在它周围的是支持这一主功能的许多非常复杂的子系统，主要包括：燃料输送系统、压缩机反熄火系统、涡轮叶片冷却系统、发动机仪表系统、电气系统等。集成件之间通过复杂的像迷宫似的管道和电线相互连接来传递功能，以便让其他的系统得以应用。

这一建构过程和计算机编程没什么不同，都是要应用一个基本原理， 即核心概念或逻辑，作为编程背后的支撑。完成这个原理需要一套具有积木一样结构的支撑程序和子程序系统。这在计算机中，常恰如其分地被称为”主程序（main）“，main又需要其子功能或子程序的支持。PS：main 中的代码 要体现或者只包含 ”基本原理“

不论是发动机还是计算机程序，所有的组件之间必须小心的保持平衡，每个部分都必须在一个由其它相关部分 设定的约束范围内运行，包括温度、流速、载荷、电压等。每部分又都要为依赖它们的那些组件建立一个适合的工作环境。

赫伯特·西蒙(Herbert Simon) 讲了一个关于两个制表匠的经典寓言故事。假设每只表都集成了1000个零件。一个名叫坦帕斯的钟表匠一个零件一个零件地安装，但是，如果他的工作被打断了，或者丢下一只没完成的手表，他就必须从头开始。相反，另一个名叫赫拉的钟表匠则是将10个模块组装在一起，最后组成手表。每个模块又由10个子模块组成，每个子模块再由10个零件组成。如果他暂停工作或者被打断工作，他只是损失了一小部分工作成果。西蒙的重点是：**将零件集成化可以更好地预防不可预知的变动，且更容易修复**。对此，我们可以进一步加以扩展,模块将允许技术的组成部分分别进步:可以对每个部分分别加以关注和改进,对工作性能分别进行试验和分析—每个“集成”可以“悄悄地”被探察或者更换而不必解体余下的技术整体。将技术进行功能性分组还简化了设计过程，如果设计者面对数以万计的零件， 那么他们将淹没在细琐零件的汪洋之中。 

将技术分割为功能单元需要付出一些代价,至少要有一些精神上的努力。只有当模块被反复使用,且反复使用的次数足够多时,才值得付出代价将技术进行分割。这和亚当·斯密的劳动分工理论类似:斯密指出,只有在生产的数量足够大的情况下,才值得将工厂的工作划分成专业工作。

总结：**每一种装置都来自于一个中心原理**，并且有一个中央集成(一个装置的整体骨干或者执行方法)，加上其他的零部件围绕其周围，令其可以工作并且规定它的功能这些组件的每个自身都是一个技术，因此它自己也有一个核心骨干，以及附着其上的其他组件。这一结构是递归性的。

## 技术的本质

从本质上看，**技术是被捕获并加以利用的现象的集合**，或者说，技术是对现象有目的编程。技术是指向某种目的的，被编程了的现象。一系列被捕获现象的互相支持、互相利用、互相“交谈”，就如同计算机子程序之间的互相“呼应”（calling）一样。

生物对基因加以编程从而产生无数的结构（任何生物都是大概21000种基因），技术对现象加以编程从而产生无数的应用。

随着现象家族被不断发掘，先前发现的现象为此后现象的发现提供了方法和启示。一个现象引发另一个，然后是下一个，直到最后相关现象的整个矿脉都被发掘出来。一个现象家族形成了一个井下硐室，硐室之间由矿层或巷道相连，一个通往另一个。然而这还不是全部,一个硐室一个现象家族，会通过巷道通往任何硐室，即使通向完全不同的现象家族也可以。比如，先前的电现象就导致了后来量子现象的发现。现象构成一个被发掘出来的相互联系的硐室和巷道系统,而且整个地下系统都是可以相互连通的。

现象在本性上就是可以做点什么。当人们意识到这个潜在用途时，就可以驯服它做事了，比如将变换磁场感应电流这一现象转译为发电的手段，两者之间其实并不特别遥远。当然不是所有的现象都能够被驯服，但是，一旦一个现象家族被发现就会有一连串的技术尾随而至。1750-1875年，主要的电现象，例如静电现象、电蚀作用、由电场和磁场导致的电流偏转、感应现象、电磁辐射以及辉光放电现象都被发现了。随着对这些现象的捕捉和驯服，随之而来的是一系列的方法、工艺及设备，其中包括电池、电容和电感、变压器、电报、发电机和电动机、电话、无线电报、阴极射线管、真空管等。

**设计就如同语言表达**：一个新的设备或方法是由一个域中适用的零部件，或者也可以说是适当的词汇聚集而成的。从这个意义上来看，一个域构成了一种语言，当某个域在产生一件新的技术产品时，就是这个域在以某种语言进行表达。这也意味着技术中的主要活动，即工程设计，变成了一种写作方式，一种(或几种)语言的表达。我们并不熟悉这种看待设计的方式，但是它值得我们考虑一下。
1. 语言中有清晰和不清晰的表达方式、恰当和不恰当的语言选择之分，设计亦是如此;
2. 语言可以简明扼要，设计亦是如此;
3. 语言有不同程度的复合句式，设计亦是如此;
4. 用语言表达一个理念可以只用一个简单句,也可以用一整本书,并辅之以若干支撑材料来呈现一个主题,设计亦是如此。
5. 语言中任何目的的表达都可以有很多选择。类似地，技术为达到任何目的也可以选择多种组合。
6. 如同语言的组织必须依据语言规则一样,设计的建构也要根据域允许的组合规则来进行我将这种规则称为语法。

**电子学的语法背后是电子运动的物理学以及电现象的规律，DNA 操作的语法背后是核苷酸和与DNA一起工作的酶的内在特性**。 PS：我服务了香港警队30年，认识不少人，也得罪不少人。不过在这30年我学会了一件事，就是每一个机构，每一个部门，每一个岗位都有自己的游戏规则。不管是明是暗，第一步学会它，不过好多人还没有走到这一步就已经死了，知道为何？自以为是。第二步，就是在这个游戏里面把线头找出来，学会如何不去犯规，懂得如何在线球里面玩，这样才能勉强保持性命。


好的设计事实上就像一首好诗。这不是指诗的崇高感，而是指从众多的可能性中为每个部分作出完全正确的选择。每个部分必须紧密配合，各部分的运行一定要准确，必须符合与其余部分的互动规则。**一个好的设计的美感在于适切性，在于为所获得的东西付出最少的努力。它源于一种感觉：恰如其分，增一分则多，减一分则少**。技术中的美不一定非得需要原创性，技术语法不论在形式上，还是短语的选用上，都大量借鉴了其他语言。如此看来，具有讽刺意味的是，设计所依赖的往往是对常用元素的组合和操纵。尽管如此，一个美的设计总是包含一些意想不到的组合，并以其适切性震撼人心。

技术就像写作、演讲或高级烹饪，存在不同程度的流畅性、表达能力和自我表现。一个建筑界的新手，就像一个学习外语的新手一样，即使有时不太适合，也会一遍又一遍地一直使用同样的组合，或者说同样的短语。一个熟练的建筑师深谙域的艺术，他会摒弃纯粹的语法规则，而诉诸直觉知识，使他们组合、搭配在一起。一位真正的大师会挑战极限， 他会在域中赋诗，会在其所使用的惯用组合中留下自己的“签名”。技术大师实际上非常难得，因为技术的语法不像语言的语法，它变化迅速。技术语法最初是原始的，只能被模糊地感知;当组成它们的基础知识发展时，它们得以深化;当发现了新的匹配良好的组合，或者发现了设计在日常使用中遇到的困难时，它们就进化了。这个过程水无终结。结果就是，即便是此中的专家，也不能完全跟上它们在“域”中组合的原则。

一个域或者一个技术体提供一种表达的语言、一套肉容和实践的词汇表，设计师可以从中作出选择。计算技术（或数字技术）是一个集合，是一个巨大的词汇表，包括硬件、软件、传输网络、语言、超大规模集成电路、算法以及所有和它们相关的组件和实践，我们可以把计算技术，或者任何与之相关的域看作是一个仓库，它们随时准备服务于某种特殊用途。

一个域便是一个想象的王国。在那里，设计者可以在思维中想象自己能做什么。那是一个充满可能性的域世界。电子设计师知道他们可以扩大信号、转换频率、减少噪音、调节载波信息、设置定时回路，还可以利用许许多多其他可靠的操作。他们依据电子世界实现内容的可能性去思考。此外，如果他们是专家，他们应该非常熟悉这个世界，因此他们几乎可以自动地组合操作并预见结果。专家们沉浸在域的世界里，就如我们写信的时候沉浸在文字中一样，他们的精神沉浸其中。他们着眼于目的性，然后在头脑中进行每一步操作，这很像一个作曲家构思出一个主题，但是却要回过头来诉诸乐器去表达它。

## 技术的发展

**技术是建构的产物，即由零部件或组件组合而成**。
1. 如果我们从技术外部 将技术看作是一个整体的对象，那么个体技术，如计算机、基因测序、蒸汽机似乎是相对固定的。但从内部看，技术的内部组件一直在变化，比如替换零部件、改进材料、改变建构方法、有了新元素可以利用等，技术是一种非常易变的东西。PS：比如苹果手机 的不停迭代
2. 如何看待技术的可能性。从外部看，每项技术都是在完成某个目标，比如如果想导航，就用北斗。但技术不仅是提供某种特定的功能而存在，它实际上还提供了一个组合或编程的词汇表，这个词汇表的存在使技术可以提供无穷无尽的新颖方法，去实现无穷无尽的新颖目的。PS：比如有一个电脑，他不单是可以看视频、听音乐，可以提供无穷无尽的可能。

设计就是关于解决方案的选择，因此，**设计与选择有关**。

我们是人类，我们需要的不只是经济上的舒适。我们需要挑战，我们需要意义，我们需要目的，我们需要和自然融为一体。