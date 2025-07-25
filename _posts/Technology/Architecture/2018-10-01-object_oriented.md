---

layout: post
title: 重新看面向对象设计
category: 架构
tags: Architecture
keywords: 面向对象

---

## 简介

我一直有一个比方，如果把程序员写程序比作史官写历史，那么面向过程就是编年体通史，而面向对象更像是纪传体通史。编年体通史以时间（或一段时间）为中心，而纪传体通史则以人物为中心，譬如《史记》的《高祖本纪》等。类似的，面向过程多认为程序由一个个函数组成（依照顺序先后调用），譬如历史由一件件大事组成。而面向对象倾向于认为程序由一个个对象组成，譬如历史由一个个人的故事组成。

[Is Inheritance Dead? A Detailed Look Into the Decorator Pattern](https://dzone.com/articles/is-inheritance-dead)未读完

[编程就是压缩](https://mp.weixin.qq.com/s/bXl9IP5CyoNHND6dyXNFvA)
1. static compression 的目标：给定所有的当前需求，最小化整体的实现代码量。方法就是能复用就复用，不能简单的复用的，创造条件也要强行复用。
2. generalization ability 的目标：给定过去的需求，进行合理的预测，最小化迁移到未来可能的新需求的成本。其实现手段就是保持代码整洁，**有结构**。但是问题在于什么叫“有结构”很难学习。相反，最小化整体的实现代码量是一个更容易销售，也更容易学习的目标。这里“合理的预测”似乎和不要预先设计是矛盾的。然而所有的“泛化”都是某种预测。整洁有结构的代码“赌”的是什么呢？赌的就是新需求大概率是 localized 的。
3. 压缩的前提是对象本身具有内在的规律性。压缩的过程就是把规律性的部分和特异性的部分分离出来。这个也就体现在了各种“中台”，“低代码”，“DSL”的说辞之中，所谓一部分人搞定 80%，剩下的领域专家（也就是处理特异性的人）去搞定特异性的 20%。然而这里的前提是处理 20% 的人要先知道之前的工序做了哪部分的 80%，才能知道自己要做哪 20%。wetware 相比 hardware 就是自然进化的速度太慢，千兆网，万兆网，升级速度杠杠的。但是人与人的沟通还是要靠非常低效率的文字理解和声波通信。除非需求内在的 Regularity 非常强，可以用极其少量的参数组合已有功能来表达新需求，否则大概率写这 20% 代码的人会非常痛苦。这也是为什么“低代码”无法流行的根本原因。
4. 一部分人处理 Regularity，一部分人处理 Specificity：这就是所谓“中台”，“低代码”，“DSL”。核心是 specificity 部分要足够小，才能弥补沟通成本。人类来提供 Regularity，机器处理 Specificity：一开始是 logical programming，人类要提供非常多的 regularity。后来是 probabilistic programming，人类需要提供 generating function 来编码专家经验到概率分布里。到后来 deep neural network，人类需要用 convolution filter 来用所谓 inductive bias 的方式 encourage 神经网络往某个方向学习。越来越多的程序员会从直接编码规则来驾驭机器，变成“声明式编程”。把规律性的部分提取成 what，让机器去找到 how。

[一文教会你领域建模](https://mp.weixin.qq.com/s/3x4fK7rtA2US9fxoKpoAiA)
1. 为什么需要建模？ 想象一个应用程序，其代码有几十万行。如果使用C语言平均每个子程序编写50到100行代码，那么子程序就有几千到几万个。如果使用面向对象，平均在每个类中汇总10个子程序，那么类的总数就比子程序的总数减少1个数量级，为几百到几千个。当然，即便是1000个类，数量依然庞大。为了可理解性，我们可以按照内聚性，进一步使用“包”、“组件”、“模块”、“服务”做归类分组，以减少我们大脑的思维负担。**这就是我们为什么需要面向对象的原因**。可是，我们很多系统还是失控了。因为他们虽然使用了面向对象语言，但干的还是面向过程的勾当。体现在代码上，就是缺少对领域概念的清晰表达、缺乏面向对象的领域封装、领域边界模糊不清、以及领域模型抽象不正确。
    ![](/public/upload/kubernetes/domain_model.png)
2. 理解业务需求。如何理解？  用户故事，挖掘概念。我们可以借助更多的辅助手段，比如四色建模、事件风暴、专家会谈、阅读领域书籍等等。但归根结底，都是语言的思维游戏。领域建模作为一种能力，是可以通过《刻意练习》习得和不断提升的。越多的锻炼就能建立越强大的**心理表征**能力，届时，你会发现你已经不再需要这些“方法”的指引，更多的时候只是在用“直觉”。

[重拾面向对象软件设计](https://mp.weixin.qq.com/s/UPGnfEYj8e1rnD1igqFe0A)
1. 流程化的设计让编码更加清晰，相比于机器码或汇编，开发效率得到了极大改善，包括现在仍然有很多场景更适合面向过程来完成。但软件工程最大的成本在于维护，**由于面向过程更多聚焦于问题的解决而非领域的设计，代码的重用性与扩展性弊端逐步彰显出来**，随着业务逻辑越来越复杂，软件的复杂性也变得越来越不可控。
2. 面向对象以分类的方式进行思考和解决问题，面向对象的核心是抽象思维。通过抽象提取共性，通过封装收敛逻辑，通过多态实现扩展。面向对象的思想本质是将数据与行为做结合，数据与行为的载体称之为对象，而**对象要负责的是定义职责的边界**。面向过程简单快捷，在处理简单的业务系统时，面向对象的效果其实并不如面向过程。但在复杂系统的设计上，通用性的业务流程，个性化的差异点，原子化的功能组件等等，更适合面向对象的编程模式。但面向对象也不是银弹，甚至有些场景用比不用还糟，一切的根源就是抽象。我们在设计抽象进行分类时，不一定能抓住最合适的切入点，**错误的抽象比没有抽象复杂度更高**。

[面向对象分析与设计的底层逻辑](https://mp.weixin.qq.com/s/aiMzDm_nJgUXuZAFW-9EQw)
1. 应对复杂事物的有一个重要的方法即是抽象，抽象在实际应用过程中，又体现在两种方法上：分层和分类。
2. 分层是通过不同的视角看事物，每一层的关注点是不一样的，这种关注点不同是由自己的视角造成的，比如我们理解计算机，并不需要深入到二进制电信号去理解计算机。
3. 我们可以把一个软件划分成三层：场景、功能和实体，场景层是经常会变的，比如发放优惠券场景就非常多，比如有天降红包领取优惠、分享有礼领取优惠券、新人注册领取优惠券等，这种场景的更迭随着业务的调整变化得非常快，因此场景层是不稳定的。功能支撑某一些的场景集合，对比场景，功能相对而言稳定些，就像前面提到的发放优惠券场景，本质就是给用户发放优惠券，只需要提供发放优惠券的功能即可，至于哪些场景来调用它并不关注，但功能还是基于场景的集合抽象出来的，如果场景场景类型变化了，功能也就随之变化，比如担保交易和预售交易就不一样。实体是稳定的，以担保交易和预售交易为例，它的订单模型大致是一样的，只是新增加了一些信息而已。因此，我们希望从问题空间到解空间，大家看到的、理解的是一致的，而且看到的是问题的本质而非表象，**往往场景、功能是不稳定的，而面向过程又是以功能驱动的**，所以在易变化的场景下，它面临的问题就比较多。比较稳定的是问题空间中的实体对象，所以面向对象分析是现实的需要。
4. 提到面向对象，有部分人会提到封装、继承、多态等特性，然后这些并不是面向对象的本质特性，比如封装，面向过程中也有封装，多态面向过程也有体现，这些特性算不上面向对象特有的特性。面向对象的底层逻辑是基于现实事物做的抽象映射：现实事物对应软件中的对象，我们讨论解空间能对应到问题空间中的对象，两者是一一直接映射的，其它的分析方法是问题空间到解空间的间接映射。
    ![](/public/upload/architecture/design_map.png)
5. 对象从哪里来？问题空间到解空间是一一映射，我们讨论解空间中的对象时，其实它映射到问题空间中的对象，而问题空间中的对象主要来源于业务概念、业务规则、关键事件。大部分的对象是显现的，我们通过理解业务能发现，有的对象是隐性的，需要我们持续对业务有更深的理解才能发掘出来。好的对象模型是需要经过多次迭代打磨出来的，并非一次就能设计得十全十美。
6. 职责是怎么来的？职责分为两类：一类是认知职责；另一类是行为职责。
    1. 认知职责包含：对私有数据封装的认知。对相关对象的认知。对其能够导出或计算的事物的认识。认知职责是基于对象属性的，正所谓"不在其位、不谋其政"，认知职责一定不会超过它的认识范围的。
    2. 行为职责包含：自己执行的行为，包括创建对象或计算。初始化其它对象的动作。控制或协调其它对象的活动。

《程序员底层思维》过程式的代码所做的事情无外乎是取数据、做计算、存数据，在这种情况下，要如何通过代码来显性化的表达业务呢？这其实很难做到，因为缺失了模型与模型之间的关系。除了抽象和封装（增强表达能力），继承和多态提高了代码的扩展性（更有利于应对变化，单单继承在应对多维度变化时是不足的）。有结构化分解要好于没有分解，结构化分解+抽象要好于单纯的结构化分解。

**突破一个类的限制，走向更多的类的协作设计，也是我们进阶的方向**。


## 抽象

抽象思维是程序员最重要的思维能力，抽象的过程就是寻找共性、归纳总结、综合分析，提炼出相关概念的过程。抽象是忽略细节的。编程要依赖于抽象而不是具体，抽象的东西不易变化，而具体的东西容易变化，抽象能在更高的层次上实现细节的隐藏。抽象类是最抽象的，忽略的细节也最多，就像抽象牛，只是几根线条而已。**在代码中可以类比到 Abstract Class 或者 Interface**。抽象具有层次性。抽象层次越高，内涵越小，外延越大，也就是说它的涵义越小，泛化能力越强。比如，牛就要比水牛更抽象，因为它可以表达所有的牛，水牛只是牛的一个种类（Class）。而设计模式是软件开发中抽象化思维的重要经验总结。

[程序员必备的思维能力：抽象思维](https://mp.weixin.qq.com/s/cJ0odiYcphhNBoAVjqpCZQ)

抽象层次/**分层抽象**：抽象是以概念（词语）来反映现实的过程，每一个概念都有一定的外延和内涵．概念的外延就是适合这个概念的一切对象的范围，而概念的内涵就是这个概念所反映的对象的本质属性的总和．例如“平行四边形”这个概念，它的外延包含着一切正方形、菱形、矩形以及一般的平行四边形，而它的内涵包含着一切平行四边形所共有的“有四条边，两组对边互相平行”这两个本质属性。**一个概念的内涵愈广，则其外延愈狭；反之，内涵愈狭，则其外延愈广**，泛化能力越强。然而，其代价就是业务语义表达能力越弱。 例如，“平行四边形”的内涵是“有四条边，两组对边互相平行”，而“菱形”的内涵除了这两条本质属性外，还包含着“四边相等”这一本质属性。“菱形”的内涵比“平行四边形”的内涵广，而“菱形”的外延要比“平行四边形”的外延狭。所谓的抽象层次就体现在概念的外延和内涵上，这种层次性，基本可以体现在任何事物上。每一个抽象层次都有它的用途，对于我们工程师来说，如何拿捏这个抽象层次是对我们设计能力的考验，抽象层次太高和太低都不行。

我经常开玩笑说，你把所有的类都叫Object，把所有的参数都叫Map的系统最通用，因为Object和Map的内涵最小，其延展性最强，可以适配所有的扩展。从原理上来说，这种抽象也是对的，万物皆对象嘛。但是这种抽象又有什么意义呢？它没有表达出任何想表达的东西，只是一句正确的废话而已。越抽象，越通用，可扩展性越强，然而其语义的表达能力越弱。越具体，越不好延展，然而其语义表达能力很强。所以，**对于抽象层次的权衡，是我们系统设计的关键所在**，也是区分普通程序员和优秀程序员的关键所在。PS：接口是定义规范的，而基类往往没啥核心方法。

[重拾面向对象软件设计](https://mp.weixin.qq.com/s/u-JEQmtFqSYFWRJ45RAFYw)面向对象的核心是抽象思维。**通过抽象提取共性，通过封装收敛逻辑，通过多态实现扩展**。面向对象的思想本质是将数据与行为做结合，数据与行为的载体称之为对象，而对象要负责的是定义职责的边界。但面向对象也不是银弹，甚至有些场景用比不用还糟，一切的根源就是抽象。我们在设计抽象进行分类时，不一定能抓住最合适的切入点，**错误的抽象比没有抽象复杂度更高**。在系统设计的早期，业务规则不复杂，逻辑复用与扩展体现得也并不强烈，而面向过程的代码在支撑这些相对简单的业务场景是非常容易的。但软件工程最大的成本在于维护，当系统足够复杂时，当初那些写起来最easy的代码，将来就是维护起来最hard的债务。PS：别没事瞎抽象。

[再谈软件设计中的抽象思维（上），从封装变化开始](https://mp.weixin.qq.com/s/zEkoDxAv8VxNexh7KiUnBg)做软件抽象设计，从分析变化开始，到沉淀新知识结束。
1. 代码为什么不能被重用? 我们写代码是为了被调用，当只有一个使用场景时，不存在重用问题。如下图所示，出现重用问题，是因为引入了新的场景，有了变化，导致老的代码不能满足新场景的需要，从而出现重用问题。**为了解决差异，我们需要重新抽象，新抽象意味着新概念、新知识**，这就是我开篇说的，从变化开始，抽象到新知识结束的含义。我写了一个吃苹果的程序eat(Apple apple)，有一天我苹果吃腻了，想吃香蕉，问题来了，原来的eat(Apple apple)并不能被重用。差异性体现在Apple和Banana的不同，针对这个变化，我们需要一个新的抽象去抹平差异，关于如何抽象，关键是要寻找共性。Apple和Banana向上抽象的共性是什么呢？这个简单，我们都知道是Fruit，这个Fruit就是我们通过抽象获得的“新知识”、“新概念”。为了让原来的eat更通用，我们可以用eat(Fruit fruit)来代替eat(Apple apple)。如果有一天我又想吃肉了，那么Fruit的抽象层次也不够了，必须要eat(Food food)才行。最后我们不断演化的过程，就是抽象层次不断提升的过程。那有同学可能会问，如果一开始能预见到这些变化，那一开始就设计成eat(Food food)岂不是更好？嗯，理论上是这样的。那又有同学说，为了更好地扩展性，我一开始设计成eat(Object object)可以吗？呃…… 一般我们不这么做，除非你是给广东人建模：）因为Object的抽象层次太高了，万物皆对象，在抹平万物的差异的同时，也失去了可理解性，以及业务语义直观表达的能力。这是一个简单的抽象案例，之所以简单，是因为我们都熟悉水果、食物的概念，抽象起来很容易。而实际工作中，并不是所有的抽象都是如此显而易见，很多时候，我们不得不深入理解问题域，了解很多的背景知识，不断犯错迭代才能挖掘（有时候是创造）出“新知识”，有时是纯抽象的存在，不像苹果、香蕉，你还能看得见摸得着，难就难在这个地方。以上，我们通过一个类型变化为案例导入，介绍变化和抽象的关系，**抽象就是一个使用新概念（新知识）统合差异的过程**。所以发现变化、分析变化、明确差异点，找到新概念抹平差异，是我们进行抽象的一般思考路径。实际上，仔细考察软件中代码层面的变化因素，主要有三类变化：数据变化：比如针对不同的场景，我们需要不同的配置数据；类型变化：比如上面提到的Apple和Banana的差异；行为变化：比如我需要用if-else来处理不同的场景。
    ![](/public/upload/architecture/abstract_thinking.jpg)
2. 如何抽象数据变化？程序=数据+算法，数据变化是最常见的变化，如果我们能分离数据变化，算法就可以变得更通用。我们可以用“数据外推”的抽象过程来处理数据变化。这个外推的过程，可以分解成4步：
    1. 数据参数化：将可能变化的变量，提升为函数的参数，可提升函数的复用性
    2. 参数结构化：将有关联的数据变量聚合成有意义的结构。
    3. 结构模型化：按照数据结构的生命周期和使用频率关系，将数据结构进行分类抽象。通过建模让数据结构之间形成关系；
    4. 数据配置化：静态数据可以外置成配置文件；动态数据可以外置到DB（DB Schema建模）
3.  如何抽象行为变化、类型变化？比如需要对学生按照身高从高到低进行排序，我们写了如下的冒泡排序代码： `public static void sort_students_by_height(Student[] students, int numOfStudents) `之后，我们又需要对老师按照年龄进行排序。Student和Teacher的差异性不必多说，这里的关键是要寻找共性，他们的共性是什么？我们第一反应可能是他们都是人Human，没错，这是一个共性。你也可以说他们都是求是小学的，没错，这也是共性。但这些抽象对于当前的对象比较问题并没有什么帮助。任何两个事物，如果不加约束的话，我们总是可以从很多角度进行抽象。比如我在《程序员的底层思维》中讲过一个笑话，问：金鱼和激光笔有什么共同之处？答：它们都不会吹口哨。类似这样天马行空的“抽象”，可以说是无穷无尽。但真正有用的抽象是在领域上下文下，对我们解决问题有帮助的抽象。针对当前的排序问题，我们可以说Student和Teacher都是Comparable（可比较的）。使用Comparable抽象，我们解决了Student和Teacher的类型变化问题，与此同时，行为变化（比较height和age的差异）也能通过compareTo( )这个行为抽象进行抹平，**基于这个抹平变化之后的新抽象**，我们就得到了一个通用的冒泡排序框架。PS：抹平差异。 
3. 很多的业务代码类似，初始场景简单，代码不复杂也很clean。但随着应用场景的变化，各种逻辑分支开始冲击原来的代码结构，最初的clean code就会慢慢地变得dirty，最后变成shit。[再谈软件设计中的抽象思维（下），从FizzBuzz到规则引擎](https://mp.weixin.qq.com/s/C_aiWaGCBc7qXPGk9s-iFg)整个设计过程，就是不断地分析变化点，抽象概念，沉淀领域知识，让系统更加通用的过程。不同的问题域，解决的问题不同，但这一套从变性入手，分析综合，以领域为核心，以概念为核心，抽象建模的方法论绝对是相通的。也是我们软件设计里最重要的核心能力之一。PS：工作有八九年了，一个感受是，再复杂的系统，最后核心概念、核心代码就那些，一定的简单的，纯粹的。反过来，如果系统一直处于一个复杂、难以理解、维护的状态，那一定就是抽象、概念没做好， 一定是设计跟不上需求变化，有些需求超出了之前的假设，cover不住。

### 评价抽象

UML的创始人Grady booch在 Object Oriented Analysis and Design with Applications 一书中，提到了评判一种抽象的品质可以通过如下5个指标进行测量：耦合性、内聚性、充分性、完整性与基础性。
1. 耦合性，一个模块与其他模块高度相关，那它就难以独立得被理解、变化或修改。但这并不意味着我们就不需要耦合。软件设计是朝着扩展性与复用性发展的，继承天然就是强耦合，但它为我们提供了软件系统的复用能力。如同摩擦力一般，起初以为它阻碍了我们前进的步伐，实则没有摩擦力，我们寸步难行。
2. 内聚性，内聚性测量的是单个模块里，各个元素的的联系程度。内聚性分为偶然性内聚与功能性内聚。金鱼与消防栓，**我们一样可以因为它们都不会吹口哨，将他们抽象在一起，但很明显我们不该这么干，这就是偶然性内聚**。最希望出现的内聚是功能性内聚，即一个类或模式的各元素一同工作，提供某种清晰界定的行为。比如我将消防栓、灭火器、探测仪等内聚在一起，他们是都属于消防设施，这是功能性内聚。
    1. 什么叫“内聚”，就是功能不是以牺牲复杂度来换取的。像 linux 的 core 很内聚，驱动即使扩展了一万个，系统复杂度也没增加，虽然代码在一直增加。所以软件设计时的抽象能力就变得极为重要。
3. 充分性，充分性指一个类或模块需要应该记录某个抽象足够多的特征，否则组件将变得不用。比如Set集合类，如果我们只有remove、get却没有add，那这个类一定没法用了，因为它没有形成一个闭环 。
4.  完整性，完整性指类或模块需要记录某个抽象全部有意义的特征。完整性与充分性相对，充分性是模块的最小内涵，完整性则是模块的最大外延。我们走完一个流程，可以清晰得知道我们缺哪些，可以让我们马上补齐抽象的充分性，但可能在另一个场景这些特征就又不够了，我们需要考虑模块还需要具备哪些特征或者他应该还补齐哪些能力。
5. 基础性，基础性指抽象底层表现形式最有效的基础性操作（似乎用自己在解释自己）。比如Set中的add操作，是一个基础性操作，在已经存在add的情况下，我们是否需要一次性添加2个元素的add2操作？很明显我们不需要，因为我们可以通过调用2次add来完成，所以add2并不符合基础性。但我们试想另一个场景，如果要判断一个元素是否在Set集合中，我们是否需要增加一个contains方法。Set已经有foreach、get等操作了，按照基础性理论，我们也可以把所有的元素遍历一遍，然后看该元素是否包含其中。但基础性有一个关键词叫“有效”，虽然我们可以通过一些基础操作进行组合，但它会消耗大量资源或者复杂度，那它也可以作为基础操作的一个候选者。

某个功能或数据，若没有归属于某个明显的职责范围，很可能就有相同的功能被多次实现，同一份数据被存放在多处。

### 软件设计原则

总体原则：**一切为了变更**，优秀的设计比糟糕的设计更容易变更。我刚刚做的事情让系统更容易改变，还是更难改变？《程序员修炼之道：通向务实的最高境界》据我们所知，无论是什么设计原则，都是ETC（Easier To Change）的一个特例。为什么命名很重要？因为好的命名可以使代码更容易阅读，而你需要通过阅读来变更代码——此谓ETC！

抽象的品质可以指导我们抽象与建模，**但总归还是不够具象，在此基础上一些更落地更易执行的设计原则涌现出来**，最著名的当属面向对象的五大设计原则 S.O.L.I.D。PS：思想 ==> 原则 ==> 框架/语言特性 越来越具象化。
1. 开闭原则OCP，对扩展开放，对修改关闭
2. 依赖倒置DIP，高层模块不应该依赖低层模块，两者都应该依赖抽象；抽象不应该依赖细节，细节应该依赖抽象。
3. PLOA最小惊讶原则，如果必要的特征具有较高的惊人因素，则可能需要重新设计该特征。
4. KISS简单原则，保持愚蠢，保持简单

《敏捷软件开发》这本书把设计模式上升到设计思想的高度，书中说“软件设计不应该是面向需求设计，而应该是面向需求变更设计”，也就是说在设计的时候，主要要考虑的是当需求变更的时候，如何用最小的代价实现变更。优秀的工程师不应该害怕需求变更，而应该欢迎需求变革，因为优秀的工程师已经为需求变更做好了设计，如果没有需求变更，那就显示不出自己和只会重复的平庸工程师的区别。

### 与应用结合

[抽象不稳定，老大徒伤悲](https://mp.weixin.qq.com/s/MJd9vVzPH_5YftavWOkPXw)抽象是基于归纳法，**所以越稳定的东西，越适合提前做好抽象复用**。因为你已经了解问题的全貌，而且问题域后续变化的可能小，所以抽象出来的东西就比较稳定，也容易被复用。而越不稳定的东西，越不适合做提前抽象。这就需要我们想清楚，**尽量延迟决策，或者压根放弃共同的抽象**。对于差异很大，又不稳定的问题域。你会发现用少量的代码重复（Repeat Yourself），来代替代码复用，可能比那个拧巴的共同抽象，有更高的效率和实用价值。PS：在业务层，比如你提一个复用的服务/组件，那么这个服务/组件没有一个产品/技术对它负责，若谁都可以改一点，那么此时还不如不复用，因为不稳定，**也就是抽象也跟团队构成有关系**。 

什么东西稳定？问题域越明确，越容易形成稳定的抽象。从这个角度而言，越偏向纯技术的领域越稳定，比如，对于文件的操作，无外乎就是打开、关闭，读写，这套玩意几十年如一日，稳定，持久，有用。再比如，服务治理中间件，解决的就是微服务互相通信的问题，包括服务注册、发现、负载均衡、熔断等。所以不管是Spring Cloud还是Dubbo，也可以很稳定的被抽象，被复用。消息中间件也是一样，就是要解决消息的生产、传递、消费问题，也很稳定。其它的技术领域也是一样，比如云计算，整个IT基础实施也是相对稳定的问题域，就是要提供可隔离的网络、高可靠的存储、可弹性的计算。所以也能在更大层面被抽象，被复用。**底层技术的难度在于技术深度**。就问题域而言，它要解决的问题是相对稳定的、清晰的。这就给合理的抽象，持续的打磨迭代带来了机会。PS：稳定不稳定就看你是否持续投入人去改。

什么东西不稳定？**上层业务应用看起来没有什么“技术含量”，但其难度在于合理的业务抽象和设计权衡**。越上层的业务越不稳定，稳定抽象的难度越大。这也是为什么当年阿里的业务中台失败的原因。因为他尝试在一个极其不稳定的电商业务领域，构建一套相对稳定，可复用的中台抽象。这种基于不稳定问题域的抽象复用，会带来极大的耦合成本，主要体现在三个方面：
1. 人员协作耦合：如果需要中台提供新的扩展点，也就是新的抽象，你就要协调一堆的人，还要排期，这叫协作耦合。
2. 代码耦合：因为业务逻辑是直接以代码的方式嵌入在中台系统，导致多个团队要在一起搞代码，其测试、联调、集成、发布成本都很高。另外，因为新框架的认知成本造成的认知复杂度，也使得coding的难度加大。
3. 部署耦合：因为是中台统一部署，从而导致不同的业务之间可能会相互影响。比如某个业务突发事件，可能导致其它业务的稳定性都受到影响。

![](/public/upload/architecture/abstract_value.jpg)

**对于高价值区，这个没有任何异议，我们要不遗余力的去构建公共的技术底座，复用公共的技术能力，从而提升整体的研发效率**。这里请Don't Repeat Yourself！这里要特别留意的是极度危险区。比如上文提到的不同电商业务的履约服务，但实际上问题域的共性很小，稳定性很差就属于极度危险区。这里虽然也可以抽象，但抽象层次往往过高，不会有多大的复用价值，再加上深度耦合带来的问题。烟囱式的解耦将是更加合适的选择！**危险区是那些复用程度很高，但稳定性不足的领域**。比如商品领域，还是以电商业务为例。虽然饿了么、淘宝、飞猪的业务差异很大，但是商品的类目管理、属性管理，商品上架，浏览，搜索，推荐还是有很大的共性所在。这里肯定是有复用的空间的，但是也要注意程度的把控。可以考虑把中台做薄，复用可以复用的部分，比如商品数据的存储，类目管理可以由平台统一做。对于商品审核、上架这些差异性比较大的部分，交给业务自己去处理。还是解耦比较好，一股脑的做厚中台，又会陷入耦合的噩梦。

Don't Repeat Yourself在大部分时候是正确的，但是复用的代价就是会引入耦合，DRY 的函数会被大量的地方引用，导致其内部逻辑需要考虑各种情况，逻辑及其复杂，修改风险也极高。正如Neal Ford在《Fundamentals of Software Architecture》一书中所说：“When an architect designs a system that favors reuse, they also favor coupling”。特别是在人员众多的大型系统中，**耦合的代价往往要比duplication的代价大的多**，天下没有免费的午餐，因此，如果共性不够大，问题域变化多，抽象不稳定。我宁愿用少量的duplication解耦，也不愿意陷入耦合的噩梦。

[代码复用：DDD视角下的平衡艺术](https://mp.weixin.qq.com/s/5gIBJByRZfNPbh6yjAvj9w)在 《架构整洁之道》中提到，**“拖延决策” 也是优秀架构设计的特点之一**。因为随着软件的开发和业务的迭代，我们掌握的信息越来越多，后期做出的决策肯定比项目早期的草率决定要靠谱。《复杂软件设计之道》中吐槽道：架构师们总是在只掌握 20% 信息的情况下，就已经做出了 80% 的决策。通过纯粹设计原则的角度是看不出来软件设计决策是否正确的，必须从更高的视角出发才行。
1. 成本角度谈复用，文件系统对上层提供了非常简单的文件模型，数据库对应用也提供了非常好理解的表模型。而他们的实现非常复杂，需要考虑并发，数据完整性，事务等一系列问题。相比理解他们的实现，学习模型和接口成本几乎可以忽略不计。上面的案例有共同的特点，**即模块的接口很简单，但是提供的功能却是深刻的。这个时候复用就非常的合算**。
这刚好就是 John Ousterhout 教授（Raft 的发明者）在其著作 《软件设计哲学》中提到 深模块 的概念。深模块在简单的接口后隐藏了许多功能。深模块代表很好的抽象，其内部复杂性只有很小一部分对其用户可见。浅模块的接口复杂度和实现复杂度接近，与其去了解模块的接口，开发人员还不如自己重新实现一遍。
2. 效益角度谈复用。Supercell 游戏公司将之前的爆款中备受玩家欢迎的风格，素材和程序逻辑沉淀下来，通过复用之前积累，可以快速产出新的爆款。上面的两个例子刚好就代表了两种提升产品核心竞争力的逻辑：复用之前具有竞争力的技术模块，让过去的成功经验助力未来的产品成功；给用户提供一致的体验，考虑用户的使用习惯，降低学习成本。复用不同模块能取得效果的程度也是不同的，复用什么样模块更有可能获得上述两点效果呢？DDD 中对子域的划分或许能够给我们答案，
    1. 核心子域，能够给公司带来核心竞争力的领域模块，拥有很高的复杂度和差异化价值，比如滴滴的司机调度算法，支付宝的交易系统，钉钉的 IM 系统等等，属于该子域的模块应该尽可能地复用，将其竞争力也注入到其他产品，甚至投入精兵强将，提升其可扩展性，进一步拉开和竞争对手差距。
    2. 支持子域，用来支撑核心子域，但是不能带来竞争力。因为不能带来核心竞争力，不如各个业务根据自己需求，使用脚手架快速搭建，定制起来还更加方便
    3. 通用子域，通用的业务或者技术问题领域， 比较复杂， 却不能给企业带来核心竞争力。好在一般有现成的解决方案，可以直接采购。比如财务系统，可以直接采购用友，金蝶；分库分表，消息队列可以直接使用开源软件，或者购买云上解决方案，尽可能复用，**但是复用的目的与核心子域不同，主要是为了降低研发成本**。
    因此 DDD 要求技术和业务深度结合，如果不了解业务的话，**单从设计原则角度，很难理解为什么要复用一个技术模块**。成功的设计来自对业务问题的深刻理解。最符合其业务子域的地方，才是类/函数应该在的地方。

### 抽象的产出

数据模型通常分为三个层次：概念模型、逻辑模型和物理模型。

![](/public/upload/architecture/abstract_data.jpg)

## 面向对象特性

面向对象质上是一种设计思想、方法，与语言细节无关。面向对象编程的基本出发点是“对现实世界的模拟”，把问题中的实体抽象出来，封装为程序里的类和对象，这样就在计算机里为现实问题建立了一个“虚拟模型”。然后以这个模型为基础不断演化，继续抽象对象之间的关系和通信，再用更多的对象去描述、模拟……直到最后，就形成了一个由许多互相联系的对象构成的系统。把这个系统设计出来、用代码实现出来，就是“面向对象编程”了。

### 封装

[Clean Code之封装：把野兽关进笼子](https://mp.weixin.qq.com/s/-dcXKWYeD-Y2SMq6MsVuSw)Dijkstra说：“软件是唯一的职业，人的思维要从一个字节大幅跨越到几百兆字节，也就是九个数量级（放在今天的话，恐怕还要再加上几个数量级）”。通过封装，我们可以实现信息隐藏（information hiding），把底层细节信息封装起来，隐藏起来，为上一层提供信息量更少的界面。通过这种方式，可以减少认知成本和大脑记忆负担，从而降低复杂度。不幸的是，软件不像收纳盒那样是一个物理的盒子，有物理边界。软件是软的，软件的收纳，只能通过“逻辑盒子”的封装来实现。复杂的系统都会呈现出层次结构（**一个基本结论**），在不同层次上，要封装不同的“逻辑盒子”。

最底层是方法，一个方法就是一个“逻辑盒子”，它封装了这个方法要实现的功能；其次，一个类也是一个“逻辑盒子”，它封装了这个类的属性和方法；再往上，一个包（package），一个模块（module），一个应用（applicaiton），都是一个个的“逻辑盒子”。从某种意义上来说，软件设计就是在设计这些逻辑边界，所谓的clean code，就是尽量让每一个“逻辑盒子”都封装合理——隐藏该隐藏的，暴露该暴露的，让系统呈现出一个清晰、可理解的结构，而不至于失控。

1. 方法封装的几个具体方法：抽象层次一致性，尤其是主方法/入口方法
2. 类封装是对数据和方法的封装（这句话没用，你的努力不是在实现需求，而是在应对混乱）
    1. 封装的要义在于信息隐藏，隐藏该隐藏的信息，暴露该暴露的信息。信息多了，就会杂乱，就会复杂，大脑就要晕菜
    1. 功能内聚，避免了散弹式修改
    2. 把系统中的重要领域概念挖掘出来，封装成可以复用的类，把业务语义显性化的表达出来，这种方式可以极大的增加系统的可理解性。这种对领域概念的类封装，也是DDD（Domain Driven Design，领域驱动设计）所倡导的，即明晰领域概念，并以领域模型为核心驱动系统设计。PS： **编程范式与 领域建模的关联**

我们是否能做一个足够灵活的设计呢？把所有需要修改或者扩展的地方通通暴露出来，一切皆可配置？但当我们这么做的时候，会发现这个设计几乎没有封装什么内容，所有的复杂度都在最外层可被感知和控制。这也是尴尬之处，实际项目中，我们并不需要无限的灵活度，因为那往往带来过多的非必要复杂度。我们只需要足够的灵活度，足够的可配置性，其他当前项目不需要的信息统统应该被封装。而项目和项目是有差异的，同类型的项目大概可以共享类似的基础封装，但可能无法期待所有的项目都可以共用某个最优的抽象与封装。

### 继承 vs 组合

继承应该用于 “属于” 关系。例如，狗属于一种动物，矩形属于一种形状，汽车属于一种交通工具。组合应该用于 “拥有 关系。例如，狗有主人，汽车有发动机，项目组有一名或多成员。反过来，狗并不是主人，汽车并不是发动机，项目组并不是成员。我们不能用继承来表示这些关系，而应该使用组合。

对于继承，子类通过super 可以访问父类的相关方法。对于java8 interface，也是类似。从这个角度看，如果将 this、super 理解为 类成员，继承父类、实现接口，像是组合的一种特殊形态。在c++里面，子类拥有父类的数据拷贝。那么java的内存对象模型和c的内存对象模型，研究一下，做个对比，还是蛮有意思的。以下图为例，UML 在展示PriorityKafkaProducer的继承和聚合关系时，将父类AbstractPriorityKafkaProducer和聚合类/成员KafkaProducer做了平级的处理。

![](/public/upload/java/priority_kafka_producer_class_diagram.png)

分解是设计的第一步，而且分解的粒度越小越好。当你可以分解出来多个关注点，每一个关注点就应该是一个独立的模块。最终的类是由这些一个一个的小模块组合而成，这种编程的方式就是面向组合编程。它相当于换了一个视角：类是由多个小模块组合而成。PS： 一个类存在两个（或多个）独立变化的维度，可以通过组合的方式，让这两个（或多个）维度可以独立进行扩展。

如果用 Ruby，组合的表现形式就会是一个 module；而在 Scala 里，就会成为一个 trait。使用 C++ 的话，表现形式则会是私有继承。学习多种程序设计语言的重要性：**Java 只有类这种组织方式，所以，很多有差异的概念只能用类这一个概念表示出来，思维就会受到限制**，而不同的语言则提供了不同的表现形式，让概念更加清晰。

许式伟：继承是个过度设计，其实继承实现了代码复用和多态两个东西，揉在一起。在 Go 里面，组合实现代码复用，接口实现多态，彼此完全独立，非常清晰。PS： go 中 struct a 嵌进 struct b，`a.b.func()` 只是 go 的语法糖，实际上a 和b 没有任何关系。

《罗剑锋的C++实战笔记》现实世界非常复杂，“面向对象编程”作为一种工程方法，是不可能完美模拟的，**纯粹的面向对象也有一些缺陷，其中最明显的就是“继承”**。“继承”的本意是重用代码，表述类型的从属关系（Is-A），但它却不能与现实完全对应，所以用起来就会出现很多意外情况。比如基类 Bird 有个 Fly 方法，所有的鸟类都应该继承它。但企鹅、鸵鸟这样的鸟类却不会飞，实现它们就必须改写 Fly 方法。各种编程语言为此都加上了一些“补丁”，像 C++ 就有“多态”“虚函数”“重载”，虽然解决了“继承”的问题，但也使代码复杂化了，一定程度上扭曲了“面向对象”的本意。**“面向对象编程”的关键点是“抽象”和“封装”，而“继承”“多态”并不是核心，只能算是附加品**。所以，我建议你在设计类的时候尽量少用继承和虚函数。特别的，如果完全没有继承关系，就可以让对象不必承受“父辈的重担”（父类成员、虚表等额外开销），轻装前行，更小更快。没有隐含的重用代码也会降低耦合度，让类更独立，更容易理解。还有，把“继承”切割出去之后，可以避免去记忆、实施那一大堆难懂的相关规则，比如 public/protected/private 继承方式的区别、多重继承、纯虚接口类、虚析构函数，还可以绕过动态转型、对象切片、函数重载等很多危险的陷阱，减少冗余代码，提高代码的健壮性。如果非要用继承不可，那么我觉得一定要**控制继承的层次**，如果继承深度超过三层，就说明有点“过度设计”了，需要考虑用组合关系替代继承关系，或者改用模板和泛型。

《程序员修炼之道》有些人认为继承是定义新类型的一种方式，他们最喜欢设计图表，展示出类的层次结构，不幸的是，这类图表很快就会为了表示类之间的细微差别而逐层添加，最终可怕的爬满墙壁。更糟糕的是多重继承的问题，汽车可以是一种交通工具，但它也可以是一种资产、保险项目、贷款抵押品等等，正确的对此建模需要多重继承，即使你很喜欢复杂的类型树，也完全无法为你的领域准确的建模。

### 面向对象和基于对象——多态

《解密java虚拟机》面向对象编程之所以要实现多态这一特性，最主要的目的是为了消除类型之间的耦合，也就是解耦，进而有一定余度应对变化。多态得以实现的一大前提是编程语言必须是面向对象的，否则哪来的对象与函数互相绑定一说呢？函数属于对象的一部分，便具备了封装的特性，有封装才有对象。一个函数能绑定多个不同的对象，意味着多个不同的对象有相同的行为，这是抽象/继承的含义，**封装、继承成全了多态**。在Python中“一切皆对象”，object是一切类的父类（包括type这个类），所有的类都直接或间接地继承自object，当你创建一个简单的类而不指定父类时，它隐式地继承自object。它是所有Python对象的共同祖先，确保了所有对象具有一些最基本的方法和属性，比如`__str__`, `__repr__`, `__del__`等。object的存在为Python的所有对象提供了一个统一的基础，使得所有对象可以共享一些基本行为。它也是面向对象编程中多态性的基石之一，因为所有对象都可以被视为object的实例。

大部分人写出的java 代码，可能只是“基于对象”。基于对象通常指的是对数据的封装，以及提供一组方法对封装过的数据操作。比如 C 的 IO 库中的 FILE * 就可以看成是基于对象的。面向对象的程序设计语言必须有描述对象及其相互之间关系的语言成分。这些程序设计语言可以归纳为以下几类：

1. 系统中一切事物皆为对象；
2. 对象是属性及其操作的封装体；《effective java》：尽可能地使每个类或者成员不被外界访问。
2. 对象可按其性质划分为类，
3. 对象成为类的实例；
4. 实例关系和继承关系是对象之间的静态关系；
5. 消息传递是对象之间动态联系的唯一形式，也是计算的唯一形式；
6. 方法是消息的序列。

《软件设计之美》只使用封装和继承的编程方式，我们称之为基于对象（Object Based）编程，而只有把多态加进来，才能称之为面向对象（Object Oriented）编程。也就是说，**多态是一个分水岭，将基于对象与面向对象区分开来**。软件设计是一门关注长期变化的学问，只有当你开始理解了多态，你才真正踏入应对长期变化的大门。

既然多态这么好，为什么很多程序员不能在自己的代码中很好地运用多态呢？因为多态需要构建出一个抽象。**构建抽象，需要找出不同事物的共同点，而这是最有挑战的部分**。而遮住程序员们双眼的，往往就是他们眼里的不同之处。在他们眼中，鸡就是鸡，鸭就是鸭。寻找共同点这件事，地基还是在分离关注点上。只有你能看出来，鸡和鸭都有羽毛，都养在家里，你才有机会识别出一个叫做“家禽”的概念。在构建抽象上，接口扮演着重要的角色。首先，接口将变的部分和不变的部分隔离开来。**不变的部分就是接口的约定，而变的部分就是子类各自的实现**。在软件开发中，对系统影响最大的就是变化。有时候需求一来，你的代码就要跟着改，一个可能的原因就是各种代码混在了一起。比如，一个通信协议的调整需要你改业务逻辑，这明显就是不合理的。对程序员来说，识别出变与不变，是一种很重要的能力。

很多程序员在接口中添加方法显得很随意，因为在他们心目中，并不存在**实现者和使用者之间的角色差异**。这也就造成了边界意识的欠缺，没有一个清晰的边界，其结果就是模块定义的随意，彼此之间互相影响也就在所难免。要想理解多态，首先要理解接口的价值，而理解接口，最关键的就是在于谨慎地选择接口中的方法。相对于封装和继承而言，多态对程序员的要求更高，需要你有长远的眼光，看到未来的变化，而理解好多态，也是程序员进阶的必经之路。

只要能够遵循相同的接口，就可以表现出来多态，所以，多态并不一定要依赖于继承。比如，在动态语言中，有一个常见的说法，叫 Duck Typing，就是说，如果走起来像鸭子，叫起来像鸭子，那它就是鸭子。“多态依赖于继承”只是某些程序设计语言自身的特点。你也看出来了，在面向对象本身的体系之中，**封装和多态才是重中之重**，而继承则处于一个很尴尬的位置。

以 Linux 文件系统接口为例

```c
struct file_operations {
  loff_t (*llseek) (struct file *, loff_t, int);
  ssize_t (*read) (struct file *, char __user *, size_t, loff_t *);
  ssize_t (*write) (struct file *, const char __user *, size_t, loff_t *);
  int (*open) (struct inode *, struct file *);
  int (*flush) (struct file *, fl_owner_t id);
  int (*release) (struct inode *, struct file *);
  ...
}
```
假设你写一个 HelloFS，那你可以这样给它赋值：
```c
const struct file_operations hellofs_file_operations = {
    .read = hellofs_read,
    .write = hellofs_write,
};
```
只要给这个结构体赋上不同的值，就可以实现不同的文件系统。但是，这种做法有一个非常不安全的地方。既然是一个结构体的字段，那就有可能改写了它，像下面这样：

```c
void silly_operation(struct file_operations* operations) {
  operations.read = sillyfs_read;
}
```

如此一来，本来应该在 hellofs_read 运行的代码，就跑到了 sillyfs_read 里，程序很容易就崩溃了。对于 C 这种非常灵活的语言来说，你根本禁止不了这种操作，只能靠人为的规定和代码检查。到了面向对象程序设计语言这里，这种做法由一种编程结构变成了一种语法。**给函数指针赋值的操作下沉到了运行时去实现**。

[强制类型转换是抽象层次有问题](https://mp.weixin.qq.com/s/cJ0odiYcphhNBoAVjqpCZQ)我们在写代码的过程中，什么时候会用到强制类型转换呢？子类的方法超出了父类的类型定义范围，为了能使用到子类的方法，只能使用类型强制转换将类型转成子类类型。举个例子，在苹果（Apple）类上，有一个isSweet()方法是用来判断水果甜不甜的；西瓜（Watermelon）类上，有一个isJuicy()是来判断水分是否充足的；同时，它们都共同继承一个水果（Fruit）类。我们需要挑选出甜的水果和有水分的习惯，因为pick方法的入参的类型是Fruit，所以为了获得Apple和Watermelon上的特有方法，我们不得不使用instanceof做一个类型判断。这里问题出在哪里？对于这样的代码我们要如何去优化呢？仔细分析一下，我们可以发现，根本原因是因为isSweet和isJuicy的抽象层次不够，站在更高抽象层次也就是Fruit的视角看，我们挑选的就是可口的水果，只是具体到苹果我们看甜度，具体到西瓜我们看水分而已。因此，解决方法就是对isSweet和isJuicy进行抽象，并提升一个层次，在Fruit上创建一个isTasty()的抽象方法，然后让苹果和西瓜类分别去实现这个抽象方法就好了。 

## 面向对象的渊源——限定修改影响的范围

面向对象的编程产生的历史原因：**由于面向过程编程在构造系统时，无法解决重用，维护，扩展的问题，**而且逻辑过于复杂，代码晦涩难懂。PS：笔者第一次看到这句话时没有感觉，后来忘记了，看《左耳听风》时自己总结了这句，再看到这句早就看到的话时，知己二字不能形容。

[《面向对象分析与设计》读书笔记 （1）- 关键的思想](https://zhuanlan.zhihu.com/p/27106866) 要点如下

1. 复杂性是面向对象主要解决的问题,复杂系统的5个属性

	* 层次结构,复杂性常常以层次结构的形式存在,层次结构的形式

		* 组成（”part of“）层次结构
		* 是一种“("is a")层次结构
	* 相对本原，这里是指构建系统的最小单位。你不需要担心基础组件是如何实现的，只要利用其外部行为即可。举个例子，你要盖一个房子，你需要砖，水泥等，这些都是一些基础组件，但是你不要自己去生产砖，水泥。
	* 分离关注,组件内的联系通常比组件间的联系更强。这一事实实际上将组件中高频率的动作（涉及组件的内部结构）和低频率的动作（涉及组件间的相互作用）区分开来
	* 共同模式,复杂系统具有共同的模式。比如小组件的复用，比如常用方案提炼为设计模式等
	* 稳定的中间形式（注意不是中间件），复杂系统是在演变中诞生的，不要一开始就期望能够构建出一个复杂的系统。要从简单系统逐步迭代到复杂的系统。

2. 思考分解的方式

	1. 系统中每个模块代表了某个总体过程的一个主要步骤。邮寄快递时，我们先将物品准备好，找到快递员，填写快递信息，进行邮寄。在这个过程中，我们分成了4个步骤，我们更注重的是事件的顺序，而非主要关注参与者。
	2. 根据问题域中的关键抽象概念对系统进行分解。针对上面的快递邮寄的例子，采用面向对象分解时，我们分解成4个对象：物品，快递单，快递员，我。我拥有物品，然后向快递员发出请求，快递员给我快递单让我填写快递信息。然后快递员进行邮递。

3. 编程风格，Bobrow将编程风格定义为“一种组织程序的方式，基于某种编程概念模型和一种适合的语言，其目的是使得用这种风格编写的程序很清晰”

4. 对象模型的4个主要要素：抽象；封装；模块化；层次结构；3个次要要素：类型、持久、并发

5. Shaw对于抽象的定义："对一个系统的一种简单的描述或指称，强调系统的某些细节或属性同时抑制另一些细节或属性。好的抽象强调了对读者或者用户重要的细节，抑制了那些至少是暂时的非本质细节或枝节" （我以前的思维漏洞 就是不知道 抑制非本质细节）
6. 封装的意义，复杂系统的每一部分，都不应该依赖于其他部分的内部细节。要让抽象能工作，必须将实现封装起来
7. 模块化的意义
8. 层次结构的意义

## 实现

问题：面向对象特性是当前很多主流语言都支持的特性。那么要在编译器和运行时上做哪些工作，来支持面向对象的特性呢？对象在内存里的表示都有哪些不同的方式？如何实现继承和多态的特性？为什么 Java 支持基础数据类型和对象类型，而有些语言中所有的数据都是对象？要在编译技术上做哪些工作来支持纯面向对象特性？

对象的概念是它的基础，然后语言的设计者再把类型体系、软件重用机制和信息封装机制给体现出来。在这个过程中，不同的设计者会有不同的取舍。所以，希望你不要僵化地理解面向对象的概念。比如，以为面向对象就必须有类，就必须有继承；以为面向对象才导致了多态，等等。这些都是错误的理解。

要实现一门面向对象的语言，我们重点要了解三个方面的关键工作：一是编译器在语法和语义处理方面要做哪些工作；二是运行期对象的内存布局的设计；三是在存在多态的情况下，如何实现方法的绑定。

编译器前端的工作
1. 从语法角度来看，语言的设计者要设计与类的声明和使用有关的语法。
2. 在语义分析阶段，每个类会作为自定义类型被加入到符号表里。这样，在其他引用到该类型的地方，比如用该类型声明了一个变量，或者一个方法的参数或返回值里用到了该类型，编译器就能够做正确的引用消解。
3. 使用类型系统的信息，在变量赋值、函数调用等地方，会进行类型检查和推断。
对象的内存布局
1. 像 Java、Python 和 Julia 的对象，一般都有一个固定的对象头。对象头里可以保存各种信息，比如到类定义的指针、与锁有关的标志位、与垃圾收集有关的标志位，等等。
2. 对象头之后，通常就是该类的数据成员。如果存在父类，那么就既要保存父类中的成员变量，也要保存子类中的成员变量。在生成的汇编代码里，如果要访问一个类的成员变量，其实都是从对象地址加上一个偏移量，来得到成员变量的地址。

方法的静态绑定和动态绑定
1. 方法带有 private、static 或 final 关键字的时候。在编译期就知道去执行哪段字节码
2. 对于重载（Overload）的情况，也就是方法名称一样、参数个数或类型等不一样的情况，也是可以在编译期就识别出来的，所以也可以通过静态绑定。
3. 动态绑定一般通过vtable


## 其它

面向对象三要素封装、继承、多态。
1. 封装，封装的意义，在于明确标识出允许外部使用的所有成员函数和数据项，或者叫接口。**有了封装，就可以明确区分内外**，使得类实现者可以修改封装内的东西而不影响外部调用者；而外部调用者也可以知道自己不可以碰哪里。这就提供一个良好的合作基础——或者说，只要接口这个基础约定不变，则代码改变不足为虑。
    1. 这和带兵打仗是类似的，班长需要知道每个战士的姓名/性格/特长，否则就不知道该派谁去对付对面山坡上的狙击手；而连长呢，只需知道自己手下哪个班/排擅长什么就行了，然后安排他们各自去守一段战线；到了师长/军长那里，他更关注战场形势的转变及预期……没有这种层层简化、而是必须直接指挥到每个人的话，累死军长都没法指挥哪怕只是一场形势明朗的冲突——光一个个打完电话就能把他累成哑巴。反过来也对：军长压根就不应该去干涉某个步兵班里、几个大头兵之间的战术配合；这不仅耽误他行使身为军长的职责，也会干扰士兵们长久以来养成的默契。他的职责是让合适的部队在合适的时机出现在合适的战场，而不是一天到晚对着几个小兵指手画脚、弄的他们无所适从。约束各单位履行各自的职责、禁止它们越级胡乱指挥，这就是封装。
    2. 什么是真正的封装？封装是不是等于“把不想让别人看到、以后可能修改的东西用private隐藏起来”？显然不是。如果功能得不到满足、或者未曾预料到真正发生的需求变更，那么你怎么把一个成员变量/函数放到private里面的，将来就必须怎么把它挪出来。真正的封装是，经过深入的思考，做出良好的抽象，给出“完整且最小”的接口，并使得内部细节可以对外透明。注意：对外透明的意思是，外部调用者可以顺利的得到自己想要的任何功能，完全意识不到内部细节的存在。一个典型的例子，就是C++的new和过于灵活的内存使用方式之间的耦合。这个耦合就导致了new[]/delete[]、placement new/placement delete之类怪异的东西：**这些东西必须成对使用，怎么分配就必须怎么释放，任何错误搭配都可能导致程序崩溃**——这是为了兼容C、以及得到更高执行效率的无奈之举；但，它更是“抽象层次过于复杂，以至于无法做出真正透明的设计”的典型案例。
1. 继承同时具有两种含义：其一是继承基类的方法，并做出自己的改变和/或扩展——号称解决了代码重用问题；其二是声明某个子类兼容于某基类（或者说，接口上完全兼容于基类），外部调用者可无需关注其差别。实践中，继承的第一种含义（实现继承）意义并不很大，甚至常常是有害的。因为它使得子类与基类出现强耦合。继承的第二种含义非常重要。它又叫“接口继承”。接口继承实质上是要求“做出一个良好的抽象，这个抽象规定了一个兼容接口，使得外部调用者无需关心具体细节，可一视同仁的处理实现了特定接口的所有对象”——这在程序设计上，叫做归一化。
    1. 没有真正的抓到一类事物（在当前应用场景下）的根本，就去设计继承结构，是必不会有所得的。不仅如此，请注意我强调了在当前应用场景下。这是因为，分类是一个极其主观的东西，不存在普适的分类法。举例来说，我要研究种族歧视，那么必然以肤色分类；换到法医学，那就按死因分类；生物学呢，则搞门科目属种……
    2. 最具重量级的炸弹则是：正方形是不是一个矩形？它该不该从矩形继承？如果可以从矩形继承，那么什么是正方形的长和宽？在这个设计里，如果我修改了正方形的长，那么这个正方形类还能不能叫正方形？它不应该自然转换成长方形吗？如果我有两个List，一个存长方形，一个存正方形，自动转换后的对象能否自动迁移到合适的list？什么语言能提供这种机制？如果不能，“一视同仁的处理某个容器中的所有元素”岂不变成了一句屁话？造成这颗炸弹的根本原因是，**面向对象中的“类”，和我们日常语言乃至数学语言中的“类”根本就不是一码事**。面向对象中的“类”，意思是“接口上兼容的一系列对象”，关注的只不过是接口的兼容性而已（可搜索 里氏代换）；关键放在“可一视同仁的处理”上（学术上叫is-a）。显然，这个定义完全是且只是为了应付归一化的需要。这个定义经常和我们日常对话中提到的类概念上重合；但，如前所述，根本上却彻彻底底是八杆子打不着的两码事。
1. 多态实质上是继承的实现细节；那么让多态与封装、继承这两个概念并列，显然是不符合逻辑的。
总结：面向对象的好处实际就这么两点。
1. 一是通过封装明确定义了何谓接口、何谓接口内部实现、何谓接口的外部调用者，使得大家各司其职，不得越界；
2. 二是通过继承+多态这种内置机制，**在语言的层面支持归一化的设计**，并使得内行可以从代码本身看到这个设计。但，注意仅仅只是支持归一化的设计。不懂如何做出这种设计的外行仍然不可能从瞎胡闹的设计中得到任何好处。


### 左耳听风

来自陈皓《左耳听风》付费课程，建议先看下[java 语言的动态性](http://qiankunli.github.io/2018/08/15/java_dynamic.html)

在面向对象编程里，计算机程序会被设计成彼此相关的对象，独立而又相互调用。传统程序主张将程序看做一系列指令，或一系列函数。面向对象设计中的每一个对象 都应该能够接受数据、处理数据并将数据传递给其它对象。

面向对象的缺点：通过对象来达到抽象效果， 把代码分散在不同的类里面。那要让它们执行起来，就需要将这些类粘合起来。设计模式以及ioc 等虽然精巧，但不得不怀疑点歪了科技树。一段代码的执行路径 `obj1.func1 ==> obj2.func2 ==> obj3.func3` 在函数式编程中 `func1(func2(func3))` 就解决了。

![](/public/upload/java/object_oriented_3.png)

换个角度想一下，架构设计从单体演化到微服务架构，固然一部分是单机无法负载，另一个原因就是单体 在维护和运维上的困难，比如一个小改动导致整个项目的重启。也就是说，架构设计之初，就越来越考虑可维护性和扩展性。**架构设计不只考虑实现功能，可维护性和扩展性影响了架构设计，对应的，可维护性和扩展性影响了代码结构。****指令序列或函数序列被解构，分散在各个对象中，以减少修改对整体的影响**。但从上图可以看到，面向对象的优点 也直接导致了其缺点。

宏观上的系统设计与类设计

![](/public/upload/java/object_oriented_1.png)

可以认为，db 就是controller-service-dao  这些类的状态。

微观上的类实现与系统模块实现

![](/public/upload/java/object_oriented_2.png)

你设计一个controller-service-dao 项目 制定http api 时，肯定会想业务层面会有哪些调用，绝不会一个http api 干了一半的活儿 然后让调用方自己 访问两个 http api 自己聚合数据。对应的，我们在设计类时，根据类持有的数据/状态，一个对象可以访问数据库、可以内部操作线程，但其对外提供的interface function 应该是自洽的（对外隐藏掉不必要的细节）。类似的概念可以看 [ddd(一)——领域驱动理念入门](http://qiankunli.github.io/2017/12/25/ddd.html)

### 多线程与对象的 关系

我以前的认知，线程只是一个 驱动者，驱动代码执行，对象跟线程没啥关系。一个典型的代码是

```java
class XXThread extends Thread{
    private Business business;
    public XXThread(Business business){
        this.business = business;
    }
    public void run(){  
        // 一般适用于 task之间没有关联，线程内可以闭环掉所有逻辑，无需共享数据，线程之间无需协作
        business code
    }
}
```
	
在apollo client 中，RemoteRepository 内部聚合线程 完成配置的周期性拉取，线程就是一个更新数据的手段，只是周期性执行下而已。 

```java
class Business{
    private Data data;
    public Business(){
        Executors.newSingleThread().execute(new Runnable(){
            public void run(){
                acquireDataTimely();
            }
        });
    }
    public void acquireDataTimely(){}
    public void useData(){}
    public void transferData(){}
}
```
	
从两段代码 看，线程与对象的 主从关系 完全相反。[程序的本质复杂性和元语言抽象](https://coolshell.cn/articles/10652.html)指出：程序=control + logic。 同步/异步 等 本质就是一个control，只是拉取数据的手段。因此，在我们理解程序时，同步异步不应成为本质的存在。


## 喷一喷面向对象

### 不要为了面向对象而面向对象

2019.1.2 补充[如此理解面向对象编程](https://coolshell.cn/articles/8745.html)，有一个需求：代码检查操作系统类型，若是linux 输出：linux很不错；若是windows，输出windows 很差

1. 过程化的方案
2. 一般面向对象方案（os 抽象为class，一个具体os 对应一个子class）
3. 面向对象进化：不仅弄子类，还弄一map 保存os 和 子类的关系
4. 大神 Rob Pike 对此的评论是：根本就不需要什么Object，只需要一张小小的配置表格，里面配置了对应的操作系统和你想输出的文本。这不就完了。所谓的代码进化相当疯狂和愚蠢的，这个完全误导了对编程的认知。

还有的人喜欢用Object来替换所有的if-else语句，他们甚至还喜欢把函数的行数限制在10行以内 [programming in the
twenty-first century](https://prog21.dadgum.com/156.html)
6. **那23个经典的设计模式和OO半毛钱关系没有**，只不过人家用OO来实现罢了。设计模式就三个准则：1）中意于组合而不是继承，2）依赖于接口而不是实现，3）高内聚，低耦合。你看，这完全就是Unix的设计准则。


[Don't Distract New Programmers with OOP](https://prog21.dadgum.com/93.html)The shift from procedural to OO brings with it a shift from thinking about problems and solutions to thinking about architecture. That's easy to see just by comparing a procedural Python program with an object-oriented one. The latter is almost always longer, full of extra interface and indentation and annotations. The temptation（诱惑） is to start moving trivial bits of code into classes and adding all these little methods and anticipating（预料） methods that aren't needed yet but might be someday. 封装对象、类、接口等对很多简单代码来说是不必要的。

When you're trying to help someone learn how to go from a problem statement to working code, the last thing you want is to get them sidetracked（转移话题） by faux（人造的）-engineering busywork（作业、额外工作）. Some people are going to run with those scraps（点滴） of OO knowledge and build crazy class hierarchies and end up not as focused on on what they should be learning. Other people are going to lose interest because there's a layer of extra nonsense（无意义的） that makes programming even more cumbersome（笨重的）.

面向对象逼着你除了思考问题本身外，还要思考结构、设计，很多人无此意识或功力不足， 滥用面向对象的特性，整出大量无意义的代码，使得代码复杂度大大超过了问题本身的复杂度。

### 走偏的controller-service-dao

controller-service-dao说白了，跟面向对象没啥关系，说面向过程也不算错。web 开发其实是在进行**数据处理**，这估计也是现在在倡导异步处理、反应式处理的初衷，而进行**逻辑处理**的框架，对代码设计是非常讲究的。习惯了Controller-service-dao这种思维，**在关键领域不会用类解决问题**，所有的代码都是把字段取出来计算，然后再塞回去。各种不同层面的业务计算混在一起，将来有一点调整，所有的代码都得跟着变，其实就是面向过程的代码。

面向对象几个基本概念：抽象、封装、继承、多态等，现在看，最难的就是抽象，抽象是在说啥？在厘定边界，什么活该什么类干是精确的，**变动被局限在一个很小的范围内**（比如，我把map 改成guava cache，变动越小越优秀）。理论上，不违背基本设计的变动，修改起来应该是很容易的。没有抽象，写出来的都是方法和方法的组合

一个类，有几个方法，有几个字段，叫啥名，哪些对外可见的，很重要，绝不是随意的，反应了你的设计理念。尤其是在重逻辑、轻数据处理的项目中。Controller-service-dao 给了很不好的恶习。

在一些从结构化编程起步的程序员的视角里，面向对象就是数据加函数。虽然这种理解不算完全错误，但理解的程度远远不够。面向对象是解决更大规模应用开发的一种尝试，它提升了程序员管理程序的尺度。谈到面向对象，你可能会想到面向对象的三个特点：封装、继承和多态。**封装，则是面向对象的根基**。对象之间就是靠方法调用来通信的。但这个方法调用并不是简单地把对象内部的数据通过方法暴露。因为，封装的重点在于对象提供了哪些行为，而不是有哪些数据。也就是说，即便我们把对象理解成数据加函数，**数据和函数也不是对等的地位**。函数是接口，而数据是内部的实现，正如我们一直说的那样，接口是稳定的，实现是易变的。**“封装”的要点是行为，数据只是实现细节**，而很多人习惯性的写法是面向数据的，这也是导致很多人在设计上缺乏扩展性思考的一个重要原因。

一个模型的封装应该是以行为为基础的。PS： 一个类不应该只当数据类用，除非目的就是数据类。

理解了这一点，我们来看一个很多人都有的日常编程习惯。他们编写一个类的方法时，把这个类有哪些字段写出来，然后，生成一大堆 getter 和 setter，将这些字段的访问暴露出去。**这种做法的错误就在于把数据当成了设计的核心**，这一堆的 getter 和 setter，是对于封装的破坏，它把一个类内部的实现细节暴露了出来。请注意，方法的命名，体现的是你的意图，而不是具体怎么做。所以，**getXXX 和 setXXX 绝对不是一个好的命名**。不过，在真实的项目中，有时确实需要暴露一些数据，所以，等到你确实需要暴露的时候，再去写 getter 也不迟，你一定要问问自己为什么要加 getter。至于 setter，首先，大概率是你用错了名字，应该用一个表示意图的名字；其次，setter 通常意味着修改，一个好的设计应该尽可能追求不变性。所以，setter 也是一个提示符，告诉我们，这个地方的设计可能有问题。


setter 的出现，是对于封装的破坏，它把一个类内部的实现细节暴露了出来。面向对象的封装，关键点是行为，而使用 setter 多半只是做了数据的聚合，缺少了行为的设计。setter 通常还意味着变化，一个好的设计应该尽可能追求不变性。所以，setter 也是一个提示符，告诉我们，这个地方的设计可能有问题。

## 其它

本文着重从面向对象渊源的角度来说明：**面向过程编程在构造系统时，无法解决重用，维护，扩展的问题**，进而说明面向对象将 重用、维护、扩展加入了设计理念中，进而体现在语言的方方面面。

