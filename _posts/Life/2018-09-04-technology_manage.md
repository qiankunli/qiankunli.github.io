---

layout: post
title: 尝试带好一个小团队
category: 生活
tags: Life
keywords: technology manage

---

## 简介

本文的逻辑比较乱，对于笔者来说，从一个人做事情到带多个人做事情，这意味着很多转变：

1. 从单人到多人协作的转变，单人时，沟通无关紧要，没有文档也问题不大，项目做的不好，自己可能加个班重构一下，但你让小伙伴重构还要做大量的说服工作。[破解遗留系统重构问题的 6 步心法](https://zhuanlan.zhihu.com/p/542092692)
2. 从一段时间一个项目到多个项目并进的转变，有些项目人力不够要搁置，不同项目处于不同的阶段，不同的项目自己承担的角色不同
3. 做技术到做管理（部分技术部分管理）的转变

就好比以前的单机系统演化为现在的分布式系统，你就不得不搞一些zookeeper、kafka、监控系统等保证正确性，且不得不接受单个请求 响应时间延长的事实，以换取整体吞吐量的增加。

![](/public/upload/life/technology_manage_xmind.png)

## 认知上

[十年技术老兵总结的自我修炼之路](https://mp.weixin.qq.com/s?__biz=MjM5MDE0Mjc4MA==&mid=2651009003&idx=2&sn=9c48e19c584d7deacb4cdfa72ad31a05&chksm=bdbed7b88ac95eae78539cdf0201133c5dc51e49dba036f4787d9c624aeab32e84a492be415e&mpshare=1&scene=23&srcid=0904sCU1C4mt5FODn2JRNPL9%23rd)

![](/public/upload/life/technology_manage_1.png)

[技术管理到底管什么](https://mp.weixin.qq.com/s/QN1OKEFT3DiA82-OAp858Q) [如何把系统优化的技术用到团队/技术管理中？](https://mp.weixin.qq.com/s/NdAb_JJI_YGwH8S7s9UY0A)

1. 无论是春秋战国时候的丞相，还是三国时候的谋士们，技能大概分成两种：谋略和理事。谋略重在想办法，理事重在做事情。但凡有名的丞相谋士至少有一样特长，两者兼顾的自然是不世功勋唾手而来。

	* 谋略：技术架构搭建、新技术演进选型等，解决该做什么和怎么做的问题。
	* 理事：任务和人员协调、分配等，解决谁来做、哪件事先做的问题。

	把握好谋略和理事这两条，技术管理才能真正落地，团队战斗力才能真正体现出来。
	
2. 我们的目的应该是把事情做好，管人只是一种方式而已。以代码规范为例，最好的方式是，通过技术手段去保证，不按照规范去做的代码没法提交成功；次好的方式是，建立规范准则，要求大家去遵守；最坏的方式是，质问和批评为什么不这么做。不要轻易尝试去改变一个人，哪怕你是为他好，哪怕改的是缺点对他也有好处。

3. **管理者不应该是职级层面的上级，而应该是分工层面的决策和协调者。**
4. 觉得不写代码会心慌的人，说明你把太多精力花在理事上了，谋略给丢了。实际上，在技术架构满足不了业务增长需求的时候，是需要你（和你的架构师一起）去拿出新的架构的；组员讨论几个方案拿不定主意的时候，也是需要你去做决策的。这些都是需要你平时花时间去不断丰富和充实你的技术的。否则一定是一将无能累死三军。
4. 你并不应该陷入与人沟通和处理杂事的漩涡中，而是应该抽时间出来保持自己的技术敏感度。**管理者和普通员工在技术上的区别，在于和技术打交道的方式变了。普通员工是写代码，技术管理者应该是学习技术和思考架构。**

[技术的成长曲线](http://zhangtielei.com/posts/blog-growth-curve.html) 熟手：技术还无法自成体系，能理解到的知识架构还有所残缺。解决问题所依靠的，更多是经验而非缜密的逻辑。而从熟手向专家的突破，则需要系统地去补习知识架构。**技巧应该建立在对于普遍规则的理解之上.**

陈皓《左耳听风》降到技术领导力时提到

1. 能够发现问题
2. 能够提供解决问题的思路和方案，并能比较这些方案的优缺点
3. 能够做正确的技术决定
4. 能够用更优雅、简单、容易的方式来解决问题
5. 能够提高代码或软件的扩展性、重用性和可维护性
6. 能够用正确的方式管理团队

	* 让正确的人做正确的事儿
	* 提高团队的生产力和人效，找到最有价值的需求，用最少的成本实现之
7. 创新能力

这个顺序也基本指出了技术人的成长路径。那么如何选择自己的人身与职业发展呢？

1. 客观的审视自己。在职场上，审视自己的最佳方式就是隔三差五的出去面试一把，看看自己在市场上能够到什么样的级别
2. 确定自己想要什么
3. 注重长期的可能性，多经历、多去选择
4. 不要和大众的思维方式一样

很多事情能做到什么程度，其实在思想的源头就被决定了，因为它会绝大程度的受思考问题的出发点、思维方式、格局观、价值观等因素影响，这才是最本源的东西。

个人感觉程序猿水平晋级 跟玩户外是一样的

1. 只是参与出去玩，但事儿多
2. 出去玩，自己管自己
3. 出去玩，主动地做点事儿
4. 当押队，交代事情做
5. 当副队，是个可以商量的人，能全面考虑事情，给出建议
6. 当领队，独自筹划大部分事情
7. 不错的领队

小伙伴若是自驱动不高的话，其实是对任务负责。而小队长是对项目负责的。项目成功跟小伙伴干完具体的活儿之间有大量的事情规划、协调、踩着石头过河。直接对结果负责的人其实最累的。

晓斌：管理者需要认识、理解以及建立这些高标准，并不断和团队成员传达。各个层级各个岗位的标准和要求是什么？员工晋升是因为他 / 她的哪些行为体现了下一层的要求？就日常细节工作的高标准，什么是优秀的设计文档？什么优秀的 PRD 文档？什么是高质量的代码？什么行为体现了客户第一？定义清晰的高标准并不是一件容易的事，更常见的情况反而是作为管理者自己也不知道高标准是什么（参考彼得原理），这可能是很多员工收到负反馈就感觉是被 PUA 的核心因素（“你都是个 P7 了，你自己应该知道怎么做好”）。我建议管理在想要给下属负反馈的时候自己先回答这个问题，“我是否可以清晰地和下属表达我期望看到哪些行为的变化？用具体的事例描述。”不要模棱两可。

**同理心，不是发慈悲，而是一种策略**。它要求你能了解下属的能力，并善于发挥下属的能力。

[这年月只能靠手艺吃饭了](https://mp.weixin.qq.com/s/5bzJR0qlkUrMHnuIsMnnXA) 以亲身经历列出了技术管理的几个阶段。

晓斌：我做leader 主要抓几个事情
1. 大，关键决策，定义问题，大方向
2. 小，工程质量，比如codereview等。中间的给小伙伴留够空间。
3. 日常，高危风险，比如服务监控

乔新亮
1. 因为当前这个时代就是乌卡（VUCA）时代，这是今天的企业面临的第一个真实的情况。在这个情况下，**对于 Leader 最大的一个要求就是方向**。所以我在公司里面经常讲一句话——没有方向感，最大的责任人就是一把手或这个高管团队。
2. 中国市场的变化，**从供不应求到供大于求**。不管是真实的物质产品，还是虚拟产品，包括技术产品，都是从供不应求到供大于求。在过去，业务的增长可能很大来自于增量用户，但如今我们很多时候需要基于存量用户去经营。
3. 企业原来是先做大再做强，现在要先做强再做大。过去因为供不应求，都是增量用户，等你精雕细琢出来，蛋糕早被瓜分完了，所以当时做企业是不计成本、不惜代价，因为有资本的助力，你可以有恃无恐。但现在是什么？现在你一定要先做强后做大，否则死路一条，我觉得这也是创业者以及 IT 部门人员要刻在骨子里的认知。
3. **应对第一点问题，你需要做敏捷。那么针对第二点，你需要足够优秀、足够卓越**。
4. 我们公司总共 4000 多人，研发团队总共就 100 人，核心的业务全部是自研的。一个小团队有 3 到 5 人，做 1 到 3 个产品，他们不断把这些产品的成熟度做高，**从支持业务慢慢地转向接管业务**。随着我的产品做得越来越好，请问，业务部门的人难道不应该去提升能力、去更进步吗？否则我为什么要研发去做这个产品？我们公司做了一件很明确的事情就是，随着科技能力建设的提升，做出了共享中心，然后进行裁员。你猜结果怎么样？产品做得更好了，因为我们把标准做在了产品里边。今天很多的 CTO 都是被干掉的，在公司里边很没地位的。为什么？因为他没有成就业务。我在公司里观点就很鲜明，因为我给公司创造价值。所以我就跟 IT 团队讲，我们要干的就是把业务部门裁掉，裁不掉他们就裁我们，这是血性，这也是方向，要是你不做到这个迟早会被裁。其实这很正常，企业要存活，等你真正面临企业的生死存亡时刻，你就会知道这是很残酷的一句话。谁来为结果负责呢？谁强谁负责。就像在公司里边，这个需求到底什么时候做，不好意思，我说了算。如果业务部门跟我谈了一堆没啥用的需求，我可以压需求三个月不做，我觉得你作为 Leader 真的要扛起这事儿。但是回过头来讲，在有的企业里面你能这样吗？**所以很多企业我是不去的**，去了我肯定死得很惨。
5. 一个产品，如果按照成熟度来划分，它可以**被高度总结为工具、助手、专家三个等级**。第一个阶段，工具，对业务部门有点帮助，对别人有点用。第二个阶段，助手，你这个人、你这个岗位百分之七八十被我的产品取代了，这叫助手。专家是什么？你，裁掉，当然这是形象化的表述，实际是指是替换掉做的事，关于相关人员，那是公司 HR 的事。可以很明确地说，我们现在已经进入到助手和专家之间。我也可以很明确地跟大家讲，我们公司就是每个月都在裁业务部门的人，你就得进步，就得转岗学习，你做的那点事有什么系统做不了的？**真的留下不能取代的是真正有价值的工作**。我们把这些角色给高度抽象了以后，你可以看看一个公司里到底有多少人是有创造力的，有的人做的不是搬数据就是搬菜。你把数据从这个部门搬到那个部门，加工一下再搬到下一个部门，这就是搬数据的，挣的工资可能是 1 万 5 到 2 万。搬菜的呢，把这个菜从工厂加工出来，然后再搬到下一个地方，一个月挣 4000 到 8000。搬数据的，IT 都可以接管，搬菜的，可以用机器人去取代，比如 AGV 搬运机器人。由此，我们可以让员工和公司一起进步，不要去做低质量的重复性工作，而是不断去做更高层级的、创新的、有创造力的工作。
6. 一定要做组织变革。 但其实面对这一点，大多数的公司会完蛋。全是利益，怎么做组织变革？因此，从这一点去看，你永远不要害怕大公司。创业公司永远要充满信心地去干掉大公司，大公司的效率就是低。其实我对创业公司都充满希望，也愿意拿这个认知去投资那些我看好的创业公司，我觉得认知可以给你带来很多的好处和回报。所以你其实要打造的是数据、用户体验驱动内部经营完善的体系。什么叫驱动内部经营完善？做得不好的要批评他、要通报他、要把他裁掉，做得那么垃圾的为什么不裁掉，是吧？不做这些事情，你的 IT 的产品能好吗？在这一点上，我们公司就推进得特别好，公司高管层意见一致。我在我们公司坚决地推进这点，**推进不下去，我滚蛋，因为推进不下去是不可能成功的，我为什么要在这里浪费时间**。
7. 我一直认为，科技向善，科技不与人争利，人跟人才争利。公司不会因为用了哪个系统效率变高了裁你，是因为你那些低质量的重复性工作被系统取代后，你还不思进取，不去做有创造性的新工作，老板觉得你没用了，他才会裁你。但是最后想一想，这些科技是不是会变成社会基础设施？过去只有少数人、少数组织能享受的，后面慢慢都会变成社会基础设施。为什么今天我们普通人比过去皇帝过得还好？都是依托于科技的进步，而随着科技越来越变成基础设施，人民的生活还会越来越好。
8. 哪些自建？让你具备核心竞争力的要自建，毕竟这种让你绝对牛逼、超越竞争对手几十号排位的产品，怎么可能会有厂商卖给你？而有的东西就没必要自建了，能买就买，所以越是偏技术性的越不要做。人家外面有做得好的产品不拿过来，非要自己下场干是吧？你这么牛逼怎么不像毕玄一样去创业？你也可以选择自建，比如一个商业产品要价 20 万，而你可以很快自建出来满足公司需要，并且只用花 10 万的研发成本。不过，现在 IT 的人力成本已经这么贵了，10 万的成本你能投入几个人？按照 2 万一个人来算，10 万也只能让 5 个人做一个月，如果要做两个月，那么就是 2.5 个人，这点人能做出什么像样的产品？
9. 不管你想做成什么事，你都要找到最优解，首先就要掌控设计。
	1. 技术上一定要关注模块化。这个模块里的数据只有这个模块可以访问，其他人访问是不被允许的，至少能把这个守住，你的架构就不会烂到家。
	2. 要关注用户体验，一定要有监控体系。 你的系统里边到底怎么样，服务响应时间怎么样，用户访问你，他有没有业务异常，有没有系统异常？如果你连用户发生了什么都不关注，你怎么可能做好？所以你要有监控体系，也要做好治理工作。
	3. 只用一只眼睛盯着 AI。 当前的 AI 大多数时候没什么用，因为它还不是真正的人工智能，是傻人工智能。在这个情况下要发挥 AI 的价值，需要有几个依赖。**第一，你的数据得够多。第二，数据得够准确**。你们公司有数据吗？连数据都没有还谈什么 AI？你说有数据，主数据管理都没做好，最基础的数据都是垃圾数据，怎么可能做好人工智能？第三，上层的大数据做好了吗？大数据没做好，AI 怎么可能做好？所以 AI 到公司里面之后全是坑
10. 三大效率
	1. 研发效率，像 CI/CD、云计算、分布式服务、DB 缓存等等，有关基础平台和组件都要持续建设完善，你投入两到三个人干这个活儿，就可以让一百多个人效率更高，何乐而不为呢？我们做这个事情的目的就是让开发人员只写代码，测试人员只写测试案例，我们在各个层面都持续观察、分析研发人员到底在忙什么，然后帮助他提升效率。
	2. 会议效率，其实公司里大量的时间都是浪费在会议上。当然，大量的有用的价值也都产生在会议中，但是你一定要持续地去优化它，比如你要引入一些会议的工具，你要去观测，你要去教大家如何开会。
	3. 决策效率，决策效率也特别重要，不要议而不决。我对团队说过一句话，**中层干部要独挡一面，不敢做决定的都要被干掉**，不许拿 A/B 方案来跟我来汇报，不好意思，A 还是 B，你必须选一个。
11. 别带着工人心态去抱怨，我们下面一个开发跟我抱怨说，Alex，业务给我们提的需求，今天说这样做，过了两天又那样做，过了几天又变了，来回反复。我问他，你有没有观点，有没有意见，你既没观点又没意见，你做不就行了吗？公司又不是没给你发工资。产品经理能力不足，业务部门对事情认知不到位，这是你不得不经历的。但是，所有的人都不要去抱怨，都要去思考这个事情我们是不是做对了，别一方面是个工人心态，一方面又抱怨这、抱怨那。所以我们的 3 到 5 人团队，每一个团队都要懂业务，我们打造的绝对是全公司最懂业务的技术团队，甚至任何一个业务部门都没我们懂。给每个小团队算财务账的阶段，一句话总结就是**拿着工资去创业**。
12. 缺乏能做顶层设计的人才。 乌卡时代具有不确定性，但你又要设计一个大的体系，去识别公司里有确定性的东西和不确定性的东西。但是你的公司里面有没有这样的人？你们的 CTO 和高层有没有把这个事情规划好让大家一起去做、一块块落地？你说，不对，**我们还是需求驱动的，那还谈什么？本来过去就是需求驱动的，你想一下把所有的需求都做了，能让企业上一个大台阶吗？很多公司从来没有失败的项目，都是成功的项目，可一年下来有什么用呢？确定的是成本，不确定的是收益**。
13. 数字化转型是滚滚长江水，谁都挡不住。老的企业总会死的，新一代的企业会出来。企业如人，有的企业已经是老人，老态龙钟，有的企业是壮年，有的企业决心返老还童，但是真正做到了返老还童的企业少之又少。再过 20 年，很多企业里面的很多业务都是有数字化思维的人在主导，因为效率高。如果用自然法则、生老病死去看一个企业，会看得很透，**对于一些达到比较高职位的人来说，你再去选企业的时候就知道如何选择了**，你要选赛道、选创始人，尤其是你带着一身抱负想要施展的，尤其要看你去了之后这个企业能不能改变，做好这个判断。

Fatescript：
1. 团队不是简单的人的集合，团队中个人的强并不代表团队的强大。按照上面提到的scale的说法，团队不是人的scale。就像是打篮球一样，巨星抱团未必能有立竿见影的效果，“化学反应”才是关键，竞技体育中的核心目的是赢球而不是某个人的得分高。在一个所有人都很厉害的团队，很容易出现互不认可然后“大路朝天，各走一边”的情况，有人戏称这种情况叫做“聚是一坨x，散是满天星”。担任这样的团队的leader，技术实力的强弱与否只是一个评价维度，相较来说，能够协调处理各方的冲突，使团队focus共同的目标，是leader更重要的能力。
2. 我工作期间有幸跟过一些从比较简单、原始的状态开始的项目。现在回看，该走的弯路、该踩的坑，一个都不会少。初期犯错，是为了后期犯更少、更严重的错。作为对比，有些技术非常好的leader可能会不认可他们管辖的人（通常也是执行具体任务的人）的提议，希望项目能够少走一些弯路。这种不认可有很多表现形式，可能是只给你灌输他认可的想法，偏离这种想法的路径都是歪门邪道；可能是默许你做一些自己的创新，但是对于你在新路径上的发现采取不管不顾的态度。虽然后者的态度能让员工保留一些自主性，但是在一个鼓励创新与探索的环境里，这两种态度都是十分有毒的，因为“试错，是走向正确道路的最佳实践，有一些学费是不得不交的。”对于leader来说，指导固然很关键，但是建立鼓励尝试、对错误宽容（但及时反思）的团队氛围的作用要远大于指导。员工很多时候需要的是催化剂，而不是导航员。
3. 在一个团队里面，如果你总觉得一个事情需要有人去做，那么你也许是最适合的人选。比如你可能会觉得文档需要有人写了，那很可能你就是最适合写文档的人（说明你对缺乏文档这件事情的耐受度比较低）；或者你觉得频繁做某项任务太麻烦了，应该写一个工具简化流程，当你在团队里表达了这个想法之后，很可能最后就是交给你来完成。


## 心态上

到今年7月份就工作3年了，回过头看过去3年，再看以后，有哪些脉络和教训。

1. 第一个坎，从学生到工作的转变。第二个坎，不用带脑子也能完成一天的工作（成了一个熟手）。那么下一步怎么搞？
1. 头三年是锻炼基本能力的三年。说白了就是单兵能力的提升

	* 基本工作组件（语言、存储）的熟悉，以能实现编码任务
	* 技术依赖组件的熟悉，以使故障时能够发现原因
	* 源码阅读及分析
	* 对程序开发有个宏观认识

2. 那么以后呢？要做成更大的系统，一定是调动多个人的力量。要做成更大的项目，一定是团结多个团队的力量。为此

	* 做好心理上的准备。作为技术人，难免觉得技术才是最牛逼的。技术，更确切的是写代码，并不是难度最大的事情。但你敢这么说的 前提一定是 你基础比较扎实

3. 自己成为驱动的源头，驱动人，驱动事儿。做驱动，心理上的准备也是很重要的，“这事儿该不该我做” 是一个很费心神的问题。

	* 有一次跟一个工作十几年的大牛交流， 大牛说我工作这么多年，只有一次，就是那个人我跟着他干就行，做什么项目都可以。大部分时候你的身边找不到这样的角色， 那你就要勇于承担这样的角色。
	* 做好驱动的一个必要条件是，能从琐碎中脱身。一个是人的注意力有限，注意好琐碎就管不好大局。一个是人都有逃避心理，比如一直负责一个项目，就很烦那个项目了。
	* **自驱动虽然心态上很心累，但好处便是，对项目有很大的主导权，可以按自己的意愿引导项目的发展。**

你必须要接受，本来你可以十分钟干完的事儿，别人要干两个小时。你设计的很有美感的代码 被破坏殆尽。但该他负责的事情还是要他去做，否则他没有参与感，永远无法主动地做工作。很多事情你做了其实没有成长，但还是要带着别人一起做，帮助别人成长也是你工作的一部分。

曾经的一个困惑：“我知道很多事儿要干，但我不知道精力投向哪里”。迷茫，就是对当前的处境没有成体系化的认识，要多人交流， 整理，发现规律，指导实践。

要兼容不同的人，有的很有想法但落地能力不够，有的想法较少但好在认真负责等等，不以自己认为的优缺点去套别人。

## 原则上

[写给工程师的十条精进原则](https://tech.meituan.com/10_principles_for_engineers.html)

1. 设计优先，比如 开发周期在3pd以上的项目必须有设计文档，开发周期在5pd以上的项目必须有设计评审。对应的，必须有n个xx级别以上的工程师参与讨论。
2. 设计的过程是一种智力上的创造，我们更希望它能成为个人与集体智慧的结晶。如何才能让我们的设计变得通俗易懂？我个人认为，设计应该尽量使用比较合理的逻辑，进而把设计中的一些点组织起来。比如可以使用从抽象到具体，由总到分的结构来组织材料。在设计过程中，**要以需求为出发点，**通过合理的抽象把问题简化，讲清楚各个模块之间的关系，再详细分述模块的实现细节。
3. “P/PC平衡”原则，即产出与产能平衡原则。伊索寓言中讲述了一个《生金蛋的鹅》的故事。产出好比“金蛋”，产能好比“会下金蛋的鹅”。“重蛋轻鹅”的人，最终可能连产蛋的资产都保不住；“重鹅轻蛋”的人，最终可能会被饿死。产出与产能必须平衡，才能达到真正的高效能。**从系统的角度看，每一个系统都是通过持续不断地叠加功能来实现其产出，而系统的产能是通过系统架构的可扩展性、稳定性等一系列特性来表征。为了达到产出与产能的平衡，需要在不断支持业务需求的过程中，持续进行技术架构层面的优化。**“P/PC平衡”原则还适用于很多其他的领域，例如团队、家庭等
4. 波克定理告诉我们，只有在争辩中，才可能诞生最好的主意和最好的决定。

许晓斌：提高团队效率的核心还是提升开发者的体验。
1. 首先，给他们设置有挑战有意思的目标，激发团队；
2. 其次，在组织结构和系统架构上设计比较合理的高内聚低耦合的边界，让团队同学可以用更多的时间去思考如何解决技术 / 用户问题，而不是浪费在沟通和消耗上；
3. 还有，就是建立比较好的工程师文化和氛围，关注工程质量，而不是整天在所谓的“代码屎山”上工作；
4. 最后，就是给他们提供最好的工具，无论是 AI 的也好，还是我们称之为平台工程也要，这也是我团队为阿里巴巴整体在做的事情。
技术管理者需要同时兼顾技术和管理，二者缺一不可。在实际的工作中我看到的优秀管理者，都是两者同时成长的。如果只做管理，缺乏对技术深度的理解，就会在风险的识别，人员的任用，系统的合理架构设计上出现许多问题。反之，如果只关注技术，缺乏对人的理解和关心，那随着团队变大，各种冲突、矛盾就会被方法（是的，有人的地方就有观念差异和冲突），管理者很多时候是在基于对每个人的理解和认同的基础上，去建立团队的共识和积极的氛围，是一件解决工程师团队规模 Scalability 的事，**即随着人员的增多怎么保障平均每个人的贡献不下降**。

### 不同项目的不同角色

负责多个项目之后，笔者曾有一段时间感觉很空虚，感觉有很多事儿要做却又不知道从何做起。不同的项目中，一个人可能呈现出不同的角色（可能是多个角色并存）：

1. 需求/技术调研
2. 唤醒高层、平级、小伙伴对项目的重视
2. 项目设计
3. coder
4. 进度推动/资源协调
5. 推广
6. 系统运营

针对不同项目的不同角色，做不同的事情，并认可其价值，还是一件蛮有挑战的事情。从一个程序员的角度来说，什么最让人安心？持续的技术进步。这里有几个认知问题

1. 技术只是达成目标的一环，不同项目不同阶段 瓶颈不一样，作为一个小队长首要的是解决这些瓶颈
2. 随着代码能力的提高，一些项目的code无助于技术提高
3. 与技术的沟通方式变了，不仅是直接code，而是通过code review、项目设计 与技术“沟通”

作为开发，鼓励同时参与到其它角色中。**作为小队长，因为精力有限，一旦决定了自己的角色，便要谨慎加入其它角色，尤其是短期内无法脱身的角色。**一定要防止：同时推进多个项目，每天很忙但每个项目都进展不大的情况，因为这样会极大的削弱工作带来的“获得感”（进而发现远没有纯做技术充实），引发挫折感。

进展不大的原因是：

1. 忙项目B时 被中断去解决项目A 的问题，负责项目越多，被中断的可能性越大，以至于完全无法开展新工作
2. 解答用户对系统的咨询
3. “调度精力”带来的开销
4. 高层决心不确定，不是不做，但不肯花大力气做
5. 依赖方太多，你有空，别人没空

因此

1. 尽量专人专职
2. 理清项目优先级，最好用图表的方式展示，适时搁置一些项目
3. 完备的文档，减少维护压力
4. 两人或多人负责一个项目，使得有一人可以专心开发（另一人负责维护的事情）
5. 不能因为技术实现上图省事，来让上层业务方案来做妥协，进而产生让用户感觉困惑的操作。
6. 对自己小组的产出能力、工作负载要有一个认识，不要盲目拉活儿。

### 个人梳理

1. 高噪声背景下提取有效信息的能力
2. 管住头尾，项目刚开始的时候都是很模糊的，这时候要你帮小伙伴明确目标，明确产出，并去校验那个产出。
3. 随着项目的增多，你会发现：**被动应对的事情太多，主动规划的太少**。这样你会很挫败，不仅是工作，人生也是如此。
4. **如果没有设计流程规范，大家习惯于会按照最简单的方案去实现。如果没有项目管理，大家习惯于去做最简单的/易做的/容易写周报的/依赖资源最少/推得动的事儿，而对紧急的、收尾的、杂碎的事儿无动于衷。**
6. 嗅出大家的认知瓶颈点， 对大家做说服工作。当年林彪问主席红旗还可以打多久？你也要类似，做很多的说服工作。说服的技巧

	* “我觉得”通常是说服不了人的
	* 你要有一个体系化的论述，并且大多数时候要形成文章，从当前的问题、场景、上下文、所有可选方案出发，系统和完整。也就是说，你一定要证明，你的理解比他人更深刻和全面
	* 借助大厂方案的背书

## 工具集

善于利用工具、日程管理、任务管理，番茄工作法等。除了用teambition 管理项目外，还需设置定时提醒，定期检查teambition上的未完成任务。

分层无处不在，不要让一个函数 里出现不同抽象层次的代码，另一个角度说， 很多时候脑子乱也是如此。人的脑容量有限，一时思考不了过多维度的事情。所以要借助 日程管理、任务管理、表格 等归纳整理

实现一个新的业务系统，理解业务，提出抽象概念 ，定义概念和名词很重要，将几个小步骤 用一个词语 归纳很重要，可以有效的降低沟通成本。

陈晨：要熟练使用工具，利用大量的工具，尽量把流程自动化，用工具管人而不是用人管人，这样管理压力会小很多。PS：有点理解为什么需要单元测试和代码覆盖率，因为通常也没有时间帮小伙伴review，这时代码质量几乎完全靠个人，这时心里是很虚的，不出事故纯靠运气。

### 分阶段管理项目

假设你同时负责几个项目，但不同的项目所处的状态是不一样的。比如策划阶段 ==> 调研阶段 ==> 设计阶段 ==> 实施阶段 ==> 维护阶段。

1. 策划和调研阶段，因为太模糊了，可能要独立完成
2. 设计阶段带着小伙伴一起做（业务架构图、类图、流程图、Swagger、数据库表设计等）
3. 实施阶段 每天检查下小伙伴的产出，确保没有跑偏即可
4. 维护阶段可以交给小伙伴，提取新的需求 进入新的循环

如果所有阶段的所有事儿 都需要你去做，那么一定有问题。如果一直如此，你可能不是纠结自己做什么，而是要换个人合作。

**当你有了一定的人力资源之后，你解决问题的工具集不能  只有一个：一有问题自己上。** 这个项目目前在什么阶段？最大的瓶颈在哪里？是否需要你直接参与？想好了再行动。

形成一套成体系的表格，进而决定轻重缓急，从哪切入、推动。我后来会在周报中整理表格体现这些内容，既用于汇报工作，也时刻提醒自己。

||所处阶段|本周产出/进展|瓶颈/问题|切入点/办法|
|---|---|---|---|---|
|项目1|开发|下周转入测试|暂无||
|项目2|维护|无|暂无|code review|
|项目3|维护|明确xx为负责人|依赖资源不到位|组织各个大佬开会|
|项目4|调研|整理文档以备开会|需求不明|开会|
|项目5|开发|无|当前算法难以支持高并发|小组讨论|


### 熟悉小伙伴

||主动性|技术能力|认真负责|项目熟悉程度、兴趣程度|
|---|---|---|---|---|
|小伙伴1|||||
|小伙伴2|||||

你如何画这个表格，本身就体现了你对这事儿的认识。

公司发展到一定阶段，能力强的员工容易离职，因为他们对公司内愚蠢的行为的容忍度不高，他们也容易找到好工作；能力差的员工倾向于留着不走，他们也不太好找工作，年头久了，他们就变中高层了。在湖畔大学第三届的第一课上，马云也讲到：小公司的成败在于你聘请什么样的人，大公司的成败在于你开除什么样的人。

阿里巴巴的人才盘点矩阵中，就明确将员工划分了成了明星、瘦狗、野狗、牛和小白兔五大类

如何提到面试“看人”的准确率？我们需要什么样的人？如何判断一个人是不是那样的人？根据他的能力交代任务时以下达指令为主还是意图为主？

小伙伴的状态评估

1. 项目设计能力，技术选型，方案选型，一定的知识广度
2. 代码设计能力，优雅、“坏味道”比较少，易懂
3. 独立做事情
3. 有没有成长性，即自我超越的心思和动力

如何帮助？

1. 直接指明工作中的问题
2. 找到跟他关系比较好的小伙伴（相对能力也强一点的），一则从他这边了解更多的信息， 二则让他们两个一起做事情。
3. 一个是准确判断小伙伴的状态，对一个人能力不停地试探，做的好就上探，做的不好就下探。你对他的期望和实际的工作不一致，他难受，你也难受。
4. 判断完毕后，可以推荐一些书籍， 限期读完，小组内做一个分享。
5. 对不同的事儿，不同的人，无所区别分析，是很大的偷懒。

每个小伙伴，进来的第一个项目最重要，你要进行观察，把他作为“客体”进行认知。性格、协作能力、设计能力、code能力（规范、以及优雅程度）。进而确定你们以后的相处模式

1. 不用管
	* 完全放心
	* 小伙伴外向，主动沟通
2. 定期汇报
3. 每日跟踪

同时，不管他优秀也好，差一点也好，第一个项目想办法做成、做好。在这个过程中，将他的code 习惯等打上自己和公司的印记。

沟通很重要，第一优先是质量，但这个比较难，需要你跟小伙伴在一个“思维”上。

对于有些人来说，你教他防患于未然，不如等他自己踩坑。

如果没有积极主动地心态，很多人要半年后才可以适应新公司的节奏。这个时候，带人耐心很重要。

如何管理能力比较差的小伙伴？

1. **很多时候，小伙伴做不好，或者没有达到你的心理预期，往往不是能力问题，而是因为他们不知道“真正的好是什么”**。他们认为好的，可能在你眼里只值60分，你眼里认为好的，他们可能听都没听说过。这不是能力问题，而是眼界问题，而影响眼界的原因非常的多，可能是出身、经历等等，所以当你意识到小伙伴存在认知错误的时候，最好的方式是“做一遍给他看”，或者是“给他看看好东西的样子，并告诉他一些简单的窍门”。一般而言，上述行为做个两三次左右并不会花你太多时间，如果两三次后还不行， 一个是可能教的不对，或者小伙伴不擅长这个，最后才可能是他的能力/态度不行。
2. 你需要以一种坦诚的方式，告诉他你的评价标准是什么，标准一定是有效地，而且可量化的，不会轻易改变的，即使改变也是有理有据可以信服的。小伙伴当前 在标准里面 处于什么位置（完全不行，中规中矩，还不错，很不错）
3. 私下批评，公共场合维护
4. 别失态别升级，失态升级就把问题拖入绩效之外的信任问题了
5. 管理的本质是“配置资源，实现既定目标”。注意：是“配置”资源，而不是“改造”资源，或者“创造”资源。大多数时候，改变一个人，比改变自己都难的多。
6. 不要让较弱的小伙伴单独做事情，找个人跟他一起。几个人一起帮助他，比你一个人帮助他 心态上要轻松。


### 团队文化

秦国自商鞅变法后设立了以首级计算军功的制度，后来发展完善为二十等爵制。针对低等爵位的战士，只要斩获敌人一个首级，便可以升一级爵位。军官的爵位是按照集体功劳计算，如果自己部队的斩首达到了三十人以上，该队的百将、屯长才能记功；如果攻城战中斩首八千人、野战中斩首两千人，指挥作战的将军才能记功。

从中可以看到，对于军官来说，不是砍人头就有爵位的（不然二十等爵制很快就到顶了），也不是打赢就有爵位的，而是要满足一定的整体要求，本质上是为了更有效率的砍人头。


对于技术开发来说，在有了一定的经验后，通常实现一个项目/系统 难度不是很大，这个时候要引导他们对自己有更高的要求，比如更优雅的代码、更好的效率等。

### 沟通

在创业公司，大家都是一个人干好几件事情，其他部门或者同事给你交代的事情，你其实很难完全理解，都是在被动中被推动着前进。这其实并没有错，抓大放小么。但是**在信息流的传播中，就不能抓大放小**，尤其对领导者来说，更是如此，任何员工对你的反馈都是他个人比较看重的地方。


### 项目管理

怎么带领一个团队去做业务分析和领域模型建模和自己写一个高可用的中间件是不一样的体验。

一般业务系统的精华其实就是数据库表设计，如果后端界面/前端接口再卡死的话，代码实现基本就是依葫芦画瓢的事情了。以前都是我一个人做， 所谓基本理念的东西都在我脑子里，不用什么文档。再后续我只是帮小伙伴把表设计定好，再后续我发现小伙伴还是理解不了，就把后端界面、前端接口定死，辅助以架构设计图说明，code review进行质量管控。这要求

1. 团队工作一定要有文档
1. 自己要对软件设计的本质有一个认识，先做到独自一人可以高质量的实施
2. 有了认识，才知道重点和难点，才能利用规律。当事情交给别人的做的时候，才知道如何把控。
3. 不要想着：把事情交给一个人，他就应该把事情做好，自己只要看结果就行

	* 而是对他的能力有一个认识，这个事适不适合他干
	* 如果你对一个人的能力没有认识、考察，把一个事情交给了一个不合适的人，那最大的错是自己。 
	* 若是他干，哪些事儿他干不了。尤其是定好边界（表设计、前后端接口、架构图）等防止跑偏
	* **如果说，衡量程序员的水平是对一个项目有没有一个全面的把控和实现。那么衡量一个架构师的水平，就是你的产出（包括但不限于表设计、接口定义、架构图等）是否对实施者水平要求太高。**
	* 不要抱怨身边人的能力，通常没有什么用。重要的是，发现问题、瓶颈（虽然是对方能力导致的），做更多的工作解决它。

	
如果你不能将一个复杂的事情拆的很简单，以至于小伙伴有信心实施，那就说明你对问题的理解不够。

**你碰到的人和事儿，对彼时当下来说，是有得失的，但对人生旅程、职业生涯上来讲，终究是有收获的。**

## 技术

要习惯从以往只要自己做好就一切都好的状态，转到大家好才是真的好的状态。

而领导力，简单来讲，除了本身对技术的掌握，还要注意人员管理和沟通相关的内容，取决于不同的组织，未必你需要管理这些人，比如薪水、绩效等，但是你一定会需要负责让这些人能够理解和正确地使用技术完成任务。以前你可能 90% 以上的精力都投入在技术上，而现在，这 90% 中的 20% 甚至更大比例的精力，都要用于提升整个团队的技术能力，毕竟，你的 20%\*1 个人，换回来的可能是 20%\*9 个人的整体效率提升。

任何的方法论都依赖于基层执行人员的能力和纪律，所以说打造好基层执行人员的能力和纪律，打造好基层的团队和个人是方法论生效的基础。

肖德时：创新能力并不是指发明一个技术，或者是写了一大堆新算法，而是涉猎广泛一点，在尝试解决公司难题的前提下，能不断的小步尝试一些未能证明的假设。

王喆：三四年前，我们还经常吐槽算法工程师的面试是“面试造航母，工作拧螺丝”，现在随着各家公司的系统复杂度都越来越高，对算法工程师的要求也确实水涨船高，作为一个合格的算法工程师，**你还真得知道怎么造航母，才敢在一艘大船上拧螺丝**。否则结果就是要么这艘船上到处都是螺丝钉，要么直接把船修沉了。

## 对整体环境的把握

一个公司的发展有几个阶段

1. 业务驱动，为用户带来价值，公司先活下去
2. 技术和业务并重，技术问题暴露出来，不解决就无法进一步发展
3. 技术驱动业务，具备极短事件内复制一个app、搞一个运营活动、为用户赋能、千人千面等。

具体体现在：

1. 具体体现在人员、精力的占比
2. 你觉得一件事需要好好搞搞，hr告诉你一个合适的候选人待遇要求很高时，才是考验你意志和决心的时候

每个阶段如何处理技术与业务的关系（以及是否进入下一阶段）是大佬的事情，但自己要认识到当前的局面 并做相应的调适

1. 认识到某个技术当前在公司的定位、公司可能的投入，以有限的资源 去解决最痛的痛点，不要极客的立即去实现技术落地的完美状态。
2. 技术要根据阶段取舍，你当前的项目也要根据阶段取舍。设计 ==> 实现 ==> 迭代 ==> 搁置 ==> 迭代，当前是什么状态？落后还是快于实际需要？自己要心里有数。
2. 以第一线的观察 分析当前的问题/瓶颈，进行技术的宣传和说服工作，推动领导的相关决策
3. 对自己计划的制定、精力的分配 有一个判断
4. 无论大环境怎样，自己要保持激情，发现问题，解决问题，给自己信心，给身边人信心，把事情打开一个新局面，新气象。