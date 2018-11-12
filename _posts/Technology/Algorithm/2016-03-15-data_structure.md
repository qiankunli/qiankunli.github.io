---

layout: post
title: 以新的角度看数据结构
category: 技术
tags: Algorithm
keywords: 数据结构

---


## 前言

* TOC
{:toc}

每本《数据结构》书上的主线都跑不了以下几点，时常回味一下，还是蛮有好处的。

数据结构分为逻辑结构、存储结构以及对应结构的数据运算，比如，你可以用一个数组表示一个图，也可以用链表存储一个图（或者一个图中包含数组和链表）。

存储结构主要包括：

1. 顺序存储：把逻辑上相邻的元素存储在物理位置上也相邻的存储单元里，元素之间的关系由存储单元的邻接关系来体现。其优点是可以实现随机存取，每个元素占用最少的存储空间；缺点是只能使用相邻的一整块存储单元，因此可能产生较多的外部碎片。

2. 链接存储：不要求逻辑上相邻的元素在物理位置上也相邻，借助指示元素存储地址的指针表示元素之间的逻辑关系。其优点是不会出现碎片现象，充分利用所有存储单元；缺点是每个元素因存储指针而占用额外的存储空间，并且只能实现顺序存取。

3. 索引存储：在存储元素信息的同时，还建立附加的索引表。索引表中的每一项称为**索引项**，索引项的一般形式是：（关键字，地址）。其优点是检索速度快；缺点是增加了附加的索引表，会占用较多的存储空间。另外，在增加和删除数据时要修改索引表，因而会花费较多的时间。

4. 散列存储：根据元素的关键字直接计算出该元素的存储地址，又称为Hash存储。其优点是检索、增加和删除结点的操作都很快；缺点是如果散列函数不好可能出现元素存储单元的冲突，而解决冲突会增加时间和空间开销。

**存储/物理结构通常绕不开寻址问题，以及增删改查的效率问题。**我们可以观察到集中存储结构的演进：

1. 顺序存储，直接放在一起
2. 链接存储，数据和位置信息放在一起
3. 索引存储，数据和位置信息完全独立，位置信息存在单独的索引表中
4. 直接寻址表，数据和位置信息完全独立，位置信息存在单独的数组中，数据直接作为数组下标。
5. 哈希表，直接寻址表的改进。

逻辑结构，**数据元素间抽象化的相互关系**，与存储无关，主要包括：

1. 集合结构中的数据元素之间除了 “同属于一个集合”的关系外，别无其他关系。
2. 线性结构结构中的数据元素之间只存在一对一的关系。
3. 树形结构结构中的数据元素之间存在一对多的关系。
4. 图状结构或网状结构结构中的数据元素之间存在多对多的关系。

每种逻辑结构包含一些基本的运算，包括遍历，增减节点等。

数据与数据之间的关系

## 队列

教科书上一说队列，就是四个字“先进先出”,这四个字是无法表述队列的巨大作用的.

### 合并请求的一种实现

假设一个方法或类负责管理一个资源,在多线程环境下,这个类便需要"线程安全"

1. 将这个类改造成线程安全的类
2. 调整这个类的调用方式.利用生产者消费者模式,所有想要使用这个资源的的线程(生产者)先提交请求到一个队列,由一个专门的线程(消费者)负责接收并处理请求.

### 轮询的一种实现

假设我们有一个集合，对集合中的元素实现轮询效果。

1. 我们可以标记一个index，记住上一次使用的元素，进而实现轮询。
2. 用环形队列存储集合，即天然具备轮询效果。

这两种方式，在多线程环境下，还需注意线程安全问题。

## 图

深度优先和广度优先遍历，{初始状态，目标状态，规则集合}在寻找最佳策略上的应用

## 碎碎念

### 设计数据结构的过程是一种“映射”

[Data Structure](https://www.encyclopedia.com/computing/dictionaries-thesauruses-pictures-and-press-releases/data-structure)Computer solution of a real-world problem involves designing some ideal data structures, and then mapping these onto available data structures (e.g. arrays, records, lists, queues, and trees) for the implementation. 先假设一种理想结构 然后再考虑着 组合基本结构去实现。


我们说面向的对象的四个基本特性：抽象、封装、继承、多态。在四个基本特性之上呢，一群类的组合，有了各种设计模式。

数组 + 哈希函数 就成了一个散列表。 也就是 基本特性的组合，是否也有设计模式一说呢？ 对于跳表来说，与其说数据结构本来就是那样子，还不如说为了在链表上提高查询速度而衍生的数据结构。亦或者说B+树，我们也可以说先有底层那一条叶子链，再有的上层索引结构。数据结构 + 数据结构 ==> 某方面更优秀的数据结构。数据结构 + 算法 ==> 更优秀的数据结构。

![](/public/upload/algorithm/data_structure_vs_object.png)

### 为什么查询和排序是数据结构的基本操作

以缓存为例，一个缓存（cache）系统主要包含下面这几个操作：

1. 往缓存中添加一个数据
2. 从缓存中删除一个数据
3. 在缓存中查找一个数据

这三个操作都涉及到查找操作，换句话说，对数据的操作无外乎crud，而cud都离不开查询。进而很明显，基于有序结构的查询速度是最快的，也就引出了排序算法的重要性。


### 查询的不同意涵

数据结构，model the application data，it is generally a requirement for any application to insert, edit and query a data store. Data structures offer different ways to store data items, while the algorithms provide techniques for managing this data. 不管什么样的数据结构，都跑不掉insert、edit、和query。尤其是query ，对于不同的数据结构，意涵很丰富

1. 查询的入参不同，可能是index、数据的某个属性、范围、数据的某个特征（比如第k大的数）
2. 查询返回的结果不同，可能是数据本身、数据集合、也可能是多个数据的组合（比如图的最优路径）

### 多重查询

1. 数组 + hash 构成了散列表，根据index 查询和 key 查询 都是O(1)
2. 假设 既想根据学号 又想根据成绩 查询学生，则可以先按学号递增组织学生数据，再按照分数构造一个散列表（该思想对应到数据库上，就是针对主键另外建了一个非聚簇索引）。

为什么散列表和链表经常会一起使用？

1. 链表只能基于一个维度来构建有序数据，也只能基于这个维度来查询数据。基于链表构建一个散列表 则相当于 构建了一个“非聚簇索引”，增加了查询维度。
2. 散列表中数据是经过散列函数打乱之后无规律存储的，在散列表上加链表（使用指针将散列表的n 个拉链 串起来），则可以支持散列表按照插入或访问顺序遍历数据，比如java中的LinkedHashMap。


### 数据结构与业务设计

一般的业务系统要建立数据库表，数据库表要建立索引。笔者有一个经验，不要一开始建立索引。而是业务代码完毕后，观察常查询的属性，然后对这些数据建立索引。如果你愿意，索引 + 数据库表，就是一种数据结构，数据结构的构建（比如建立索引） 要反应业务的“意志”。

[Definition of a Data Structure & Algorithms](https://smallbusiness.chron.com/definition-data-structure-algorithms-27214.html)Computing applications use many different types of data. Some applications model data within database systems, in which case the database system handles the details of choosing data structures, as well as the algorithms to manage them. However, in many cases, applications model their own data. When programmers decide what type of data structure to use for a particular set of data in an application, they need to take into account the specific data items to be stored, the relationships between data items and how the data will be accessed from within the application logic. [ddd(一)——领域驱动理念入门](http://qiankunli.github.io/2017/12/25/ddd.html) 也提到，一般业务以数据库ER设计为驱动，数据库设计代表了对业务的认识深度，也是业务的精华所在。 由此看， 数据库完成了一个项目本身应有的数据结构与算法的活儿，我们不用直接进行数据结构设计（这由数据库封装），而是通过ER表设计来间接影响数据库进行实际的数据结构设计。

Algorithms for managing data structures sometimes involve recursion. With recursion, an algorithm calls itself, which means it repeats its own processes as part of a looping structure, with each step simplifying the problem at hand. Recursive algorithms can allow programmers to implement efficient sorting and searching techniques within their applications. However, writing recursive algorithms can be difficult for beginners, as it does require a significant amount of practice. 这段有一个很有意义的观点，Recursive algorithms 一般要对应 looping structure，是不是可以笼统的说：递归的算法一般对应着可以递归的数据结构。 

## 数据结构的"基本类型化"

较早出现的编程语言,比如c语言,基本类型只包括:int,char,long等,string,map等则需要引用库.而对于新兴语言,比如go和python等,则在语言层面支持string,map等复杂结构.这一趋势,甚至扩展到了一些内存数据库.



