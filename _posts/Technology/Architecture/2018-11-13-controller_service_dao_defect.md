---

layout: post
title: ddd(三)——controller-service-dao的败笔
category: 技术
tags: Architecture
keywords: ddd cqrs

---

## 简介

* TOC
{:toc}

[领域驱动设计(DDD:Domain-Driven Design)](https://www.jdon.com/ddd.html)提到服务器后端发展三个阶段

1. UI+DataBase的两层架构，这种面向数据库的架构(上图table	module )没有灵活性。
2. UI+Service+DataBase的多层SOA架构，这种服务+表模型的架构易使服务变得囊肿，难于维护拓展，伸缩性能差
3. DDD+SOA的事件驱动的CQRS读写分离架构，应付复杂业务逻辑，**以聚合模型替代数据表模型，以并发的事件驱动替代串联的消息驱动**。真正实现以业务实体为核心的灵活拓展。

但一下子说ddd，很多人可能观念转变不过来，我们今天吐槽一下controller-service-dao的“坑”，挖一挖它的墙角。如果你觉得controller-service-dao 很不错，那说明你应对的场景还不够复杂，暂时还不适合谈论ddd。

## controller-service-dao的败笔

[Spring Web 应用的最大败笔](https://www.jdon.com/45857)

大部分Spring的Web应用程序，常见的错误的设计如下：

1. 领域模型对象用来存储应用的数据(当作DTO使用)，领域模型是贫血模型这样的反模式。
2. 服务层每个实体有一个服务。

该应用程序有一个整体的服务层，它有太多的责任。更具体地，服务层有两个主要问题：

1. 在服务层发现业务逻辑，**业务逻辑被分散在各个服务层**
2. 每个领域模型一个服务。**每一个类都应该有一个责任**，不应将原属于领域模型的行为方法等划放在服务中实现，对象不但有属性还有行为。

大道理说完了，讲点实际的例子。

## 贫血模型 VS 充血模型

我们必须将应用程序的业务逻辑从服务层迁移到领域模型类中，为何呢？ 先来看看贫血模型和充血模型的对比。

假设我是一个服务类，你是一个域模型对象。如果我让你从屋顶上跳下来，你会喜欢我这样的决定吗？跳下来会摔伤，自己没有脑子或被洗脑，变成僵尸，只听从执行，不思考自己的安全，这就是贫血模型的问题。

举个具体的例子，假设一个用户有很多收获地址

	class User{
		List<Address> addresses;
		setter
		getter
	}
	
那么在为用户添加收获地址时，不得不有很多判空操作
	
	class UserService{
		void addAddress(User user,Address address){
			List<Address> addresses = user.getAddresses();
			if(null == addresses){
				addresses = new ArrayList<Address>();
				user.setAddresses(addresses);
			}
			addresses.add(address);
		}
	}


想象一下

1. 如果有多个位置操作User的Address（这个例子针对这一点不是很适当），`if(null == addresses){...}` 会大量出现，代码量不大， 但会很丑。如果是电商业务，每一次购物都要做优惠券、红包、满减检查、余额不足检查等，这些逻辑有可能重复在各个Service中。
1. 更复杂的成员变量 `List<List>` 或者 `List<Map<String,String>>`
2. 更复杂的逻辑，比如设定默认地址，地址判重等。

`UserService.addAddress`表示，我只想添加个地址而已。 

换成充血模型

	class User{
		List<Address> addresses;
		public User(){
			addresses = new ArrayList<Address>();
		}
		void addAddress(Address address){
			addresses.addAddress(address)
		}
	}
	class UserService{
		void addAddress(User user,Address address){
			...
			user.addAddress(address);	
			...
		}
	}

	
从中可以看到，addresses的 初始化和 添加都由User 负责，代码简洁很多。

将业务逻辑从服务层迁移到域模型类有下面三个优势：

1. 我们的代码将以逻辑方式切割，服务层只要关注应用逻辑（这个词儿不是很懂，比如哪几个操作一定要放在一起以保证事务安全？），而我们的领域模型关注业务逻辑。
2. 业务逻辑只存在一个地方，容易发现修改。
3. 服务层的源代码是清洁的，不包含任何复制粘贴代码


## 搞得好像一切为了持久化

笔者在一篇文章中看到一个问题：如果内存足够大，且永不宕机，你还会用数据库么？不会， 因为：

1. 数据库表不支持继承和多态，表达能力有限。假设用户的联系方式可以是邮箱、电话（包括国家码，后续可以考虑扩展支持运营商信息）、qq任意一种，则用对象表示

		class User{
			Contact contact;
			setter
			getter
		}
		class Contact{
			int contactType
		}
		class QQ extends Contact{
			String qq;
		}
		class phone extends Contact{
			String country;
			String phone;
		}
	
	用数据库表示就很尴尬了，因为多态的感觉不太好弄，你只能：

	1. 建一个contact表，所有的字段都放在里面
	2. 建一个contact表，一种联系方式建一个表
	
2. 表达一对多关系要额外加字段，表达多对多关系要额外建一个表


我们回想一下controller-service-dao的实现过程

1. model + dao 借助自动化工具生成
2. 有一个添加地址的需求
3. 然后controller实现，进而在UserService 里加一个addAddress方法，进而自然地 逻辑就写在`UserService.addAddress` 里了，直到调用dao 为止。

搞得我们一切操作像是为了持久化，持久化是编程的目的么？有时候不是

还以上文的User为例，对每一个新来的用户，我们需要保存用户身份信息（身份证号、性别等）、收货地址信息、画像信息等。为了用户操作友好

1. 用户信息 按类别 在不同的页面上输入。比如填完身份信息，点击下一步，让用户填写收获地址信息。
2. 用户可以添加任意多个收货地址，可以让用户在地图上选择地址，考虑到页面空间有限，一个页面只添加一个收货地址。一个收货地址添加完毕后， 用户可以选择下一步（添加兴趣信息）或者 新增下一个收货地址。
3. 每一个操作 都可以上一步，以便用户修改

针对这个需求，有几个实现方式

1. 每一步操作都保存到数据库，回显时从数据库中读取数据。这涉及到 用户请求对象 和 数据库对象的 相互转换。
2. 内存中有一个User 充血对象，在最后一步保存到db之前，其它所有的步骤只操作User 即可，包括但不限于

	1. 添加/回显身份证信息
	2. 添加/回显收货地址
	3. 添加/回显联系方式

为简单起见，你甚至可以将每一个步骤中页面发你的请求 数据直接保存在user 中，回显时原封不动直接返回给页面（用户的修改类似）。只有在最后保存的时刻， `user.sync` 同步到数据库。

![](/public/upload/architecture/ddd_step_save.png)

持久化就是持久化，本身不是业务逻辑的一部分（用户才不关心，甚至上层逻辑也不关心你将数据保存在msyql还是文件里，也不关心你是否做了分库分表），因此

1. 尽量的集中，对于整个User数据（包括n个收货地址和某种类型的联系方式）

	* 执行的时间集中
	* 代码的位置集中
2. 不要干预业务逻辑的处理过程，比如回显的时候不用从数据库获取。

## 碎碎念

只有架构分层是不够的，还需要更详细的**逻辑分层**，DDD领域驱动设计正是一个详细帮助建立丰富的有行为的领域模型的方法学。

数据驱动SQL ---->服务驱动SOA ----->领域驱动

聚合 >松耦合>重用 ==> 事件驱动>依赖注入>继承

事件驱动优于 依赖注入，依赖注入（也就是组合）优于继承

过去系统分析和系统设计都是分离的，这样割裂的结果导致，需求分析的结果无法直接进行设计编程，而能够进行编程运行的代码却扭曲需求，导致客户运行软件后才发现很多功能不是自己想要的，而且软件不能快速跟随需求变化。

DDD最大的好处是：接触到需求第一步就是考虑领域模型，而不是将其切割成数据和行为，然后数据用数据库实现，行为使用服务实现，最后造成需求的首肢分离。DDD让你首先考虑的是业务语言，而不是数据。重点不同导致编程世界观不同。

## 疑问

1. “我们的代码将以逻辑方式切割，服务层只要关注应用逻辑，而我们的领域模型关注业务逻辑。”这里的应用逻辑指的什么？业务逻辑指的什么？
2. “只有架构分层是不够的，还需要更详细的逻辑分层”。这里架构分层说的什么？逻辑分层说的什么？










