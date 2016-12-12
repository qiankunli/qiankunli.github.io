---

layout: post
title: 《docker源码分析》小结
category: 技术
tags: Docker
keywords: Docker,graphdriver

---

## 简介

第一个阶段，读docker源码分析，不求了解一切细节，但最起码对一个细节有困惑的时候，可以通过追溯代码去释疑。

学习《Docker源码分析》除了加深对docker的理解外，一个比较大的收获是：从一个java程序猿的角度，学到了如何理解c/go语言中的一些程序设计技巧。设计模式是通的，c/go语言也大量的的使用了设计模式，但因为没有默认的面向对象语法、spring/ioc等支持，设计模式的实现隐藏在了大量的代码细节中，跟业务代码混杂在一起。

几个基本问题：

1. 容器与传统虚拟机的相同与不同
2. 容器与进程的关系问题。容器不能脱离进程而存在  ==> 先有进程后有容器 ==> namespace在clone子进程时传入,cgroup在子进程创建后设置。
3. daemon启动docker init（拥有自己独立的network,mount等空间），至于这些空间的具体内容，daemon向docker init提供数据（通过管道），init自己一步步完成。
4. docker daemon、docker init、entrypoint以及Cmd之间的关系问题。参见[从 Linux 进程的角度看 Docker](http://chuansong.me/n/1739346),不完全对，但提到的这个点很有必要了解下。


## 容器的创建


容器的创建分为两个部分：create和start。为什么要分开呢？因为docker在启动的时候，会加载以前的container数据，并将它们启动，可能是为了复用启动的过程。另外，与进程类似，docker也分为静态数据和动态数据。阳光底下没有新鲜事。

create是创建容器数据，这个数据体现在两个方面：

1. 文件数据
2. 创建docker自己定义的一些数据结构

||进程|容器|备注|
|---|---|---|---|
|copy from|parent process|base image||
|struct|pcb|container|聚合必要的能力对象|
|数据准备|输入、输出、打开的文件等|容器网络环境，创建后，cgroup设置等|谁让容器比进程封装的多呢|
|切换|寄存器的数据存在一个结构体中|container 结构体json化后存在磁盘上|


## Driver

<table>
	<tr>
		<td>graph driver</td>
		<td>repositoris,layers,diff,mnt</td>
	</tr>
	<tr>
		<td>network driver</td>
		<td>bridge etc</td>
	</tr>
	<tr>
		<td>exec driver</td>
		<td>libcontainer</td>
	</tr>

</table>

docker的各种Driver，在对应driver包的根部有一个driver.go负责通用逻辑（给daemon调用），同时会自定义一些Driver接口，搞一些抽象操作。然后各个实际的Driver根据实际的情况去实现。Driver.go通用逻辑中会调用这些Driver接口。**这相当于模板模式**，或者说，Driver.go 类似于java中的AbstractxxServie。

	type Driver interface {
		Run(c *Command, pipes *Pipes, startCallback StartCallback) (int, error) // Run executes the process and blocks until the process exits and returns the exit code
		Kill(c *Command, sig int) error
		Pause(c *Command) error
		Unpause(c *Command) error
		Name() string                                 // Driver name
		Info(id string) Info                          // "temporary" hack (until we move state from core to plugins)
		GetPidsForContainer(id string) ([]int, error) // Returns a list of pids for the given container.
		Terminate(c *Command) error                   // kill it with fire
	}
	
参数为什么是command而不是container，因为container实现了libcontainer的container接口。用native execdriver在Run()时，command还得转换为container对象。但要是用原来的lxc就不用了，这里使用command，估计是execdriver的一个统一抽象吧。**结构体定义在哪个位置，跟它自己的抽象所在的维度有关系。**

## docker 和libcontainer的关系

到目前为止，我们可以看到，docker daemon要处理的东西很多:

1. io与业务
2. 文件、系统信息、配置信息加载
3. daemon、container等能力对象的聚合与管理

作为docker重头戏的容器管理部分，反倒是隐藏在一片繁杂中。所以，将容器操作部分摘出，单独作为一个工具集。所谓工具集：定义好接口，自己不存储信息，只有功能实现。

libcontainer定义了一些接口，供上层的execDriver实现，方法主要有两种

1. 返回container的一些状态，libcontainer通过它们感知容器信息。**所以这个接口，也是既有数据部分，又有操作部分。**以前我自己定义的方法，只有操作方法，所有接口一般叫做xxService。
2. 实现一些功能，execdriver在合适的时机调用它们

		type Container interface {
			// Returns the ID of the container
			ID() string
			Status() (Status, error)
			State() (*State, error)
			Config() configs.Config
			Processes() ([]int, error)
			Stats() (*Stats, error)
			Set(config configs.Config) error
			Start(process *Process) (err error)
			Checkpoint(criuOpts *CriuOpts) error
			Restore(process *Process, criuOpts *CriuOpts) error
			Destroy() error
			Pause() error
			Resume() error
			NotifyOOM() (<-chan struct{}, error)
		}

类似的interface还有factory、process等。

一个docker请求的处理过程，`docker client ==> docker daemon ==> driver ==> ==> libcontainer ==> linux kernel`

struct daemon 和 struct container

1. daemon 聚合所有的能力对象，提供命令对应的操作
2. container 聚合容器相关的能力对象，提供容器相关的操作
最终，准备好相关的数据，交给libcontainer执行。



docker 几个struct的关系，本质是一个类似controller-service-dao的结构，每种类型的Driver也确实存在接口的定义。问题就在于不清晰，真正实现业务的代码和维持依赖关系的代码混杂，并且还是混杂在类似于controller-service-dao（也就是实际干活的代码中），在java中依赖关系以及维护依赖关系的代码是由ioc专门维护的。

那么我们在分析docker 源码的时候，就要定位好哪些代码是哪个领域的功能，尽可能集中在它的业务模块。

## docker网络

||主要工作|配置来源|
|---|---|---|
|docker daemon/host 网络|1. 网桥创建;2. iptables 规则设置|daemon读取用户配置（文件或daemon参数指定）|
|container 网络|1. 创建veth pair;2. veth pair1 ==> bridge,veth pair2 ==>  container namespace|docker client设置|

## 引用


