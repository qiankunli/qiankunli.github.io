---

layout: post
title: 《docker源码分析》小结
category: 技术
tags: Container
keywords: Docker,source

---

## 简介

第一个阶段，读docker源码分析，不求了解一切细节，但最起码对一个细节有困惑的时候，可以通过追溯代码去释疑。

学习《Docker源码分析》除了加深对docker的理解外，一个比较大的收获是：从一个java程序猿的角度，学到了如何理解c/go语言中的一些程序设计技巧。设计模式是通的，c/go语言也大量的的使用了设计模式，但因为没有默认的面向对象语法、spring/ioc等支持，设计模式的实现隐藏在了大量的代码细节中，跟业务代码混杂在一起。

几个基本问题：

1. 容器与传统虚拟机的相同与不同
2. 容器与进程的关系问题。容器不能脱离进程而存在  ==> 先有进程后有容器 ==> namespace在clone子进程时传入,cgroup在子进程创建后设置。
3. daemon作为父进程，启动docker init子进程。dockerinit是容器的第一个进程，拥有自己独立的network,mount等空间。dockerinit启动后，docker daemon向其传入容器config数据，dockerinit据此完成容器的初始化，譬如初始化网络栈等。
4. docker daemon、docker init、Entrypoint以及Cmd之间（这里的Entrypoint/Cmd,指的是容器启动后，我们自己要容器执行的程序）的关系问题。参见[从 Linux 进程的角度看 Docker](http://chuansong.me/n/1739346),不完全对，但提到的这个点很有必要了解下。


## 数据的表现形式


容器的创建分为两个部分：create和start。为什么要分开呢？因为docker在启动的时候，会加载以前的container数据，并将它们启动，可能是为了复用启动的过程。

另外，与进程类似，docker也分为静态数据和动态数据。对于容器来说，容器在运行时，内存中对应一个struct contianer，容器stop后，struct container会json化后存在磁盘上。

对于镜像来说，

||表现形式|实际内容|
|---|---|---|
|网络传输或repository存储时|archive file|a list of file（类似于，winrar可以将一个目录下文件压缩为一个rar文件，解压后，文件依然具有自己的目录结构） + jsonData file|
|本地，因graph driver而异，以aufs drvier为例|layers，diff，mnt三个目录|见下文|

假设存在image1和image2，base ubuntu是image1的父镜像，image1是image2的父镜像，image2相对于image1新增了一个`/root/data`和`/root/conf/abc.conf`文件

	var/lib/docker/aufs
							/layers
								/imageId1
								/imageId2
							/diff
								/imageId1
								/imageId2
									/root/data
										  /conf/abc.conf
							/mnt
								/imageId1
								/imageId2
									/usr
									/var		// mount 过的目录
									...


对于image2来说，三个目录的内容如下：

||文件/目录|内容|
|---|---|---|
|layers|文件|一行一个parent imageId，这里就是imageId1|
|diff|目录|相对于imageId2不同的文件，每个文件有自己的路径|
|mnt|union 目录|一个ubuntu的根目录|

换句话说，当根据image2创建容器时，`var/lib/docker/aufs/mnt/imageId2`就是容器的根目录（当然，实际上还要在`var/lib/docker/aufs/mnt/imageId2`之上加一个init layer和读写layer）。

union mount 的一个实例`sudo mount -t aufs -o br=/tmp/dir1=ro:/tmp/dir2=rw none /tmp/aufs` 类似到 docker 中（未验证），就是`sudo mount -t aufs -o br=/xx/diff/imageId1=ro:/xx/diff/imageId2/root=rw none /xx/mnt/imageId2`

## Driver和模板模式

docker的各种Driver，在对应driver包的根部有一个driver.go负责通用逻辑（给daemon调用），同时会自定义一些Driver接口，搞一些抽象操作。然后各个实际的Driver根据实际的情况去实现。Driver.go通用逻辑中会调用这些Driver接口。**这相当于模板模式**，或者说，Driver.go 类似于java中的AbstractxxServie。以execDriver为例

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
	

**结构体定义在哪个位置，跟它自己的抽象所在的维度有关系。**

## docker 和libcontainer的关系

到目前为止，我们可以看到，docker daemon要处理的东西很多:

1. http request等业务处理
2. 文件、系统信息、配置信息加载
3. struct daemon、struct container等能力对象的聚合与管理

作为docker重头戏的容器管理部分，反倒是隐藏在一片繁杂中。所以，将容器操作部分摘出，单独作为一个工具集。所谓工具集：定义好接口，自己不存储信息，只有功能实现。类似的还有libnetwork等。

## controller-service-dao

struct daemon 和 struct container

1. daemon 聚合所有的能力对象，提供docker command命令对应的操作（例如`docker run`等）
2. container 聚合容器相关的能力对象，提供具体的容器相关的实现（交给libcontainer执行）。

一个docker请求的处理过程，`docker client ==> docker daemon ==> xxdriver  ==> libxx ==> linux kernel`

struct daemon虽说是一个struct，但daemon.go却可以做到一个能力对象（比较叫做daemon interface）。docker 几个struct的关系，本质是一个类似controller-service-dao的结构，姑且视为daemon ==> container ==> driver。问题就在于不清晰，真正实现业务的代码和维持依赖关系的代码混杂，在java中依赖关系以及维护依赖关系的代码是由ioc专门维护的。

那么我们在分析docker 源码的时候，就要定位好哪些代码是哪个领域的功能，尽可能聚焦在它的业务模块，而不是一些依赖关系的维护之类的代码中。


## docker和操作系统的神似

对于linux操作系统，linux依次启动进程0、进程1（ldt和tss等写死在os中），提供（linux基本抽象）进程和文件运行所需要的环境。对于docker，容器启动后，在执行用户命令Entrypoint/Cmd之前，要为容器设置cgroup和namespace。其基本流程是，docker daemon创建fork一个子进程，子进程执行exec执行docker预定义好的dockerinit可执行文件。

docker daemon为dockerinit设置cpgroup，至于namespace，**因为go语言运行时不具备跨namespace操作资源的能力**，docker daemon将namespace及其它数据传给docker init，dokerinit据此配置自己网络、挂载等空间。命名空间初始化完毕后，执行用户命令Entrypoint/Cmd。

除了init流程的神似，[有容云——窥探Docker中的Volume Plugin内幕](http://geek.csdn.net/news/detail/74847)提到的volumn的实现原理，也可以看到docker和linux共通之处。linux提供进程运行环境，有一系列进程与进程所用资源的**全局组织与关联结构** 。docker提供容器的运行环境，docker daemon提供容器与容器所用资源的全局组织与关联结构。 


## 其它（待完善）

||主要工作|配置来源|
|---|---|---|
|docker daemon/host 网络(提供容器间连通性)|1. 网桥创建;2. iptables 规则设置|daemon读取用户配置（文件或daemon参数指定）|
|container 网络（容器本身的网络准备）|1. 创建veth pair;2. veth pair1 ==> bridge,veth pair2 ==>  container namespace|docker client设置|




一个容器有独立的命名空间，各种命名空间的表现形式。

||结构体||
|---|---|---|
|网络|type NetworkState struct{VethHost,VethChild,NsPath}||
|挂载|设备、文件系统、挂载点|



