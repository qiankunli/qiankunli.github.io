---

layout: post
title: 《趣谈Linux操作系统》笔记
category: 技术
tags: Linux
keywords: linux命令 

---

## 简介（持续更新）


* TOC
{:toc}

所谓成长，就是知道自己目前**在哪里**，清楚将要**去哪里**，然后通过学习和行动到达目的地。

## 进程——为啥创建进程的 系统调用起名叫fork（分支） 

**一句看似废话的废话：进程是进程创建出来的**

创建进程的系统调用叫fork。这个名字很奇怪，中文叫“分支”为啥启动一个新进程叫“分支”呢？在 Linux 里，要创建一个新的进程，需要一个老的进程调用fork 来实现，其中老的进程叫作父进程（Parent Process），新的进程叫作子进程（Child Process）。当父进程调用 fork 创建进程的时候，子进程将各个子系统为父进程创建的数据结构也全部拷贝了一份，甚至连程序代码也是拷贝过来的。


对于 fork 系统调用的返回值，如果当前进程是子进程，就返回0；如果当前进程是父进程，就返回子进程的进程号。这样首先在返回值这里就有了一个区分，然后通过 if-else 语句判断，如果是父进程，还接着做原来应该做的事情；如果是子进程，需要请求另一个系统调用execve来执行另一个程序，这个时候，子进程和父进程就彻底分道扬镳了，也即产生了一个分支（fork）了。

    public static void main(String[] args) throws IOException {
        Process process = Runtime.getRuntime().exec("/bin/sh -c ifconfig");
        //
        //  jvm这里隐藏了一个 父子进程  判断的过程
        //
        Scanner scanner = new Scanner(process.getInputStream());
        while (scanner.hasNextLine()) {
            System.out.println(scanner.nextLine());
        }
        scanner.close();
    }

新进程 都是父进程fork出来的，那到底谁是第一个呢？这就是涉及到系统启动过程了。

突然想起来，linux 和 git 都是大佬Linus的 杰作。

## 内存管理 brk和mmap

内存空间都是”按需分配“的，但在OS层面上，都是整存整取的。对于`int[] array = new int[100];`

|层次|表现|array对应的感觉|
|---|---|---|
|java 语言|申请一个数组，得到一个引用array|进程标识 + 进程内 偏移地址|
|jvm|在堆里申请一段空间|进程数据段地址 + 偏移地址|
|os|对于一般进程，就是申请时给进程的堆一段空间。<br>对于jvm 就是jvm 启动时申请一段连续的空间，然后再由jvm自行管理内存分配|物理内存地址|

## 一切皆文件

“一切皆文件”的优势，就是统一了操作的入口

## x86 架构

![](/public/upload/linux/hardware_architecture.jpeg)

### “指令格式-cpu结构-总线”的暧昧关系

![](/public/upload/linux/cpu_architecture.jpeg)

我们以一段x86汇编代码为例

    mov [ebp-4], edi   ; Move EDI into the local variable
    add [ebp-4], esi   ; Add ESI into the local variable
    add eax, [ebp-4]   ; Add the contents of the local variable

CPU 的控制单元里面，有一个指令指针寄存器，执行的是下一条指令在内存中的地址。控制单元会不停地将代码段的指令拿进来，先放入指令寄存器。当前的指令分两部分，一部分是做什么操作，例如是加法还是位移；一部分是操作哪些数据。**要执行这条指令，就要把第一部分交给运算单元，第二部分交给数据单元**。数据单元根据数据的地址，从数据段里读到数据寄存器里，就可以参与运算了。运算单元做完运算，产生的结果会暂存在数据单元的数据寄存器里。最终，会有指令将数据写回内存中的数据段。

CPU 里有两个寄存器，专门保存当前处理进程的代码段的起始地址，以及数据段的起始地址。这里面写的都是进程 A，那当前执行的就是进程 A 的指令，等切换成进程 B，就会执行 B的指令了，这个过程叫作进程切换（Process Switch）（注意跟线程切换做区别）

CPU 和内存来来回回传数据，靠的都是总线。其实总线上主要有两类数据，一个是地址数据，也就是我想拿内存中哪个位置的数据，这类总线叫地址总线（Address Bus）；另一类是真正的数据，这类总线叫数据总线（Data Bus）。

|程序|算法|数据结构|
|---|---|---|
|指令|操作码|立即数/地址|
|cpu|运算单元|数据单元|
|整体结构|cpu|内存|地址总线/数据总线|

![](/public/upload/linux/cpu_architecture_detail.jpeg)

程序是代码写的，所以一定要有”代码段“。代码的执行过程 会产生临时数据，所以要有”数据段“（当然数据段根据数据特点，一般分为堆和栈）。PS：这种描述方式很有感觉。[进程管理信息数据结构](http://qiankunli.github.io/2017/02/14/linux_art_review.html) 二进制文件分段 ==> 进程分段 ==> 指令操作码/操作数 ==> cpu运算单元/数据单元 ==> cpu代码段寄存器/数据段寄存器/堆栈段寄存器等 有一种软硬件融合的味道。

一切运算即加法，一切分支代码即jump