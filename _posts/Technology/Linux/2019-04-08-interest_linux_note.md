---

layout: post
title: 《趣谈Linux操作系统》
category: 技术
tags: Linux
keywords: linux命令 

---

## 简介（持续更新）

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

程序是代码写的，所以一定要有”代码段“。代码的执行过程 会产生临时数据，所以要有”数据段“（当然数据段根据数据特点，一般分为堆和栈）。PS：这种描述方式很有感觉

内存空间都是”按需分配“的，但在OS层面上，都是整存整取的。对于`int[] array = new int[100];`

|层次|表现|array对应的感觉|
|---|---|---|
|java 语言|申请一个数组，得到一个引用array|进程标识 + 进程内 偏移地址|
|jvm|在堆里申请一段空间|进程数据段地址 + 偏移地址|
|os|对于一般进程，就是申请时给进程的堆一段空间。<br>对于jvm 就是jvm 启动时申请一段连续的空间，然后再由jvm自行管理内存分配|物理内存地址|


## 一切皆文件

“一切皆文件”的优势，就是统一了操作的入口