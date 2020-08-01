---

layout: post
title: 程序猿应该知道的
category: 技术
tags: Code
keywords: 程序猿应该知道的

---

## 前言

本文汇总了几位大家的说法



## 左耳听风

1. 广度的知识是深度研究的副产品
2. 很多时候，你缺少的不是知识而是热情
3. 很多东西在概念上是相通的，在哲学层次上是相通的，这是你需要去追求的学习知识的境界。
4. 永远和高手一起工作
5. 不要只寄望于在工作中学习，工作没有覆盖的地方你就不学了。真正的高手在工作之余都会花很多时间去自己研究点东西的。
6. 在合适的技术而不是熟悉的技术上工作
7. 学习到一定程度，就是要从书本中走出去，到社区里和大家一起学习，而且需要自己找食吃了。


## Teach Yourself Programming in Ten Years

[Teach Yourself Programming in Ten Years](http://norvig.com/21-days.html)

A language that doesn't affect the way you think about programming, is not worth knowing. 

The key is deliberative practice: not just doing it again and again, but challenging yourself with a task that is just beyond your current ability, trying it, analyzing your performance while and after doing it, and correcting any mistakes. Then repeat. And repeat again.

Learning by Doing,he most effective learning requires a well-defined task with an appropriate difficulty level for the particular individual, informative feedback, and opportunities for repetition and corrections of errors.

**Talk with other programmers; read other programs. This is more important than any book or training course.** 多与人聊天，多看代码是多么的重要。

Computer science education cannot make anybody an expert programmer any more than studying brushes and pigment can make somebody an expert painter

Learn at least a half dozen programming languages. Include one language that emphasizes class abstractions (like Java or C++), one that emphasizes functional abstraction (like Lisp or ML or Haskell), one that supports syntactic abstraction (like Lisp), one that supports declarative specifications (like Prolog or C++ templates), and one that emphasizes parallelism (like Clojure or Go).

Remember that there is a "computer" in "computer science". Know how long it takes your computer to execute an instruction, fetch a word from memory (with and without a cache miss), read consecutive words from disk, and seek to a new location on disk.


Approximate timing for various operations on a typical PC:

|||
|---|---|
|execute typical instruction|	1/1,000,000,000 sec = 1 nanosec|
|fetch from L1 cache memory|	0.5 nanosec|
|branch misprediction|	5 nanosec|
|fetch from L2 cache memory|	7 nanosec|
|Mutex lock/unlock|	25 nanosec|
|fetch from main memory|	100 nanosec|
|send 2K bytes over 1Gbps network	|20,000 nanosec|
|read 1MB sequentially from memory|	250,000 nanosec|
|fetch from new disk location (seek)|	8,000,000 nanosec|
|read 1MB sequentially from disk	|20,000,000 nanosec|
|send packet US to Europe and back	|150 milliseconds = 150,000,000 nanosec|



[97-things-every-programmer-should-know](https://github.com/97-things/97-things-every-programmer-should-know/blob/master/en/SUMMARY.md)

As long as we know that something exists, we can always infer, look-up and work-out everything we need to be able to use it. This attitude is fine, but I believe it is wrong to apply it when it comes to fundamental knowledge.“只要知道有这么个东西，用到了再学” 这个观点对基础知识是不适用的。

## 基础的重要性

[On The Value Of Fundamentals In Software Development](https://www.skorks.com/2010/04/on-the-value-of-fundamentals-in-software-development/) A couple of years ago I was asked the following question in an interview. Explain how optimistic locking works, in the context of Hibernate. At the time I had done a bit of work with Hibernate, but I was by no means an expert, so I drew a complete blank. What a total FAIL on my part that was, because optimistic locking is not really a Hibernate-centric idea, it is a fundamental concurrency control concept. I should have been able to fire-off an explanation of optimistic locking in general and I would have been 90% of the way there (regarding how it works within Hibernate).

What does it mean to internalise something completely? There is a difference between really understanding something and learning just enough to get by (i.e. having a working knowledge). 

We jump from technology to technology, wanting to know and be aware of everything (so we don't fall behind), but all the while, we would probably have been better off practicing our "basic moves". It doesn't matter how many object oriented languages you know, if you're not precisely sure what coupling, cohesion or abstraction means, your code is likely to be equally crappy in all of them. It doesn't matter how many inversion of control containers you have used, if you don't know what problem dependency injection is trying to solve, then you're just another sheep ready to follow the leader off a cliff (it's true, sheep will actually do that). 我们说要面向接口而不是面向实现编程，同样的，如果你不理解面向对象，那不管你会多少门语言，写出来的还是面向过程的代码，依然会出现一个函数几百行代码。这里，面向对象思想是“接口”，语言是“实现”。

Certainly there must be a good balance as you must keep yourself current but stopping to invest on the fundamentals.