---

layout: post
title: jvm crash分析
category: 技术
tags: JVM
keywords: jvm crash

---

## 简介

* TOC
{:toc}

## 什么是jvm crash

Symptoms include:

1. webapps are completely unavailable.
2. The Java process isn't even running. It was not shut down manually.
3. Files with names like hs_err_pid20929.log being created in your app server's bindirectory (wherever you start it from), containing text like:

        #
        # An unexpected error has been detected by HotSpot Virtual Machine:
        #
        #  SIGSEGV (0xb) at pc=0xfe9bb960, pid=20929, tid=17
        #
        # Java VM: Java HotSpot(TM) Server VM (1.5.0_01-b08 mixed mode)
        # Problematic frame:
        # V  [libjvm.so+0x1bb960]
        #

        ---------------  T H R E A D  ---------------

        Current thread (0x01a770e0):  JavaThread "JiraQuartzScheduler_Worker-1" [_thread_in_vm, id=17]

        siginfo:si_signo=11, si_errno=0, si_code=1, si_addr=0x00000000

        Registers:
        O0=0xf5999882 O1=0xf5999882 O2=0x00000000 O3=0x00000000
        O4=0x00000000 O5=0x00000001 O6=0xc24ff0b0 O7=0x00008000
        G1=0xfe9bb80c G2=0xf5999a48 G3=0x0a67677d G4=0xf5999882
        G5=0xc24ff380 G6=0x00000000 G7=0xfdbc3800 Y=0x00000000
        PC=0xfe9bb960 nPC=0xfe9bb964
        ....
        this indicates Java is crashing.


[Java crashes](https://confluence.atlassian.com/confkb/java-crashes-235669496.html)

The virtual machine is responsible for emulating a CPU, managing memory and devices, just like the operating system does for native applications (MS Office, web browsers etc).

A Java virtual machine crash is analogous to getting a Windows Blue Screen of Death when using MS Office or a web browser. jvm crash 就好比 windows 蓝屏

## Troubleshoot System Crashes

[Troubleshoot System Crashes](https://docs.oracle.com/javase/10/troubleshoot/troubleshoot-system-crashes.htm#JSTGD314)

A crash, or fatal error, causes a process to terminate abnormally. There are various possible reasons for a crash. For example, a crash can occur due to a bug in the Java HotSpot VM, in a system library, in a Java SE library or an API, in application native code, or even in the operating system (OS). External factors, such as resource exhaustion in the OS can also cause a crash.

Crashes caused by bugs in the Java HotSpot VM or in the Java SE library code are rare.

## tomcat 进程突然关闭（未完成）

[Tomcat进程意外退出的问题分析](https://www.jianshu.com/p/0ed131c2e76e)

监控软件跟踪到 日志显示 如下

	{
	    "appName": "xxx", 
	    "blockedCount": 8656, 
	    "blockedTime": -1, 
	    "hostname": "xx", 
	    "inNative": false, 
	    "lockOwnerId": 0, 
	    "stackTrace": "sun.misc.Unsafe.park(Native Method)
	java.util.concurrent.locks.LockSupport.parkNanos(LockSupport.java:215)
	java.util.concurrent.locks.AbstractQueuedSynchronizer$ConditionObject.awaitNanos(AbstractQueuedSynchronizer.java:2078)
	java.util.concurrent.LinkedBlockingQueue.poll(LinkedBlockingQueue.java:467)
	org.springframework.amqp.rabbit.listener.BlockingQueueConsumer.nextMessage(BlockingQueueConsumer.java:188)
	org.springframework.amqp.rabbit.listener.SimpleMessageListenerContainer.doReceiveAndExecute(SimpleMessageListenerContainer.java:466)
	org.springframework.amqp.rabbit.listener.SimpleMessageListenerContainer.receiveAndExecute(SimpleMessageListenerContainer.java:455)
	org.springframework.amqp.rabbit.listener.SimpleMessageListenerContainer.access$300(SimpleMessageListenerContainer.java:58)
	org.springframework.amqp.rabbit.listener.SimpleMessageListenerContainer$AsyncMessageProcessingConsumer.run(SimpleMessageListenerContainer.java:548)
	java.lang.Thread.run(Thread.java:745)
	", 
	    "suspended": false, 
	    "threadID": 419, 
	    "threadName": "SimpleAsyncTaskExecutor-171", 
	    "threadState": "TIMED_WAITING", 
	    "timestamp": 1526904360000, 
	    "waitedCount": 3685416, 
	    "waitedTime": -1
	}

看懂这个公式 表示的问题，有助于了解问题，比如waitedCount 的含义等

java 进程挂掉的几个分析方向

1. 在jvm 脚本中 增加 -XX:+HeapDumpOnOutOfMemoryError 看看进程退出时的堆内存数据
2. 被Linux OOM killer 干掉。该killer 会在内存不足的时候 kill 掉任何不受保护的进程，从而释放内存，保证kernel 的运行。若进程 是被 killer 干掉的，则`/var/log/messages` 会有相关的日志，比如`Out of memory：Kill process 31201(java) score 783 or sacrifice child`，也可以使用`dmesg | egrep -i ‘killed process’` 此时通常有两个办法：将自己的进程设置为受保护；找一个内存富余点的服务器。==> 分析下挂掉的几次 是否是同一台服务器。
	