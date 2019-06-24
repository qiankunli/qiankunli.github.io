---

layout: post
title: java exception
category: 技术
tags: Java
keywords: JAVA exception

---

## 前言

异常有两种：

1. 非运行时异常（Checked Exception）: 这种异常必须在方法声明的throws语句指定，或者在方法体内捕获。例如：IOException和ClassNotFoundException。**《clean code》建议在写代码时不使用Checked Exception，因为catch语句和抛出异常语句经常隔的很远，而你必须在它们之间的每个方法签名中声明该异常。**
2. 运行时异常（Unchecked Exception）：这种异常不必在方法声明中指定，也不需要在方法体中捕获。例如：NumberFormatException

因为run()方法不支持throws语句，所以当线程对象的run()方法抛出非运行异常时，我们必须捕获并且处理他们。当运行时异常从run()方法中抛出时，默认行为是在控制台输出堆栈记录并且退出程序。

## 打印异常

Throwable.printStackTrace的实现：

	 Set<Throwable> dejaVu =
        Collections.newSetFromMap(new IdentityHashMap<Throwable, Boolean>());
    dejaVu.add(this);
	
    synchronized (s.lock()) {
        // Print our stack trace
        s.println(this);
        StackTraceElement[] trace = getOurStackTrace();
        for (StackTraceElement traceElement : trace)
            s.println("\tat " + traceElement);
	
        // Print suppressed exceptions, if any
        for (Throwable se : getSuppressed())
            se.printEnclosedStackTrace(s, trace, SUPPRESSED_CAPTION, "\t", dejaVu);
	
        // Print cause, if any
        Throwable ourCause = getCause();
        if (ourCause != null)
            ourCause.printEnclosedStackTrace(s, trace, CAUSE_CAPTION, "", dejaVu);
    }

其中StackTraceElement的代码

	public final class StackTraceElement implements java.io.Serializable {
	    // Normally initialized by VM (public constructor added in 1.5)
	    private String declaringClass;
	    private String methodName;
	    private String fileName;
	    private int    lineNumber;
	}
	
 An element in a stack trace, as returned by {@link
 Throwable#getStackTrace()}.  Each element represents a single stack frame.All stack frames except for the one at the top of the stack represent a method invocation.  The frame at the top of the stack represents the execution point at which the stack trace was generated.  Typically,this is the point at which the throwable corresponding to the stack trace was created.
 
 看完这段，对我们平时看异常，打的“堆栈”就有直观的感觉了。堆栈是一个数组，以多行的方式打印，每一行包含了执行路径里的类名、方法名、代码行号等信息。

## 堆栈打印和jvm优化

因为jvm优化的原因，一些异常出现次数较多时，将不打印完整堆栈。

The compiler in the server VM now provides correct stack backtraces（回溯） for all “cold” built-in exceptions. For performance purposes, when such an exception is thrown a few times, the method may be recompiled. After recompilation, the compiler may choose a faster tactic（策略） using preallocated（预先分配的） exceptions that do not provide a stack trace. To disable completely the use of preallocated exceptions, use this new flag: `-XX:-OmitStackTraceInFastThrow`.


jvm 如何处理异常 [JVM4——《深入拆解java 虚拟机》笔记](http://qiankunli.github.io/2018/07/20/jvm_note.html)

异常捕捉 对性能是有影响的，

## InterruptedException

[Dealing with InterruptedException](https://www.ibm.com/developerworks/library/j-jtp05236/index.html)

**为什么会有InterruptedException？cancellation mechanism**

When a method throws InterruptedException, It is telling you that it is a blocking method and that it will make an attempt to unblock and return early 

The completion of an ordinary method is dependent only on how much work you've asked it to do and whether adequate computing resources (CPU cycles and memory) are available. The completion of a blocking method, on the other hand, is also dependent on some external event, such as timer expiration, I/O completion, or the action of another thread (releasing a lock, setting a flag, or placing a task on a work queue). Ordinary methods complete as soon as their work can be done, but blocking methods are less predictable because they depend on external events. blocking method 何时结束是难以预测的，因为要等待external events


Because blocking methods can potentially take forever if the event they are waiting for never occurs, it is often useful for blocking operations to be cancelable.(It is often useful for long-running non-blocking methods to be cancelable as well.)  A cancelable operation is one that can be externally moved to completion in advance of when it would ordinarily complete on its own. The interruption mechanism provided by Thread and supported by Thread.sleep() and Object.wait() is a cancellation mechanism; it allows one thread to request that another thread stop what it is doing early. When a method throws InterruptedException, it is telling you that if the thread executing the method is interrupted, it will make an attempt to stop what it is doing and return early and indicate its early return by throwing InterruptedException. Well-behaved blocking library methods should be responsive to interruption and throw InterruptedException so they can be used within cancelable activities without compromising responsiveness. 既然blocking method 何时结束难以预测，那就有必要提供一个取消机制

Every thread has a Boolean property associated with it that represents its interrupted status. The interrupted status is initially false; when a thread is interrupted by some other thread through a call to `Thread.interrupt()`

**Interruption is a cooperative mechanism**. When one thread interrupts another, the interrupted thread does not necessarily stop what it is doing immediately. Instead, interruption is a way of **politely asking** another thread to stop what it is doing if it wants to, at its convenience.  You are free to ignore an interruption request, but doing so may compromise responsiveness. 当你在实现一个block 或者 long-running方法，发现当前线程 interrupted status 被置为true了，你可以不理它，但最好是处理一下，停下“手头”的工作，开始收尾。

