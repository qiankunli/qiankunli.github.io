---

layout: post
title: java exception
category: 技术
tags: Java
keywords: JAVA exception

---

## 前言（未完成）

## 其它

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