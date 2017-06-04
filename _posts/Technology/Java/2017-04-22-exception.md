---

layout: post
title: java exception
category: 技术
tags: Java
keywords: JAVA exception

---

## 前言（未完成）


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
