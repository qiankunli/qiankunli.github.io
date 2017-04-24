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
 
 看完这段，对我们平时看异常，打的“堆栈”就有直观的感觉了。

