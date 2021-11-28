---

layout: post
title: Unsafe
category: 技术
tags: JVM
keywords: java unsafe

---

## 简介

* TOC
{:toc}

我们通常对Java语言的认知是：Java语言是安全的，所有操作都基于JVM，在安全可控的范围内进行。然而，Unsafe这个类会打破这个边界，使Java拥有C的能力，可以操作任意内存地址，是一把双刃剑。

## 实例创建

[Java Magic. Part 4: sun.misc.Unsafe](http://ifeve.com/sun-misc-unsafe/) 英文版 [Java Magic. Part 4: sun.misc.Unsafe](http://mishadoff.com/blog/java-magic-part-4-sun-dot-misc-dot-unsafe/)要点如下：

1. 创建Unsafe 实例，不能`Unsafe unsafe = new Unsafe()`，有静态方法 `Unsafe.getUnsafe()`，但直接调用 也会抛 SecurityException。为何？你只能从 受信任的代码 中执行`Unsafe.getUnsafe()`
2. Java如何验证代码是否可信。它检查我们的代码是否由主要的类加载器加载。
3. 你可以 运行`java -Xbootclasspath:/usr/jdk1.7.0/jre/lib/rt.jar:. xx.UnsafeClient` 使`xx.UnsafeClient ` 受信任，但通常不要这么玩
4. Unsafe类包含一个私有的、名为theUnsafe的实例，我们可以通过Java反射窃取该变量。

		Field f = Unsafe.class.getDeclaredField("theUnsafe");
		f.setAccessible(true);
		Unsafe unsafe = (Unsafe) f.get(null);


## unsafe 能干啥

### 直接操作堆内存/堆外内存

Unsafe 提供 Direct memory access methods.

```
allocateMemory
copyMemory
freeMemory
put/getAddress
getInt/Byte/Object/Char/Float
getInt/Byte/Object/Char/FloatVolatile // 直接到主存中拿数据
putInt/Byte/Object/Char/Float
// 用例
Unsafe unsafe = getUnsafe();
Field f = guard.getClass().getDeclaredField("field_name");
unsafe.putInt(guard, unsafe.objectFieldOffset(f), 42); // memory corruption
```
	
	

如果知道 某个对象某个属性的内存地址，那么连对象的引用都不需要，可以直接设置值

使用堆外内存。 Java数组大小的最大值为Integer.MAX_VALUE，使用直接内存分配，我们创建的数组大小受限于堆大小

	class SuperArray {
		private final static int BYTE = 1;
		private long size;
		private long address;
		public SuperArray(long size) {
		    this.size = size;
		    address = getUnsafe().allocateMemory(size * BYTE);
		}
		public void set(long i, byte value) {
		    getUnsafe().putByte(address + i * BYTE, value);
		}
		public int get(long idx) {
		    return getUnsafe().getByte(address + idx * BYTE);
		}
		public long size() {
		    return size;
		}
	}
		
### 类操作

提供 object/fields/classes/static fields/Arrays manipulation（该单词是 操纵；操作；处理；篡改 的意思）

初始化类`A o3 = (A) unsafe.allocateInstance(A.class);`
动态加载类，从任何位置拿到class 文件的二进制内容，即可得到class 对象

		byte[] classContents = getClassContent();
		Class c = getUnsafe().defineClass(
		              null, classContents, 0, classContents.length);
		    c.getMethod("a").invoke(c.newInstance(), null); // 1


### synchronization

**Unsafe 提供 Low level primitives for synchronization.**

	compareAndSwapInt/Long/Object
	monitorEnter
	tryMonitorEnter
	monitorExit
	putOrderedInt

在使用大量线程的共享对象上增长值

	interface Counter {
		void increment();
		long getCounter();
	}

1. 无锁版本，最快，但结果不正确
2. increment 新增 synchronized 关键字
3. increment 使用 WriteLock
4. 使用 AtomicLong 计数
5. 使用cas，性能和 AtomicLong 就基本等价了。 **Atomic 类的实现也基本如此。**

		class CASCounter implements Counter {
			private volatile long counter = 0;
			private Unsafe unsafe;
			private long offset;
			public CASCounter() throws Exception {
			    unsafe = getUnsafe();
			    offset = unsafe.objectFieldOffset(CASCounter.class.getDeclaredField("counter"));
			}
			@Override
			public void increment() {
			    long before = counter;
			    while (!unsafe.compareAndSwapLong(this, offset, before, before + 1)) {
			        before = counter;
			    }
			}
			@Override
			public long getCounter() {
			    return counter;
			}
		}

## park/unpark

### java 状态图

[JVM内存与执行](http://qiankunli.github.io/2015/08/02/jvm_physical.html)

### 为什么需要 park 和 unpark

[The java.util.concurrent synchronizer framework](http://gee.cs.oswego.edu/dl/papers/aqs.pdf) aqs 作者关于 aqs 的论文 有一个blocking 小节，重点如下：

1. Until JSR166, there was no Java API available to block and unblock threads for purposes of creating synchronizers that are not based on built-in monitors 除了jvm 内置 monitor 对象，当时java 没有上层api 提供 block/unblock 线程的能力。所以再重复下重点：Unsafe 提供 Low level primitives for synchronization.
2. The only candidates were Thread.suspend and Thread.resume, which are unusable because they encounter an unsolvable race problem: If an unblocking thread invokes resume before the blocking thread has executed suspend, the resume operation will have no effect. Thread.suspend 和 Thread.resume 倒是行，但提前执行 Thread.resume 就比较容易尴尬
3. **this applies per-thread, not per-synchronizer.** 这或许解释了 很多场景下，synchronized 除了性能依然不够用的原因。

其实Solaris-9/WIN32/Linux NPTL 都有类似的thread library，java 支持的比较晚，总之，java 的synchronized 关键字 不是java并发的全部， java也是在不断发展的。PS：线程库不单是 submit 一个task 执行，线程的创建、中断、阻塞、从阻塞中恢复 都是基本的对线程的操作。毕竟底层 都是对task_struct state 的操作。

### 工作原理

![](/public/upload/jvm/hospot_thread_object.png)

[Understanding Java and native thread details](https://www.ibm.com/support/knowledgecenter/en/SSB23S_1.1.0.15/com.ibm.java.vm.80.doc/docs/javadump_tags_javaandnative_thread_detail.html) A Java thread runs on a native thread, java thread 和native thread 有一个Attach 和Unattach 的过程。native thread 驱动 java thread 代码序列

[打通JAVA与内核系列之一ReentrantLock锁的实现原理](https://mp.weixin.qq.com/s/224pnAA6e8LvFcbJNHpWug)每个java线程都有一个Parker实例 Unsafe.park ==> `thread->parker()->park(isAbsolute != 0, time);` 即 获取java线程的parker对象，然后执行它的park方法。parker内部有个关键字段_counter, 这个counter用来记录所谓的“permit”，当_counter大于0时，意味着有permit，然后就可以把_counter设置为0，就算是获得了permit，可以继续运行后面的代码。如果_counter=0，则把线程的状态设置成_thread_in_vm并且_thread_blocked。_thread_in_vm 表示线程当前在JVM中执行，_thread_blocked表示线程当前阻塞了。拿到mutex之后，再次检查_counter是不是>0，如果是，则把_counter设置为0，unlock mutex并返回。如果_counter还是不大于0，调用相应的pthread_cond_wait系列函数进行等待，如果等待返回（即有人进行unpark，则pthread_cond_signal来通知），则把_counter设置为0，unlock mutex并返回。本质上来讲，LockSupport.park 是通过pthread库的条件变量pthread_cond_t来实现的。无论是pthread_cond_wait还是pthread_cond_signal 都必须得先pthread_mutex_lock。pthread_mutex_lock使用了称为Futex(快速用户空间互斥锁的简称)的系统，futex的解决思路是：在无竞争的情况下操作完全在user space进行，不需要系统调用，仅在发生竞争的时候进入内核去完成相应的处理(wait 或者 wake up)。所以说，futex是一种user mode和kernel mode混合的同步机制。


```c++
class Parker : public os::PlatformParker {
private:
  volatile int _counter ;
  ...
public:
  void park(bool isAbsolute, jlong time);
  void unpark();
  ...
}
class PlatformParker : public CHeapObj<mtInternal> {
  protected:
    enum {
        REL_INDEX = 0,
        ABS_INDEX = 1
    };
    int _cur_index;  // which cond is in use: -1, 0, 1
    pthread_mutex_t _mutex [1] ;
    pthread_cond_t  _cond  [2] ; // one for relative times and one for abs.
}
```

### 其它

锁是服务于共享资源的；而semaphore是服务于多个线程间的执行的逻辑顺序的。

[java并发编程之LockSupport](https://www.jianshu.com/p/ceb8870ef2c5)

[LockSupport的park和unpark的基本使用,以及对线程中断的响应性](https://blog.csdn.net/aitangyong/article/details/38373137) 线程如果因为调用park而阻塞的话，能够响应中断请求(中断状态被设置成true)，但是不会抛出InterruptedException。

[java并发包源码学习之AQS框架——LockSupport](https://www.cnblogs.com/zhanjindong/p/java-concurrent-package-aqs-locksupport-and-thread-interrupt.html)

LockSupport的park/unpark和Object的wait/notify：

1. 面向的对象不同；
2. 跟Object的wait/notify不同，**LockSupport的park/unpark直接以线程为参数**，不需要获取对象的监视器； 
3. 实现的机制不同，因此两者没有交集(也就是notify 不能 释放一个park 的线程)。

## jvm 中，那些“水面下的”对象

1. Every object has an intrinsic lock associated with it.By convention, a thread that needs exclusive and consistent access toan object's fields has to acquire the object's intrinsic lock beforeaccessing them, and then release the intrinsic lock when it's done withthem.  笼统的说，在jvm中，每个对象都有一个monitor 对象。 
2. [Java的LockSupport.park()实现分析](https://blog.csdn.net/hengyunabc/article/details/28126139)每个java线程都有一个Parker实例



java 作为一个计算机语言，线程这块，先不说并发，单就线程管理本身，理应提供启动、暂停、中止（interrupt）、销毁（代码自然运行结束）一个线程的能力。暂停是 current 线程 自己停自己，中止是别人停自己。
