---

layout: post
title: Unsafe
category: 技术
tags: Java
keywords: java forkjoin

---

## 简介


### 实例创建

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

	allocateMemory
	copyMemory
	freeMemory
	put/getAddress
	getInt/Byte/Object/Char/Float
	putInt/Byte/Object/Char/Float
	
	
	// 用例
	Unsafe unsafe = getUnsafe();
	Field f = guard.getClass().getDeclaredField("field_name");
	unsafe.putInt(guard, unsafe.objectFieldOffset(f), 42); // memory corruption

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

![](/public/upload/java/thread_status.jpg)


[Java和操作系统交互细节](https://mp.weixin.qq.com/s/fmS7FtVyd7KReebKzxzKvQ)对进程而言，就三种状态，就绪，运行，阻塞，而在 JVM 中，阻塞有四种类型，我们可以通过 jstack 生成 dump 文件查看线程的状态。

1. BLOCKED （on object monitor)  通过 synchronized(obj) 同步块获取锁的时候，等待其他线程释放对象锁，dump 文件会显示 waiting to lock <0x00000000e1c9f108>
2. TIMED WAITING (on object monitor) 和 WAITING (on object monitor) 在获取锁后，调用了 object.wait() 等待其他线程调用 object.notify()，两者区别是是否带超时时间
3. TIMED WAITING (sleeping) 程序调用了 thread.sleep()，这里如果 sleep(0) 不会进入阻塞状态，会直接从运行转换为就绪
4. TIMED WAITING (parking) 和 WAITING (parking) 程序调用了 Unsafe.park()，线程被挂起，等待某个条件发生，waiting on condition

### 为什么需要 park 和 unpark

[The java.util.concurrent synchronizer framework](http://gee.cs.oswego.edu/dl/papers/aqs.pdf) aqs 作者关于 aqs 的论文 有一个blocking 小节，重点如下：

1. Until JSR166, there was no Java API available to block and unblock threads for purposes of creating synchronizers that are not based on built-in monitors 除了jvm 内置 monitor 对象，当时java 没有上层api 提供 block/unblock 线程的能力。所以再重复下重点：Unsafe 提供 Low level primitives for synchronization.
2. The only candidates were Thread.suspend and Thread.resume, which are unusable because they encounter an unsolvable race problem: If an unblocking thread invokes resume before the blocking thread has executed suspend, the resume operation will have no effect. Thread.suspend 和 Thread.resume 倒是行，但提前执行 Thread.resume 就比较容易尴尬
3. **this applies per-thread, not per-synchronizer.** 这或许解释了 很多场景下，synchronized 除了性能依然不够用的原因。

其实Solaris-9/WIN32/Linux NPTL 都有类似的thread library，java 支持的比较晚，总之，java 的synchronized 关键字 不是java并发的全部， java也是在不断发展的。

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
