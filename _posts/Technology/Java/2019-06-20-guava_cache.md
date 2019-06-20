---

layout: post
title: guava cache 源码分析
category: 技术
tags: Java
keywords: kafka

---

## 简介

[Guava LocalCache 缓存介绍及实现源码深入剖析](https://ketao1989.github.io/2014/12/19/Guava-Cache-Guide-And-Implement-Analyse/)

guava LocalCache与ConcurrentHashMap有以下不同

1. ConcurrentHashMap ”分段控制并发“是隐式的，而LocalCache 是显式的
2. 在Cache中，使用ReferenceEntry来封装键值对，并且对于值来说，还额外实现了ValueReference引用对象来封装对应Value对象。
3. 在Cache中，在segment 粒度上支持了LRU机制， 体现在Segment上就是 writeQueue 和 accessQueue。队列中的元素按照访问或者写时间排序，新的元素会被添加到队列尾部。如果，在队列中已经存在了该元素，则会先delete掉，然后再尾部add该节点

## Cache

![](/public/upload/java/guava_cache.png)

AbstractCache：This class provides a **skeletal implementation** of the  Cache interface to minimize the effort required to implement this interface.To implement a cache, the programmer needs only to extend this class and provide an implementation for the get(Object) and getIfPresent methods.  getAll are implemented in terms of get; getAllPresent is implemented in terms of getIfPresent; putAll is implemented in terms of put, invalidateAll(Iterable) is implemented in terms of invalidate. 这是一个典型抽象类的使用。

![](/public/upload/java/guava_cache_segment_get.png)

![](/public/upload/java/guava_cache_localcache_segment.jpg)

可以直观看到cache是以segment粒度来控制并发get和put等操作的

![](/public/upload/java/guava_cache_segment.png)

## segment

![](/public/upload/java/guava_cache_reference_entry.png)


![](/public/upload/java/guava_cache_value_reference.png)

为了减少不必须的load加载，在value引用中增加了loading标识和wait方法等待加载获取值。这样，就可以等待上一个调用loader方法获取值，而不是重复去调用loader方法加重系统负担，而且可以更快的获取对应的值。

在Cache分别实现了基于Strong,Soft，Weak三种形式的ValueReference实现。

这里ValueReference之所以要有对ReferenceEntry的引用是因为在WeakReference、SoftReference被回收时，需要使用其key将对应的项从Segment段中移除； copyFor()函数的存在是因为在expand(rehash)重新创建节点时，对WeakReference、SoftReference需要重新创建实例（C++中的深度复制思想，就是为了保持对象状态不会相互影响），而对强引用来说，直接使用原来的值即可，这里很好的展示了对彼变化的封装思想； notifiyNewValue只用于LoadingValueReference，它的存在是为了对LoadingValueReference来说能更加及时的得到CacheLoader加载的值。

## segment.get

![](/public/upload/java/guava_cache_segment_get_flow.png)
