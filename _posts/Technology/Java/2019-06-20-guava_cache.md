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

## Cache的核心是LocalCache

![](/public/upload/java/guava_cache.png)

1. 我们一般用的LoadingCache 实际实现类是 LocalLoadingCache，整个guava cache的核心是LocalCache。
2. AbstractCache：This class provides a **skeletal implementation** of the  Cache interface to minimize the effort required to implement this interface.To implement a cache, the programmer needs only to extend this class and provide an implementation for the get(Object) and getIfPresent methods.  getAll are implemented in terms of get; getAllPresent is implemented in terms of getIfPresent; putAll is implemented in terms of put, invalidateAll(Iterable) is implemented in terms of invalidate. 这是一个典型抽象类的使用：skeletal implementation，封装通用逻辑，承接大部分interface 方法， 剩下几个个性化的方法留给子类实现。 

## LocalCache 的核心是Segment

![](/public/upload/java/guava_cache_segment_get.png)

![](/public/upload/java/guava_cache_localcache_segment.jpg)

可以直观看到cache是以segment粒度来控制并发get和put等操作的

## Segment 的基本元素ReferenceEntry 和 ValueReference

![](/public/upload/java/guava_cache_segment.png)

Map类结构简单说就是数组 + 链表，最基本的数据单元是entry

![](/public/upload/java/guava_cache_reference_entry.png)


![](/public/upload/java/guava_cache_value_reference.png)

为了减少不必须的load加载，在value引用中增加了loading标识和wait方法等待加载获取值。这样，就可以等待上一个调用loader方法获取值，而不是重复去调用loader方法加重系统负担，而且可以更快的获取对应的值。

在Cache分别实现了基于Strong,Soft，Weak三种形式的ValueReference实现。

1. 这里ValueReference之所以要有对ReferenceEntry的引用是因为在WeakReference、SoftReference被回收时，需要使用其key将对应的项从Segment段中移除； 
2. copyFor()函数的存在是因为在expand(rehash)重新创建节点时，对WeakReference、SoftReference需要重新创建实例（C++中的深度复制思想，就是为了保持对象状态不会相互影响），而对强引用来说，直接使用原来的值即可，这里很好的展示了对彼变化的封装思想； 
3. notifiyNewValue只用于LoadingValueReference，它的存在是为了对LoadingValueReference来说能更加及时的得到CacheLoader加载的值。

## segment.get

![](/public/upload/java/guava_cache_segment_get_flow.png)

主要逻辑就两个：lockedGetOrLoad 和 waitForLoadingValue

### lockedGetOrLoad

下列代码只保留了load部分

    V lockedGetOrLoad(K key, int hash, CacheLoader<? super K, V> loader) throws ExecutionException {
        ReferenceEntry<K, V> e;
        ValueReference<K, V> valueReference = null;
        LoadingValueReference<K, V> loadingValueReference = null;
        boolean createNewEntry = true;

        lock();
        int newCount = this.count - 1;
        AtomicReferenceArray<ReferenceEntry<K, V>> table = this.table;
        // 计算key在数组中的落点
        int index = hash & (table.length() - 1);
        ReferenceEntry<K, V> first = table.get(index);
        // 沿着某个index 链表依次遍历
        for (e = first; e != null; e = e.getNext()) {
            K entryKey = e.getKey();
            if (e.getHash() == hash
                && entryKey != null
                && map.keyEquivalence.equivalent(key, entryKey)) {
                valueReference = e.getValueReference();
                V value = valueReference.get();
                if (value == null || map.isExpired(e, now){
                    enqueueNotification(...);
                } else {
                    return value;
                }
                this.count = newCount; // write-volatile
                break;
            }
        }
        loadingValueReference = new LoadingValueReference<>();
        if (e == null) {
            e = newEntry(key, hash, first);
            e.setValueReference(loadingValueReference);table.set(index, e);
        } else {
            e.setValueReference(loadingValueReference);
        }
        unlock();   
        synchronized (e) {
            return loadSync(key, hash,loadingValueReference, loader);
        }
    }

segment 简单说也是数组加链表，只是元素类型是ReferenceEntry，根据key 计算index，然后沿着链表匹配value，若相同，判断value元素是否有效，无效（null or 过期）则创建loadingValueReference 并更新到 ReferenceEntry。loadingValueReference.loadFuture 开始执行load逻辑。

只有ReferenceEntry 更新 其value引用 loadingValueReference 的部分是需要加锁的，之后线程竞争便转移到了 loadingValueReference 上

    V loadSync(K key,int hash,
        LoadingValueReference<K, V>,loadingValueReference,CacheLoader<? super K, V> loader)throws ExecutionException {
        ListenableFuture<V> loadingFuture = loadingValueReference.loadFuture(key, loader);
        return getAndRecordStats(key, hash,loadingValueReference, loadingFuture);
    }

### waitForLoadingValue

    V waitForLoadingValue(ReferenceEntry<K, V> e, K key, ValueReference<K, V> valueReference)
        throws ExecutionException {
        checkState(!Thread.holdsLock(e), "Recursive load of: %s", key);
        V value = valueReference.waitForValue();
        if (value == null) {
            throw new InvalidCacheLoadException("CacheLoader returned null for key " + key + ".");
        }
        ...
        return value;
    }

    static class LoadingValueReference<K, V> implements ValueReference<K, V> {
        volatile ValueReference<K, V> oldValue;
        final SettableFuture<V> futureValue =SettableFuture.create();
        final Stopwatch stopwatch =Stopwatch.createUnstarted();

        public boolean set(@Nullable V newValue) {
            return futureValue.set(newValue);
        }
        public V get() {
            return oldValue.get();
        }
        public V waitForValue() throws ExecutionException {
            // 对future.get的封装
            return getUninterruptibly(futureValue);
        }
        public boolean setException(Throwable t) {
            return futureValue.setException(t);
        }
        public void notifyNewValue(@Nullable V newValue) {
            if (newValue != null) {
                // future.get ==> waitForValue即可立即返回
                set(newValue);
            } else {
                oldValue = unset();
            }
        }
    }

所谓请求合并：当多个线程请求同一个key时，第一个线程执行loader逻辑，其余线程等待。

从上述代码可以看到

1. “其它线程等待”的效果，不是对key 加锁， 其它线程得不到锁而等待
2. LoadingValueReference 持有了 future对象，线程发现value 处于loading状态时 便直接 `LoadingValueReference.waitForValue` ==> `future.get` 准备等结果了

## 如果不想线程排队

[Guava Cache内存缓存使用实践-定时异步刷新及简单抽象封装](http://www.voidcn.com/article/p-xifknifw-brg.html)

### 只有一个用户线程排队

refreshAfterWrite 注意不是 expireAfterWrite

如果缓存过期，恰好有多个线程读取同一个key的值，那么guava只允许一个线程去加载数据，其余线程阻塞。这虽然可以防止大量请求穿透缓存，但是效率低下。使用refreshAfterWrite可以做到：只阻塞加载数据的线程，其余线程返回旧数据。

    LoadingCache<String, Object> caches = CacheBuilder.newBuilder() 
        .maximumSize(100) 
        .refreshAfterWrite(10, TimeUnit.MINUTES) 
        .build(new CacheLoader<String, Object>() { 
            @Override 
            public Object load(String key) throws Exception { 
                return generateValueByKey(key); 
            } 
        }); 

### 另起线程拉新值

真正加载数据的那个线程一定会阻塞，可以让这个加载过程是异步的，这样就可以让所有线程立马返回旧值

    ListeningExecutorService backgroundRefreshPools = MoreExecutors.listeningDecorator(Executors.newFixedThreadPool(20)); LoadingCache<String, Object> caches = CacheBuilder.newBuilder() 
        .maximumSize(100) 
        .refreshAfterWrite(10, TimeUnit.MINUTES) 
        .build(new CacheLoader<String, Object>() { 
            @Override 
            public Object load(String key) throws Exception { 
                return generateValueByKey(key); 
            } 
            @Override 
            public ListenableFuture<Object> reload(String key, Object oldValue) throws Exception { 
                return backgroundRefreshPools.submit(new Callable<Object>() { 
                    @Override 
                    public Object call() throws Exception { 
                        return generateValueByKey(key); 
                    } 
                }); 
            } 
        }); 