---

layout: post
title: 缓存系统
category: 技术
tags: Storage
keywords: 缓存 redis

---

## 简介 

* TOC
{:toc}

在计算机和网络领域，缓存无处不在。可以这么说，只要硬件性能不对等的地方都会有缓存的身影。

[缓存那些事](https://tech.meituan.com/cache_about.html)

![](/public/upload/data/cache_xmind.png)

## 缓存系统

使用缓存系统，最理想的效果是：**应用系统尽量只与缓存系统交互，只有在查询缓存失败时，才访问数据库。进而，将读写压力从数据库转移到缓存系统上。**

缓存系统有以下几类：

1. 作为一个组件存在（或者说，本地缓存。比如一个jar提供的java类）
2. 单机的、独立的应用
3. 跨主机的、独立的应用

一个缓存系统应该考虑如下特性：

1. 是否可以线性扩展，即通过增加主机，来增加缓存系统的存储能力，这涉及到分布式缓存系统。一旦涉及到分布式缓存系统，那么涉及到
    - 如何将缓存的数据均摊到所有缓存节点
    - 如果某个节点失效，如何处理

2. 线程安全，在线程操作时，维护数据的一致性
3. 当实际数据发生改变时，如何及时感知并更新缓存
4. 如果缓存系统容量一定，当添加新的数据时，没有剩余空间，如何处理？数据是否有有效期？
4. 最重要的一点，不能太复杂，如果访问延迟稍高，缓存系统便失去了存在的意义。

### 缓存系统与数据库的一致性

1. 数据加入缓存

    - 客户端查询缓存，如果缓存中没有，则查询数据库，并将查询结果加入到缓存中。
    - 独立的定时任务 负责数据库与缓存之间的数据同步

2. 数据从缓存清除或更新

    - 客户端在向数据库写入数据的同时，告诉缓存该数据应失效
    - 缓存中数据设置过期时间

### 缓存系统的数据模型

很多事情联系起来想很有意思，比如rpc，跨主机进程通信。然后一些大牛搞出redis，可以理解为**跨主机访问内存**，360推出一个pika，可以理解为跨主机访问磁盘（支持redis协议）。

跨主机通信，当然免不了网络通信协议的一些约定，这不是本文的重点，所以不多谈。不管跨主机访问内存还是磁盘，**都不是提供一个`byte[]`让客户端随便用**，而是像rpc一样，传输一些约定好的数据结构。区别是，rpc传输的数据结构描述了调用信息，redis的客户端与服务端传输的数据结构是为了存储和使用。

把一些数据结构存在本机或存在远程主机，有一些隐含的意味：

1. "本机的"数据结构包括：基本数据类型，复合类型（string，list，map等）。基本数据类型往往用不着跨主机存储，因为不值当。
2. 对于本地访问内存而言，访问一个数据结构要指明两个要素：内存地址和类型。内存地址说明去哪取数据，类型说明取多少数据，取出的数据如何处理。远端访问内存类似，只不过”地址“不再是一个内存地址，而是一个具备唯一性的key，由远端主机完成key到该主机的内存地址的映射。

上述逻辑或许能够解释，很多类似redis的工具为什么是key-value的，并且value可以是各种数据结构。

### 缓存系统带来的一些问题

1. 穿透，主要有两种情况

    - 比如系统刚启动时，缓存中没有数据，突如其来的大量请求直接冲过缓存访问数据库
    - 对于一个热门数据，缓存中没有，在第一个线程还未完成“查询数据库，写入缓存”过程时，便有多个线程冲过缓存访问数据库

    解决办法主要是做请求合并

2. 非法查询，缓存中存的大多数是有效的数据，那么对于一个非法的数据（或者说合法，但数据库中没有），缓存中没有，则查询压力还是由数据库承担。

## 不同位置的缓存
    
![](/public/upload/data/cache_location.png)

### 本地缓存

在java中，经常拿来当缓存用的是HashMap。不过，建议使用WeakHashMap，而不是HashMap，当然，更好的选择是使用框架，例如Guava Cache [Guava 学习笔记](https://legacy.gitbook.com/book/skyao/learning-guava/details)。

```java
public void TestLoadingCache() throws Exception{
    // Cache 类更加灵活
    LoadingCache<String,String> cahceBuilder=CacheBuilder
    .newBuilder()
    .build(new CacheLoader<String, String>(){
        // 如果key值不在缓存中，则调用该方法获取key的实际值
        @Override
        public String load(String key) throws Exception {        
            String strProValue="hello "+key+"!";                
            return strProValue;
        }
    });  
}
```

使用时，事先设定缓存的大概容量，可以有效地提高性能。   

2018.12.02 补充：guava cache 的清理逻辑 [When Does Cleanup Happen?](https://github.com/google/guava/wiki/CachesExplained) 

Caches built with CacheBuilder do not perform cleanup and evict values "automatically," or instantly after a value expires, or anything of the sort. Instead, it performs small amounts of maintenance during write operations, or during occasional read operations if writes are rare.

The reason for this is as follows: if we wanted to perform Cache maintenance continuously, we would need to create a thread, and its operations would be competing with user operations for shared locks. Additionally, some environments restrict the creation of threads, which would make CacheBuilder unusable in that environment.

Instead, we put the choice in your hands. If your cache is high-throughput, then you don't have to worry about performing cache maintenance to clean up expired entries and the like. If your cache does writes only rarely and you don't want cleanup to block cache reads, you may wish to create your own maintenance thread that calls Cache.cleanUp() at regular intervals.

If you want to schedule regular cache maintenance for a cache which only rarely has writes, just schedule the maintenance using ScheduledExecutorService.

你对缓存设置一个最大容量（entry/key的个数）之后，  guava cache 只有在write 操作时才会去清理 过期的expire。如果是读多写少的业务，read 操作也会触发清理逻辑occasionally。在一些场景下，guava cache put the choice in your hands，所以不可无脑使用。


### 单机缓存系统

在不考虑任何异常、简化特性的情况下，以下Go代码便可以实现一个简单的缓存系统。

服务端
```go
package main 
import (
    "fmt"
    "github.com/gorilla/mux"
    "net/http"
    "strings"
) 
var m map[string]string                      //缓存key-value
func main() {
    m = make(map[string]string, 10)
    r := mux.NewRouter()
    r.HandleFunc("/", HomeHandler)           // 将客户端发来的请求交给HomeHandler处理
    fmt.Println("listen...")
    http.ListenAndServe(":8080", r)
}    
func HomeHandler(rw http.ResponseWriter, r *http.Request) {  
    // 解析客户端发来的命令
    argStr := r.RequestURI
    argStartIndex := strings.LastIndex(argStr, "?") + 1
    args := strings.Split(argStr[argStartIndex:], "|")
    for _, arg := range args {
        fmt.Println(arg)
    }
    command := args[0]
    // 处理命令
    switch command {
    case "get":
        fmt.Fprintf(rw, "%s = %s\n", args[1], handleGet(args[1]))
    case "set":
        handleSet(args[1], args[2])
        fmt.Fprintf(rw, "%s = %s\n", args[1], args[2])
    default:
        fmt.Println("command error")
    }
}
func handleGet(key string) string {
    if val, ok := m[key]; ok {
        return val
    }
    return ""
}
func handleSet(key, val string) {
    m[key] = value
}
```

客户端
```go
package main   
import (
    "flag"
    "fmt"
    "io/ioutil"
    "net/http"
    "strings"
)
func main() {
    flag.Parse()
    reqStr := strings.Join(flag.Args(), "|")    
    fmt.Printf("reqStr = %s\n", reqStr)    
    httpGet(reqStr)    
}    
func httpGet(reqStr string) {
    resp, err := http.Get("http://localhost:8080?" + reqStr)
    if err != nil {
        // handle error
    }  
    defer resp.Body.Close()
    body, err := ioutil.ReadAll(resp.Body)
    if err != nil {
        // handle error
    }  
    fmt.Println(string(body))
}
```
    

执行过程为：
```shell
$ mycache set name lqk
# “客户端”解析命令行，并向服务端发送“http://localhost:8080/?get|name”
# “服务端”解析出"get|name"并返回key为name的值
$ name = lqk
```    
### 分布式缓存系统

如果cache system架在多个主机上，问题就复杂了，因为会有主机宕机，也有的新的主机加进来。因此，要**尽可能（有些损失无法避免）**满足以下四个特性：

1. 平衡性，每个主机存储的key（或者说负载）都差不多
2. 单调性，当增加新的主机时，能够将某些key（旧有的或新的）弄到新的主机上
3. 分散性，**有待进一步了解**，大意是尽量避免重复存储相同的key值
4. 负载，**有待进一步了解**

关键：**将一个key存在哪个节点上**。

- 取模算法（`hash(key)%N`）其弊端很明显：当某个主机宕机时，其存储的数据将无法找到（这个是任何缓存系统都无法避免的），问题是，其保有的存储地址空间也将失效（即该主机宕机后，一些key值还是会继续被映射到该主机，然后发现无法存储）。
- 一致性哈希算法
    
### varnish 缓存

位于服务与 nginx 之间
    
### 全站缓存

cdn位于站点和用户之间， 也是一种变相的缓存系统。[看不见摸不着的cdn是啥](http://qiankunli.github.io/2018/03/29/cdn.html)

## 一致性哈希算法/就近路由算法

《分布式协议与算法实战》一致哈希本质上是一种路由寻址算法（实现上一般会有一个“路由表”，路由规则是“就近”），适合简单的路由寻址场景。

![](/public/upload/data/consistent_cache.jpg)

假设 key-01、key-02、key-03 三个 key，根据一致哈希算法，key-01 将寻址到节点 A，key-02 将寻址到节点 B，key-03 将寻址到节点 C。

那一致哈希是如何避免哈希算法“在节点变更的情况下要求数据迁移”的问题呢？

1. 节点宕机。可以看到，key-01 和 key-02 不会受到影响，只有 key-03 的寻址被重定位到 A。受影响的数据是会寻址到节点 B 和节点 C 之间的数据（例如 key-03）

    ![](/public/upload/data/consistent_cache_delete.jpg)
2. 增加节点。key-01、key-02 不受影响，只有 key-03 的寻址被重定位到新节点 D。

    ![](/public/upload/data/consistent_cache_add.jpg)

使用一致哈希的话，对于 1000 万 key 的 3 节点 KV 存储，如果我们增加 1 个节点，变为 4 节点集群，只需要迁移 24.3% 的数据。**当节点数越多的时候，使用哈希算法时，需要迁移的数据就越多，使用一致哈希时，需要迁移的数据就越少**。当我们向 10 个节点集群中增加节点时，如果使用了哈希算法，需要迁移高达 90.91% 的数据，使用一致哈希的话，只需要迁移 6.48% 的数据。

当节点数较少时，可能会出现节点在哈希环上分布不均匀的情况。这样每个节点实际占据环上的区间大小不一，最终导致业务对节点的访问冷热不均。这个问题可以通过引入更多的虚拟节点来解决：就是对每一个服务器节点计算多个哈希值，在每个计算结果位置上，都放置一个虚拟节点，并将虚拟节点映射到实际节点。比如，可以在主机名的后面增加编号，分别计算 “Node-A-01”“Node-A-02”“Node-B-01”“Node-B-02”“Node-C-01”“Node-C-02”的哈希值，于是形成 6 个虚拟节点：

![](/public/upload/data/consistent_cache_virtual_node.jpg)