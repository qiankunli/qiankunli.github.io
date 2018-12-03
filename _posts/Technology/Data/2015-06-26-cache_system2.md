---

layout: post
title: 缓存系统——具体组件
category: 技术
tags: Data
keywords: 缓存 redis

---

## 简介

建议看下前文[缓存系统](http://qiankunli.github.io/2015/06/26/cache_system.html)

[缓存那些事](https://tech.meituan.com/cache_about.html)

![](/public/upload/data/cache_location.png)

## 本地缓存

在java中，经常拿来当缓存用的是HashMap。不过，建议使用WeakHashMap，而不是HashMap，当然，更好的选择是使用框架，例如Guava Cache [Guava 学习笔记](https://legacy.gitbook.com/book/skyao/learning-guava/details)。

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
        

使用时，事先设定缓存的大概容量，可以有效地提高性能。   

2018.12.02 补充：guava cache 的清理逻辑 [When Does Cleanup Happen?](https://github.com/google/guava/wiki/CachesExplained) 

Caches built with CacheBuilder do not perform cleanup and evict values "automatically," or instantly after a value expires, or anything of the sort. Instead, it performs small amounts of maintenance during write operations, or during occasional read operations if writes are rare.

The reason for this is as follows: if we wanted to perform Cache maintenance continuously, we would need to create a thread, and its operations would be competing with user operations for shared locks. Additionally, some environments restrict the creation of threads, which would make CacheBuilder unusable in that environment.

Instead, we put the choice in your hands. If your cache is high-throughput, then you don't have to worry about performing cache maintenance to clean up expired entries and the like. If your cache does writes only rarely and you don't want cleanup to block cache reads, you may wish to create your own maintenance thread that calls Cache.cleanUp() at regular intervals.

If you want to schedule regular cache maintenance for a cache which only rarely has writes, just schedule the maintenance using ScheduledExecutorService.

你对缓存设置一个最大容量（entry/key的个数）之后，  guava cache 只有在write 操作时才会去清理 过期的expire。如果是读多写少的业务，read 操作也会触发清理逻辑occasionally。在一些场景下，guava cache put the choice in your hands，所以不可无脑使用。


## 单机缓存系统

在不考虑任何异常、简化特性的情况下，以下Go代码便可以实现一个简单的缓存系统。

服务端

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

客户端

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
    

执行过程为：

    $ mycache set name lqk
    # “客户端”解析命令行，并向服务端发送“http://localhost:8080/?get|name”
    # “服务端”解析出"get|name"并返回key为name的值
    $ name = lqk
    
本例虽然简单，但与现在大部分软件的设计方式雷同，比如memcache和docker等。

如果读者熟悉java netty，使用netty实现上述例子也非常容易。

    public class MyHandler extends ChannelHandlerAdapter{
        private Map<String,String> kvs;
        public MyHandler(Map<String,String> kvs){
            this.kvs = kvs;
        }
        public void channelRead(ChannelHandlerContext ctx, Object msg)
        		throws Exception {
        	1. 解析用户请求数据
        	2. 操作容器
        	3. 写回结果
        }
    }

## 分布式缓存系统

如果cache system架在多个主机上，问题就复杂了，因为会有主机宕机，也有的新的主机加进来。因此，要**尽可能（有些损失无法避免）**满足以下四个特性：

1. 平衡性，每个主机存储的key（或者说负载）都差不多
2. 单调性，当增加新的主机时，能够将某些key（旧有的或新的）弄到新的主机上
3. 分散性，**有待进一步了解**，大意是尽量避免重复存储相同的key值
4. 负载，**有待进一步了解**

关键：**将一个key存在哪个节点上**。

- 取模算法（`hash(key)%N`）其弊端很明显：当某个主机宕机时，其存储的数据将无法找到（这个是任何缓存系统都无法避免的），问题是，其保有的存储地址空间也将失效（即该主机宕机后，一些key值还是会继续被映射到该主机，然后发现无法存储）。
- 一致性哈希算法

    [每天进步一点点——五分钟理解一致性哈希算法(consistent hashing)][]
    
## varnish 缓存

位于服务与 nginx 之间
    
## 全站缓存

位于站点和用户之间

[看不见摸不着的cdn是啥](http://qiankunli.github.io/2018/03/29/cdn.html)

## 引用

[应用系统数据缓存设计][]

[Guava学习笔记：Guava cache][]

[面向GC的Java编程][]

[Apache Commons Pool的入门例子][]

[应用系统数据缓存设计]: http://www.tuicool.com/articles/nYvy2a
[Guava学习笔记：Guava cache]: http://www.cnblogs.com/peida/p/Guava_Cache.html
[面向GC的Java编程]: http://coolshell.cn/articles/11541.html
[Apache Commons Pool的入门例子]: http://blog.csdn.net/fwing/article/details/5525124
[每天进步一点点——五分钟理解一致性哈希算法(consistent hashing)]: http://blog.csdn.net/cywosp/article/details/23397179