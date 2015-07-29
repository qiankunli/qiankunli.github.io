---

layout: post
title: 缓存系统
category: 技术
tags: Architecture
keywords: 缓存 redis

---

## 简介 

第一次接触缓存，是在学习hibernate的时候，hibernate可以将查询结果缓存起来，加快下次查询的速度。第一次自己写缓存（其实就是个map），是因为被告知某个查询操作，批量查询的效率更高，这个map用来暂存批量查询的处理结果。但要说将缓存系统作为应用服务器和数据库服务器的中间层（缓存系统作为单独的一层，并上升成为一个独立的组件），还是在学习redis的时候。

在计算机和网络领域，缓存无处不在。可以这么说，只要硬件性能不对等的地方都会有缓存的身影。

## 缓存系统

使用缓存系统，最理想的效果是：应用系统尽量只与缓存系统交互，只有在查询缓存失败时，才访问数据库。进而，将读写压力从数据库转移到缓存系统上。

缓存系统有以下几类：

1. 作为一个组件存在（或者说，本地缓存。比如一个jar提供的java类）
2. 单机的、独立的应用
3. 跨主机的、独立的应用

一个缓存系统应该考虑如下特性：

1. 是否可以线性扩展，即通过增加主机，来增加缓存系统的存储能力，这涉及到分布式缓存系统。

    一旦涉及到分布式缓存系统，那么涉及到
    
       - 如何将缓存的数据均摊到所有缓存节点
       - 如果某个节点失效，如何处理

2. 线程安全，在线程操作时，维护数据的一致性
3. 当实际数据发生改变时，如何及时感知并更新缓存
4. 如果缓存系统容量一定，当添加新的数据时，没有剩余空间，如何处理？数据是否有有效期？
4. 最重要的一点，不能太复杂，如果访问延迟稍高，缓存系统便失去了存在的意义。

## 缓存组件 一 

在java中，经常拿来当缓存用的是HashMap。不过，建议使用WeakHashMap，而不是HashMap，当然，更好的选择是使用框架，例如Guava Cache。

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

## 缓存系统 二 对象池

我曾经有一种潜意识，缓存只用于两个速度不同的组件之间，其实缓存也可以作为一个临时的存储区域，存储一些创建开销比较大的对象（比如，线程、数据库连接、tcp连接等）。成熟的开源框架有很多，例如Apache Commons Pool。

    package pool;  
    import org.apache.commons.pool.BasePoolableObjectFactory;  
    import org.apache.commons.pool.impl.GenericObjectPool;  
    public class Test {  
        public static void main(String[] args) {  
            // GenericObjectPool把我们需要的东西基本都实现了，可能我们要做的只是了解其中的参数含义，然后具体设置一下就行了  
            final GenericObjectPool pool = new GenericObjectPool(  
                    new TestPoolableObjectFactory());  
            pool.setMaxActive(20);  
            // pool.setMaxWait(1000);  
            for (int i = 0; i < 40; i++) {  
                new Thread(new Runnable() {  
                    @Override  
                    public void run() {  
                        try {  
                            // 注意，如果对象池没有空余的对象，那么这里会block，可以设置block的超时时间  
                            Object obj = pool.borrowObject();
                            System.out.println(obj);  
                            Thread.sleep(5000);  
                            // 申请的资源用完了记得归还 
                            pool.returnObject(obj);
                        } catch (Exception e) {  
                            e.printStackTrace();  
                        }  
                    }  
                }).start();  
            }  
        }  
        static class TestPoolableObjectFactory extends BasePoolableObjectFactory {  
            public Object makeObject() throws Exception {  
                return new Resource();  
            }  
        } 
        // 代表复用的对象 
        static class Resource {  
            private int rid;  
            public Resource() {  
            }  
            public int getRid() {  
                return rid;  
            }  
            @Override  
            public String toString() {  
                return "id:" + rid;  
            }  
        }  
    }  

使用池的时候要慎重，因为从线程安全的角度考虑，通常池都是会被并发访问的，那么你就需要处理好同步的问题，这又是一个大坑，并且同步带来的开销，未必比你重新创建一个对象小。

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