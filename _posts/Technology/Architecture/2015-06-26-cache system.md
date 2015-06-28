---

layout: post
title: 缓存系统
category: 技术
tags: Architecture
keywords: 缓存 redis

---

## 简介 

第一次接触缓存，是在学习hibernate的时候，hibernate可以将查询结果缓存起来，加快下次查询的速度。第一次自己写缓存（其实就是个map），是因为被告知某个查询操作，批量查询的效率更高，这个map用来暂存批量查询的处理结果。但要说将缓存系统作为应用服务器和数据库服务器的中间层（缓存系统作为单独的一层，并上升成为一个独立的组件），还是在学习redis的时候。

## 缓存系统

使用缓存系统，最理想的效果是：应用系统尽量只与缓存系统交互，只有在查询缓存失败时，才访问数据库。进而，将读写压力从数据库转移到缓存系统上。

一个缓存系统应该考虑如下特性：

1. 是否可以线性扩展，即通过增加主机，来增加缓存系统的存储能力，这涉及到分布式缓存系统。

    一旦涉及到分布式缓存系统，那么涉及到
    
       - 如何将缓存的数据均摊到所有缓存节点
       - 如果某个节点失效，如何处理

2. 线程安全，在线程操作时，维护数据的一致性
3. 当数据库数据发生改变时，如何及时感知并更新缓存
4. 如果缓存系统容量一定，当添加新的数据时，没有剩余空间，如何处理？数据是否有有效期？
4. 最重要的一点，不能太复杂，如果访问延迟稍高，缓存系统便失去了存在的意义。


缓存系统的使用：

应用系统首先访问缓存系统，如果未找到需要的数据，则访问数据库，并将返回结果加入到缓存系统中。


## 单机缓存系统

在不考虑任何异常、简化特性的情况下，以下代码便可以实现一个简单的缓存系统。

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

未完待续

## 引用

[应用系统数据缓存设计][]

[应用系统数据缓存设计]: http://www.tuicool.com/articles/nYvy2a