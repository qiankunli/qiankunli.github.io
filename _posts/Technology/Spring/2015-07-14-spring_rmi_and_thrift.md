---

layout: post
title: spring rmi和thrift
category: 技术
tags: Spring
keywords: spring rmi thrift

---

## 前言 

## 基本思路

代理模式

### client
    interface{
        hello()
    }
    
    impl{
        hello(){
            连接服务器端
            发送调用需求（接口的名字，方法的名字，方法的参数值）
            等待结果
        }
    }
    client{
        直接使用impl即可
    }
    
### server  

    
    server{
        work(){
            注册impl
            解析请求
            新建一个impl实例，执行方法
            将结果返回
        }
    }
    interface{
        hello()
    }
    impl{
        // 实际实现
        hello(){
        
        }
    }
    
### 基本要求

- 对方法的参数类型和返回值类型，要能够序列化
- 封装socket通信细节（当然，有的远程调用框架用的不是socket通信）

## thrift

其实套路是一样的，只是呈现出来的抽象不同，尤其是thrift支持

1. 跨语言运用
2. 异步通信
    



## 引用

