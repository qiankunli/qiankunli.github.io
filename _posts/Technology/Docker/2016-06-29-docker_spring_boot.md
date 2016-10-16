---

layout: post
title: 基于spring boot和Docker搭建微服务
category: 技术
tags: Docker
keywords: Docker,plugin

---

## 前言（未完待续）

docker发展的真是一日千里，maven出了一款docker插件，以前`mvn clean package`产出的是一个war包，现在产出的是一个image。

spring boot的作用简单解释就是，如果你想创建一个返回“hello world”的web项目，只需

maven配置一下

   
    <dependencies>
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-web</artifactId>
        </dependency>
    </dependencies>


写一个类

    @Controller
    @EnableAutoConfiguration
    public class SampleController {
    
        @RequestMapping("/")
        @ResponseBody
        String home() {
            return "Hello World!";
        }
    
        public static void main(String[] args) throws Exception {
            SpringApplication.run(SampleController.class, args);
        }
    }
    
运行main方法，就可以启动一个内置的tomcat，部署应用。直接`http://localhost:8080`就可以访问了。

这样，web.xml,application-context.xml等全部按约定来（当然，可以更改约定）。Spring Boot最大的特色是“约定优先配置”，大大简化spring项目开发的工作。

将这两者结合起来，就可以快速部署一个微服务。

