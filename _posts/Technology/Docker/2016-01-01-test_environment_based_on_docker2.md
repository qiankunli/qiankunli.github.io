---

layout: post
title: 基于docker搭建测试环境(二)
category: 技术
tags: Docker
keywords: Docker jenkins python

---
## 简介

前文`基于docker搭建测试环境`展示了docker ci环境的搭建过程，本文将进一步丰富细节，主要讲述“将web项目编译，创建镜像以及将镜像部署到某个主机上，创建容器并运行”的脚本实现。

## 流程

![Alt text](/public/upload/docker/docker_ci.png)

## 脚本语言的选择

1. linux shell,操作swarm和registry，使用remote api
2. linux shell,操作swarm使用`docker -H xxx command`，操作registry使用remote api
3. python脚本,操作swarm使用docker-py，操作registry使用remote api

作者一开始采用第一种方案实现了一个shell脚本，但在使用过程中发现如下问题

1. 处理请求返回的响应值比较困难，即便在linux中安装jq等工具，代码还是比较复杂。

    - 有些请求不返回任何结果，表示正常运行
    - 有些请求返回一个id，表示正常运行
    - 有些请求返回一个json字符串，需要解析并获取其中的key值
2. 异常处理代码非常丑陋，有时甚至无法及时发现异常（即实际运行失败，在jenkins仍显示build成功）。

使用python后，python的语言处理能力自不用多说，还可以使用`try except`语句块优雅的处理异常。

## 脚本思路

### 设置环境变量

脚本中用到的一些环境变量需要预先设置（jenkins需要环境变量相关插件）

    DOCKER_ADDRESS=192.168.56.154:2375
    REGISTRY_ADDRESS=192.168.56.154:5000
    # 需要将容器的8080端口映射为主机的哪个端口
    PORT=3000
    
### 一些约定

1. web项目编译成war包后，基于`REGISTRY_ADDRESS/tomcat7`镜像生成`REGISTRY_ADDRESS/JOB_NAME`镜像。
2. 容器在某个主机运行时，容器的name是JOB_NAME（JOB_NAME表示jenkins job的名字，可从jenkins自带的环境变量中获取）
3. 容器对外暴漏8080和8000端口，8080在主机上的映射端口由PORT变量指定，8000端口作为调试端口，其映射端口由容器自动指定
4. web项目在容器的`/logs`目录下记录日志，`/logs`目录映射到主机的`/logs`目录

## 脚本实现

    from docker import AutoVersionClient
    import os
    import sys
    # python 3.x
    # import http.client
    # python 2.x
    import httplib
       
    PORT = os.getenv('PORT')
    JOB_NAME = os.getenv('JOB_NAME')
    DOCKER_ADDRESS = os.getenv('DOCKER_ADDRESS')
    REGISTRY_ADDRESS = os.getenv('REGISTRY_ADDRESS')
    WORKSPACE = os.getenv('WORKSPACE')
    
    DOCKERFILE = WORKSPACE + '/Dockerfile'
    IMAGE_NAME = JOB_NAME
    FULL_IMAGE_NAME=REGISTRY_ADDRESS + '/' + IMAGE_NAME
    
    # http://docker-py.readthedocs.org/en/latest/api/#containers
    jenkinsCli = AutoVersionClient(base_url='localhost:2375',timeout=10)
    swarmCli = AutoVersionClient(base_url=DOCKER_ADDRESS,timeout=10)
    # registryConn = http.client.HTTPConnection(REGISTRY_ADDRESS)
    registryConn = httplib.HTTPConnection(REGISTRY_ADDRESS)
    
    def remove_image_in_registry():
        try:
            registryConn.request('GET','v1/repositories/' + IMAGE_NAME + '/tags/latest')
            response = registryConn.getresponse()
            print("query image %s in registry,http status %d" %(FULL_IMAGE_NAME,response.status))
            # if image is in registry
            if response.status / 100 == 2 :
                print("registry image %s id : %s" %(FULL_IMAGE_NAME,response.read()))
                registryConn.request('DELETE','v1/repositories/'+ IMAGE_NAME + '/tags/latest')
                r = registryConn.getresponse()
                print("delete image %s in registry,http status %d" %(FULL_IMAGE_NAME,r.status))
            else :
                print("query image %s in registry fail,resason %s" %(FULL_IMAGE_NAME,response.reason))
        except Exception as inst:
            print(type(inst))    # the exception instance
            print(inst.args)     # arguments stored in .args
            print(inst)          # __str__ allows args to be printed directly
            pass
        finally:
          	registryConn.close()
    
    def remove_container_with_imageName():
        response = swarmCli.containers(all=True)
        container_num = len(response)
        if container_num > 0 :
            i=0
            while i < container_num :  
                 if response[i]['Image'] == FULL_IMAGE_NAME :
                     print('remove container :' + response[i]['Names'][0])
                     print('remove container id :' + response[i]['Id'])
                     swarmCli.remove_container(response[i]['Id'],force=True)
                 i=i+1
    
    def clear():
        # delete image in jenkins
        try:
            print("remove image : %s in jenkins" %(FULL_IMAGE_NAME))
            jenkinsCli.remove_image(FULL_IMAGE_NAME,force=True,noprune=False)
        except Exception as inst:
            print(type(inst))    # the exception instance
            print(inst.args)     # arguments stored in .args
            print(inst)          # __str__ allows args to be printed directly
            pass
        # remove image in registry
        remove_image_in_registry()
        # remove container in swarm
        remove_container_with_imageName()
        # remove image in swarm
        try:
            print("remove image : %s in swarm" %(FULL_IMAGE_NAME))
            swarmCli.remove_image(FULL_IMAGE_NAME,force=True,noprune=False)
        except Exception as inst:
            print(type(inst))    # the exception instance
            print(inst.args)     # arguments stored in .args
            print(inst)          # __str__ allows args to be printed directly
            pass
          
    def build():
        # remove dockerfile
        if(os.path.isfile(DOCKERFILE)):
            os.remove(DOCKERFILE)
        file = open(DOCKERFILE,mode='a+')
        file.write("FROM %s \n"%(REGISTRY_ADDRESS + '/tomcat7'))
        file.write('ADD target/*.war $TOMCAT_HOME/webapps/ \n')
        file.write('CMD  bash start.sh')
        file.close()
        print('build dockerfile')
        response = [line for line in jenkinsCli.build(path=WORKSPACE, rm=True, tag=FULL_IMAGE_NAME)]
        response
    
    def push():
        print("push image : %s from jenkins to registry" %(FULL_IMAGE_NAME))
        response = [line for line in jenkinsCli.push(FULL_IMAGE_NAME, stream=True)]
        response
    
    def run():
        # pull image
        for line in swarmCli.pull(FULL_IMAGE_NAME, stream=True):
            print(line)
        # create container
        config = swarmCli.create_host_config(binds=['/logs:/logs'],port_bindings={8080:PORT,8000:None},publish_all_ports=True)
        container = swarmCli.create_container(image=FULL_IMAGE_NAME,name=IMAGE_NAME,ports=[8080,8000],volumes=['/logs'],host_config=config)
        print(container)
        # start container
        response = swarmCli.start(container=container.get('Id'))
        print(response)
    def main():
      try:
          clear()
          # build dockerfile
          build()
          # push image from jenkins to registry
          push()
          # pull image,create container,start container
          run()
          sys.exit(0)
      except Exception as inst:
          print(type(inst))    # the exception instance
          print(inst.args)     # arguments stored in .args
          print(inst)          # __str__ allows args to be printed directly
          sys.exit(1)
      finally:
          jenkinsCli.close()
          swarmCli.close()
    main()
    

## 回滚

基于docker ci的回滚。其实就是docker镜像保存最后一次正确的历史版本（需要将hudson build序号跟镜像的tag联系起来），一旦最新代码运行有问题，就找到最后一次正确的docker镜像，部署该镜像即可。

## 小结

脚本的shell版本和python版本可在`git@github.com:qiankunli/dockerci.git`中下载。

代码中，为了可读性加了许多注释，代码本身还是比较简单的。

作者第一次使用python写程序，有问题的地方，欢迎大家指正。