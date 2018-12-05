---

layout: post
title: docker dns
category: 技术
tags: Network
keywords: Docker dns

---

## 简介

公司的测试环境运行了很多docker服务，很多服务内在的使用域名访问其它主机上的服务（很明显，你不能限定同事们只能使用ip）。这时，便需要一个dns，使得容器中的服务可以方便的根据域名得到ip。

如果这个dns服务运行在一个容器中就更好了，毕竟容器访问容器要比容器访问主机更自然。

bind9，不要提它了，不是专业的运维人员很难运用自如，实际上，我们也不需要那么强大的功能。

我们使用一个轻量级的dnsmasq来提供dns服务（dnsmasq还可以提供dhcp服务，但这不是本节的重点了）。

## dnsmasq使用

假设dnsmasq运行在`172.17.0.2`上，那么你只需要更改`172.17.0.2`上的`/etc/hosts`文件，并重启dnsmasq服务即可。

复杂点的方式参见`https://segmentfault.com/a/1190000000629231`

## dockerfile

为了保证dnsmasq服务的持续运行，我们使用supervisor来管理dnsmasq服务

    FROM ubuntu
    MAINTAINER bert.li
    RUN apt-get update
    RUN apt-get install -y dnsmasq supervisor
    RUN apt-get install -y vim
    RUN mkdir -p /var/log/supervisor
    COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf
    EXPOSE 53
    CMD ["/usr/bin/supervisord"]
    
    
supervisord.conf 

	[supervisord]
	nodaemon=true
	[program:dnsmasq]
	command=/usr/sbin/dnsmasq --no-daemon

这样，一个提供dns服务的容器便制作成功，运行其它容器时，只需

	docker run -d --dns 172.17.0.2 -name container_name image_name


## 注意的问题

1. 当你需要更新`172.17.0.2`上的`/etc/hosts`文件时

	supervisod会保证dnsmasq的持续启动，所以，当你更新`/etc/hosts`后，只需将原来的进程杀死即可。直接`/etc/init.d/dnsmasq restart`会失败，因为老的实例在关闭后，很快就会被supervisord重启。新的进程实例，虽然加载了新的数据，单因为supervisord重启的实例已经占用了53端口，导致无法正常启动。
    
2. 为dns容器分配固定的ip

	这个涉及到另一个范畴的问题，本节不做讨论