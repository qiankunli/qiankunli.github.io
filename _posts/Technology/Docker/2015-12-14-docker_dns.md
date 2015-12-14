---

layout: post
title: docker dns
category: 技术
tags: Docker
keywords: Docker dns

---

## 简介（未完待续）

公司的测试环境运行了很多docker服务，很多服务内在的使用域名访问其它主机上的服务（很明显，你不能限定同事们只能使用ip）。这时，便需要一个dns，使得容器中的服务可以方便的根据域名得到ip。

bind9，不要提它了，不是专业的运维人员很难运用自如，实际上，我们也不需要那么强大的功能。

我们使用一个轻量级的dnsmasq。

## dnsmasq使用



## dockerfile


https://segmentfault.com/a/1190000000629231

## 注意的问题


supervisod 会保证 dnsxx的持续启动

所以，当你更新/etc/hosts后，只需将原来的进程杀死即可。

直接/etc/init.d/dnsxx restart会失败后，因为老的实例在关闭后，很快就会被suxx重启。新的进程实例，虽然加载了新的数据，单因为suxx重启的实例已经占用了53端口，导致无法正常启动