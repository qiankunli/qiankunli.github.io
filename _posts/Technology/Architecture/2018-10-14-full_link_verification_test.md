---

layout: post
title: 全链路压测1——案例研究
category: 技术
tags: Architecture
keywords: 全链路压测

---

## 简介（未完成）



项目隔离跟压测 有一个共性，就是打标签，并将标签在整个调用链路中传输

[干货 | 一文带你了解携程第四代全链路测试系统](https://mp.weixin.qq.com/s?__biz=MjM5MDI3MjA5MQ==&mid=2697267581&idx=1&sn=a81ae51a7633c5970e6e5510afbb43f2&chksm=8376f649b4017f5fa1e71f4b89cb0f9839b55e212a1d39304671e674b9a5c8e2e6cca93d5f4d&mpshare=1&scene=23&srcid=1012gRT6WwfdGz8g4k1ctmVi%23rd)


第二代，提高应用所在集群中某台机器的权重，使其承担远高于其他机器的真实流量负载，进而分析在该负载下，测试机应用的性能表现。
第三代，生产环境流量回放，生产环境流量回放的基本原理是对应用所在集群中，流经一台或多台机器的流量进行拷贝，并以实时的方式将流量的副本分发到测试机上
第四代，便是文中提到的系统