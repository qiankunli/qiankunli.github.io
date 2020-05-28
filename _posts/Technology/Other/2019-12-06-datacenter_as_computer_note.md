---

layout: post
title: 《datacenter as a computer》笔记
category: 技术
tags: Other
keywords: datacenter

---

## 前言（未完成）

The objectives of this book are to introduce readers to this new design space(新的设计领域), describe some of the requirements and characteristics of WSCs(ware house-scale computers), highlight some of the important challenges unique to this space, and share some of our experience designing, programming, and operating them within Google

### ot just a collection of servers

互联网时代，有比较重的pc端，然后是互联网时代，计算主要放在server-side，然后服务端应用的规模开始越来越大。

一开始的时候， 机房只是提供一个物理机托管。公司规模变大之后，一家公开开始拥有一整个或多个机房，那就有机会在一个机房内采取统一的硬件、平台软件等。成本核算也从物理机维度 过渡到 整个机房的运营成本。not just a collection of servers. 

