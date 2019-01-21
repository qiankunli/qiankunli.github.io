---

layout: post
title: 《数据结构与算法之美》——算法新解
category: 技术
tags: Algorithm
keywords: 数据结构与算法之美

---

## 简介（未完成）

建议看下前文 [《数据结构与算法之美》——数据结构笔记](http://qiankunli.github.io/2018/09/19/beauty_of_algorithm_1.html)


## 字符串匹配

先将主串拆分为 n-m+1 个子串，然后模式串与字串一一匹配 

![](/public/upload/algorithm/string_match_bf.jpg)

模式串与字串的匹配方式

1. 暴力匹配
2. 先全部算一遍哈希值，哈希值匹配
3. 前一个字串与后一个字串的哈希值计算有关联关系， 省去部分计算

针对a~z组成的字符串 设计一个哈希算法，将其转换为一个数字

1. 字符串对应一个26进制数字，数字可能很大， 计算机表示不了
2. a~z 对应1~26，将所有字母对应的数字求和，容易冲突
3. a~z 对应素数（这就引出了素数的价值），这样求和时冲突的概率就很低了



![](/public/upload/algorithm/beauty_of_algorithm_post.JPG)

个人微信订阅号

![](/public/upload/qrcode_for_gh.jpg)