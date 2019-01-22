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

先将主串拆分为 n-m+1 个子串，然后模式串与子串一一匹配 

![](/public/upload/algorithm/string_match_bf.jpg)

模式串与字串的匹配方式

1. Brute Force 算法：暴力匹配
2. Rabin-Karp 算法，在匹配效率（快速判断两个字符串是否相等 ==> 哈希 ==> 如何快速求哈希）上做文章
3. Boyer-Moore 算法，常用于文本编辑器中的查找替换。当遇到不匹配的字符时，有什么固定的规律，跳过一些肯定不会匹配的情况，将模式串往后多滑动几位呢？BM 算法构建的规则有两类：坏字符规则和好后缀规则。好后缀规则可以独立于坏字符规则（耗内存、某些场景下失效）使用
4. Knuth Morris Pratt/KMP 算法，当遇到不匹配的字符时，将模式串往后多滑动几位呢？好前缀规则。 这里有两个事情：模式串（长度为n）有n-1个好前缀；每个前缀有它自己的后移位数

    ![](/public/upload/algorithm/string_match_kmp.jpg)

Rabin-Karp 算法

1. 先全部算一遍哈希值，哈希值匹配
2. 前一个字串与后一个字串的哈希值计算有关联关系， 省去部分计算

针对a~z组成的字符串 设计一个哈希算法，将其转换为一个数字
1. 字符串对应一个26进制数字，数字可能很大， 计算机表示不了
2. a~z 对应1~26，将所有字母对应的数字求和，容易冲突
3. a~z 对应素数（这就引出了素数的价值），这样求和时冲突的概率就很低了



![](/public/upload/algorithm/beauty_of_algorithm_post.JPG)

个人微信订阅号

![](/public/upload/qrcode_for_gh.jpg)