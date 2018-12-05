---

layout: post
title: calico 问题排查
category: 技术
tags: Network
keywords: Docker, calico

---


## 简介

背景：calico 网络网段172.32.0.0/16。容器内网卡名 cali0，容器containerA 运行在hostA 上，containerA calic0 对应的 hostA 网卡名为 cali2b77f29827c。

## ping 不通

### node status 有问题

containerA跨主机ping hostB上的containerB，不通，检查步骤

1. tcpdump(`tcpdump -nn -i 网卡名 -e`) 判断 hostA 网卡是否收到包
2. 检查 hostA 路由规则
3. tcpdump 判断 hostB 网卡是否收到包
4. 检查 hostA 路由规则

已知原因:

`calicoctl node status` 某个节点 状态 不是up

	IPv4 BGP status
	+---------------+-------------------+-------+----------+--------------------------------+
	| PEER ADDRESS  |     PEER TYPE     | STATE |  SINCE   |              INFO              |
	+---------------+-------------------+-------+----------+--------------------------------+
	| 192.168.60.42 | node-to-node mesh | start | 06:03:39 | Active Socket: Connection      |
	|               |                   |       |          | refused                        |
	| 192.168.60.83 | node-to-node mesh | wait  | 06:03:40 | Established                    |
	+---------------+-------------------+-------+----------+--------------------------------+
	
### ping www.baidu.com 不通

 192.168.60.6 和 192.168.3.2 是 hostA 配置的两个nameserver

	root@hostA:~# tcpdump -nn -i cali2b77f29827c -e
	tcpdump: verbose output suppressed, use -v or -vv for full protocol decode
	listening on cali2b77f29827c, link-type EN10MB (Ethernet), capture size 262144 bytes
	20:27:20.475417 ee:ee:ee:ee:ee:ee > d6:19:05:0b:f4:72, 172.32.61.66.60637 > 192.168.60.6.53: 37653+ A? www.baidu.com. (31)
	20:27:20.475481 ee:ee:ee:ee:ee:ee > d6:19:05:0b:f4:72, 172.32.61.66.60637 > 192.168.3.2.53: 37653+ A? www.baidu.com. (31)
	20:27:20.475516 ee:ee:ee:ee:ee:ee > d6:19:05:0b:f4:72, 172.32.61.66.60637 > 192.168.60.6.53: 38287+ AAAA? www.baidu.com. (31)
	
从中可以看到，只有发往name server 53 端口的数据包，想必因为name server 没有配置bgp，进而没有 route 数据，导致返回的数据包回不来。

一个佐证是，直接ping baidu的 域名ip `115.239.210.27` 是可以ping 通的。
