---
layout: post
title: zabbix 使用
category: 技术
tags: WEB
keywords: 集群 监控 zabbix
---

## some basic defination ##

- Item
 
	a particular piece of data that you want to receive off of a host, a **metric** of data.

- Trigger

	**a logical expression** that defines a problem threshold and is used to “evaluate” data received in items

	When received data are above the threshold, triggers go from 'Ok' into a 'Problem' state. When received data are below the threshold, triggers stay in/return to an 'Ok' state.

- Template

	a **set** of entities (items, triggers, graphs, screens, applications, low-level discovery rules) ready to be applied to one or several hosts

	The job of templates is to speed up the deployment of monitoring tasks on a host; also to make it easier to apply mass changes to monitoring tasks. Templates are linked directly to individual hosts.

- Host
	
	a networked device that you want to monitor, with IP/DNS.
- host group

	a logical grouping of hosts; it may contain hosts and templates. Hosts and templates within a host group are not in any way linked to each other. Host groups are used when assigning access rights to hosts for different user groups.
## add custom item ##

for example,we add a item which could collect the number of "sshd" running on a machine.

1. add **UserParameter** in `/etc/zabbix/zabbix_agentd.conf`

		Format:UserParameter=<key>,<shell command>
		UserParameter=sshdnum,"ps -A | grep sshd | wc -l"

2. config item in WebUI

	![Alt text](/public/upload/zabbix_add_item.png)

## send email ##

 
