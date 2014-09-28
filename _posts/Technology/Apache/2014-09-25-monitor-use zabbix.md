---
layout: post
title: zabbix 使用
category: 技术
tags: WEB
keywords: 集群 监控 zabbix
---

## some basic defination ##

It is necessary for us to have a basic understand on some definition of zabbix.

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
	
## some general configure ##

### create template ###

if you have configed hostA so that you could collect some data you need from it,and there is also a another host need to configed as same as hostA. you may need create a template and link it to the hosts you want to monitor,then ,the zabbix server could receive the same item from all the hosts.

if you change the setting of template,the data collected by zabbix-agent will change immediately.

### add custom item ###

for example,we add a item which could collect the number of "sshd" running on a machine.

1. add **UserParameter** in `/etc/zabbix/zabbix_agentd.conf`

		# Format:UserParameter=<key>,<shell command>
		UserParameter=sshdnum,"ps -A | grep sshd | wc -l"

2. config item in WebUI

	![Alt text](/public/upload/zabbix_add_item.png)
	
now, the zabbix agent will run the command in order to get the data of item `sshnum`.

### add trigger ###

if the number of "sshd" greate than the threshold you have defined, you want to receive a reminder,you may need add a trigger.

1. add trigger "sshd"

    ![Alt text](/public/upload/zabbix_add_trigger.png)
2. check the trigger

    ![Alt text](/public/upload/zabbix_check_trigger.png)
    
    if the matrix is green, it represents the number of "sshd" is equal or less than threshold.