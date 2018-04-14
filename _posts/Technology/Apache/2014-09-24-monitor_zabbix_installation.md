---
layout: post
title: zabbix 安装
category: 技术
tags: WEB
keywords: 集群 监控 zabbix
---

## 1. Introduction ##

Why zabbix? firstly, it's free,it is important for us Chinese company;secondly, there is a community to support it ,so you hava a question, you can find a person to ask for help;lastly,it may be enough to do what we want to do.so, if you want to monitor a cluster, it is necessary for you to move on.

## 2. Installation ##

the "zabbix" consist of zabbix server and zabbix agent. zabbix proxy may be needed in big cluster.

### 2.1 config yum repo ###

we need to config epel repo to acquire zabbix rpm packages, at the same time, we also need to config CentOS repo to install softwares which the zabbix depend on.  

- CentOS repo: `wget http://mirrors.163.com/.help/CentOS6-Base-163.repo`
	
	you may need replace the variable `$releasever` in CentOS6-Base-163.repo using command `sed -i "s/$releasever/6/g"  CentOS6-Base-163.repo`

- epel repo: `wget http://dl.fedoraproject.org/pub/epel/6/x86_64/epel-release-6-8.noarch.rpm`

you may need config proxy in `/etc/yum.conf` if necessary.

### 2.2 install zabbix-server and zabbix-agent in “zabbix server” ###

#### 2.2.1 preparation ####

please make sure you have config firewall(zabbix use port 10050 and 10051，httpd use port 80) and selinux properly.

	# cat /etc/sysconfig/iptables
	-A INPUT -m conntrack --ctstate NEW -m tcp -p tcp --dport 10050 -j ACCEPT
	-A OUTPUT -m conntrack --ctstate NEW -m tcp -p tcp --dport 10050 -j ACCEPT
	-A INPUT -m conntrack --ctstate NEW -m tcp -p tcp --dport 10051 -j ACCEPT
	-A INPUT -m conntrack --ctstate NEW -m tcp -p tcp --dport 80 -j ACCEPT
 
As for selinux,you could use command `setenforce 0` to close selinux temporarily.

#### 2.2.2 install zabbix server ####

- install zabbix-server and zabbix-agent in zabbix server machine.
	`yum install zabbix zabbix-server zabbix-web-mysql zabbix-web zabbix-agent`

- install and config mysql(zabbix use mysql to store the data collected from zabbix agent).
	1. install mysql-server: `yum install mysql-server`
	2. config `/etc/my.cnf`
		
			[mysqld]
			datadir=/var/lib/mysql
			socket=/var/lib/mysql/mysql.sock
			user=mysql
			# Disabling symbolic-links is recommended to prevent assorted security risks
			symbolic-links=0
			
			character-set-server=utf8 # added
			innodb_file_per_table=1   # added
			
			[mysqld_safe]
			log-error=/var/log/mysqld.log
			pid-file=/var/run/mysqld/mysqld.pid

	3. start mysql serivce
	
			chkconfig mysqld on
			service mysqld start
	4. create database
			
			# mysqladmin -uroot password admin
			# mysql -uroot -padmin
			mysql> create databse zabbix character set utf8;
			mysql> grant all privileges on zabbix.* to zabbix@localhost identified by 'zabbix';
    		mysql> flush privileges; 
	5. import meatadata(such as structure of tables) from scripts
	
			# mysql -uzabbix -pzabbix
			mysql> use zabbix
			mysql> 
			未完成
	6. config `/etc/zabbix/zabbix_server.conf`

		It is ok for you to use default value for most of parameters, what you should modify are as following:
			
			DBHost=localhost
			DBName=zabbix
			DBPassword=zabbix
			StartPollers=5
			CacheSize=256M

	7. start zabbix server service
		
			# chkconfig zabbix-server on
			# chkconfig httpd on
			# service zabbix-server start
			# service httpd start


#### 2.2.3 config php ####

zabbix could show us data through WebUI using php running in httpd,so we also need config them properly.It is ok for you to use default value for most of parameters, what you should modify are as following:
	
	# cat /etc/php.ini
	date.timezone = Asia/Shanghai
	max_input_time = 300
	mbstring.func_overload = 2
	mbstring.language = Chinese
	memory_limit = 128M
	post_max_size = 16M

	  

### 2.3 install zabbix-agent in slave node ###

#### 2.3.1 preparation ####

please make sure you have config firewall(zabbix-agent use port 10050) and selinux properly.

	# cat /etc/sysconfig/iptables
	-A INPUT -m conntrack --ctstate NEW -m tcp -p tcp --dport 10050 -j ACCEPT
	-A OUTPUT -m conntrack --ctstate NEW -m tcp -p tcp --dport 10050 -j ACCEPT
 
As for selinux,you could use command `setenforce 0` to close selinux temporarily.

#### 2.3.2 install zabbix-agent ####

1. install zabbix-agent

		yum install zabbix zabbix-agent
2. config zabbix-agent
	
	1. there are two mode for zabbix server to collect data from zabbix agent.
		
		- passive mode: zabbix server pull data from zabbix agent.
		- active mode:zabbix agent push data to zabbix server.

	2. config `/etc/zabbix/zabbix_agentd.conf`.It is ok for you to use default value for most of parameters, what you should modify are as following:

			# cat /etc/zabbix/zabbix_agentd.conf
			Server=xxx 
			#passvie mode. if you config it,it defines which zabbix server could pull data from the machine.
			ServerActive=xxx 
			#active moed. if you config it,it defines which zabbix server the machine will push data to. 
			Hostname=xxx # the hostname of the machine
  	
	3. it is may be tricky for some linux OS, such as fedora, you may need:
		
			# mkdir /var/run/zabbix
			# chown -R zabbix:zabbix /var/run/zabbix
			# chmod -R g+w /var/run/zabbix

	4. start zabbix-agent service
			
		for redhat:
		
			# chkconfig zabbix-agent on
			# service zabbix-agent start.

		for fedora:

			# /usr/sbin/zabbix_agentd
	

### 2.4 config WebUI in browser###

please make sure you have start up httpd service in zabbix server machine.

1. input the url in brower: `http://zabbix server ipaddress/zabbix`

2. config zabbix acount

	![Alt text](/public/upload/zabbix_1.jpg)