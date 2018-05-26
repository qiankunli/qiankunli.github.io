---

layout: post
title: 异构数据库表在线同步
category: 技术
tags: Data
keywords: mysql binlog

---

## 简介

假设存在数据库表table1，要将其同步到table2。有以下要求：

1. table1作为线上数据库表，不能停止服务。这意味着table1在不停的被修改。
2. table1和table2表结构不同。比如删减或修改字段等。

比较常用的做法是：

1. 记一个时间点A
2. 先进行一次全量同步
3. 从时间点A开始进行一次增量同步

本文阐述下使用mysql binlog实现增量同步。

## 简单使用和查看

binlog是一种类型的文件，记录对数据发生或潜在发生更改的SQL语句，并以二进制的形式保存在磁盘中。

查看mysql的配置文件my.cnf，跟binlog相关的部分如下

    ## server-id 必须设置
    server-id		= 10
    log_bin			= /var/log/mysql/mysql-bin.log
    expire_logs_days	= 5
    max_binlog_size         = 100M
    #binlog_do_db		= include_database_name
    #binlog_ignore_db	= include_database_name
    
1. binlog文件的大小

    如果你的二进制文件的大小超过了max_binlog_size，它就是自动创建新的二进制文件。当然如果恰好在日志文件到达它的最大尺寸时写入了大的事务，那么日志文件还是会超过max_binlog_size的大小。默认情况下当二进制日志写满了或者数据库重启了才会进行切换，但是也可以手工的进行切换的动作。（在mysql命令行下执行`mysql> flush logs;`）
    
        # 查看binlog日志
        root@ubuntu:/var/log/mysql# ls
        mysql-bin.001337  mysql-bin.001338  mysql-bin.001339   mysql-bin.index

2. binlog文件的格式    

    binlog可以有多种格式，通过my.cnf文件的属性binlog_format指定。不同的格式包含的信息不太一样，有各自的优缺点。建议使用mixed  
    
    - statement，默认格式，保存SQL语句  
    - row 保存影响记录数据  
    - mixed 前面两种的结合  

3. binlog信息的查看

    binlog不能直接用文本的方式打开，mysql提供了相应的查看工具

    - mysql命令

            mysql> show master status;
            +------------------+----------+--------------+------------------+-------------------+
            | File             | Position | Binlog_Do_DB | Binlog_Ignore_DB | Executed_Gtid_Set |
            +------------------+----------+--------------+------------------+-------------------+
            | mysql-bin.000008 |     2066 |              |                  |                   |
            +------------------+----------+--------------+------------------+-------------------+
            
        
             mysql> show binlog events in 'mysql-bin.000009';
            +------------------+-----+-------------+-----------+-------------+---------------------------------------------------------------+
            | Log_name         | Pos | Event_type  | Server_id | End_log_pos | Info                                                          |
            +------------------+-----+-------------+-----------+-------------+---------------------------------------------------------------+
            | mysql-bin.000009 |   4 | Format_desc |         1 |         120 | Server ver: 5.6.28-log, Binlog ver: 4                         |
            | mysql-bin.000009 | 120 | Query       |         1 |         199 | BEGIN                                                         |
            | mysql-bin.000009 | 199 | Query       |         1 |         322 | use `test`; update tb_test0 set username= 'lqk3' where id = 3 |
            | mysql-bin.000009 | 322 | Xid         |         1 |         353 | COMMIT /* xid=8 */                                            |
            +------------------+-----+-------------+-----------+-------------+---------------------------------------------------------------+


    - mysqlbinlog命令，`mysqlbinlog filename`

        mysqlbinlog有丰富的参数可以提取出部分日志（其实就是sql语句），和其它命令组合就可以实现增量备份或还原。`mysqlbinlog –start-date=”2010-09-29 18:00:00″ –stop-date=”2010-09-29 23:00:00″ binlogfilename |mysql -u root -p`



## 使用java处理

java客户端使用binlog时，相当于该客户端是目标数据库（master）的一个slave。

binlog的java客户端采用github上大牛的开源作品`https://github.com/shyiko/mysql-binlog-connector-java`。mysql-binlog-connector-java的api比较简单，此处不再详谈。主要有以下几个概念：

1. BinaryLogClient，使用binlog的客户端
2. Event，数据增删改查等都被抽象为一个事件
3. EventListener，事件监听者，当事件发生时得到通知
4. EventFilter，事件过滤器，这样EventListener就可以只监听符合条件的事件

我在巨人的肩膀上做了进一步的封装，将原来的BinaryLogClient,EventListener和EventFilter等“配置化”，并实现了AbstractCudEventListener。AbstractCudEventListener负责监听create，update和delete event，并将源event数据转化为源表的model，用户可以在此基础上做进一步处理。经过封装后，用户无需再了解binlog的相关细节。

demo地址:`git@code.csdn.net:lqk654321/mysql-binlog-connector-demo.git`


## 数据库表同步的一些细节

可以优化的地方有很多，主要有以下几个方面：

1. 数据库使用文件存储数据，其性能的关键就是根据sql找到其操作的数据的位置。
2. 尽量减少数据库连接建立次数（或者说尽量一次数据库操作干尽可能多的事），事务的开启也是如此。

具体的说

1. 针对精确查找的情况，`select * from table1 where id in (xx,xx)`，批量查询比单个查询更提高性能。
2. 针对批量查找的情况，`select * from table1 where id > xx and id < xx`，比limit更提高性能，因为每次`limit startrow，rows`,数据库引擎都要遍历数据文件以找到开始的行。
3. 针对批量插入的情况，可以先开启事务，插入一千行，然后提交事务。这样可以减少事务的操作次数。当然，这样做的一个问题是，一条记录插入失败将导致一个事务的所有记录插入失败。因此要专门开启一个服务，将漏掉的记录找到并再次插入。
4. 如果数据库表有索引，导数据时可以先不创建索引，数据同步完毕后再创建
5. 控制操作数据表的线程数，防止数据库压力过大。

## 引用

[MySQL Binlog的介绍][]

[MySQL grant用户授权 和 MYSQL binlog日志 实操讲解][]

[MySQL Binlog的介绍]: http://www.linuxidc.com/Linux/2014-09/107095.htm
[MySQL Binlog三种格式介绍及分析]: http://www.linuxidc.com/Linux/2012-11/74359p2.htm
[MySQL日志格式 binlog_format][]

[MySQL日志格式 binlog_format]: http://blog.csdn.net/mycwq/article/details/17136997


[MySQL grant用户授权 和 MYSQL binlog日志 实操讲解]: http://www.lxway.com/18948251.htm