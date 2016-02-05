---

layout: post
title: docker volume
category: 技术
tags: Docker
keywords: Docker volume

---

## 简介（存储管理也是一个重要课题，还没做什么工作）

## docker volume


    // 创建一个容器，包含两个数据卷
    $ docker run -v /var/volume1 -v /var/volume2 -name Volume_Container ubuntu14.04 linux_command
    // 创建App_Container容器，挂载Volume_Container容器中的数据卷
    $ docker run -t -i -rm -volumes-from Volume_Container -name App_Container ubuntu14.04  linux_command
    // 这样两个容器就可以共用这个数据卷了
    
    // 最后可以专门安排一个容器，在应用结束时，将数据卷中的内容备份到主机上
    docker run -rm --volumes-from DATA -v $(pwd):/backup busybox tar cvf /backup/backup.tar /data
    
在默认方式下，volume就是在`/var/lib/docker/volumes`目录下创建一个文件夹，并将该文件夹挂载到容器的某个目录下（以UFS文件系统的方式挂载）。当然，我们也可以指定将主机的某个特定目录（该目录要显式指定）挂载到容器的目录中。