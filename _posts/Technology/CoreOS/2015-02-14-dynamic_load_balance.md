---
layout: post
title: 使用etcd + confd + nginx做动态负载均衡
category: 技术
tags: CoreOS
keywords: CoreOS Docker confd etcd nginx
---

## 简介

原文地址：[How To Use Confd and Etcd to Dynamically Reconfigure Services in CoreOS](https://www.digitalocean.com/community/tutorials/how-to-use-confd-and-etcd-to-dynamically-reconfigure-services-in-coreos "")，有删改。

CoreOS可以让我们在一个集群中很方便的运行docker contaienr 服务。我们通常启动一个或多个实例，并将这些实例注册到etcd中（etcd是CoreOS使用的一种分布式存储系统）。利用这个过程，相关的服务可以获取到架构的状态信息，今儿改变自己的行为。也就是说，当etcd中的数据改变时，相关服务可以动态的改变自己的配置。

在这个guide中，我们将讨论使用一个叫做confd的工具，专为监听分布式键-值存储系统的变化。它运行在一个container中，用来改变配置和重启服务。

## Creating the Nginx Container

### Installing the Software

通过`docker run -i -t ubuntu:14.04 /bin/bash`启动一个新的contaienr。更新apt缓存，安装Nginx和curl。同时，从github中获取confd的安装包。
    
    apt-get update
    apt-get install nginx curl -y
    
通过浏览器进入github，浏览confd的release page，找到最新的版本，撰写本文时，confd最新为v.0.5.0，右键复制链接地址。回到docker container中，使用复制的链接下载confd，并将其放在`/usr/local/bin`目录下。为confd文件添加可执行权限，同时在`/etc`下为confd创建configuration structure：

    mkdir -p /etc/confd/conf.d
    mkdir -p /etc/confd/templates
    
### Create a Confd Configuration File to Read Etcd Values

现在，我们所需的应用已经安装完毕了，我们开始配置confd，创建一个配置文件或是模板文件。

confd的配置文件是建立一个服务，监控特定的etcd的键值以及当值发生变化时执行某些操作。其文件格式是TOML，易于使用，（其规则）跟人的直觉也相近。

首先我们先创建一个名为nginx.toml的配置文件`vi /etc/confd/conf.d/nginx.toml`，文件内容如下：

    [template]
    # The name of the template that will be used to render the application's configuration file
    # Confd will look in `/etc/conf.d/templates` for these files by default
    src = "nginx.tmpl"
    
    # The location to place the rendered configuration file
    dest = "/etc/nginx/sites-enabled/app.conf"
    
    # The etcd keys or directory to watch.  This is where the information to fill in
    # the template will come from.
    keys = [ "/services/apache" ]
    
    # File ownership and mode information
    owner = "root"
    mode = "0644"
    
    # These are the commands that will be used to check whether the rendered config is
    # valid and to reload the actual service once the new config is in place
    check_cmd = "/usr/sbin/nginx -t"
    reload_cmd = "/usr/sbin/service nginx reload"
    
<table>
    <tr>
        <th>Directive</th>
        <th>Required?</th>
        <th>Type</th>
        <th>Description</th>
    </tr>
    <tr>
        <td>src</td>
        <td>Yes</td>
        <td>String</td>
        <td>The name of the template that will be used to render the information. If this is located outside of "/etc/confd/templates", the entire path is should be used.</td>
    </tr>
      <tr>
        <td>dest</td>
        <td>Yes</td>
        <td>String</td>
        <td>The file location where the rendered configuration file should be placed.</td>
    </tr>
    <tr>
        <td>keys</td>
        <td>Yes</td>
        <td>Array of strings</td>
        <td>The etcd keys that the template requires to be rendered correctly. This can be a directory if the template is set up to handle child keys.</td>
    </tr>
    <tr>
        <td>owner</td>
        <td>No</td>
        <td>String</td>
        <td>The username that will be given ownership of the rendered configuration file.</td>
    </tr>
     <tr>
        <td>group</td>
        <td>No</td>
        <td>String</td>
        <td>The group that will be given group ownership of the rendered configuration file.</td>
    </tr>
     <tr>
        <td>mode</td>
        <td>No</td>
        <td>String</td>
        <td>The octal permissions mode that should be set for the rendered file.</td>
    </tr>
     <tr>
        <td>check_cmd</td>
        <td>No</td>
        <td>String</td>
        <td>The command that should be used to check the syntax of the rendered configuration file.</td>
    </tr>
     <tr>
        <td>reload_cmd</td>
        <td>No</td>
        <td>String</td>
        <td>The command that should be used to reload the configuration of the application.</td>
    </tr>
     <tr>
        <td>prefix</td>
        <td>No</td>
        <td>String</td>
        <td>A part of the etcd hierarchy that comes before the keys in the keys directive. This can be used to make the .toml file more flexible.</td>
    </tr>
    
    
</table>

这个文件指出了我们的confd如何工作，我们的Nginx contaienr使用`/etc/confd/templates/nginx.tmpl`来渲染并生成`/etc/nginx/sites-enabled/app.conf`，文件的权限的0644，所有者为root。

confd将监控`/services/apache`（etcd路径）的变化。当发生变化时，confd将查询`/services/apache`节点的信息，并为Nginx渲染一个新的配置文件，检查该配置文件的语法并重启Nginx服务（以应用新的配置文件）。
接下来，我们创建一个实际的confd template文件。(confd)用它来渲染Nginx配置文件。

### Create a Confd Template File

创建我们在`/etc/confd/conf.d/nginx.toml`提到的模板文件，将其放在`/etc/confd/templates`目录下。`vi /etc/confd/templates/nginx.tmpl`。

在这个文件中，我们只是简单的重写了一个标准的Nginx反向代理配置文件。但是，我们使用一些Go 模板语法来代替confd从etcd中拉取的信息。

首先，我们配置"upstream" servers block（代码块，配置块）。这个小节定义了Nginx可以发送请求的服务器池。格式如下

    upstream pool_name {
        server server_1_IP:port_num;
        server server_2_IP:port_num;
        server server_3_IP:port_num;
    }
    
This allows us to pass requests to the pool_name and Nginx will select one of the defined servers to hand the request to.


对于template文件来说，不是静态的定义upstream servers，这些信息在渲染时再动态的填充（我们从etcd中获取apache服务器的ip地址和端口）。我们先用Go template来表示这些动态内容。

    upstream apache_pool {
    {\{ range getvs "/services/apache/*" \}}
        server {\{ . \}};
    {\{ end \}}
    }

我们定义了一个upstream pool of servers 叫做`apache_pool`，在这个快内，我们用双大括号表示里面填充的是Go代码。

在大括号里，我们指定了我感兴趣的信息所在的etcd节点，并使用range来遍历它们。（具体可以参见GO语法）

设置完server pool之后，我们可以设置一个proxy pass将所有的请求导向这个pool。于是一个标准的反向代理的server block就好了。

    upstream apache_pool {
    {{ range getvs "/services/apache/*" }}
        server {{ . }};
    {{ end }}
    }
    
    server {
        listen 80 default_server;
        listen [::]:80 default_server ipv6only=on;
    
        access_log /var/log/nginx/access.log upstreamlog;
    
        location / {
            proxy_pass http://apache_pool;
            proxy_redirect off;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        }
    }

同时还要记着移除默认的nginx文件

    rm /etc/nginx/sites-enabled/default
    
同时还要配置下日志的格式（我们在template文件中有引用到），这必须在http block中配置（在`/etc/nginx/nginx.conf`文件中）。

我们将添加一个`log_format`的directive来定义我们想要记录信息。它记录了客户端请求以及处理请求的服务端的信息，以及处理请求耗费的时间。

    . . .
    http {
        ##
        # Basic Settings
        ##
        log_format upstreamlog '[$time_local] $remote_addr passed to: $upstream_addr: $request Upstream Response Time: $upstream_response_time Request time: $request_time';
    
        sendfile on;
        . . .

## Creating a Script to Run Confd

我们需要创建一个脚本文件，以在合适的时机触发confd来操作template resource文件和template文件。

这个脚本必须做两件事：

- 在nginx根据后端（提供服务的所有server）情况启动之前运行
- 必须能够持续监控etcd的变化，重新配置nginx，以确保为nginx配置的后端server都是可用的。

我们将该文件命名为`confd-watch`，跟confd 可执行文件放在一个目录下（`/usr/local/bin`）。

首先，我们制定bash作为解释器，并为bash设置一些参数，确保其如果执行时发生错误，会立即退出，并返回最后执行指令的返回值。

接下来，我们设置一些变量。通过使用bash parameter substitution（参数替换），我们先设置默认值，当执行脚本时再替换硬编码的值。 This will basically just set up each component of the connection address independently and then group them together to get the full address needed.

The parameter substitution is created with this syntax: `${var_name:-default_value}`. This has the property of using the value of var_name if it is given and not null, otherwise defaulting to the default_value.

我们设置了etcd的默认值，这样，脚本没有其他信息（执行时不输入参数）也可以正常工作。当然，必要的时候，也可以自定义参数的值。

    #!/bin/bash
    
    set -eo pipefail
    
    export ETCD_PORT=${ETCD_PORT:-4001}
    export HOST_IP=${HOST_IP:-172.17.42.1}
    export ETCD=$HOST_IP:$ETCD_PORT
    
    echo "[nginx] booting container. ETCD: $ETCD."
    
    # Try to make initial configuration every 5 seconds until successful
    until confd -onetime -node $ETCD -config-file /etc/confd/conf.d/nginx.toml; do
        echo "[nginx] waiting for confd to create initial nginx configuration."
        sleep 5
    done
    
    # Put a continual polling `confd` process into the background to watch
    # for changes every 10 seconds
    confd -interval 10 -node $ETCD -config-file /etc/confd/conf.d/nginx.toml &
    echo "[nginx] confd is now monitoring etcd for changes..."
    
    # Start the Nginx service using the generated config
    echo "[nginx] starting nginx service..."
    service nginx start
    
    # Follow the logs to allow the script to continue running
    tail -f /var/log/nginx/*.log
    
接下来，从etcd中读取数据，并使用cond实例化nginx的配置文件。我们会使用一个until循环来不停的执行这个操作。


为了防止etcd突然失效或者nginx 容器先于后端运行，这个循环是必要的。有了这个循环，脚本将不停的从etcd拉取数据直到可以（为nginx）构建一个有效的初始化配置为止。

单就confd而言，confd 命令执行一次就退出了（指的是onetime参数）。两次运行的间隔为5秒，为后端的服务器（将服务注册到etcd）留一点时间。我们连接etcd（指的是-node参数），其值可以是默认或者我们通过参数指定的。我们使用模板文件来描述我们想做什么（指的是-config-file参数）。

初始化配置完毕后，接下来是实现一个持久拉取数据的机制。我们要确保及时发现（后端）所有的变化并更新nginx。

我们再一次调用confd，这一次，我们设定一个轮询间隔，并让这条命令后台运行（这样它就可以一直运行）。因为（跟初始化过程的）目标的一样，我们为其配置了同样的etcd节点和template source文件。

之后，我们可以安全的启动nginx了。因为这个脚本由`docker run`命令触发（nginx和confd运行在一个容器中），我需要它保持运行（ running in the foreground）以防容器退出。`tail -f /var/log/nginx/*.log`可以很好的做到这一点（我们也可以方便的查看日志信息）。

接着，便是为文件添加执行权限了。

    chmod +x /usr/local/bin/confd-watch

## Commit and Push the Container

将上述步骤制作的镜像（ubuntu + nginx + confd + 相关文件和脚本）制作成镜像并提交。

## Build the Nginx Static Unit File

接下来就是为这个container 创建unit文件，然后通过fleet来控制它（从而提供负载均衡服务），unit文件命名为`nginx_lb.service`。

    [Unit]
    Description=Nginx load balancer for web server backends
    
    # Requirements
    Requires=etcd.service
    Requires=docker.service
    
    # Dependency ordering
    After=etcd.service
    After=docker.service
    
    [Service]
    # Let the process take awhile to start up (for first run Docker containers)
    TimeoutStartSec=0
    
    # Change killmode from "control-group" to "none" to let Docker remove
    # work correctly.
    KillMode=none
    
    # Get CoreOS environmental variables
    EnvironmentFile=/etc/environment
    
    # Pre-start and Start
    ## Directives with "=-" are allowed to fail without consequence
    ExecStartPre=-/usr/bin/docker kill nginx_lb
    ExecStartPre=-/usr/bin/docker rm nginx_lb
    ExecStartPre=/usr/bin/docker pull user_name/nginx_lb
    ExecStart=/usr/bin/docker run --name nginx_lb -p ${COREOS_PUBLIC_IPV4}:80:80 \
    user_name/nginx_lb /usr/local/bin/confd-watch
    
    # Stop
    ExecStop=/usr/bin/docker stop nginx_lb
    
    [X-Fleet]
    Conflicts=nginx.service
    Conflicts=apache@*.service

（unit小节略过），在service小节中，设置timeout和killmode，把`/etc/environment`文件拉进来（因为unit文件接下来的内容会用到）。然后依次是配置ExecStartPre，ExecStart和ExecStop。

在X-Fleet小节，为了让其有效的分派负载，我们要求该容器运行在没有运行负载平衡和后端服务器的主机上。

## Running the Nginx Load Balancer

接下来便是，fleet如何控制其运行了。
