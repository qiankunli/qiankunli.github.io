---

layout: post
title: Golang开发环境搭建（Windows下）
category: 技术
tags: Go
keywords: Go

---

## 一 前言（已过时）

搭建go开发环境主要有以下方式：

1. goEclipse
2. sublime text + gosublime + gocode
3. liteIDE

第一种，速度较慢；第三种，还得新装一个软件；推荐使用第二种方式。

## 二 步骤

1. 安装go环境，配置GOROOT和GOPATH，添加PATH变量

    - （题外话）在ubuntu下安装

            $ sudo apt-get install golang
            # 在/etc/profile下配置环境变量
                
            GOROOT=/usr/lib/go
            GOBIN=/usr/bin/go
            # go get 得到的第三方库就存在/usr/local/gorepo路径下
            GOPATH=/usr/local/gorepo
            export GOPATH GOBIN GOROOT

2. 安装package controll    （` crtrl + 反引号`进入命令）
    
    输入以下内容并回车（不同版本的sublime，该内容貌似不一样）。
    
        import urllib2,os; pf='Package Control.sublime-package'; ipp=sublime.installed_packages_path(); os.makedirs(ipp) if not os.path.exists(ipp) else None; urllib2.install_opener(urllib2.build_opener(urllib2.ProxyHandler())); open(os.path.join(ipp,pf),'wb').write(urllib2.urlopen('http://sublime.wbond.net/'+pf.replace(' ','%20')).read()); print 'Please restart Sublime Text to finish installation'    
    
2. 安装gosublime pakcage    （`ctrl + shift + p` 进入包管理器）

    输入`install`回车，进入一个安装pakcage的对话框
    输入`GoSublime` 回车
3. 安装gocode（语言自动补全守护程序）（使用`go get`前提是已安装git环境）

    1. `go get -u github.com/nsf/gocode` 获取项目文件
    2. `go install github.com/nsf/gocode` 编译项目文件得到可执行文件
    3. 配置gosublime 使用它

        Preferences ==> package settings ==> GoSublime ==> settings-Default 
        
        将该文件的
        
            "env":{},
        改为

            "env":{
		         "path":"E:\\GoRepo\\bin"
		         },
		         
        其中，`E:\\GoRepo`是笔者存放下载的go库的总目录（`go get`前要将该路径添加到GOPATH环境变量中（如果GOPATH包含多个路径，则该路径必须是第一位的（因为`go get`只会向GOPATH指向的第一个路径里存放文件））），`E:\\GoRepo\\bin`包含了gocode的可执行文件。
        
4. 编写`hello.go`文件
5. `ctrl + b` 切换到侧边栏显示状态

        [ E:/workspaces/golang/hello/ ] go build hello.go
        [ E:/workspaces/golang/hello/ ] hello
            
## 三 第一个Go项目

比如在`E:\\workspaces\\golang`中新建一个web项目：

    $ cd /e/workspaces/golang
    // 创建以下目录结构
    $ mkdir myweb
    $ mkdir myweb/src
    $ touch myweb/src/server.go 
    $ mkdir myweb/bin
    
    // 进入bin目录下编译源文件
    $ cd myweb/bin
    $ go build server   // 编译server.go文件
    // 此时bin目录下便生成了该项目的可执行文件
    $ ./bin/server            // 运行server
    
** 注意： ** 为了构建这个工程，必须将"E:\\workspaces\\golang\\myweb"加入到GOPATH环境变量中。

在sublime下开发时，则可以 Preferences ==> package settings ==> GoSublime ==> settings-User，在文件中添加如下内容：

    {
    	"env": {
    		"GOPATH":"E:\\workspaces\\golang\\myweb"
    	}
    }
    
当然，如果只是编译一个go文件，那么可以进入文件所在目录，`go build 文件名.go`，在该目录下将生成对应的可执行文件。

## 四 引入第三方包
假设所有第三方库文件存放在`E:\\GoRepo`目录下，将该路径加入到GOPATH环境变量中（注意，如果存在其他路径，要将其放在第一的位置）

    $ go get github.com/cihub/seelog
    
执行完毕后，`E:\\GoRepo`将包含如下内容:

    /e/GoRepo/pkg
    /e/GoRepo/src/github.com/cihub/seelog
    
接着执行
    
    $ go get github.com/go-sql-driver/mysql
    
执行完毕后，`E:\\GoRepo`将包含如下内容:

    /e/GoRepo/pkg
    /e/GoRepo/src/github.com/cihub/seelog
    /e/GoRepo/src/github.com/go-sql-driver/mysql
    
一般，go代码托管网站除了github.com外，还有code.google.com，而国内无法直接下载code.google.com上的库，需要时可以到[golang中国][]下载。

对于golang.org/x上的代码，github上也有一份，可以`go get github.com/golang`,然后再`GOPATH/src/golang`下建一个软件链接，`x ==> GOPATH/github.com/golang`

[golang中国]: http://www.golangtc.com/download/package