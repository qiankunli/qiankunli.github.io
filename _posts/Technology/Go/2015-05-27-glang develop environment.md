---

layout: post
title: Golang开发环境搭建（Windows下）
category: 技术
tags: Go
keywords: Go

---

## 一 前言

搭建go开发环境主要有以下方式：

1. goEclipse
2. sublime text + gosublime + gocode
3. liteIDE

第一种，速度较慢；第三种，还得新装一个软件；推荐使用第二种方式。

## 二 步骤

1. 安装go环境，配置GOROOT和GOPATH，添加PATH变量
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
		         "path":"E:\\GoRepo\\gocode\\bin"
		         },
		         
        其中，`E:\\GoRepo`是笔者存放下载的go库的总目录，E:\\GoRepo\\gocode\\bin`包含了gocode的可执行文件。
        
4. 编写hello文件
5. `ctrl + b` 切换到侧边栏显示状态

        [ E:/workspaces/golang/hello/ ] go build hello.go
        [ E:/workspaces/golang/hello/ ] hello
            
## 三 第一个Go项目

比如在`E:\\workspaces\\golang`中新建一个web项目：

    $ cd /e/workspaces/golang
    $ mkdir myweb
    $ cd myweb
    $ mkdir src
    $ mkdir bin
    $ cd src
    $ mkdir server      // 弄个server包
    $ // 在server包下创建server.go文件
    $ cd ../bin
    $ go build server   // 编译server.go文件（如果server.go引入了其它包的文件，则编译其它包）
    $ ./server            // 运行server
    
** 注意： ** 为了构建这个工程，必须将"E:\\workspaces\\golang\\myweb"加入到GOPATH环境变量中。

在sublime下开发时，则可以 Preferences ==> package settings ==> GoSublime ==> settings-User，在文件中添加如下内容：

    {
    	"env": {
    		"GOPATH":"E:\\workspaces\\golang\\myweb"
    	}
    }

    

    


