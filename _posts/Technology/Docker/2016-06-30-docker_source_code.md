---

layout: post
title: Docker0.1.0源码分析
category: 技术
tags: Docker
keywords: Docker

---

## 前言

一般学习一个较早版本的源码，有助于我们忽略非核心细节，学习一个工具的基本使用。

下载docker某个版本的源码，进入`https://github.com/docker/docker`,进入release或tag，就可以看到下载`tar.gz`或`zip`的按钮。

本文主要以0.1.0为基础，一共二三十个go文件。


## 基本的tcp通信

docker是client与daemon合二为一，通过flag区分。其server部分源码有一段（去掉了错误处理）
    
    // tcp.go
    func ListenAndServe(proto, addr string, service Service) error {
    	listener, err := net.Listen(proto, addr)
    	defer listener.Close()
    	// for循环监听连接，开启新的“线程”处理连接
    	for {
    		if conn, err := listener.Accept(); err != nil {
    			return err
    		} else {
    		    // 使用GoRoutine执行一个匿名函数处理连接
    			go func() {
    				if err := Serve(conn, service); err != nil {
    					log.Printf("Error: " + err.Error() + "\n")
    					fmt.Fprintf(conn, "Error: "+err.Error()+"\n")
    				}
    				conn.Close()
    			}()
    		}
    	}
    	return nil
    }

这跟我们常写的通信代码一样一样的，甚至还没有netty的nio来的复杂。笔者看到这，就对docker代码的畏惧感小了很多。

通过跟踪Serve的执行，我们会发现docker会解析请求中的数据，根据请求中的参数执行相应的方法。（用到了go里的反射），比如一个`docker ps`命令，commands.go中就会有一个CmdPs方法。

    // types.go
    func LocalCall(service Service, stdin io.ReadCloser, stdout io.Writer, args ...string) error {
    	method := getMethod(service, cmd)
    	if method != nil {
    		return method(stdin, stdout, flags.Args()[1:]...)
    	}
    	return errors.New("No such command: " + cmd)
    }
    
    func getMethod(service Service, name string) Cmd {
    	methodName := "Cmd" + strings.ToUpper(name[:1]) + strings.ToLower(name[1:])
    	method, exists := reflect.TypeOf(service).MethodByName(methodName)
    	if !exists {
    		return nil
    	}
    	// 这个待理解
    	return func(stdin io.ReadCloser, stdout io.Writer, args ...string) error {
    		ret := method.Func.CallSlice([]reflect.Value{
    			reflect.ValueOf(service),
    			reflect.ValueOf(stdin),
    			reflect.ValueOf(stdout),
    			reflect.ValueOf(args),
    		})[0].Interface()
    		if ret == nil {
    			return nil
    		}
    		return ret.(error)
    	}
    }

接下来，我们就可以在commands.go里找到每个方法的执行逻辑。

## docker命令的执行

最关键的就是start命令的实现


    // container.go
    func (container *Container) Start() error {
        if err := container.EnsureMounted(); err != nil {
		    return err
	    }
    	if err := container.allocateNetwork(); err != nil {
    		return err
    	}
    	if err := container.generateLXCConfig(); err != nil {
    		return err
    	}
    	// 省略
    	// 为cmd赋值
    	container.cmd = exec.Command("/usr/bin/lxc-start", params...)
    	// 省略
    	if container.Config.Tty {
		    err = container.startPty()
    	} else {
    	    // 启动
    		err = container.start()
    	}
        // 省略
    	go container.monitor()
	    return nil
    }
    func (container *Container) start() error {
    	// 配置一些环境
    	return container.cmd.Start()
    }
    func (c *Cmd) Start() error {
    	// 配置Cmd struct的其它参数，Cmd封装了"/usr/bin/lxc-start"所需的所有参数，os.StartProcess执行它
    	c.Process, err = os.StartProcess(c.Path, c.argv(), &os.ProcAttr{
    		Dir:   c.Dir,
    		Files: c.childFiles,
    		Env:   c.envv(),
    		Sys:   c.SysProcAttr,
    	})
    	// 其它后续工作
    }
    
如果你跟踪这些方法，你会发现，运行一个容器之前的所有工作都是为`/usr/bin/lxc-start`准备各种数据。有些地方看不懂，应该是对lxc提供的接口及实现不太懂（主要是cgroup和namespace特性，笔者没怎么用过go语言，更别提用go操作os的高级特性了，从这个角度看，docker基本不可能用java实现，除非jvm支持），整体的实现流程是清晰的。


## 重要的结构体

    // runtime.go
    type Runtime struct {
    	root           string
    	repository     string
    	containers     *list.List
    	networkManager *NetworkManager
    	graph          *Graph
    	repositories   *TagStore
    	authConfig     *auth.AuthConfig
    }
    // container.go
    type Container struct {
    	root string
    	Id string
    	Created time.Time
    	Path string
    	Args []string
    	Config *Config
    	State  State
    	Image  string
    	network         *NetworkInterface
    	NetworkSettings *NetworkSettings
    	SysInitPath string
    	cmd         *exec.Cmd
    	stdout      *writeBroadcaster
    	stderr      *writeBroadcaster
    	stdin       io.ReadCloser
    	stdinPipe   io.WriteCloser
    	stdoutLog *os.File
    	stderrLog *os.File
    	runtime   *Runtime
    }
    
看每个文件中的struct以及对应的方法，将docker复杂的功能分解成一个个小模块，或者说，将lxc的复杂使用（参数配置、文件、网卡等环境准备等）如何向上简化，程序设计真的是一门艺术。
    
## go语言写大型项目的基本特点

1. 一个文件有一个主要的struct，以及这个struct相关的操作。类似java的一个类文件
2. 通过struct方法的参数实现struct之间的相互依赖（面向对象的三个基本特征：封装，继承和多态，go只支持封装，因此没有java那样复杂的类间关系）
