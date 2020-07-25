---

layout: post
title: Gotty源码分析
category: 技术
tags: Go
keywords: Gotty

---

## 前言

* TOC
{:toc}

## 手感

[容器Web Console技术实现](https://cloud.tencent.com/developer/article/1416063)

### 基本使用

1. 本机启动`gotty -w bash`
2. 本机启动浏览器，`localhost:8080` 浏览器便是一个全黑的console 界面

### 和kubernetes 结合使用

1. 本机启动`gotty -w --permit-arguments kubectl exec -it`
2. 本机启动浏览器 `http://localhost:8080/?arg=fm-barge-backend-stable-69f77f6767-jw69l&arg=%2fbin%2fbash`

传入两个arg，一个是pod 的名字，一个是`/bin/bash` ，其实就是把 两个arg 拼在`kubectl exec -it` 之后

## 代码结构

```
/main.go        // 入口文件，命令行工具库用的"github.com/codegangsta/cli"
/server
    /server.go  // 定义server struct
    /ws_wrapper.go  // 定义wsWrapper struct
/backend
    /localcommand
        /local_command.go
/webtty
    /webtty.go  // 定义WebTTY struct
```

![](/public/upload/go/gotty_object.png)

## 启动流程

main.go  的核心逻辑app.Run ==>  app.Action，去掉参数校验、日志等逻辑，核心流程为创建Server 并run 起来

```go
app.Action = func(c *cli.Context) {
    configFile := c.String("config")
    utils.ApplyFlags(cliFlags, flagMappings, c, appOptions, backendOptions)
    args := c.Args()
    factory, err := localcommand.NewFactory(args[0], args[1:], backendOptions)
    srv, err := server.New(factory, appOptions)
    errs := make(chan error, 1)
    go func() {
        errs <- srv.Run(ctx, server.WithGracefullContext(gCtx))
    }()
    err = waitSignals(errs, cancel, gCancel)
    if err != nil && err != context.Canceled {
        fmt.Printf("Error: %s\n", err)
        exit(err, 8)
    }
}
```

![](/public/upload/go/gotty_serv.png)

## 交互命令是如何执行的

### http 升级为web socket

![](/public/upload/go/gotty_interact.png)

可见，gotty 要处理两种请求 http 和 websocket 请求， 它们的处理逻辑在`Server.setupHandlers` 中指定。基于 siteMux 创建siteHandler，然后siteHandler wrap wsMux 等，使其对http、websocket、staticFile 都具备处理能力。 可以看到 `server.generateHandleWS` 提供了处理 websocket 请求的handler

```go
func (server *Server) setupHandlers(ctx context.Context, cancel context.CancelFunc, pathPrefix string, counter *counter) http.Handler {
    staticFileHandler := http.FileServer(
        &assetfs.AssetFS{Asset: Asset, AssetDir: AssetDir, Prefix: "static"},
    )

    var siteMux = http.NewServeMux()
    siteMux.HandleFunc(pathPrefix, server.handleIndex)
    siteMux.Handle(pathPrefix+"js/", http.StripPrefix(pathPrefix, staticFileHandler))
    siteMux.Handle(pathPrefix+"favicon.png", http.StripPrefix(pathPrefix, staticFileHandler))
    siteMux.Handle(pathPrefix+"css/", http.StripPrefix(pathPrefix, staticFileHandler))

    siteMux.HandleFunc(pathPrefix+"auth_token.js", server.handleAuthToken)
    siteMux.HandleFunc(pathPrefix+"config.js", server.handleConfig)

    siteHandler := http.Handler(siteMux)

    if server.options.EnableBasicAuth {
        log.Printf("Using Basic Authentication")
        siteHandler = server.wrapBasicAuth(siteHandler, server.options.Credential)
    }

    withGz := gziphandler.GzipHandler(server.wrapHeaders(siteHandler))
    siteHandler = server.wrapLogger(withGz)
    // 处理websocket 请求
    wsMux := http.NewServeMux()
    wsMux.Handle("/", siteHandler)
    wsMux.HandleFunc(pathPrefix+"ws", server.generateHandleWS(ctx, cancel, counter))
    siteHandler = http.Handler(wsMux)

    return siteHandler
}
```
本机启动 `gotty -w bash`，然后浏览器访问 `localhost:8080`，浏览器发出请求下载一系列js文件， 其中的关键是 发出了`ws://localhost:8080/ws`，然后服务端返回http status=101（Switching Protocols 服务器将遵从客户的请求转换到另外一种协议）进行了协议升级。

![](/public/upload/go/gotty_ws_upgrade.png)

### 处理websocket 请求的逻辑

webSocketConn 代表浏览器websocket 连接，localcommand 代表 用户命令的执行。

```go
func (server *Server) generateHandleWS(ctx context.Context, cancel context.CancelFunc, counter *counter) http.HandlerFunc{
    ...
    return func(w http.ResponseWriter, r *http.Request) {
        ...
        conn, err := server.upgrader.Upgrade(w, r, nil)
        defer conn.Close()
        ...
        err = server.processWSConn(ctx, conn)
        ...
    }
}
func (server *Server) processWSConn(ctx context.Context, conn *websocket.Conn) error {
    typ, initLine, err := conn.ReadMessage()
    err = json.Unmarshal(initLine, &init)
    queryPath := "?"
    if server.options.PermitArguments && init.Arguments != "" {
        queryPath = init.Arguments
    }
    query, err := url.Parse(queryPath)
    ...
    params := query.Query()
    var slave Slave
    slave, err = server.factory.New(params)
    ...
    tty, err := webtty.New(&wsWrapper{conn}, slave, opts...)
    if err != nil {
        return errors.Wrapf(err, "failed to create webtty")
    }
    err = tty.Run(ctx)
    return err
}
```

根据`gotty -w $GOTTY_PERMIT_WRITE` 中指定的command 以及arg 创建cmd 并接上 `/dev/ptmx`（参见文末的终端和伪终端）。 

```go
func (wt *WebTTY) Run(ctx context.Context) error {
    err := wt.sendInitializeMessage()
    go func() {
        errs <- func() error {
            buffer := make([]byte, wt.bufferSize)
            for {
                n, err := wt.slave.Read(buffer)
                err = wt.handleSlaveReadEvent(buffer[:n])
            }
        }()
    }()
    go func() {
        errs <- func() error {
            buffer := make([]byte, wt.bufferSize)
            for {
                n, err := wt.masterConn.Read(buffer)
                err = wt.handleMasterReadEvent(buffer[:n])
            }
        }()
    }()
    ...
}
```

GoTTY在收到用户请求后，会执行启动时设置的参数，得到进程的stdin和stdout。随后会在单独的goroutine中，循环读取进程的输出写到websocket中，循环从websocket中读取写到进程的输入中

数据流如下

1. 发送指令： websocket.Conn ==> `/dev/ptmx` ==> `/dev/pts/xx` ==> localCommand
2. 接收响应： websocket.Conn <== `/dev/ptmx` <== `/dev/pts/xx` <== localCommand

## 终端和伪终端

PS： 笔者一开始mac上试验，一些细节和linux 有所不同。 这里的主从理解起来比较难受，可以不用太关注。

### 终端

tty, tty原意是远程输入机（teletypewriter)，现在在unix系统中是 text terminal 的意思。在 GNU/Linux 和 Mac OS X 上，都有terminal程序，打开一个 terminal 程序就对应一个 tty (text terminal) 设备文件。往 `/dev/tty` 写入内容会在当前terminal里回显。

    $ echo 'haha' > /dev/tty
    haha

每次打开terminal时会有个唯一的tty文件与其对应，比如`/dev/ttys000` 、`/dev/ttys001`等，`/dev/tty` 会根据当前活动的terminal去找到对应文件ttys000或者ttys001。 

terminal（终端）可以等同于 tty。terminal 是 shell 的包裹器（wrapper），terminal 接收用户输入的命令，并将命令传给 shell。


### 伪终端

**伪终端**(Pseudo Terminal)是**终端**的发展，它是成对出现的逻辑终端设备，对master的操作会反映到slave，pts和ptmx 配合使用实现 pty。

[深入理解sshd创建pty的过程](https://www.cnblogs.com/MrVolleyball/p/10024540.html)历史上，有两套伪终端软件接口：

1. BSD接口：较简单，master为`/dev/pty[p-za-e][0-9a-f]` ;slave为 `/dev/tty[p-za-e][0-9a-f]`
2. Unix 98接口：使用一个`/dev/ptmx`作为master设备，在每次打开操作时会得到一个master设备fd，并在`/dev/pts/`目录下得到一个slave设备如 `/dev/pts/3`


master  `/dev/ptmx`  Master 将命令传给 slave 或者将 slave 的数据显示出来。
slave   `/dev/pts/xx`  Slave 就是 pts（pseudo terminal slave），不同在于 terminal 直接连接在主机上，pts 通过一些软件连接到主机上。

```go
// github.com/kr/pty/run.go
func Start(c *exec.Cmd) (pty *os.File, err error) {
    pty, tty, err := Open()
    if err != nil {
        return nil, err
    }
    defer tty.Close()
    c.Stdout = tty
    c.Stdin = tty
    c.Stderr = tty
    c.SysProcAttr = &syscall.SysProcAttr{Setctty: true, Setsid: true}
    err = c.Start()
    if err != nil {
        pty.Close()
        return nil, err
    }
    return pty, err
}

func open() (pty, tty *os.File, err error) {
    p, err := os.OpenFile("/dev/ptmx", os.O_RDWR, 0)
    sname, err := ptsname(p)
    err = grantpt(p)
    err = unlockpt(p)
    // 看样子是 根据/dev/ptmx 创建一个/dev/pts/xx
    t, err := os.OpenFile(sname, os.O_RDWR, 0)
    return p, t, nil
}
```

tty file 是根据 pty file 创建，assigns a pseudo-terminal tty os.File to cmd.Stdin, cmd.Stdout,and cmd.Stderr, calls c.Start, and returns the File of the tty's corresponding pty.

1. 发送指令：sshd ==> /dev/ptmx ==> /dev/pts/xx ==> bash 
2. 接收指令的数据： sshd <== /dev/ptmx <== /dev/pts/xx <== bash

