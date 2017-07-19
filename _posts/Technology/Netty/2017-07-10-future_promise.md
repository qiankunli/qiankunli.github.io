---

layout: post
title: netty中的future和promise
category: 技术
tags: Netty
keywords: JAVA netty

---

## 前言

## Future 模式

（来自引用）**并发编程中，我们通常会用到一组非阻塞的模型：Promise，Future 和 Callback。其中的 Future 表示一个可能还没有实际完成的异步任务的结果，针对这个结果可以添加 Callback 以便在任务执行成功或失败后做出对应的操作，而Promise交由任务执行者，任务执行者通过 Promise 可以标记任务完成或者失败。 这一套模型是很多异步非阻塞架构的基础。** 因为netty中也有类似“调用者和执行部件以异步的方式交互通信结果”的需求（要知道eventloop本质上是一个ScheduledExecutorService，ExecutorService是一种“提交-执行”模型实现，也存在线程间异步方式通信和线程安全问题），所以netty自己实现了一套。

Netty中的所有的IO操作都是异步的（比如write（object），因为一个线程要服务多个Sockethannel，所以通过`socketchannel.xx`只是提交一个任务，何时返回结果自然是不确定的），而不是像传统BIO那样同步等待操作完成。同步操作的行为是可以预期的（一般你可以从方法的返回值中拿到结果）

1. 要么就等着操作完成
2. 可以设置timeout

异步指的是，调用者向执行部件发出一个调用，不等执行部件返回结果，继续向下执行。但异步操作带来一个问题：调用者如何获取异步操作的结果？

1. 轮询状态，调用者每隔一段时间检查下执行部件的状态
2. 通知，执行部件保有调用者引用，执行完毕后，执行调用者的相关方法
2. 回调，跟通知没多大区别，调用者向执行部件传递一个callback
3. promise参数

事实上，Netty Future的建议操作模式就是赤裸裸的通知，执行部件改变状态时，会执行注册在future上的Listener（变相的观察者模式）。

scala在语言层面提供对Promise，Future和Callback模型的支持，`https://bitbucket.org/qiyi/commons-future.git`作者自定义实现了该模型，去除了Netty的Future模型对EventLoop的依赖。


## netty中的future

a future is a read-only placeholder view of a variable, while a promise is a writable。Promise是可写的Future，提供写操作相关的接口，用于设置IO操作的结果。Future，Promise，callback抽象出一套调用者与执行部件间的通信模型（不只是netty中），**Future像是给调用者用的（拿结果），Promise像是给执行部件用的（设置结果），它们简化了调用者和执行部件对其的调用（调用者get，执行部件set），但本身要封装很多事**。比如Future必须是一个线程安全的类（大部分时候，调用者和执行部件身处两个线程），比如执行callback（或者listener）。

![Alt text](/public/upload/java/netty_future.png) 

比如DefaultPromise的setSuccess方法

    public Promise<V> setSuccess(V result){
    	// setSuccess0返回是否设置状态成功，其中涉及到一些锁操作
        if(setSuccess0(result)){
            notifyListeners();
            return this;
        }
        throw new IllegalStateException("complete already: " + this);
    }

Future和Promise常见的使用模式是

1. 调用创建一个Future（Promise）
2. 传给执行部件，执行部件根据执行情况，设置Promise
3. 调用者从Future中获取执行状态或结果

但也有很多其它使用方式。

很多类里定义了一些future，比如VoidFutrue，SuccessedFuture

是不是一个操作成功了，直接返回一个successedfuture，而不是future.setResult(success);

比如AbstractChannel有一个CloseFuture成员，`channel.closeFuture().sync();`，实际的代码就是加锁并object.wait。

这CloseFuture定义在AbstractChannel内部

    static final class CloseFuture extends DefaultChannelPromise {
        CloseFuture(AbstractChannel ch) {
            super(ch);
        }
        @Override
        public ChannelPromise setSuccess() {
            throw new IllegalStateException();
        }
        @Override
        public ChannelPromise setFailure(Throwable cause) {
            throw new IllegalStateException();
        }
        @Override
        public boolean trySuccess() {
            throw new IllegalStateException();
        }
        @Override
        public boolean tryFailure(Throwable cause) {
            throw new IllegalStateException();
        }
        boolean setClosed() {
            return super.trySuccess();
        }
    }

然后在AbstractChannel很多操作的catch代码块，都会有`closeFuture.setClosed`，实际会执行`object.notifyAll`。在此处，closeFuture就不是一个操作的结果，或者说，本来应该有个`CloseFuture close()`，而AbstractChannel为省事，直接弄成内部成员了，这样CloseFuture也就不用外部专门定义了。

还有一种用来做标识的，比如AbstractChannel中的connectPromise成员，会暂存`connect(
                final SocketAddress remoteAddress, final SocketAddress localAddress, final ChannelPromise promise)`传进来的promise，当connect第二次被触发时，如果connectPromise成员不为空，标识已经连过了。

## netty promise 执行listener

netty中的promise通常由eventloop创建，也就是promise通常会绑定executor。为何呢？因为netty保证listener的执行，一定是在channel对应的eventloop中。


