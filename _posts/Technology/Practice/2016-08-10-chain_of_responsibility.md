---

layout: post
title: 链式处理的那些套路
category: 技术
tags: Practice
keywords: commons-chain

---

## 简介

* TOC
{:toc}

几个基本问题

1. 事先构造好每个步骤之间的顺序，步骤之间如何连接？数组、链表。
2. 每个步骤是否可以异步执行， 若是异步执行， 一个步骤完成后如何通知下一个步骤
3. 中间某个步骤执行异常时，如何处理？

## 用责任链模式来简化web开发逻辑

The essence of computing might be that for any expected input (A), we return the expected output (B). The challenge is getting from (A) to (B). For a simple program, (A) to (B) might be a single transformation. Say, shifting a character code 32 digits so that "a" becomes "A". In a complex application, A to B can be a long and winding road.

To be useful, most applications need to run a process and then tell the client what happened. In practice, we find mixing "running" and "telling" together creates code that can be hard to test and maintain. If we can have one component run (or execute) the process, and another component report (or present) the result, then we can test, create, and maintain each component separately. But, how can we cleanly separate the execution and presentation layers without complicating the design of an application?

Most application frameworks, especially web application frameworks, rely on the Command pattern. An incoming HTTP request is mapped to some type of "command" object. The command object takes whatever action is required, using information passed in the HTTP request.

大部分application frameworks使用的是命令模式，这里举HTTP的例子不太恰当，观察一下FTP的java实现，可以清晰的发现其完全基于命令模式。

In practice, there are usually commands within commands. A Command object in a web application often looks like a sandwich. First, it does some things for the benefit of the presentation layer, then it executes the business logic, and then it does some more presentation layer things. The problem many developers face is how to cleanly separate the business logic in the middle of a web command from other necessary tasks that are part of the request/response transaction.

在实践中，一个Command类通常很复杂，贯穿显示层、业务逻辑层和其它必要的任务。我们面临的问题是，将位于中间位置的业务逻辑代码从request/response的组装和写出中隔离出来。

The Chain of Responsibility package combines the Command pattern with the classic Chain of Responsibility pattern（将命令模式与经典的责任链模式结合起来） to make it easy to call a business command as part of a larger application command.

## commons-chain

http://commons.apache.org/proper/commons-chain/cookbook.html

![](/public/upload/java/commons_chain_object.png)

Context. A Context represents the state of an application. In the Chain package, Context is a **marker interface for a java.util.Map**(Map-style context). The Context is an envelope containing the attributes needed to complete a transaction. In other words, **a Context is a stateful object** with member values.

Command. A Command represents a unit of work. A Command has a single entry method: public boolean execute(Context context). **A Command acts upon the state passed to it through a context object, but retains no state of its own.**（一个command在一个传给它的state上工作，但自身是无状态的） Commands may be assembled into a Chain, so that a complex transaction can be created from discrete（离散的、不连续的） units of work. 

1. If a Command returns true, then other Commands in a Chain should not be executed. 
2. If a Command returns false, then other Commands in the Chain (if any) may execute.

Chain.Chain implements the Command interface, so a Chain can be used interchangeably（可交换的） with a Command. An application doesn't need to know if it's calling a Chain or a Command, so you can refactor from one to the other. A Chain can nest other Chains as desired. This property is known as the Liskov substitution principle.

Filter. Ideally, every command would be an island. In real life, we sometimes need to allocate resources and be assured the resources will be released no matter what happens. **A Filter is a specialized Command that adds a postProcess method.** A Chain is expected to call the postProcess method of any filters in the chain before returning. A Command that implements Filter can safely release any resources it allocated through the postProcess method, even if those resources are shared with other Commands.

Catalog. Many applications use "facades" and "factories" and other techniques to avoid binding layers too closely together. **Layers need to interact, but often we don't want them to interact at the classname level.** A Catalog is a collection of logically named Commands (or Chains) that a client can execute, without knowing the Command's classname. PS：**Catalog 这里起到的就是一个spring ioc容器的作用，根据name 而不是class name依赖**。

### 示例代码

    public class TestChain {
        @Test
        public void test() throws Exception {
            Chain chain = new ChainBase();
            chain.addCommand(new AddCommand());
            chain.addCommand(new LogCommand());
            chain.execute(new ContextBase());
        }
    }

### ChainBase

维护一个Command数组，触发Command执行。责任链模式的经典实现，不再赘述。

比较有意思的是，Command完全执行完毕后**或抛出异常时**，触发Filter（跟Web中的Filter不同）执行，一般用来释放资源。

### contextBase

ContextBase,In addition to the minimal functionality required by the Context interface, this class implements the recommended support for **Attribute-Property Transparency**. This is implemented by analyzing the available JavaBeans properties of this class (or its subclass), exposes them as key-value pairs in the Map,with the key being the name of the property itself.

举个例子

    public class UserContext extends ContextBase{
        private String name;
        public String getName() {
            return name;
        }
        public void setName(String name) {
            this.name = name;
        }
    }
    
    public class TestUserContext {
        @Test
        public void testGet(){
            UserContext context = new UserContext();
            context.setName("abc");
            System.out.println(context.get("name"));
        }
	}
    
此处，我们setName时，name ==> abc 就加入到了map中。

简单说就是，context有很多种风格，Map-Style或者JavaBean，ContextBase将两者结合起来，在predefine一些属性的同时，保留了map的扩展性。

### 其它

commons-chain中还提供了许多web工具类，用来与Web开发集成。

通过commons-digester与commons-chain结合，可以将Chain的生成配置化。参见[commons-chain 应用记录](http://coffeelover.iteye.com/blog/710615)

## commons-pipeline（未完成）

[Apache Commons Pipeline 使用学习（一）](http://caoyaojun1988-163-com.iteye.com/blog/2124833)

几个问题：

1. 为何线程 模型能够随意的更改？
2. 如果 想屏蔽 同步与异步的不同，应该如何改造？

### demo 代码

	Pipeline pipeline = createPipeline();
	pipeline.getSourceFeeder().feed("Hello world! Remove 10 unnecessary 9 punctuations. ! ! ");
	pipeline.run();
	
	
    private static Pipeline createPipeline() throws ValidationException {
        Stage tokenizer = new Tokenizer();
        Stage caseCorrector = new CaseCorrector();
        Stage punctuationCorrector = new PunctuationCorrector();
        StageDriverFactory sdf = new SynchronousStageDriverFactory();
        Pipeline instance = new Pipeline();
        instance.addStage(tokenizer, sdf);
        instance.addStage(punctuationCorrector, sdf);
        instance.addStage(caseCorrector, sdf);
        return instance;
    }


### 基本原理

Pipeline

	几个重要方法，run、finish
PipelineLifecycleJob  run 和finish 方法的 单纯作为回调接口，用于上层 跟踪Pipeline 的生命周期
Stage
Feeder

	Pipeline.run 方法
    public void run() {
        try {
            start();
            finish();
        } catch (StageException e) {
            throw new RuntimeException(e);
        }
    }
    
    start 逻辑
   
  for (PipelineLifecycleJob job : lifecycleJobs) job.onStart(this);
        for (StageDriver driver: this.drivers) driver.start();
        for (Pipeline branch : branches.values()) branch.start();
        
## netty中的pipeline

![](/public/upload/java/netty_pipeline_object.png)

1. ChannelHandler. A ChannelHandler is provided with a ChannelHandlerContext object. A ChannelHandler is supposed to interact with the ChannelPipeline it belongs to via a context object. Using the context object, the ChannelHandler can pass events upstream or downstream, modify the pipeline dynamically, or store the information (using AttributeKeys) which is specific to the handler.
2. ChannelHandlerContext. Enables a ChannelHandler to interact with its ChannelPipeline and other handlers. A handler can notify the next ChannelHandler in the ChannelPipeline, modify the ChannelPipeline it belongs to dynamically.

![](/public/upload/java/netty_pipeline_diagram.png)

可以看到ChannelHandler 基本就只管业务处理了， ChannelHandler逻辑处理完，进行下一步工作，都是`ctx.writexxx,ctx.firexxx` 来进行



