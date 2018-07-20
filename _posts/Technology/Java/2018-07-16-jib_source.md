---

layout: post
title: jib 源码分析
category: 技术
tags: Java
keywords: jib

---

## 简介

## maven 插件的 基本抽象

[博客园首页联系订阅管理
随笔 - 90  文章 - 0  评论 - 234
Maven提高篇系列之（六）——编写自己的Plugin](http://www.cnblogs.com/davenkin/p/advanced-maven-write-your-own-plugin.html)

## 主干代码

`mvn compile jib:build` 触发 BuildImageMojo execute 方法执行

中间有一个分步执行 框架

BuildStepsRunner 包括一个BuildSteps 属性，外界通过  `BuildStepsRunner.build` ==> `BuildSteps.run` 触发build 过程的执行

  	BuildStepsRunner(BuildSteps buildSteps) {
    	this.buildSteps = buildSteps;
  	}
  	public void build(HelpfulSuggestions helpfulSuggestions){
  	
  	}

BuildSteps  主要是 触发 `StepsRunnerConsumer.accept` ==> StepsRunner 的执行序列
  	
	private BuildSteps(
		String description,
		BuildConfiguration buildConfiguration,
		SourceFilesConfiguration sourceFilesConfiguration,
		Caches.Initializer cachesInitializer,
		String startupMessage,
		String successMessage,
		StepsRunnerConsumer stepsRunnerConsumer){
			...
		}
		
StepsRunner 是有一个基本的构造函数之后

	class StepsRunner{
		private 各种Step
		private 基本属性(主要是上下文参数)
    	public StepsRunner(
      		BuildConfiguration buildConfiguration,
      		SourceFilesConfiguration sourceFilesConfiguration,
      		Cache baseLayersCache,
      		Cache applicationLayersCache) {
      			...
      		}
    }

执行runxxx 方法，runxx方法 构造一个对应的xxxStep 赋给  StepsRunner 对应xxStep 成员。new xxStep 时， 构造方法中触发了xxStep 逻辑的异步执行    	

	stepsRunner
        .runRetrieveTargetRegistryCredentialsStep()
        .runAuthenticatePushStep()
        .runPullBaseImageStep()
        .runPullAndCacheBaseImageLayersStep()
        .runPushBaseImageLayersStep()
        .runBuildAndCacheApplicationLayerSteps()
        .runBuildImageStep(getEntrypoint(buildConfiguration, sourceFilesConfiguration))
        .runPushContainerConfigurationStep()
        .runPushApplicationLayersStep()
        .runFinalizingPushStep()
        .runPushImageStep()
        .waitOnPushImageStep()
        
        
 每一个AsyncStep 的大致组成是
 
 	class xxStep implements AsyncStep<Void>, Callable<Void>{
 		private 完成本Step所需基本属性
 		private 依赖Step
 		private final ListenableFuture<Void> listenableFuture;
 		xxStep(基本属性,依赖Step){
	 		赋值code
	 		// 依赖任务执行完毕后，执行本Step 的call 方法
	 		listenableFuture = Futures.whenAllSucceed(
	                依赖Step.getFuture(),
	                依赖Step.getFuture())
	            .call(this, listeningExecutorService);
 		}
 	}
 	
 
最优意思的部分就是， 本来十几个step 具有复杂的依赖关系，有的需要同步执行，有的可以异步执行。而通过代码的腾挪， 表面调用起来却是平铺直叙的。
 
 从另一个角度说，代码调用可以是顺序的，但业务不是顺序的。
      	