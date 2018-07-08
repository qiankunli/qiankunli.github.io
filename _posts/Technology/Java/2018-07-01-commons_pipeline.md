---

layout: post
title: commons-pipeline 源码分析
category: 技术
tags: Java
keywords: commons-pipeline

---

## 简介（未完成）

[Apache Commons Pipeline 使用学习（一）](http://caoyaojun1988-163-com.iteye.com/blog/2124833)

几个问题：

1. 为何线程 模型能够随意的更改？
2. 如果 想屏蔽 同步与异步的不同，应该如何改造？

## demo 代码

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


## 基本原理

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
        
        
        
      

## 线程实现

