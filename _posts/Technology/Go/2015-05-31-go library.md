---

layout: post
title: Go 常用的一些库
category: 技术
tags: Go
keywords: Go

---

## 一 前言

本文主要阐述一下golang中常用的库。

## 二 日志
golang中涉及到日志的库有很多，除了golang自带的log外，还有glog和log4go等，不过由于bug、更新缓慢和功能不强等原因，笔者推荐使用seelog。

### 2.1 安装

    $ go get github.com/cihub/seelog
    
### 2.1 使用

    package main
    import (
    	"fmt"
    	log "github.com/cihub/seelog"
    )
    func main() {
        // xml格式的字符串，xml配置了如何输出日志
    	testConfig := `
    <seelog type="sync">
        // 配置输出项，本例表示同时输出到控制台和日志文件中
    	<outputs formatid="main">
    		<console/>
    		<file path="log.log"/>
    	</outputs>
    	<formats>
    	    // 日志格式，outputs中的输出项可以通过id引用该格式
    		<format id="main" format="%Date/%Time [%LEV] %Msg%n"/>
    	</formats>
    </seelog>
    `   // 根据配置信息生成logger（应该是配置信息的对象表示）
    	logger, err := log.LoggerFromConfigAsBytes([]byte(testConfig))
    	if err != nil {
    		fmt.Println(err)
    	}
    	// 应该是配置日志组件加载配置信息，然后输出函数比如Info在输出时，会加载该配置
    	log.ReplaceLogger(logger)
    	log.Info("Hello from Seelog!")
    }

## 三 json与结构体的转换