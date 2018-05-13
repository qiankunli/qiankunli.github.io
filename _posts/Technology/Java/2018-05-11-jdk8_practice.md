---

layout: post
title: java8 实践
category: 技术
tags: Java
keywords: java forkjoin

---

## 简介(未完成)

## optional

假设有一个接口`User queryByName(String name)`，name 可能不存在，有以下改进点

1. `Optional<User> queryByName(String name)`
2. 从使用角度看，假设要获取User 的 age，仍需要

		Optional<User> user = queryByName("zhangsan");
		if(user.isPresent()){
			long uid = user.get().getId();
			business code
		}
3. 一行解决问题的办法是`long uid = queryByName("zhangsan").map(User::getId).orElse(0l)`
		
stream map 之后是stream，optional map 之后是optional ，不要彼此相互干扰