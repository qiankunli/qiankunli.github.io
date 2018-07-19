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


##  filter

假设对一个用户列表，先找到所有的成年人。

`users.stream().filter(user -> user.getAge > 18).collect(Collectors.toList())`

但若是还有一个需求，对18岁以下的人打个日志，以方便debug， 一般会

	users.stream().filter(user -> {
		if(user.getAge < 18){
			log...
			return false;
		}
		return true;
	}).collect(Collectors.toList());
	
这样就破坏了stream 的美感，同时实际需求 通常比这个更复杂，因此可以

	class Logable{
		private User user;
		private Logable(User user){
			this.user = user;
		}
		public static Logable of(User user){
			return new Logable(user);
		}
		public boolean isGreaterThan18(){
			if(user.getAge < 18){
				log...
				return false;
			}
			return true;
		}
	}

然后`users.stream().map(Logable::of).filter(logable:: isGreaterThan18).collect(Collectors.toList())`

这个可以称之为，给对象加一个方法。如果加上观察者，就可以演化为rxjava。
