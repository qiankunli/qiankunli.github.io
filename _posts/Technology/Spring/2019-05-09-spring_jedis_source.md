---

layout: post
title: spring redis 源码分析
category: 技术
tags: Spring
keywords: kafka

---

## 简介

[Spring Data Redis](https://spring.io/projects/spring-data-redis#overview) 源代码 [spring-projects/spring-data-redis](https://github.com/spring-projects/spring-data-redis)示例代码

    ListOperations<String, Person> listOps = template.listOps();
    listOps.rightPush(random, new Person("Jane", "Smith"));
    List<Person> peopleOnSecondFloor = listOps.range("users:floor:2", 0, -1);

背景 [Jedis源码分析](http://qiankunli.github.io/2016/06/07/jedis_source.html)

## spring-data-redis实现

下文仅以redis 五种类型的 String和List 数据结构为例：

![](/public/upload/java/sdr_redisTemplate_diagram.png)

RedisTemplate 和ValueOperations、ListOperation 互相持有对方的引用， RedisTemplate 作为用户操作的入口对象， ValueOperations、ListOperation 等负责分担五种数据类型的 操作。

![](/public/upload/java/sdr_redisConnection_diagram.png)

1. 数据操作最终由 JedisConnection 来完成
2. 从spring-data-redis 实现看，spring-data-redis 只是将jedis 作为 redis 访问的工具之一，并没有严格绑定
3. JedisConnection 与 JedisStringCommands、JedisListCommands 等互相持有对方的引用，JedisStringCommands、JedisListCommands 等负责分担五种数据类型的 操作。

![](/public/upload/java/sdr_sequence_diagram.png)

	public <T> T execute(RedisCallback<T> action, boolean exposeConnection, boolean pipeline) {
		RedisConnectionFactory factory = getRequiredConnectionFactory();
		RedisConnection conn = null;
		try {
			if (enableTransactionSupport) {
				conn = RedisConnectionUtils.bindConnection(factory, enableTransactionSupport);
			} else {
				conn = RedisConnectionUtils.getConnection(factory);
			}
			boolean existingConnection = TransactionSynchronizationManager.hasResource(factory);
			RedisConnection connToUse = preProcessConnection(conn, existingConnection);
			boolean pipelineStatus = connToUse.isPipelined();
			if (pipeline && !pipelineStatus) {
				connToUse.openPipeline();
			}
			RedisConnection connToExpose = (exposeConnection ? connToUse : createRedisConnectionProxy(connToUse));
			T result = action.doInRedis(connToExpose);
			if (pipeline && !pipelineStatus) {
				connToUse.closePipeline();
			}
			return postProcessResult(result, connToUse, existingConnection);
		} finally {
			RedisConnectionUtils.releaseConnection(conn, factory);
		}
	}

RedisTemplate使用模板模式，提供了高层调用结构和基本的逻辑实现（execute逻辑就是：建连接，序列化，发请求数据，拿数据，返回）。ValueOperations、ListOperation 对应的命令由它们自己实现，双方的结合点就是callback。这和spring-jdbc中的JdbcTemplate非常相像。

## ScriptingCommands

redis中提供对lua脚本的支持，jedis和sdr自然也不甘落后，也都提供了支持。

反应在jedis上，就是ScriptingCommands接口，从中可以看出一种感觉，eval命令和set命令并无本质不同，在实现上都是完成参数的传递即可，并没有因为其功能不同，有特别的处理。

    public interface ScriptingCommands {
    	// 以eval方式执行lua脚本
        Object eval(String script, int keyCount, String... params);
        Object eval(String script, List<String> keys, List<String> args);
        Object eval(String script);
        // 以evalsha方式执行lua脚本
        Object evalsha(String sha1);
        Object evalsha(String sha1, List<String> keys, List<String> args);
        Object evalsha(String sha1, int keyCount, String... params);
        // 判断lua脚本是否存在
        Boolean scriptExists(String sha1);
        List<Boolean> scriptExists(String... sha1);
        // 加载lua脚本
        String scriptLoad(String script);
    }

