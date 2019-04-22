---

layout: post
title: Jedis源码分析
category: 技术
tags: Java
keywords: jedis,spring-data-redis

---

## 简介（待整理）

* TOC
{:toc}

[Intro to Jedis – the Java Redis Client Library](https://www.baeldung.com/jedis-java-redis-client-library)

## 简单实现

![](/public/upload/java/jedis_class_diagram.png)

从下到上，是一个复杂功能需求如何被逐步分解的过程，比如JedisCommands 的`void set(String key,String value)` 和BinaryJedisCommands 的`void set(byte[] key,byte[] value)`

在网络通信层面，jedis与其它rpc组件是一样一样的

![](/public/upload/java/jedis_sequence_diagram.png)


jedis协议支持的操作，称为Command，反映在代码中，抽象出了一系列的Command接口，负责不同的操作。

- BasicCommands，比如ping，save，bgsave等
- BinaryJedisCommands，负责各种数据结构的set和get
- MultiKeyBinaryCommands，参数为多个key的命令

        Long del(String... keys);
        List<String> blpop(int timeout, String... keys);

- AdvancedJedisCommands

          List<Slowlog> slowlogGet();
          String clientList();

- 其它的Command类一般用不着

### set 命令实例

client负责数据io，command定义业务接口，jedis负责整合两者。command没有XXXCommand实现类，其最终由jedis实现，jedis实现几个Command，该jedis就支持几种操作，比如ShardedJedis就没有支持所有的Command。

    //jedis
    public String set(String key, String value) {
        this.checkIsInMultiOrPipeline();
        this.client.set(key, value);
        return this.client.getStatusCodeReply();
    }
    // Client
    public void set(String key, String value) {
        this.set(SafeEncoder.encode(key), SafeEncoder.encode(value));
    }
    // BinaryClient
    public void set(byte[] key, byte[] value, byte[] nxxx, byte[] expx, long time) {
        this.sendCommand(Command.SET, new byte[][]{key, value, nxxx, expx, Protocol.toByteArray(time)});
    }
    Connection sendCommand(Command cmd, byte[]... args) {
        try {
            this.connect();
            // 协议层
            Protocol.sendCommand(this.outputStream, cmd, args);
            ++this.pipelinedCommands;
            return this;
        }catch(Exception e){...}
    }
    // Protocol，提供了一些额外的write方法，将command变成符合redis协议的二进制数据，并发送
    private static void sendCommand(RedisOutputStream os, byte[] command, byte[]... args){        
        try {
            os.write((byte)42);
            os.writeIntCrLf(args.length + 1);
            os.write((byte)36);
            os.writeIntCrLf(command.length);
            os.write(command);
            os.writeCrLf();
            byte[][] e = args;
            int len$ = args.length;
            for(int i$ = 0; i$ < len$; ++i$) {
                byte[] arg = e[i$];
                os.write((byte)36);
                os.writeIntCrLf(arg.length);
                os.write(arg);
                os.writeCrLf();
            }

        } catch (IOException var7) {
            throw new JedisConnectionException(var7);
        }
    }
    
整个代码看下来，真是太流畅了，类似的client-server工具程序可以借鉴下。

## shard jedis 实现

![](/public/upload/java/sharded_jedis_class_diagram.png)


这里主要用到了一致性哈希，完成`key ==> 虚拟节点 ==> 实际节点`的映射

涉及到的类: ShardedJedis extends BinaryShardedJedis，Sharded，ShardInfo

    // ShardedJedis
    public String set(String key, String value) {
        Jedis j = this.getShard(key);
        return j.set(key, value);
    }
    // Sharded，先在TreeMap中找到对应key所对应的ShardInfo，然后通过ShardInfo在LinkedHashMap中找到对应的Jedis实例。
    public R getShard(byte[] key) {
        return this.resources.get(this.getShardInfo(key));
    }

    Sharded<R, S extends ShardInfo<R>> {
        private TreeMap<Long, S> nodes;                  hash(虚拟shardinfo)与shardinfo的映射
        private final Hashing algo;    					 // 哈希算法
        private final Map<ShardInfo<R>, R> resources;    // shardInfo与Jedis的映射
    }


虚拟节点的生成以及与实际节点的映射在系统初始化时完成，至于key到虚拟节点的映射：treemap.tailMap(hash(key))，会返回所有大于hash(key)的key-value,选择第一个即可。

## Pipeline

pipeline的简单实例

    Jedis jedis = new Jedis(host, port);
    jedis.auth(password);
    Pipeline p = jedis.pipelined();
    p.get("1");
    p.get("2");
    p.get("3");
    List<Object> responses = p.syncAndReturnAll();
    System.out.println(responses.get(0));
    System.out.println(responses.get(1));
    System.out.println(responses.get(2));

![](/public/upload/java/pipeline_class_diagram.png)

提供Queable接口，负责暂存响应结果

    Queable {
        private Queue<Response<?>> pipelinedResponses =  new LinkedList<Response<?>>();
    }

将model包装成response，然后提供从list上加入和取出response的操作接口

Response

    Response<T> {
    	T response = null;
    	JedisDataException exception = null;
    	boolean building = false;
    	boolean built = false;
    	boolean set = false;
    	Builder<T> builder;
    	Object data;
    	Response<?> dependency = null; 
    }

pipeline包含client成员，因此具备数据的收发能力，但在收发数据的逻辑上与jedis的不同。pipeline的基本原理是：在pipeline中的请求，会直接发出去，同时加一个response进入list（相当于约定好返回结果存这里）。网络通信嘛，返回的结果本质上是inputstream。等到syncAndReturnAll的时候集中解析inputstream。因为redis server端是单线程处理的，所以也不用担心get("2")的结果跑在get("1")的前面。

    public List<Object> syncAndReturnAll() {
        if (getPipelinedResponseLength() > 0) {
            List<Object> unformatted = client.getMany(getPipelinedResponseLength());
            List<Object> formatted = new ArrayList<Object>();
            for (Object o : unformatted) {
                try {
                    formatted.add(generateResponse(o).get());
                } catch (JedisDataException e) {
                    formatted.add(e);
                }
            }
            return formatted;
        } else {
            return java.util.Collections.<Object> emptyList();
        }
    }

## spring-data-redis实现

[Spring Data Redis](https://spring.io/projects/spring-data-redis#overview) 源代码 [spring-projects/spring-data-redis](https://github.com/spring-projects/spring-data-redis)示例代码

    ListOperations<String, Person> listOps = template.listOps();
    listOps.rightPush(random, new Person("Jane", "Smith"));
    List<Person> peopleOnSecondFloor = listOps.range("users:floor:2", 0, -1);

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

## pool实现

基于common pool2实现

## redis的其它应用

http://www.blogjava.net/masfay/archive/2012/07/03/382080.html

1. pipeline
2. 跨jvm的id生成器 
3. 跨jvm的锁实现
