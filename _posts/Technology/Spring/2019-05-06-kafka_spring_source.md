---

layout: post
title: spring kafka 源码分析
category: 技术
tags: Spring
keywords: kafka

---

## 简介（未完成）


源码地址 [spring-projects/spring-kafka](https://github.com/spring-projects/spring-kafka)

官网地址 [Spring for Apache Kafka](https://spring.io/projects/spring-kafka)

spring 对框架的封装 套路很一致，参见 [spring redis 源码分析](http://qiankunli.github.io/2019/05/09/spring_jedis_source.html) 以及 spring 对rabbitmq 的代码。

问题

1. 官方如何多线程消费consumer

## 代码示例

kafka-producer.xml

    <!--基本配置 -->
    <bean id="producerProperties" class="java.util.HashMap">
        <constructor-arg>
            <map>
                <!-- kafka服务地址，可能是集群-->
                <entry key="bootstrap.servers" value="192.168.62.212:9092,192.168.62.213:9092,192.168.62.214:9092"/>
                <!-- 有可能导致broker接收到重复的消息,默认值为3-->
                <entry key="retries" value="10"/>
                <!-- 每次批量发送消息的数量-->
                <entry key="batch.size" value="1638"/>
                <!-- 默认0ms，在异步IO线程被触发后（任何一个topic，partition满都可以触发）-->
                <entry key="linger.ms" value="1"/>
                <!--producer可以用来缓存数据的内存大小。如果数据产生速度大于向broker发送的速度，producer会阻塞或者抛出异常 -->
                <entry key="buffer.memory" value="33554432 "/>
                <!-- producer需要server接收到数据之后发出的确认接收的信号，此项配置就是指procuder需要多少个这样的确认信号-->
                <entry key="acks" value="all"/>
                <entry key="key.serializer" value="org.apache.kafka.common.serialization.StringSerializer"/>
                <entry key="value.serializer" value="org.apache.kafka.common.serialization.StringSerializer"/>
            </map>
        </constructor-arg>
    </bean>
    <!-- 创建kafkatemplate需要使用的producerfactory bean -->
    <bean id="producerFactory"
          class="org.springframework.kafka.core.DefaultKafkaProducerFactory">
        <constructor-arg>
            <ref bean="producerProperties"/>
        </constructor-arg>
    </bean>
    <!-- 创建kafkatemplate bean，使用的时候，只需要注入这个bean，即可使用template的send消息方法 -->
    <bean id="KafkaTemplate" class="org.springframework.kafka.core.KafkaTemplate">
        <constructor-arg ref="producerFactory"/>
        <!--设置对应topic-->
        <property name="defaultTopic" value="bert"/>
    </bean>
    // 测试类
    @RunWith(SpringJUnit4ClassRunner.class)
    @ContextConfiguration(locations = "classpath:kafka-producer.xml")
    public class KafkaTemplateTest {
        @Autowired
        private KafkaTemplate<Integer, String> kafkaTemplate;
        @Test
        public void hello(){
            kafkaTemplate.sendDefault("hello world");
        }
    }


kafka-consumer.xml

    <bean id="consumerProperties" class="java.util.HashMap">
            <constructor-arg>
                <map>
                    <!--Kafka服务地址 -->
                    <entry key="bootstrap.servers" value="192.168.62.212:9092,192.168.62.213:9092,192.168.62.214:9092" />
                    <!--Consumer的组ID，相同goup.id的consumer属于同一个组。 -->
                    <entry key="group.id" value="bert.mac" />
                    <!--如果此值设置为true，consumer会周期性的把当前消费的offset值保存到zookeeper。当consumer失败重启之后将会使用此值作为新开始消费的值。 -->
                    <entry key="enable.auto.commit" value="true" />
                    <!--网络请求的socket超时时间。实际超时时间由max.fetch.wait + socket.timeout.ms 确定 -->
                    <entry key="session.timeout.ms" value="15000 " />
                    <entry key="key.deserializer"
                        value="org.apache.kafka.common.serialization.StringDeserializer" />
                    <entry key="value.deserializer"
                        value="org.apache.kafka.common.serialization.StringDeserializer" />
                </map>
            </constructor-arg>
        </bean>
        <!--指定具体监听类的bean -->
        <bean id="messageListernerConsumerService" class="com.ximalaya.queue.KafkaConsumerListener" />
        <!-- 创建consumerFactory bean -->
        <bean id="consumerFactory" class="org.springframework.kafka.core.DefaultKafkaConsumerFactory">
            <constructor-arg>
                <ref bean="consumerProperties"/>
            </constructor-arg>
        </bean>
        <bean id="containerProperties" class="org.springframework.kafka.listener.config.ContainerProperties">
            <!-- 要消费的 topic -->
            <constructor-arg value="bert"/>
            <property name="messageListener" ref="messageListernerConsumerService"/>
        </bean>
        <bean id="messageListenerContainer" class="org.springframework.kafka.listener.KafkaMessageListenerContainer" init-method="doStart">
            <constructor-arg ref="consumerFactory"/>
            <constructor-arg ref="containerProperties"/>
        </bean>

## 生产者

![](/public/upload/spring/spring_kafka_producer_class_diagram.png)

1. 相对于 [《Apache Kafka源码分析》——Producer与Consumer](http://qiankunli.github.io/2017/12/08/kafka_learn_1.html)直接使用KafkaProducer， DefaultKafkaProducerFactory 典型的工厂模式， 封装了kafka producer 配置
2. KafkaTemplate 来了一个 经典的单例模式

        public class KafkaTemplate<K, V> implements KafkaOperations<K, V> {
            private final ProducerFactory<K, V> producerFactory;
            private volatile Producer<K, V> producer;
            private Producer<K, V> getTheProducer() {
                if (this.producer == null) {
                    synchronized (this) {
                        if (this.producer == null) {
                            this.producer = this.producerFactory.createProducer();
                        }
                    }
                }
                return this.producer;
            }
        }

发送逻辑

    public ListenableFuture<SendResult<K, V>> send(String topic, V data) {
        ProducerRecord<K, V> producerRecord = new ProducerRecord<>(topic, data);
        return doSend(producerRecord);
    }
    protected ListenableFuture<SendResult<K, V>> doSend(final ProducerRecord<K, V> producerRecord) {
        getTheProducer();
        final SettableListenableFuture<SendResult<K, V>> future = new SettableListenableFuture<>();
        getTheProducer().send(producerRecord, new Callback() {
            public void onCompletion(RecordMetadata metadata, Exception exception) {
                if (exception == null) {
                    future.set(new SendResult<>(producerRecord, metadata));
                    if (KafkaTemplate.this.producerListener != null
                            && KafkaTemplate.this.producerListener.isInterestedInSuccess()) {
                        KafkaTemplate.this.producerListener.onSuccess(producerRecord.topic(),
                                producerRecord.partition(), producerRecord.key(), producerRecord.value(), metadata);
                    }
                }else {
                    future.setException(new KafkaProducerException(producerRecord, "Failed to send", exception));
                    if (KafkaTemplate.this.producerListener != null) {
                        KafkaTemplate.this.producerListener.onError(producerRecord.topic(),
                                producerRecord.partition(), producerRecord.key(), producerRecord.value(), exception);
                    }
                }
            }
        });
        if (this.autoFlush) {
            flush();
        }
        return future;
    }

1. 将KafkaProducer 的send callback 转换为ListenableFuture
2. 使用 producerListener 将“事件处理”逻辑与发送主流程解耦

## 多线程 消费

[【原创】探讨kafka的分区数与多线程消费](https://raising.iteye.com/blog/2252456) 未读