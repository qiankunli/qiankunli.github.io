---

layout: post
title: 让kafka支持优先级队列
category: 技术
tags: Java
keywords: kafka

---

## 简介（未完成）


kafka 官方需求  Kafka Improvement Proposals



[KIP-349: Priorities for Source Topics](https://cwiki.apache.org/confluence/display/KAFKA/KIP-349%3A+Priorities+for+Source+Topics)


## 内部使用优先级队列重新缓冲

## 从高优先级到低优先级依次拉取

[flipkart-incubator/priority-kafka-client](https://github.com/flipkart-incubator/priority-kafka-client)

![](/public/upload/java/priority_kafka_producer_class_diagram.png)

![](/public/upload/java/priority_kafka_consumer_class_diagram.png)

背景

![](/public/upload/java/priority_kafka_subscribe.png)

1. 优先级以数字表示，值越高优先级越高
2. 一个topic 对应有一个优先级，对应一个consumer

### 循环拉取

    // 同事维护多个consumer
    private Map<Integer, KafkaConsumer<K, V>> consumers;
    public ConsumerRecords<K, V> poll(long pollTimeoutMs) {
        Map<TopicPartition, List<ConsumerRecord<K, V>>> consumerRecords = 
            new HashMap<TopicPartition, List<ConsumerRecord<K, V>>>();
        long start = System.currentTimeMillis();
        do {
            for (int i = maxPriority - 1; i >= 0; --i) {
                ConsumerRecords<K, V> records = consumers.get(i).poll(0);
                for (TopicPartition partition : records.partitions()) {
                    consumerRecords.put(partition, records.records(partition));
                }
            }
        } while (consumerRecords.isEmpty() && System.currentTimeMillis() < (start + pollTimeoutMs));
    }

从上述代码可以看到， 单纯的循环拉取，只是做到了局部的先消费优先级搞的，再消费优先级低的。但还是无法做到，当高中低都有消息待消费时，集中全力先消费高优先级的消息。


CapacityBurstPriorityKafkaConsumer 的类描述

1. regulates capacity (record processing rate) across priority levels. 跨优先级管控 记录消费速度
2. When we discuss about priority topic XYZ and consumer group ABC, here XYZ and ABC are the logical names. For every logical priority topic XYZ one must define max supported priority level via the config ClientConfigs#MAX_PRIORITY_CONFIG property.**This property is used as the capacity lever（杠杆） for tuning processing rate across priorities**. Every object of the class maintains KafkaConsumer instance for every priority level topic [0, ${max.priority - 1}].

### 一次拉取多少


