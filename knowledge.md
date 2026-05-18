---
layout: page
title: 知识地图
permalink: /knowledge/
---

<section class="knowledge-hero">
  <p class="knowledge-hero__eyebrow">Knowledge Map</p>
  <h1>知识资产总览</h1>
  <p class="knowledge-hero__lead">
    这里按长期主题而不是按发布时间组织内容。它更像一张主题地图，用来查看我已经沉淀了哪些知识主线，以及哪些主题仍在持续演进。
  </p>
</section>

<div class="knowledge-grid">
  <section class="knowledge-card">
    <header class="knowledge-card__header">
      <h2>MachineLearning</h2>
      <p class="knowledge-card__meta">107+ posts · LLM / Agent / Infra</p>
    </header>

    <p>
      围绕 LLM、Agent、训练推理基础设施、评测与工程化的长期研究线，是当前最核心的知识资产主干。
    </p>

    <h3>代表文章</h3>
    <ul>
      <li><a href="{% post_url MachineLearning/Infra/2025-06-21-llm_observability %}">大模型可观测性</a></li>
      <li><a href="{% post_url MachineLearning/Infra/2025-07-20-llm_infra %}">大模型infra综述</a></li>
      <li><a href="{% post_url MachineLearning/Agent/2023-10-30-llm_agent %}">增强型LLM——Agent</a></li>
      <li><a href="{% post_url MachineLearning/Agent/2024-11-20-llm_memory %}">上下文记忆——AI Agent native 的任务存储机制</a></li>
      <li><a href="{% post_url MachineLearning/Agent/2025-04-26-agent_framework %}">agent框架</a></li>
      <li><a href="{% post_url MachineLearning/Agent/2025-07-19-context_engineer %}">提升Agent能力——上下文工程</a></li>
      <li><a href="{% post_url MachineLearning/LLM/2025-06-28-llm_evaluation %}">llm评测</a></li>
    </ul>

    <h3>持续演进中</h3>
    <ul class="knowledge-tags">
      <li>llm-observability</li>
      <li>agent-observability</li>
      <li>context-engineering</li>
      <li>agent-evaluation</li>
      <li>code-agent</li>
    </ul>
  </section>

  <section class="knowledge-card">
    <header class="knowledge-card__header">
      <h2>Kubernetes</h2>
      <p class="knowledge-card__meta">87+ posts · Controller / Scheduler / Platform</p>
    </header>

    <p>
      围绕容器编排、控制面、调度、扩展机制以及平台化实践的长期积累，是系统工程视角下最完整的一条知识主线之一。
    </p>

    <h3>代表文章</h3>
    <ul>
      <li><a href="{% post_url Kubernetes/Controller/2019-03-07-kubernetes_controller %}">Kubernetes 控制器模型</a></li>
      <li><a href="{% post_url Kubernetes/Scheduler/2019-03-03-kubernetes_scheduler %}">Kubernetes资源调度——scheduler</a></li>
      <li><a href="{% post_url Kubernetes/Kubelet/2020-06-24-create_pod %}">Pod是如何被创建出来的？</a></li>
      <li><a href="{% post_url Kubernetes/Controller/2020-08-10-controller_runtime %}">controller-runtime源码分析</a></li>
      <li><a href="{% post_url Kubernetes/Extension/2019-10-21-kubernetes_operator %}">kubernetes operator</a></li>
      <li><a href="{% post_url Kubernetes/Tools/2022-04-08-kustomize_helm %}">K8S YAML 资源清单管理方案</a></li>
    </ul>

    <h3>持续演进中</h3>
    <ul class="knowledge-tags">
      <li>controller-runtime</li>
      <li>scheduler</li>
      <li>operator</li>
      <li>application-platform</li>
      <li>multi-cluster</li>
    </ul>
  </section>

  <section class="knowledge-card">
    <header class="knowledge-card__header">
      <h2>Technology</h2>
      <p class="knowledge-card__meta">306+ posts · Architecture / Go / Storage / Concurrency</p>
    </header>

    <p>
      围绕通用软件工程、架构设计、编程语言与基础系统问题的长期知识资产，是很多业务与平台实践背后的方法论底盘。
    </p>

    <h3>代表文章</h3>
    <ul>
      <li><a href="{% post_url Technology/Architecture/2018-09-28-business_system_design %}">业务系统设计的一些体会</a></li>
      <li><a href="{% post_url Technology/Architecture/2018-10-01-object_oriented %}">重新看面向对象设计</a></li>
      <li><a href="{% post_url Technology/Go/2017-02-04-go_concurrence %}">Go并发机制及语言层工具</a></li>
      <li><a href="{% post_url Technology/Go/2015-04-29-goroutine_scheduler_1 %}">Goroutine 调度模型</a></li>
      <li><a href="{% post_url Technology/Storage/2016-09-21-db %}">数据库的一些知识</a></li>
      <li><a href="{% post_url Technology/Concurrency/2017-09-05-learn_concurrency %}">学习并发</a></li>
    </ul>

    <h3>持续演进中</h3>
    <ul class="knowledge-tags">
      <li>architecture</li>
      <li>go-engineering</li>
      <li>storage</li>
      <li>distributed-systems</li>
      <li>observability</li>
    </ul>
  </section>

  <section class="knowledge-card">
    <header class="knowledge-card__header">
      <h2>Life</h2>
      <p class="knowledge-card__meta">41+ posts · Cognition / Management / Growth</p>
    </header>

    <p>
      围绕认知、技术管理、方法论与个人成长的长期思考，是技术主线之外持续沉淀的另一组核心资产。
    </p>

    <h3>代表文章</h3>
    <ul>
      <li><a href="{% post_url Life/2018-09-04-technology_manage %}">尝试带好一个小团队</a></li>
      <li><a href="{% post_url Life/2018-11-05-cognition %}">认知的几点规律</a></li>
      <li><a href="{% post_url Life/2018-11-28-thinking %}">一切瓶颈都是思维瓶颈</a></li>
      <li><a href="{% post_url Life/2019-07-16-dart_time_note %}">《暗时间》笔记</a></li>
      <li><a href="{% post_url Life/2018-11-13-technical_leadership_note_1 %}">《技术领导力300讲》笔记</a></li>
      <li><a href="{% post_url Life/2021-09-27-technology_life %}">《阿里技术人生》系列小结</a></li>
    </ul>

    <h3>持续演进中</h3>
    <ul class="knowledge-tags">
      <li>认知</li>
      <li>技术管理</li>
      <li>成长方法</li>
      <li>长期主义</li>
    </ul>
  </section>

  <section class="knowledge-card">
    <header class="knowledge-card__header">
      <h2>Product</h2>
      <p class="knowledge-card__meta">3 posts · Product Thinking / Growth</p>
    </header>

    <p>
      围绕产品理解与业务视角的补充性主题，规模不大，但对技术与业务协同的理解有明显补强作用。
    </p>

    <h3>代表文章</h3>
    <ul>
      <li><a href="{% post_url Product/2019-01-23-product_thinking %}">技术人员的产品思维</a></li>
      <li><a href="{% post_url Product/2019-10-22-behind_product %}">《幕后产品》笔记</a></li>
      <li><a href="{% post_url Product/2020-06-14-user_growth_note %}">《用户增长》笔记</a></li>
    </ul>

    <h3>持续演进中</h3>
    <ul class="knowledge-tags">
      <li>product-thinking</li>
      <li>user-growth</li>
      <li>business-awareness</li>
    </ul>
  </section>
</div>
