---

layout: post
title: microduck 学习
category: 技术
tags: MachineLearning
keywords: deepresearch deepsearch

---

<script>
  MathJax = {
    tex: {
      inlineMath: [['$', '$']], // 支持 $和$$ 作为行内公式分隔符
      displayMath: [['$$', '$$']], // 块级公式分隔符
    },
    svg: {
      fontCache: 'global'
    }
  };
</script>
<script async src="/public/js/mathjax/es5/tex-mml-chtml.js"></script>

* TOC
{:toc}

## 简介
Microduck 是一台装在双足身体里的小电脑，连接着 15 个智能舵机、姿态传感器和摄像头。
## body

身体结构：两条腿，加上能活动的脖子、头和嘴**，分成 15 个可控制的关节

| 身体部位 | 关节 | 数量 | 直观作用 |
|---|---|---:|---|
| 左腿 | hip_yaw、hip_roll、hip_pitch、knee、ankle | 5 | 转向、侧摆、前后摆腿、屈膝和调整脚部 |
| 右腿 | 同上 | 5 | 与左腿配合支撑、迈步 |
| 脖子和头 | neck_pitch、head_pitch、head_yaw、head_roll | 4 | 调整头的位置与朝向 |
| 嘴 | mouth | 1 | 张合嘴部 |

其中，`yaw` 是左右转向，`pitch` 是前后俯仰，`roll` 是侧倾。





## 系统

对应的onboard computer是 Radxa Zero 3W，采用 Rockchip RK3566 SoC。它承担运行程序、读取状态、执行策略推理和发送目标等工作。

## 策略推理

## 策略训练