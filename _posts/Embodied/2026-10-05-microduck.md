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

| 传感器 | 提供的信息 | 对学习走路的意义 |
|---|---|---|
| 舵机内部位置传感器 | 各关节实际角度，以及相关运动反馈 | 知道腿现在弯到了哪里 |
| 躯干 IMU | 角速度、姿态等 | 知道身体正在向哪个方向倾斜、转动 |
| 头部摄像头 | 外部环境图像 | 用于视觉功能|

躯干 IMU 使用 `imu_to_dxl v2` 板，传感器为 **LSM6DSV16X**。板子把 IMU 数据通过 Dynamixel 协议提供给主控，代码据此得到角速度、姿态和身体坐标系下的重力方向。

摄像头帮助机器人看外界，关节反馈和 IMU 帮助机器人感知自己的身体。15 个舵机和 IMU 转接板共享同一条通信总线，通过设备 ID 区分。

```
                     robotd — control thread
                              │
                              │  duck_control::bus::DynamixelIo
                              │  serialport · TIOCEXCL
                              ▼
         /dev/ttyS2 · 1 Mbps · Dynamixel protocol v2
                              │
    ┌────────────┬────────────┴───────┬──────────────────┐
    │            │                    │                  │
  id 200       20–24                30–34             10–14
 imu_to_dxl   left leg        neck · head · mouth     right leg
  v2 board    5 servos             5 servos           5 servos
```

## 系统

对应的onboard computer是 Radxa Zero 3W，采用 Rockchip RK3566 SoC。它承担运行程序、读取状态、执行策略推理和发送目标等工作。

1. Armbian Linux，
2. Linux 上运行哪些进程？由 systemd 启动、重启并收集日志。
   1. robotd，运动控制的核心服务，身体状态读取、策略推理、运动控制、安全检查
   2. padd，读取游戏手柄，将操作转换为控制意图
   3. mediad，摄像头媒体管线、WebRTC 视频与控制通道，视频服务和远程接入入口
   4. btd，通过 BLE 转发部分控制、配置接口，蓝牙接入入口
   5. tofd，读取头部 ToF 深度传感器并提供数据	传感器服务
   6. configd，Wi-Fi、机器人名称、配对等配置，系统配置服务
   7. updaterd，软件更新、校验、健康检查与回滚，部署与恢复服务
3. 进程之间怎样通信？本机主要通过 Unix domain socket + JSON-RPC 通信，例如 robotd 监听：`/run/robotd.sock`。调用方提交的是“控制意图”，例如：
   1. robot.move：期望向前、侧向移动或转向。robotd 再根据身体状态和策略计算关节目标。
   2. robot.head：调整头部。
   3. robot.stop：停止运动。
   4. robot.subscribe：订阅状态。
4. robotd 内部怎样驱动身体？核心是一条 目标频率为 50 Hz，即约每 20 ms 一轮的控制循环：
    ```
    读取舵机和 IMU
      ↓
    取得当前控制意图
        ↓
    构造 Observation
        ↓
    调用 Policy，得到动作
        ↓
    转换为关节目标，执行必要的限制
        ↓
    通过 Dynamixel 总线写入目标位置
        ↓
    提供状态反馈，进入下一轮
    ```



```
手柄           手机           命令行           远程客户端
 │              │               │                 │
padd           btd           robotctl           mediad
 │              │               │                 │
 └──────────────┴───────────────┴─────────────────┘
          JSON-RPC 2.0 / Unix domain sockets
          各服务有自己的 socket，客户端直接连接
                 │
        ┌────────┼───────────┐
        ▼        ▼           ▼
      robotd   configd    updaterd
      运动控制   系统配置     软件更新
        │
   Dynamixel / UART
        │
   15 个舵机 + IMU
```


仿真（MuJoCo）如何接进来？仿真器独立运行的 robotd 协作
```
                       robotd 控制循环
                              │
                           RobotIo(硬件接口抽象)
                              │
                ┌─────────────┴──────────────┐
                │                            │
          DynamixelIo                    RemoteIo
                │                            │
          串口 / Dynamixel              TCP + JSON
                │                            │
          实体舵机 + IMU               duck-body / MuJoCo
```

真机中，`DynamixelIo` 把目标转换成 Dynamixel 指令，经 Linux 串口发给舵机；仿真中，`RemoteIo` 把目标送给 `duck-body / MuJoCo`，由物理仿真和执行器模型计算运动，再返回模拟的关节、IMU 数据。

## 策略推理

假设通过手柄要求 Microduck「以 0.2 m/s 向前走」。`robotd` 收到的是期望速度，而舵机需要的是各关节的目标位置。**Policy（策略）负责结合当前身体状态，把运动意图转换成这一轮的关节动作。** 即使期望速度没有变化，身体倾斜、关节角度发生变化时，策略给出的动作也会变化。

策略是已经训练好的神经网络，以 ONNX 文件保存，由 ONNX Runtime 执行推理。可以把它看成一个函数：`action = policy(observation)`。运行时反复调用这个函数，模型参数保持不变；训练则负责学习这些参数。板载电脑执行这里的推理，并不需要在机器人身上重新训练。

### Observation 和 Action

以当前 Microduck 的运动策略接口为例，输入是 61 个数，输出是 14 个数（MLP，输入输出定义和训练后的权重针对 Microduck）。Observation 同时包含身体反馈、上一轮动作和用户指令，顺序、单位、关节排列必须与训练时一致。

| 输入内容 | 维度 | 含义 |
|---|---:|---|
| 躯干角速度 | 3 | 身体正在怎样转动，来自 IMU |
| 身体坐标系下的重力方向 | 3 | 身体相对竖直方向怎样倾斜，由姿态换算 |
| 关节位置相对默认姿态的偏移 | 14 | 腿、脖子和头现在处于什么姿态 |
| 关节速度 | 14 | 各关节正在怎样运动 |
| 上一轮 Action | 14 | 上一轮策略输出的原始值 |
| Command | 13 | 期望前进、侧移、转向速度，以及头部和身体姿态指令 |

这里的行走策略使用关节反馈和 IMU，不读取摄像头图像。14 个输出分别对应两条腿的 10 个关节，以及脖子、头部的 4 个关节；嘴部由另外的逻辑控制。

Action 经过缩放，成为相对默认姿态的关节角度偏移：

```text
目标关节角度 = 默认关节角度 + action_scale × Action
```

例如，某关节默认角度是 0.4 rad，策略输出 0.2，若 `action_scale = 0.9`，得到的目标角度就是 0.58 rad。这是相对**默认姿态**的偏移，不是在上一轮角度上再累加 0.18 rad。实际写入前还会经过目标滤波、安全限制等处理。

### 一个简化的控制循环

伪代码表达 `robotd` 的核心思路，省略了启动校准、策略切换、目标滤波、跌倒保护和通信异常处理。

```python
policy = load_onnx("walking.onnx")      # 启动时加载一次，文件名为示意
io = connect_robot_io()                 # 真机用 DynamixelIo，仿真用 RemoteIo
home = default_pose_for_policy_joints() # 14 个关节，顺序与策略一致
action_scale = load_action_scale()      # 使用与部署策略匹配的配置
last_action = zeros(14)

for tick in every(period=0.02):         # 目标频率 50 Hz，周期包含本轮计算和 I/O
    state = io.read()                  # 关节反馈 + IMU
    command = latest_command()         # 最近收到的意图，例如 vx=0.2 m/s

    observation = concat(
        state.imu.gyro,                # 3
        state.imu.projected_gravity,   # 3
        policy_positions(state) - home,  # 14，跳过嘴部
        policy_velocities(state),     # 14
        last_action,                  # 14
        encode_command(command),      # 13
    )

    action = policy.infer(observation) # 61 个输入 → 14 个输出
    target = home + action_scale * action
    targets = merge_mouth_target(target, command)  # 补齐 15 个关节
    targets = apply_safety_limits(targets, state)
    io.write(targets)                  # 下发目标；实际角度在后续 read() 中反馈
    last_action = action              # 保存原始策略输出，而非限幅后的目标角度
```

## 策略训练

Microduck 的策略训练，就是让神经网络在仿真里反复控制“虚拟身体”，根据动作产生的结果打分，再更新网络参数。我们以“按要求的速度走路，同时保持平衡”为例。
1. 先准备身体、任务和评分规则

    | 内容 | Microduck 中的例子 |
    |---|---|
    | 身体和环境 | 关节、质量、惯量、舵机模型、地面与接触 |
    | Observation | 前面讨论的 61 个数 |
    | Action | 14 个关节的目标偏移 |
    | Command | 期望前进、侧移、转向速度等 |
    | Reward | 速度跟踪得好、身体保持直立得分；打滑、动作突变等扣分 |
    | 重置规则 | 一轮结束后恢复初始状态，开始下一次尝试 |

    一个简化的评分思路：

    ```python
    reward = (
        + 速度接近_command_的程度
        + 身体保持直立的程度
        - 脚部打滑程度
        - 相邻两次动作变化过大的程度
    )
    ```

    Microduck 的实际奖励还包括抬脚、姿态跟踪等项，各项有权重。
2. 用当前策略，在仿真里收集一段经历。假设这次 Command 是“向前 0.2 m/s”：

    ```text
    读取 Observation
          ↓
    Actor 产生动作分布，从中采样 Action
          ↓
    转换为关节目标，交给仿真器
          ↓
    MuJoCo 计算身体运动、碰撞、关节反馈
          ↓
    环境生成下一时刻 Observation 和 Reward
          ↓
    记录下来，继续下一步
    ```
    训练时采样动作，是为了探索不同的动作效果。刚开始网络参数未经训练，动作通常很差；随着训练，能保持平衡、跟踪速度的动作逐渐更容易被选中。每一步记录的数据大致是：`当前观察、采取的动作、动作概率、奖励、下一观察、是否结束`
3. PPO 根据经历更新 Actor 和 Critic


把训练循环写成伪代码

```python
envs = make_parallel_microduck_envs(count=1024)
actor = MLP(input=61, hidden=[512, 256, 128], output=14)
critic = ValueNetwork()
ppo = PPO(actor, critic)

obs = envs.reset()

for iteration in range(training_iterations):
    rollout = []

    # 收集数据期间，使用同一版本的策略
    for step in range(24):
        action, log_prob = actor.sample(obs)
        value = critic.predict(envs.critic_observation())

        next_obs, reward, done, info = envs.step(action)

        rollout.append(
            (obs, action, log_prob, value, reward, done)
        )
        obs = next_obs  # 环境负责重置结束的机器人

    # 利用后续奖励和价值估计，计算训练目标
    advantages, returns = estimate_returns(
        rollout, bootstrap=critic.predict(envs.critic_observation())
    )

    # 内部进行若干轮小批量梯度更新
    ppo.update(rollout, advantages, returns)

export_actor_to_onnx(actor, include_observation_normalizer=True)
```

