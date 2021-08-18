---

layout: post
title: tf_operator源码分析
category: 架构
tags: Kubernetes
keywords: tf_operator,tensorflow,kubeflow
---

## 简介

* TOC
{:toc}

## 从kube-flow说起

[炼丹师的工程修养之五：KubeFlow介绍和源码分析](https://zhuanlan.zhihu.com/p/98889237)

KubeFlow是运行在K8S之上的一套技术栈。这套技术栈包含了很多组件，组件之间的关系比较松散，我们可以配合起来用，也可以单独用其中的一部分。比如Jupyter Notebook本身可以单机运行，但是和K8S结合之后就能组建一个分布式的Jupyter Notebook服务，供多个用户同时使用。

KubeFlow不是什么
1. KubeFlow不是机器学习框架，跟TensorFlow，PyTorch，XGBoost不是一个层次的东西
2. KubeFlow不负责调度，他还是要依赖K8S来调度，但是可以在KubeFlow这个层次上，使用K8S提供的机制针对机器学习来调整资源调度的策略和算法
3. KubeFlow不是一个软件，他更像是一个技术栈，用户从这个技术栈中挑选符合需求的组件使用，各个组件是非常松耦合的
4. KubeFlow不是机器学习算法，这个不解释了

阿里推出类似 kube-flow 的 [KubeDL 加入 CNCF Sandbox，加速 AI 产业云原生化](https://mp.weixin.qq.com/s/7SUhnW4cnk_3G9Q7lIytcA)

## 案例

官方文档 [TensorFlow Training (TFJob)](https://www.bookstack.cn/read/kubeflow-1.3-en/10609d8d54cc9ad7.md)

```yml
apiVersion: kubeflow.org/v1
kind: TFJob
metadata:
  generateName: tfjob
  namespace: kubeflow
spec:
  tfReplicaSpecs:
    PS:
      replicas: 1
      restartPolicy: OnFailure
      template:
        spec:
          containers:
          - name: tensorflow
            image: gcr.io/your-project/your-image
            command:
              - python
              - -m
              - trainer.task
              - --batch_size=32
              - --training_steps=1000
    Worker:
      replicas: 3
      restartPolicy: OnFailure
      template:
        spec:
          containers:
          - name: tensorflow
            image: gcr.io/your-project/your-image
            command:
              - python
              - -m
              - trainer.task
              - --batch_size=32
              - --training_steps=1000
```


## 源码分析

实现crd 有两派思路，tf-operator 都有体现：tf-operator.v1 是目前在用的版本，training-operator.v1 是下一个大版本的设计，还不太稳定，没有发布出去。training-operator.v1 是基于 controller runtime 写的，tf-operator.v1 是基于 informer 和 client 裸写的。 当然都有大量的复用部分，因此抽了一个 `github.com/kubeflow/commmon`。

tf-operator.v1  核心struct 是 TFController，training-operator.v1 的核心struct 是TFJobReconciler， 它们都实现了ControllerInterface， 都聚合了JobController。同时 JobController 也聚合了 ControllerInterface。 具体的说TFController 和 JobController 互为成员。可以初步认为 在新老切换阶段，JobController 沉淀了TFController 和 TFJobReconciler 的一些通用逻辑，Reconcile 的核心逻辑是：ReconcileJobs ==> ReconcilePods ==> ReconcileServices，流程由TFController/TFJobReconciler触发，这几个方法TFController/TFJobReconciler 自己实现就用自己的，自己没实现就用JobController的。不必纠结这些细节。

以下分析以 tf-operator.v1 为主。

```go
// pkg/controller.v1/common/job_controller.go
type JobController struct {
    WorkQueue workqueue.RateLimitingInterface
}
// pkg/controller.v1/common/job.go
func (jc *JobController) ReconcileJobs(...){...}
// pkg/controller.v1/common/pod.go
// AddPod/UpdatePod/DeletePod 会作为pod event 的回调，从pod 从获取 resolveControllerRef 并将对应的tf job 加入workqueue
func (jc *JobController) AddPod(obj interface{}) {...}
func (jc *JobController) UpdatePod(old, cur interface{}) {...}
func (jc *JobController) DeletePod(obj interface{}) {...}
func (jc *JobController) ReconcilePods(...){...}
// pkg/controller.v1/common/service.go
```
pod/service/tfjob 的变化都会转换 jobname 加入到WorkQueue 触发reconcileTFJobs

```go
// pkg/controller.v1/tensorflow/controller.go
func (tc *TFController) processNextWorkItem() bool {
	...
	// 这个Handler其实是SyncTFJob这个函数
	forget, err := tc.syncHandler(key)
    ...
}
func (tc *TFController) syncTFJob(key string) (bool, error) {
	...
 	// 调用reconcileTFJobs这个函数来“启动”TFJobs
	reconcileTFJobsErr = tc.reconcileTFJobs(tfjob)
	...
}
// pkg/controller.v1/common/job.go
func (jc *JobController) ReconcileJobs(...)error{
    pods, err := jc.Controller.GetPodsForJob(job)
    services, err := jc.Controller.GetServicesForJob(job)
    // If the Job is succeed or failed, delete all pods and services.
    // If the Job exceeds backoff limit or is past active deadline delete all pods and services, then set the status to failed
    // 正常 Reconcile 逻辑
    if jc.Config.EnableGangScheduling {
        _, err := jc.SyncPodGroup(metaObject, pgSpec)
    }
    // Diff current active pods/services with replicas.
    for rtype, spec := range replicas {
        err := jc.Controller.ReconcilePods(metaObject, &jobStatus, pods, rtype, spec, replicas)
        err = jc.Controller.ReconcileServices(metaObject, services, rtype, spec)
    }
    // 如果状态变更了就更新到apiserver
}
```
### reconcilePods

```go
// pkg/controller.v1/tensorflow/pod.go
func (tc *TFController) reconcilePods(...) {
	...
    // 调用createNewPod函数创建Pod
	err = tc.createNewPod(tfjob, rt, strconv.Itoa(index), spec, masterRole)
	...
}

func (tc *TFController) createNewPod(...) error {
	...
	// 生成集群的配置信息，这里最关键，看一下实现
	setClusterSpec(podTemplate, tfjob, rt, index)
    if tc.Config.EnableGangScheduling {... podTemplate.Spec.SchedulerName = gangSchedulerName ...}
	// 使用上面的配置信息，真正启动Pod的创建
	tc.PodControl.CreatePodsWithControllerRef(tfjob.Namespace, podTemplate, tfjob, controllerRef)
	...
}

func setClusterSpec(podTemplateSpec *v1.PodTemplateSpec, tfjob *tfv1.TFJob, rt, index string) error { 
	...
	// 有没有看到熟悉的字眼：这里生成了TF_CONFIG的Json串
	tfConfigStr, err := genTFConfigJSONStr(tfjob, rt, index)
	
	// 接着把TF_CONFIG作为Env环境变量，在启动时传递给容器
	for i := range podTemplateSpec.Spec.Containers {
		if podTemplateSpec.Spec.Containers[i].Name == tfv1.DefaultContainerName {
			podTemplateSpec.Spec.Containers[i].Env = append(podTemplateSpec.Spec.Containers[i].Env, v1.EnvVar{
				Name:  tfConfig, // tfConfig == “TF_CONFIG”
				Value: tfConfigStr,
			})
			break
		}
	}
	...
}
// TF_CONFIG的生成过程，很直接
func genTFConfigJSONStr(tfjob *tfv1.TFJob, rtype, index string) (string, error) {
	i, err := strconv.ParseInt(index, 0, 32)
	cluster, err := genClusterSpec(tfjob)
    // 组装形成TF_CONFIG
	tfConfig := TFConfig{
		Cluster: cluster,
		Task: TaskSpec{
			Type:  rtype,
			Index: int(i),
		},
		Environment: "cloud",
	}
}

func genClusterSpec(tfjob *tfv1.TFJob) (ClusterSpec, error) {
	...
		// 这里循环生成了TF_CONFIG里面的Cluster信息。注意看注释，使用DNS配合Service，解决的还是各个节点IP不固定的问题
		for i := int32(0); i < *spec.Replicas; i++ {
			// As described here: https://kubernetes.io/docs/concepts/services-networking/dns-pod-service/#a-records.
			// Headless service assigned a DNS A record for a name of the form "my-svc.my-namespace.svc.cluster.local".
			// And the last part "svc.cluster.local" is called cluster domain
			// which maybe different between kubernetes clusters.
			hostName := jobcontroller.GenGeneralName(tfjob.Name, rt, fmt.Sprintf("%d", i))
			svcName := hostName + "." + tfjob.Namespace + "." + "svc"
			cluserDomain := os.Getenv(EnvCustomClusterDomain)
			if len(cluserDomain) > 0 {
				svcName += "." + cluserDomain
			}

			endpoint := fmt.Sprintf("%s:%d", svcName, port)
			replicaNames = append(replicaNames, endpoint)
		}
        // rt是角色，比如worker， ps等
		clusterSpec[rt] = replicaNames
	...
}
```

### reconcileServices

```go
// pkg/controller.v1/tensorflow/service.go
func (tc *TFController) reconcileServices(...) {
	...
	// 循环创建多个Service
	serviceSlices := tc.GetServiceSlices(services, replicas, tflogger.LoggerForReplica(tfjob, rt))
	for index, serviceSlice := range serviceSlices {
		if len(serviceSlice) > 1 {
			...
		} else if len(serviceSlice) == 0 {
			tflogger.LoggerForReplica(tfjob, rt).Infof("need to create new service: %s-%d", rt, index)
			// 调用createNewService创建Service
			err = tc.createNewService(tfjob, rtype, strconv.Itoa(index), spec)
		}
	}
	...
}

func (tc *TFController) createNewService(...) {
	...
	// 这里就直接生成了Service的配置信息。注意Selector
	service := &v1.Service{
		Spec: v1.ServiceSpec{
			ClusterIP: "None",
			Selector:  labels,
			Ports: []v1.ServicePort{
				{
					Name: tfv1.DefaultPortName,
					Port: port,
				},
			},
		},
	}
    // Service的名字
	service.Name = jobcontroller.GenGeneralName(tfjob.Name, rt, index)
	service.Labels = labels
    // 调用后面的接口创建Service
	err = tc.ServiceControl.CreateServicesWithControllerRef(tfjob.Namespace, service, tfjob, controllerRef)
}
```

## Running jobs with gang-scheduling

Take tf-operator for example, enable gang-scheduling in tf-operator by setting true to `--enable-gang-scheduling` flag. Volcano scheduler and operator in Kubeflow achieve gang-scheduling by using PodGroup. operator will create the PodGroup of the job automatically. operator 会创建PodGroup，Volcano scheduler（较早版本是kube-batch）  会根据PodGroup 实现gang-scheduling，TFJob yaml 本身无区别。