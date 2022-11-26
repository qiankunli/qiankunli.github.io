---

layout: post
title: controller-runtime细节分析
category: 架构
tags: Kubernetes
keywords: controller-runtime 
---

## 简介（未完成）

* TOC
{:toc}


以下部分是controller-runtime 组件

1. Cache，Kubebuilder 的核心组件，负责在 Controller 进程里面根据 Scheme 同步 Api Server 中所有该 Controller 关心 GVKs 的 GVRs，其核心是 GVK -> Informer 的映射，Informer 会负责监听对应 GVK 的 GVRs 的创建/删除/更新操作，以触发 Controller 的 Reconcile 逻辑。
2. Controller，Kubebuidler 为我们生成的脚手架文件，我们只需要实现 Reconcile 方法即可。
3. Clients，在实现 Controller 的时候不可避免地需要对某些资源类型进行创建/删除/更新，就是通过该 Clients 实现的，其中查询功能实际查询是本地的 Cache，写操作直接访问 Api Server。
4. Index，由于 Controller 经常要对 Cache 进行查询，Kubebuilder 提供 Index utility 给 Cache 加索引提升查询效率。
5. Finalizer，在一般情况下，如果资源被删除之后，我们虽然能够被触发删除事件，但是这个时候从 Cache 里面无法读取任何被删除对象的信息，这样一来，导致很多垃圾清理工作因为信息不足无法进行，K8s 的 Finalizer 字段用于处理这种情况。在 K8s 中，**只要对象 ObjectMeta 里面的 Finalizers 不为空，对该对象的 delete 操作就会转变为 update 操作**，具体说就是 update deletionTimestamp 字段，其意义就是告诉 K8s 的 GC“在deletionTimestamp 这个时刻之后，只要 Finalizers 为空，就立马删除掉该对象”。所以一般的使用姿势是
    1. 在DeletionTimestamp 为空时， 若对象没有Finalizers 就把 Finalizers 设置好（任意 string），
    2. 在DeletionTimestamp 不为空时， 根据 Finalizers 的值执行完所有的 pre-delete hook（此时可以在 Cache 里面读取到被删除对象的任何信息），之后将 Finalizers 置为空。
    一个使用场景时：正常情况下 A 创建B，则B的 ownerreference 指向A，删除A时会自动删除B。但 ownerreference 不能跨ns，因此在对 跨ns 进行级联删除时，可以使用
6. OwnerReference，K8s GC 在删除一个对象时，任何 ownerReference 是该对象的对象都会被清除，与此同时，Kubebuidler 支持所有对象的变更都会触发 Owner 对象 controller 的 Reconcile 方法。

## client

使用：client.Get 可以根据 obj 获取到对应的 gvk client，然后获取到 obj的真实数据，赋值给 obj。

```go
pod := &core.Pod{}
err := r.Client.Get(ctx, req.namesapce, pod); 
```

初始化

```
// controller-runtime/pkg/manager/manager.go
func New(config *rest.Config, options Options) (Manager, error) 
	// controller-runtime/pkg/cluster/cluster.go
	func New(config *rest.Config, opts ...Option) (Cluster, error) 
		cache, err := options.NewCache(config, ...）
		cli, err := client.New(restConf, client.Options{Scheme: scheme.Scheme,})
			func newClient(config *rest.Config, options Options) (*client, error)
```
Get 实现
```go
// controller-runtime/pkg/client/client.go
func (c *client) Get(ctx context.Context, key ObjectKey, obj Object, opts ...GetOption) error {
	switch obj.(type) {
	case *unstructured.Unstructured:
		return c.unstructuredClient.Get(ctx, key, obj, opts...)
	case *metav1.PartialObjectMetadata:
		// Metadata only object should always preserve the GVK coming in from the caller.
		defer c.resetGroupVersionKind(obj, obj.GetObjectKind().GroupVersionKind())
		return c.metadataClient.Get(ctx, key, obj, opts...)
	default:
		return c.typedClient.Get(ctx, key, obj, opts...)
	}
}
// controller-runtime/pkg/client/typed_client.go
func (c *typedClient) Get(ctx context.Context, key ObjectKey, obj Object, opts ...GetOption) error {
	r, err := c.cache.getResource(obj)
	getOpts := GetOptions{}
	getOpts.ApplyOptions(opts)
	return r.Get().
		NamespaceIfScoped(key.Namespace, r.isNamespaced()).
		Resource(r.resource()).
		VersionedParams(getOpts.AsGetOptions(), c.paramCodec).
		Name(key.Name).Do(ctx).Into(obj)
}
```
clientCache  是 k8s teype与client 的cache，不是数据的cache。
```go
// controller-runtime/pkg/client/client_cache.go
// clientCache creates and caches rest clients and metadata for Kubernetes types.
type clientCache struct {
	config *rest.Config			// config is the rest.Config to talk to an apiserver
	scheme *runtime.Scheme		// scheme maps go structs to GroupVersionKinds	
	mapper meta.RESTMapper		// mapper maps GroupVersionKinds to Resources
	codecs serializer.CodecFactory		// codecs are used to create a REST client for a gvk
	structuredResourceByType map[schema.GroupVersionKind]*resourceMeta		// structuredResourceByType caches structured type metadata
	unstructuredResourceByType map[schema.GroupVersionKind]*resourceMeta	// unstructuredResourceByType caches unstructured type metadata
	mu                         sync.RWMutex
}
// resourceMeta caches state for a Kubernetes type.
type resourceMeta struct {
	// client is the rest client used to talk to the apiserver
	rest.Interface
	// gvk is the GroupVersionKind of the resourceMeta
	gvk schema.GroupVersionKind
	// mapping is the rest mapping
	mapping *meta.RESTMapping
}
// getResource returns the resource meta information for the given type of object.If the object is a list, the resource represents the item's type instead.
func (c *clientCache) getResource(obj runtime.Object) (*resourceMeta, error) {
	gvk, err := apiutil.GVKForObject(obj, c.scheme)
	_, isUnstructured := obj.(*unstructured.Unstructured)
	_, isUnstructuredList := obj.(*unstructured.UnstructuredList)
	isUnstructured = isUnstructured || isUnstructuredList
	// It's better to do creation work twice than to not let multiple people make requests at once
	c.mu.RLock()
	resourceByType := c.structuredResourceByType
	if isUnstructured {
		resourceByType = c.unstructuredResourceByType
	}
	r, known := resourceByType[gvk]
	c.mu.RUnlock()
	if known {
		return r, nil
	}
	// Initialize a new Client
	c.mu.Lock()
	defer c.mu.Unlock()
	r, err = c.newResource(gvk, meta.IsListType(obj), isUnstructured)
	resourceByType[gvk] = r
	return r, err
}
```
