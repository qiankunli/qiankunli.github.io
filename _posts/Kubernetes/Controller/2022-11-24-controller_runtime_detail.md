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

```go
pod := &core.Pod{}		// 底层 通过反射获取 到pod 类型，进而获取到 pod gvk，拿到对应的client 或informer，再根据 objName 获取真实数据。
err := r.Client.Get(ctx, req.podName, pod); 
```

使用：client.Get 可以根据 obj 获取到 gvk对应的  client 或informer ，进而获取到 obj的真实数据，赋值给 obj。client的厉害之处就在于 无论带缓存（informer） 还是不带缓存（直连apiserver/restclient） ，都屏蔽了 gvk 的差异。用户只要提供一个 空的go struct 以及 资源name ，client 即可为 空的go struct 赋值。

### 初始化

manager.Manager 聚合了 cluster.Cluster，对应的 controllerManager聚合了 cluster。manager 除了健康检查、选主之外，**主要的能力是cluster 提供的**，即manager.GetXX ==> manager.client.GetXX

```
ctrl.NewManager(ctrl.GetConfigOrDie(), ctrl.Options{ Scheme: scheme,...})
	cluster, err := cluster.New(config, func(clusterOptions *cluster.Options)
		mapper, err := options.MapperProvider(config)
		cache, err := options.NewCache(config, ...)
		apiReader, err := client.New(config, clientOptions)
		writeObj, err := options.NewClient(cache, config, clientOptions, options.ClientDisableCacheFor...)	// 即DefaultNewClient
			c, err := client.New(config, options)
			return client.NewDelegatingClient(client.NewDelegatingClientInput{
				CacheReader:     cache,
				Client:          c,
				UncachedObjects: uncachedObjects,
			})
		return &cluster{
			config:           config,
			scheme:           options.Scheme,
			cache:            cache,
			fieldIndexes:     cache,
			client:           writeObj,
			apiReader:        apiReader,
			mapper:           mapper,
		}, nil
```
其中 apiReader 是直连apiserver的client， writeObj 即包装了缓存后的client

### 带缓存

带缓存的实现，主力是delegatingClient

```go
type delegatingClient struct {
	Reader
	Writer
	StatusClient

	scheme *runtime.Scheme
	mapper meta.RESTMapper
}
func NewDelegatingClient(in NewDelegatingClientInput) (Client, error) {
	uncachedGVKs := map[schema.GroupVersionKind]struct{}{}
	for _, obj := range in.UncachedObjects {
		gvk, err := apiutil.GVKForObject(obj, in.Client.Scheme())
		uncachedGVKs[gvk] = struct{}{}
	}

	return &delegatingClient{
		scheme: in.Client.Scheme(),
		mapper: in.Client.RESTMapper(),
		Reader: &delegatingReader{
			CacheReader:       in.CacheReader,	// 实际为cache
			ClientReader:      in.Client,
			scheme:            in.Client.Scheme(),
			uncachedGVKs:      uncachedGVKs,
			cacheUnstructured: in.CacheUnstructured,
		},
		Writer:       in.Client,
		StatusClient: in.Client,
	}, nil
}
```
以 Get 方法为例，如果 对象是缓存的，则把请求转发到 cache.Get
```go
func (d *delegatingReader) Get(ctx context.Context, key ObjectKey, obj Object, opts ...GetOption) error {
	if isUncached, err := d.shouldBypassCache(obj); err != nil {
		return err
	} else if isUncached {	// 直连apiserver 读取
		return d.ClientReader.Get(ctx, key, obj, opts...)
	}
	return d.CacheReader.Get(ctx, key, obj, opts...)	// 执行cache.Get
}
```


### 不带缓存Get 实现

```
// controller-runtime/pkg/manager/manager.go
func New(config *rest.Config, options Options) (Manager, error) 
	// controller-runtime/pkg/cluster/cluster.go
	func New(config *rest.Config, opts ...Option) (Cluster, error) 
		cache, err := options.NewCache(config, ...）
		cli, err := client.New(restConf, client.Options{Scheme: scheme.Scheme,})
			func newClient(config *rest.Config, options Options) (*client, error)
```

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

## cache 是如何实现的

cache 实现了 client.Reader 接口，具体实现是  informerCache，它聚合了 InformersMap
```go
type Cache interface {
	// Cache acts as a client to objects stored in the cache.
	client.Reader
	// Cache loads informers and adds field indices.
	Informers
}
// controller-runtime/pkg/cache/informer_cache.go
func (ip *informerCache) Get(ctx context.Context, key client.ObjectKey, out client.Object, opts ...client.GetOption) error {
	gvk, err := apiutil.GVKForObject(out, ip.Scheme)
	started, cache, err := ip.InformersMap.Get(ctx, gvk, out)
		specificInformersMap.Get(ctx,gvk,obj)	// 返回MapEntry，有informer 则返回，无则创建
			i, ok := ip.informersByGVK[gvk]
			if !ok {
				ip.addInformerToMap(gvk, obj)
				lw, err := ip.createListWatcher(gvk, ip)
				ni := cache.NewSharedIndexInformer(lw, obj,...,cache.Indexers{...})
				go i.Informer.Run(ip.stop)
			}
			cache.WaitForCacheSync(...)
	return cache.Reader.Get(ctx, key, out)		// CacheReader.Get
		storeKey := objectKeyToStoreKey(key)
		obj, exists, err := c.indexer.GetByKey(storeKey)
		outVal := reflect.ValueOf(out)
		objVal := reflect.ValueOf(obj)
		reflect.Indirect(outVal).Set(reflect.Indirect(objVal))
}
```
InformersMap create and caches Informers for (runtime.Object, schema.GroupVersionKind) pairs. 如果informer 已经存在则返回informer，否则新建一个 informer 并加入到map中，后续的Get 就交给 informer 了。 
```go
type InformersMap struct {
	structured   *specificInformersMap
	unstructured *specificInformersMap
	metadata     *specificInformersMap

	// Scheme maps runtime.Objects to GroupVersionKinds
	Scheme *runtime.Scheme
}
// Get will create a new Informer and add it to the map of InformersMap if none exists.  Returns the Informer from the map.
func (m *InformersMap) Get(ctx context.Context, gvk schema.GroupVersionKind, obj runtime.Object) (bool, *MapEntry, error) {
	switch obj.(type) {
	case *unstructured.Unstructured:
		return m.unstructured.Get(ctx, gvk, obj)
	case *unstructured.UnstructuredList:
		return m.unstructured.Get(ctx, gvk, obj)
	case *metav1.PartialObjectMetadata:
		return m.metadata.Get(ctx, gvk, obj)
	case *metav1.PartialObjectMetadataList:
		return m.metadata.Get(ctx, gvk, obj)
	default:
		return m.structured.Get(ctx, gvk, obj)
	}
}
```

##  index（未完成）

因为存储的缘故，k8s的性能 没有那么优秀，假设集群有几w个pod，list 就是一个巨耗时的操作，有几种优化方式
1. list 时加上 label 限定范围。k8s 支持根据 label 对object 进行检索
2. 使用client-go 本地cache，再进一步，建立本地index。PS：apiserver 确实对label 建了索引，但是本地并没有。