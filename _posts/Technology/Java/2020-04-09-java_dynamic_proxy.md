---

layout: post
title: Java动态代理
category: 技术
tags: Java
keywords: java dynamic proxy

---

## 简介

* TOC
{:toc}

为什么需要用到代理模式？是因为类太多了，业务逻辑分散在各个类中，需要有个代理类，能够收口这些处理逻辑。所谓函数式编程，是指能够将方法当做参数传递给方法，前面“方法”是业务逻辑，后面“方法”是代理，将业务逻辑传递给代理，就实现了统一收口的目的。能够实现此功能的接口有四个，分别是：Consumer、Supplier、Predicate与Function。

动态代理到底是解决什么问题？首先，它是一个代理机制。代理可以看作是对调用目标的一个包装，这样我们**对目标代码的调用不是直接发生的**，而是通过代理完成。**通过代理可以让调用者与实现者之间解耦**。比如进行 RPC 调用，框架内部的寻址、序列化、反序列化等，对于调用者往往是没有太大意义的，通过代理，可以提供更加友善的“界面”。代理的发展经历了静态到动态的过程，源于静态代理引入的额外工作。类似早期的 RMI 之类古董技术，还需要 rmic 之类工具生成静态 stub 等各种文件，增加了很多繁琐的准备工作，而这又和我们的业务逻辑没有关系。**利用动态代理机制，相应的 stub 等类，可以在运行时生成，对应的调用操作也是动态完成**，极大地提高了我们的生产力。

部分来自极客时间 《RPC实战与核心原理》

## 从示例代码开始说起


```java
public class MyDynamicProxy {
    public static  void main (String[] args) {
        HelloImpl hello = new HelloImpl();
        MyInvocationHandler handler = new MyInvocationHandler(hello);
        // 构造代码实例，底层实现是 generateProxyClass byte[] 并进行了类加载
        Hello proxyHello = (Hello) Proxy.newProxyInstance(HelloImpl.class.getClassLoader(), HelloImpl.class.getInterfaces(), handler);
        // 调用代理方法
        proxyHello.sayHello();
    }
}
interface Hello {
    void sayHello();
}
class HelloImpl implements  Hello {
    @Override
    public void sayHello() {
        System.out.println("Hello World");
    }
}
 class MyInvocationHandler implements InvocationHandler {
    private Object target;
    public MyInvocationHandler(Object target) {
        this.target = target;
    }
    @Override
    public Object invoke(Object proxy, Method method, Object[] args)
            throws Throwable {
        // 在生产系统中，我们可以轻松扩展类似逻辑进行诊断、限流等。
        System.out.println("Invoking sayHello");
        Object result = method.invoke(target, args);
        return result;
    }
}
```

1. 实现对应的 InvocationHandler；
2. 以Hello interface为纽带，为被调用目标HelloImpl构建代理对象proxyHello
    1. 原有类的方法的调用：生成后的代理类是使用反射（method.invoke）来完成被代理类的方法调用。PS：有性能损失，但是通用
    2. 自定义扩展逻辑：method.invoke前后为应用插入额外逻辑（这里是 println）提供了便利的入口。

## Proxy.newProxyInstance 里面究竟发生了什么？


```java
Proxy.newProxyInstance
    ProxyClassFactory.apply
        byte[] proxyClassFile = ProxyGenerator.generateProxyClass(
            proxyName, interfaces, accessFlags);
        defineClass0(loader, proxyName,
                                proxyClassFile, 0, proxyClassFile.length);
```

动态代理 就是在执行代码的过程中，动态生成了 代理类 Class 的字节码`byte[]`，然后通过defineClass0 加载到jvm 中。

```java
public class Proxy implements java.io.Serializable {
    private static final Class<?>[] constructorParams =
        { InvocationHandler.class };
    public static Object newProxyInstance(ClassLoader loader,
                                            Class<?>[] interfaces,
                                            InvocationHandler h){
        final Class<?>[] intfs = interfaces.clone();
        final SecurityManager sm = System.getSecurityManager();
        if (sm != null) {
            checkProxyAccess(Reflection.getCallerClass(), loader, intfs);
        }
        // Look up or generate the designated proxy class.
        Class<?> cl = getProxyClass0(loader, intfs);
        // Invoke its constructor with the designated invocation handler.
        if (sm != null) {
            checkNewProxyPermission(Reflection.getCallerClass(), cl);
        }
        final Constructor<?> cons = cl.getConstructor(constructorParams);
        final InvocationHandler ih = h;
        if (!Modifier.isPublic(cl.getModifiers())) {
            AccessController.doPrivileged(new PrivilegedAction<Void>() {
                public Void run() {
                    cons.setAccessible(true);
                    return null;
                }
            });
        }
        return cons.newInstance(new Object[]{h});
    }
    private static Class<?> getProxyClass0(ClassLoader loader,
                                           Class<?>... interfaces) {
        if (interfaces.length > 65535) {
            throw new IllegalArgumentException("interface limit exceeded");
        }
        // If the proxy class defined by the given loader implementing
        // the given interfaces exists, this will simply return the cached copy;
        // otherwise, it will create the proxy class via the ProxyClassFactory
        return proxyClassCache.get(loader, interfaces);
    }
}
```


Proxy 包括一个内部类 ProxyClassFactory
```java
private static final class ProxyClassFactory
    implements BiFunction<ClassLoader, Class<?>[], Class<?>{
    // prefix for all proxy class names
    private static final String proxyClassNamePrefix = "$Proxy";
    // next number to use for generation of unique proxy class names
    private static final AtomicLong nextUniqueNumber = new AtomicLong();
    @Override
    public Class<?> apply(ClassLoader loader, Class<?>[] interfaces) {
        Map<Class<?>, Boolean> interfaceSet = new IdentityHashMap<>(interfaces.length);
        for (Class<?> intf : interfaces) {
            Class<?> interfaceClass = null;
            interfaceClass = Class.forName(intf.getName(), false, loader);
            ...
        }
        String proxyPkg = null;     // package to define proxy class in
        int accessFlags = Modifier.PUBLIC | Modifier.FINAL;
        ...
        if (proxyPkg == null) {
            // if no non-public proxy interfaces, use com.sun.proxy package
            proxyPkg = ReflectUtil.PROXY_PACKAGE + ".";
        }
        // Choose a name for the proxy class to generate.
        long num = nextUniqueNumber.getAndIncrement();
        String proxyName = proxyPkg + proxyClassNamePrefix + num;
        // Generate the specified proxy class.    
        byte[] proxyClassFile = ProxyGenerator.generateProxyClass(
            proxyName, interfaces, accessFlags);
        return defineClass0(loader, proxyName,
                                proxyClassFile, 0, proxyClassFile.length);
    }
}
```
可以看到，ProxyClassFactory 显示生成了 Class文件 的byte[]，然后通过defineClass0 加载到jvm 中。

`sun.misc.ProxyGenerator.saveGeneratedFiles` 控制是否把生成的字节码保存到本地磁盘。动态生成的类会保存在工程根目录下的 `com/sun/proxy`目录里面,我们找到刚才生成的 $Proxy0.class，通过反编译工具打开 class 文件

```java
package com.sun.proxy;
import com.proxy.Hello;
import java.lang.reflect.InvocationHandler;
import java.lang.reflect.Method;
import java.lang.reflect.Proxy;
import java.lang.reflect.UndeclaredThrowableException;
public final class $Proxy0 extends Proxy implements Hello {
    private static Method m3;
    private static Method m1;
    private static Method m0;
    private static Method m2;
    public $Proxy0(InvocationHandler paramInvocationHandler) {
        super(paramInvocationHandler);
    }
    public final String say() {
        try {
            return (String)this.h.invoke(this, m3, null);
        } catch (Error|RuntimeException error) {
            throw null;
        } catch (Throwable throwable) {
            throw new UndeclaredThrowableException(throwable);
        } 
    }
    public final boolean equals(Object paramObject) {
        try {
            return ((Boolean)this.h.invoke(this, m1, new Object[] { paramObject })).booleanValue();
        } catch (Error|RuntimeException error) {
            throw null;
        } catch (Throwable throwable) {
            throw new UndeclaredThrowableException(throwable);
        } 
    }
    public final int hashCode() {
        try {
            return ((Integer)this.h.invoke(this, m0, null)).intValue();
        } catch (Error|RuntimeException error) {
            throw null;
        } catch (Throwable throwable) {
            throw new UndeclaredThrowableException(throwable);
        } 
    }
    public final String toString() {
        try {
            return (String)this.h.invoke(this, m2, null);
        } catch (Error|RuntimeException error) {
            throw null;
        } catch (Throwable throwable) {
            throw new UndeclaredThrowableException(throwable);
        } 
    }
    static {
        try {
            m3 = Class.forName("com.proxy.Hello").getMethod("say", new Class[0]);
            m1 = Class.forName("java.lang.Object").getMethod("equals", new Class[] { Class.forName("java.lang.Object") });
            m0 = Class.forName("java.lang.Object").getMethod("hashCode", new Class[0]);
            m2 = Class.forName("java.lang.Object").getMethod("toString", new Class[0]);
            return;
        } catch (NoSuchMethodException noSuchMethodException) {
            throw new NoSuchMethodError(noSuchMethodException.getMessage());
        } catch (ClassNotFoundException classNotFoundException) {
            throw new NoClassDefFoundError(classNotFoundException.getMessage());
        } 
    }
}
```

综上 可以得到一个类图

![](/public/upload/java/java_dynamic_proxy.png)

[动态代理的本质](https://www.jianshu.com/p/60e283ca765b) **为什么要用 InvocationHandler 插一脚呢？**因为`sun.misc.ProxyGenerator` 在生成 proxy 字节码 `byte[]`时，自然希望具体的方法实现是一个**模式化的code**，这样才方便自动生成代码。所以**将差异化的逻辑转移到了 InvocationHandler** 。

## dynamic interface implementations

[Java动态代理的实现机制](http://developer.51cto.com/art/201509/492614.htm)所谓的动态代理就是这样一种class，它是在运行时生成的class，在生成它时你必须提供一组interface给它，然后该 class就宣称它实现了这些interface，但是其实它不会替你做实质性的工作，而是根据你在生成实例时提供的参数handler(即 InvocationHandler接口的实现类),由这个Handler来接管实际的工作。 


[Java Reflection - Dynamic Proxies](http://tutorials.jenkov.com/java-reflection/dynamic-proxies.html)Using Java Reflection you create dynamic implementations of interfaces at runtime. You do so using the class `java.lang.reflect.Proxy`. The name of this class is why I refer to these dynamic interface implementations as dynamic proxies. **我们常说的 dynamic proxies 总让人跟代理模式扯上联系，但实际上说dynamic interface implementations 更为直观。**

我们按下jdk 官方对 `java.lang.reflect.Proxy` 的注释：proxy instance has an associated invocation handler object, which implements the interface  InvocationHandler. A method invocation on a proxy instance through one of its proxy interfaces will be **dispatched** to the  InvocationHandler#invoke invoke method of the instance's invocation handler, passing the proxy instance, a ` java.lang.reflect.Method` object identifying the method that was invoked, and an array of type Object containing the arguments.  The invocation handler processes the encoded method invocation as appropriate and the result that it returns will be returned as the result of the method invocation on the proxy instance.

动态接口实现有什么用处呢？一些接口实现 取决于 用户的配置，只有加载了用户的配置才可以确认处理逻辑。此时你可以：

1. 根据用户配置穷举所有可能性， 然后根据配置调用 不同的实现类来处理。
2. 等运行时加载了用户的配置之后，再实现该接口。也就是 将实现接口的行为 放在 运行时。