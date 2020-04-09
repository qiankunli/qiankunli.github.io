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

主要来自极客时间 《RPC实战与核心原理》

## 从示例代码开始说起

给 Hello 接口生成一个动态代理类，并调用接口 say() 方法，但真实返回的值居然是来自 Other 里面的 speak() 方法返回值。


```java
// 要代理的接口
public interface Hello {
    String say();
}
// 真实调用对象
public class Other {
    public String speak(){
        return "i'm proxy";
    }
}
// JDK代理类生成
public class HelloInvocationHandler implements InvocationHandler {
    private Object target;
    HelloInvocationHandler(Object target) {
        this.target = target;
    }
    @Override
    public Object invoke(Object proxy, Method method, Object[] paramValues) {
        return ((Other)target).speak();
    }
}
// 测试例子
public class TestProxy {
    public static void main(String[] args){
        // 构建代理器
        HelloInvocationHandler proxy = new HelloInvocationHandler(new Other());
        ClassLoader classLoader = ClassLoaderUtils.getCurrentClassLoader();
        // 把生成的代理类保存到文件
        System.setProperty("sun.misc.ProxyGenerator.saveGeneratedFiles","true");
        // 生成代理类
        Hello test = (Hello) Proxy.newProxyInstance(classLoader, new Class[]{Hello.class}, proxy);
        // 方法调用
        System.out.println(test.say());
    }
}
```

1. `Hello.say` 的执行会通过 `InvocationHandler.invoke` 被转到 `Other.speak`。用AOP 里的术语 就是`Other.speak` 是pointcut， `InvocationHandler.invoke` 除`Other.speak`  之外的其它部分是advise。
2. 在一般的 动态代理 示例中，Other 会实现 Hello interface，`InvocationHandler.invoke`最后会执行 `method.invoke(target,args)`。但此处实例表明，**`InvocationHandler.invoke` 可以执行任意逻辑，可以自己实现所有逻辑，也可以是完全不相关的类的 不相关的方法**， invoke 方法参数 更多是提供被代理 方法的信息。

## Proxy.newProxyInstance 里面究竟发生了什么？

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
    implements BiFunction<ClassLoader, Class<?>[], Class<?>>{
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