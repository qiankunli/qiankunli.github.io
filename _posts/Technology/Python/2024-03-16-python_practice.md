---

layout: post
title: Python实践
category: 技术
tags: Python
keywords: Python

---

* TOC
{:toc}

## yield

return和yield异同
1. 相似的是：yield 和 return 都可以在一个函数里将值返回给调用方；
2. 不同的是：return 后，函数运行就终止了，而 yield 则只是暂停运行。
含有 yield 的函数，不再是普通的函数，直接调用含有 yield 的函数，返回的是一个生成器对象（generator object）。可以使用 for 循环（实际还可以使用 list 或者 next 函数）来遍历该生成器对象，将 yield 的内容一个一个打印出来。

## 闭包

闭包和嵌套函数类似，不同的是，这里外部函数返回的是一个函数，而不是一个具体的值。返回的函数通常赋于一个变量，这个变量可以在后面被继续执行调用。使用闭包的一个原因，是让程序变得更简洁易读。其次，和嵌套函数优点类似，函数开头需要做一些额外工作，而你又需要多次调用这个函数时，将那些额外工作的代码放在外部函数，就可以减少多次调用导致的不必要的开销，提高程序的运行效率。    

## 装饰器

所谓的装饰器，其实就是通过装饰器函数，**来修改原函数的一些功能，使得原函数不需要修改**。Python装饰器（decorator）在实现的时候，被装饰后的函数其实已经是另外一个函数了（函数名等函数属性会发生改变），为了不影响，Python的functools包中提供了一个叫wraps的decorator来消除这样的副作用。

```python
def timer(func):
"""装饰器：打印函数耗时"""
def decorated(*args, **kwargs): # 一般把 decorated 叫作“包装函数”，接收任意数目的可变参数 (*args, **kwargs)，主要通过调用原始函数 func 来完成工作。在包装函数内部，常会增加一些额外步骤，比如打印信息、修改参数等。
    st = time.perf_counter()
    ret = func(*args, **kwargs)
    print('time cost: {} seconds'.format(time.perf_counter() - st))
    return ret
return decorated
```

绝大多数情况下，我们会选择用嵌套函数来实现装饰器，但这并非构造装饰器的唯一方式。事实上，某个对象是否能通过装饰器（@decorator）的形式使用只有一条判断标准，那就是 decorator 是不是一个可调用的对象。类同样也是可调用对象。

```python
class Count:
    def __init__(self, func):   # 初始化时传入原函数 func()
        self.func = func
        self.num_calls = 0

    def __call__(self, *args, **kwargs):
        self.num_calls += 1
        print('num of calls is: {}'.format(self.num_calls))
        return self.func(*args, **kwargs)
@Count
def example():
    print("hello world")
example()

# 输出
num of calls is: 1
hello world
example()

# 输出
num of calls is: 2
hello world
...
```

装饰器将额外增加的功能，封装在自己的装饰器函数或类中；如果你想要调用它，只需要在原函数的顶部，加上 @decorator 即可。显然，这样做可以让你的代码得到高度的抽象、分离与简化。


## with和上下文管理器

当对象使用 with 声明创建时，上下文管理器允许类做一些设置和清理工作。上下文管理器的行为由下面两个魔法方法所定义： `__enter__()` 和`__exit__()`。PS：魔术方法的支持下的一种语法糖。

with 语句适用于对资源进行访问的场合，确保不管使用过程中是否发生异常都会执行必要的“清理”操作，释放资源。类似于java的 `try(xx){...}` 或 `try...finally...`
```python
with open('path','读写模式‘) as f:
    do something
```
等价于
```python
f = open('path','读写模式')
do something
f.close()
```
要使用 with 语句，首先要明白上下文管理器 以及 上下文管理协议（Context Management Protocol）：
1. 在有两个相关的操作需要在一部分代码块前后分别执行的时候，可以使用 with 语法自动完成。`__enter__()`方法会在with的代码块执行之前执行，`__exit__()`会在代码块执行结束后执行。如果使用了 as 子句，则将 enter() 方法的返回值赋值给 as 子句中的 target。
2. 使用 with 语法可以在特定的地方分配和释放资源

基于类的上下文管理器
```python
class FileManager:
    def __init__(self, name, mode):
        print('calling __init__ method')
        self.name = name
        self.mode = mode 
        self.file = None
        
    def __enter__(self):
        print('calling __enter__ method')
        self.file = open(self.name, self.mode)
        return self.file

    def __exit__(self, exc_type, exc_val, exc_tb):
        print('calling __exit__ method')
        if self.file:
            self.file.close()
            
with FileManager('test.txt', 'w') as f:
    print('ready to write to file')
    f.write('hello world')
    
## 输出
calling __init__ method
calling __enter__ method
ready to write to file
calling __exit__ method
```

基于生成器的上下文管理器，可以使用装饰器 contextlib.contextmanager，来定义自己所需的基于生成器的上下文管理器，用以支持 with 语句。

```python
from contextlib import contextmanager
@contextmanager
def file_manager(name, mode):
    try:
        f = open(name, mode)
        yield f
    finally:
        f.close()
        
with file_manager('test.txt', 'w') as f:
    f.write('hello world')
```

需要注意的是，当我们用 with 语句执行上下文管理器的操作时，一旦有异常抛出，异常的类型、值等具体信息，都会通过参数传入“__exit__()”函数中。你可以自行定义相关的操作对异常进行处理，而处理完异常后，也别忘了加上“return True”这条语句，否则仍然会抛出异常。