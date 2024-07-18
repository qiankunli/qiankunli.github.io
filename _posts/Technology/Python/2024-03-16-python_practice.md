---

layout: post
title: Python实践
category: 技术
tags: Python
keywords: Python

---

* TOC
{:toc}

## 简介

Paul Graham： 代码中任何外加的形式都是一个信号，表明我对问题的抽象还不够深入，也经常提醒我，自己正在手动完成的事情，本应该写代码通过宏的扩展自动实现。

## yield和生成器

生成器在很多常用语言中，并没有相对应的模型。调用生成器函数（带有yield关键字的函数），返回一个生成器对象 generator（由python编译器构建，提供了·__next__`方法实现），也就是说，生成器函数是generator工厂。generator 实现了Iterator 接口，因此generator 也可以迭代。生成器函数中的return 语句触发generator抛出StopIteration 异常。生成器是懒人版本的迭代器，在你调用 next() 函数的时候，才会生成下一个变量。生成器并不会像迭代器一样占用大量内存，只有在被使用的时候才会调用。
1. next(generator) 调用生成器的 `__next__()` 方法，每次遇到 yield 时函数会暂停并保存当前所有的运行信息，**返回 yield 的值后挂起**, 并在下一次执行 next(generator) 方法时从当前位置继续运行。    
1. Python 引入生成器的最初目的是作为迭代器的替代。Python 中，可以迭代处理（比如在 for 循环中）的对象称为可迭代对象，可迭代对象实现了 `__iter__()` 特殊方法，返回一个迭代器。生成器允许用户使用 yield 关键字，**将函数作为迭代器，而不用定义类并实现对应的特殊方法。Python 会自动帮用户填充特殊方法，使生成器变成迭代器**。
2. 生成器以懒加载模式按需返回值，因而更节约内存空间（甚至可以用于生成无限序列 ==> 迭代器是一个有限集合，生成器则可以成为一个无限集），可以暂停（只能将控制权移交给调用者next 函数）和恢复（被next 调用）。

```python
class Sentence:
    def __init__(self, text):
        self.text = text
        self.words = self.text.split()

    def __iter__(self):
        return SentenceIterator(self.words)
class Sentence:
    def __init__(self, text):
        self.text = text
        self.words = self.text.split()
    # __iter__ 返回一个generator（也是一个Iterator）
    def __iter__(self):
        for word in self.words:
            yield word
```

return和yield异同
1. 相似的是：yield 和 return 都可以在一个函数里将值返回给调用方；
2. 不同的是：return 后，函数运行就终止了，而 yield 则只是暂停运行。
含有 yield 的函数，不再是普通的函数，直接调用含有 yield 的函数，返回的是一个生成器对象（generator object）。可以使用 for 循环（实际还可以使用 list 或者 next 函数）来遍历该生成器对象，将 yield 的内容一个一个打印出来。

yield from 表达式句法可把一个生成器的工作委托给一个子生成器。引入 yield from 之前，如果一个生成器根据另一个生成器的值产出值，则需要使用for 循环。
```python
def sub_gen():
    yield 1.1
    yield 1.2
def gen():
    yield 1
    for i in sub_gen():
        yield i
    yield 2
def gen():
    yield 1
    yield from sub_gen()
    yield 2
```

## 闭包

在虚拟机中，函数机制的实现都离不开 FunctionObject 和 FrameObject 这两个对象。有了 FunctionObject，一个函数就可以像普通的整数或者字符串那样，作为参数传递给另外一个函数，也可以作为返回值被传出函数。所以，在 Python 语言中，函数也和整数类型、字符串类型一样，是第一类公民（first class citizen）。把函数作为第一类公民，使新的编程范式成为可能，但它也引入了一些必须要解决的问题，例如自由变量和闭包的问题。

```python
def func():
    x = 2
    
    def say():
        print(x)

    return say

f = func()
f()
```

当 say 函数在 func 函数的定义之外执行的时候，依然可以访问到 x 的值。这就好像在定义 say 函数的时候，把 x（更准确的说是x的reference） 和 say 打包在一起了，我们把这个包裹叫做闭包（closure）。

在 Python 字节码中，外部函数中定义并且被内部函数使用的变量被记录在 cell_vars 中，我们把它叫做 cell 变量。而对于内部函数，这些变量就是自由变量。自由变量是指在某个作用域内使用的变量，但该变量在这个作用域内并没有被定义或赋值，它的定义存在于当前作用域的外部。在函数中查找变量的时候，遵循 LEBG 规则。其中 L 代表局部变量，E 代表闭包变量（Enclosing），G 代表全局变量，B 则代表虚拟机内建变量。

## 装饰器

所谓的装饰器，其实就是通过装饰器函数，**来修改原函数的一些功能，使得原函数不需要修改**。装饰器实际上是通过创建一个新方法执行旧方法的代码而已。Python装饰器（decorator）在实现的时候，被装饰后的函数其实已经是另外一个函数了（函数名等函数属性会发生改变 ==> 函数签名变了），为了不影响，Python的functools包中提供了一个叫wraps的decorator来消除这样的副作用。

```python
def func():
    ...
def timer(func, *args, **kwargs):
     def decorated(*args, **kwargs): 
        st = time.perf_counter()
        ret = func(*args, **kwargs)
        print('time cost: {} seconds'.format(time.perf_counter() - st))
        return ret
    # 返回一个函数指针，这样才能赋值给func
    return decorated
func = timer(func1)
# func = timer(func1) 这样的写法麻烦且不具有共通性，所以python提供了一种装饰器的标准用法
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

装饰器是一种特殊类型的函数，它接受另一个函数作为输入参数。这意味着你可以在修饰器内部**访问和操作这个被传入的函数**。装饰器将额外增加的功能，封装在自己的装饰器函数或类中；当使用 @decorator 语法应用装饰器到一个函数上的时候，Python 会用装饰器返回的新函数来替换原始函数。这样一来，每当尝试调用原始函数的时候，实际上是调用了装饰器返回的那个新函数。修饰器不过是类似函数调用`add = call_cnt(add)`的一种语法上的简写、语法糖。在vm实现，如果一个方法被装饰器修饰，则函数调用的字节码会从CALL_FUNCTION改为CALL_FUNCTION_EX，解释器会帮忙将add 调用改为 `add=call_cnt(add)` 后的add。PS：所以是不是可以认为，语法糖的实现都有解释器帮忙？

原始的装饰器一般简单写法是两层嵌套，如果使用decorator库，将原始嵌套的写法改造成单层的写法，两层的参数合并了，且使用decorator库实现的装饰器实现了签名不变。
```python
def log(func):
    def wrapper(*args, **kw):
        print 'before run'
        return func(*args, **kw)
    return wrapper
###########################
from decorator import decorator
@decorator
def log(func, *args, **kw):
    print 'before run'
    return func(*args, **kw)
```

装饰器的用途：一般写完一个类，创建对象时可以通过init方法，也可以 定义`from_xx(cls,xx)` 类方法，也可以通过装饰器，将一个函数转换成类对象（只要这个类对象实现了 `__call__`），以下面代码为例，func_p 即可以作为一个函数，也可以作为一个Tool 对象使用。
```python
def tool(func):
    t = Tool(func,xxx)
    return t
@tool
def func_xx():
    business
func_p = func_xx
``` 

## with和上下文管理器

with 是一种不常见的控制流功能， 目的是简化常用的一些try/finally结构，

with 语句对一段定义在上下文管理器的代码进行包装（封装 try-catch-finally），当对象使用 with 声明创建时，上下文管理器允许类做一些设置和清理工作。上下文管理器的行为由下面两个魔法方法所定义： `__enter__()` 和`__exit__()`。PS：魔术方法的支持下的一种语法糖。

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

基于生成器的上下文管理器，可以使用装饰器 contextlib.contextmanager，来定义自己所需的基于生成器的上下文管理器，用以支持 with 语句。使用@contextmanager 能减少创建上下文管理器的样板代码，不用编写一个完整的类来定义`__enter__()` 和`__exit__()` 方法，而只需实现一个含有yield 语句的生成器，生成想让`__enter__()` 方法返回的值。yield 把函数主体分为两部分：在yield 之前的所有代码在调用`__enter__()` 方法时执行，yield 之后的代码在调用`__exit__()` 方法时执行。

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