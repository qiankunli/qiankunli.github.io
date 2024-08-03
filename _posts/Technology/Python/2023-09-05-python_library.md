---

layout: post
title: Python一些比较有意思的库
category: 技术
tags: Python
keywords: Python

---

* TOC
{:toc}

## 前言

## 安装第三方包

现在变成语言，大都会有一些对应的库，比如java可以使用第三方jar（通过配置maven依赖），go可以引入第三方包（`go get xx`），python安装第三方包有以下方法：

1. 使用pip工具（`pip install xx`）
2. 下载第三方包文件，执行其对应的安装脚本，`python setup.py install`（跟python安装pip的方式一样）

其本质都是将第三方包文件放在约定目录。


sys.path 是一个 Python 列表，用于指定解释器在导入模块时搜索模块的路径。当你尝试导入一个模块时，Python 解释器会按照 sys.path 中的路径顺序从前往后（越靠前的路径优先级越高）进行搜索，直到找到对应的模块为止。sys.path 的值通常是由以下几部分组成：
1. Python 安装目录下的 site-packages 目录，用于存放第三方库的安装包。
2. 当前目录（即运行 Python 解释器的目录）。
3. 通过 PYTHONPATH 环境变量设置的额外路径。
4. 其他自定义的路径，可以通过在代码中使用 sys.path.append() 方法添加。
可以在 Python 解释器中执行 `print(sys.path)`，查看当前 Python 解释器的 sys.path。

1. requirements.txt。requirements.txt 是一个纯文本文件，它列出了项目所需的所有Python包及其版本。适合小型到中型项目，或者是那些不需要复杂依赖管理的项目。它的缺点是不支持条件依赖（例如，某些依赖只在特定操作系统上需要），也不支持包的替代。这个文件通常与pip工具一起使用
2. pyproject.toml 是一个TOML格式的配置文件，它是Python包管理工具pipenv使用的配置文件。 支持更复杂的依赖管理，例如条件依赖、开发依赖和包的替代。

## streamlit

streamlit是一个开源的python库，它能够快速的帮助我们创建定制化的web应用，而且还非常便于和他人分享，特别是在机器学习和数据科学领域。整个过程不需要你了解任何前端的知识，包括html、css、javascript等，**对非前端开发人员非常的友好**。PS：nodejs让前端人员开发后端服务很方便。 

```python
import streamlit as st
# 前端页面涉及到的几乎任何元素都有相应方法 
st.text_input('请输入最喜欢的编程语言', key="name")
```

运行上述代码`streamlit run app.py` 即可在浏览器看到

![](/public/upload/python/streamlit_text_input.jpg)

## 类型注解 typing

Python 3.5前，为弱类型语言，类型不显式声明，运行时可根据上下文推断变量或参数类型；Python 3.5后，引入的typing模块支持Python的静态类型注解，可显式注明变量、函数参数和返回值的类型；

Python是一门动态语言，很多时候我们可能不清楚函数参数类型或者返回值类型，很有可能导致一些类型没有指定方法，在写完代码一段时间后回过头看代码，很可能忘记了自己写的函数需要传什么参数，返回什么类型的结果。typing提供了类型提示和类型注解的功能，用于对代码进行静态类型检查和类型推断。
1. 类型注解：typing包提供了多种用于类型注解的工具，包括基本类型（如int、str）、容器类型（如List、Dict）、函数类型（如Callable、Tuple）、泛型（如Generic、TypeVar）等。通过类型注解，可以在函数声明、变量声明和类声明中指定参数的类型、返回值的类型等，以增加代码的可读性和可靠性。
    1. 数据容器：typing模块提供了多种数据容器类型，如List、Tuple、Dict和Set。
2. 泛型支持：typing模块提供了对泛型的支持，使得可以编写更通用和灵活的代码。通过泛型，可以在函数和类中引入类型参数，以处理各种类型的数据。
3. 类、函数和变量装饰器：typing模块提供了一些装饰器，如@overload、@abstractmethod、@final等，用于修饰类、函数和变量，以增加代码的可读性和可靠性。

泛型:
1. Generic: 泛型基类，用于创建泛型类或泛型函数。
2. TypeVar: 类型变量，用于创建表示不确定类型的占位符

    ```python
    # 定义一个泛型类型T
    T = TypeVar('T')
    # 表示入参 和 出参 list内 什么数据类型都可以，但必须一致
    def reverse_list(items: List[T]) -> List[T]:
        """反转列表"""
        return items[::-1]
    ```

3. Callable: 可调用对象类型，用于表示函数类型
4. Optional: 可选类型，表示一个值可以为指定类型或None
5. Iterable: 可迭代对象类型
6. Mapping: 映射类型，用于表示键值对的映射
7. Sequence: 序列类型，用于表示有序集合类型
8. Type:泛型类，用于表示类型本身

### TypedDict

```python
class Point2D(TypedDict):
    x: int
    y: int
    label: str
a: Point2D = {'x': 1, 'y': 2, 'label': 'good'}  # OK
b: Point2D = {'z': 3, 'label': 'bad'}           # Fails type check
```

python dict 有时候被当做记录使用，以key表示字段名称，value 可以是不同的类型。 比如用json 描述一本书的记录
```json
{
    "isbc":"xx",
    "title":"xx",
    "authors": ["xx","xx"],
    "pagecount":478
}
```
在python 3.8 之前没有什么好方法可以注解这段记录，因为dict value 必须是同一类型。下面两个注解都不完美。
1. Dict[str,Any]，值可以是任何类型
2. Dict[str,Union[str,int,List[str]]]，难以理解
TypedDict 解决了这个问题。
```python
from typing import TypedDict 
class BookDict(TypedDict):
    isbc:str
    title:str
    authors:list[str]
    pagecount:int
```
TypedDict 仅为类型检查工具而生，在运行时没有作用。

### 泛型编程

在 Python 中，T 通常用作一个占位符，表示一个类型变量，这是泛型编程的一部分。T 通常与 typing.Generic 类一起使用来定义泛型类型，`typing.Generic[T]` 来定义一个可以接收任何类型参数 T 的泛型类或函数。继承自Generic的类在编写需要处理多种数据类型的代码时非常有用，它们提供了一种类型安全的方式来编写灵活和可重用的组件。

```python
from typing import Generic, TypeVar, List

# 定义一个类型变量 T
T = TypeVar('T')

# 定义一个泛型类，它可以持有任何类型的数据
class Stack(Generic[T]):
    def __init__(self):
        self.items: List[T] = []

    def push(self, item: T) -> None:
        self.items.append(item)

    def pop(self) -> T:
        if not self.items:
            raise IndexError("pop from empty stack")
        return self.items.pop()

# 使用泛型 Stack 类
int_stack = Stack[int]()  # 指定 T 为 int
int_stack.push(1)
int_stack.push(2)
print(int_stack.pop())  # 输出 2

str_stack = Stack[str]()  # 指定 T 为 str
str_stack.push("hello")
str_stack.push("world")
print(str_stack.pop())  # 输出 "world"
```

## pydantic(py+pedantic=Pydantic)

对象的属性，我们都是通过把变量值赋值给对象本身来实现的。直接赋值会存在一个问题，就是无法对属性值进行合法性较验，比如我给 age 赋值的是负数，在业务上这种数据是不合法的。

```
>>> class Student:pass
...
>>>
>>> s = Student()
>>> s.name = "xx"
>>> s.age = 27
```

一个实现了 描述符协议 的类就是一个描述符。描述符协议：在类里实现了 __get__()、__set__()、__delete__() 其中至少一个方法。
1. `__get__`： 用于访问属性。它返回属性的值，若属性不存在、不合法等都可以抛出对应的异常。
2. `__set__`：将在属性分配操作中调用。不会返回任何内容。
3. `__delete__`：控制删除操作。不会返回内容。

```python
class Student:
    def __init__(self, name, math, chinese, english):
        self.name = name
        self.math = math
        self.chinese = chinese
        self.english = english
class Score:
    def __init__(self, default=0):
        self._score = default
    def __set__(self, instance, value):
        if not isinstance(value, int):
            raise TypeError('Score must be integer')
        if not 0 <= value <= 100:
            raise ValueError('Valid value must be in [0, 100]')
        self._score = value

    def __get__(self, instance, owner):
        return self._score
    def __delete__(self):
        del self._score
```
Student类里的三个属性，math、chinese、english，三个变量的合法性逻辑都是一样的，只要大于0，小于100 就可以。Score 类是一个描述符，当从 Student 的实例访问 math、chinese、english这三个属性的时候，都会经过 Score 类里的三个特殊的方法。这里的 Score 避免了 使用Property 出现大量的校验代码无法复用的尴尬。我们熟悉的@property 、@classmethod 、@staticmethod 和 super 等特性的底层实现机制都是基于 描述符协议 的。

pydantic 是一个基于Python类型提示来定义数据验证、序列化和文档(使用JSON模式)的库； 使用Python的类型提示来进行数据校验和settings管理； 可以在代码运行的时候提供类型提示，数据校验失败的时候提供友好的错误提示；
1. 所有基于pydantic的数据类型本质上都是一个BaseModel类
2. pydantic中的一些常用的基本类型 Dict, List, Sequence, Set, Tuple
3. 高级数据结构：Enum, Optional、Union
4. 可以使用validator和config方法来实现更为复杂的数据类型定义以及检查。
5. 使用field可以灵活地定义模型中的字段，指定字段类型、默认值，添加校验函数、文档字符串等

```python
from pydantic import BaseModel, EmailStr
class Person(BaseModel):
    name: str #必填字段（无默认值的时候，其为必填字段）
    email: Optional[EmailStr] = None # 有默认值表示是可选字段
# 直接传值，此时就不用定义 __init__ 了
p = Person(name="Tom") 
# 通过字典传入
p = {"name": "Tom"} 
p = Person(**p)
# 通过其他的实例化对象传入
p2 = Person.copy(p) 
```

### 数据验证

除了内置的验证器，我们还可以为我们的模型定义自定义验证器。假设我们想要确保用户年龄在18岁以上。我们可以使用@validator装饰器创建一个自定义验证器：

```python
from pydantic import BaseModel, EmailStr, validator
class User(BaseModel):
    name: str
    age: int
    email: EmailStr
    phone: Optional[str] = None

    @validator("age")
    def check_age(cls, age):
        if age < 18:
            raise ValueError("用户年龄必须大于18岁")
        return age
```

在 Pydantic 模型中，root_validator 装饰器用于定义根验证器，它可以用来在整个模型的数据被验证之前执行一些额外的验证逻辑。这对于需要跨越多个字段进行验证的情况非常有用。下面是一个示例：

```python
from pydantic import BaseModel, root_validator
class User(BaseModel):
    username: str
    password: str
    password_confirmation: str
    @root_validator
    def passwords_match(cls, values):
        if 'password' in values and 'password_confirmation' in values:
            if values['password'] != values['password_confirmation']:
                raise ValueError('密码与确认密码不匹配')
        return values
        # 实际上还可以 使用 values['password'] 为User.password 赋值

# 使用示例
user_data = {
    'username': 'user123',
    'password': 'password123',
    'password_confirmation': 'password123'
}

user = User(**user_data)
print(user)
```
Field用于在数据模型中定义字段的附加信息和约束。例如，你可以使用Field来定义字段的描述、默认值、取值范围等。

```python
from pydantic import BaseModel, Field
class Product(BaseModel):
    name: str
    price: float = Field(..., description="商品价格", gt=1, lt=1000) # ...表示该字段是必填项。
```

Field函数提供了许多参数来定制字段的行为。以下是一些常用的参数：
1. default：定义字段的默认值。如果未提供该值，则默认为None。
2. alias：定义字段的别名。这在处理不符合Python变量命名规则的字段名时非常有用（例如，包含空格或连字符的字段名）。
3. title：定义字段的标题。这在生成文档时非常有用。
4. description：定义字段的描述信息。这在生成文档时非常有用。
5. min_length和max_length：针对字符串类型的字段定义最小和最大长度限制。
6. gt、ge、lt和le：针对数值类型的字段定义大于（gt）、大于等于（ge）、小于（lt）和小于等于（le）的限制。

### 序列化

BaseModel模型具有以下方法和属性：
1. dict() 返回模型字段和值的字典；
2. json() 返回一个 JSON 字符串表示dict()；
3. copy() 返回模型的副本（默认为浅拷贝）；
4. parse_obj() 如果对象不是字典，则用于将任何对象加载到具有错误处理的模型中的实用程序；
5. parse_raw() 用于加载多种格式字符串的实用程序；
6. parse_file() 喜欢parse_raw()但是对于文件路径；
7. from_orm() 将数据从任意类加载到模型中；
8. schema() 返回将模型表示为 JSON Schema 的字典；
9. schema_json() schema()返回;的 JSON 字符串表示形式 
10. construct() 无需运行验证即可创建模型的类方法；
11. `__fields_set__` 初始化模型实例时设置的字段名称集
12. `__fields__` 模型字段的字典
13. `__config__` 模型的配置类，cf。模型配置

