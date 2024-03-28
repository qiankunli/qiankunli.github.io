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

## streamlit

streamlit是一个开源的python库，它能够快速的帮助我们创建定制化的web应用，而且还非常便于和他人分享，特别是在机器学习和数据科学领域。整个过程不需要你了解任何前端的知识，包括html、css、javascript等，**对非前端开发人员非常的友好**。PS：nodejs让前端人员开发后端服务很方便。 

```python
import streamlit as st
# 前端页面涉及到的几乎任何元素都有相应方法 
st.text_input('请输入最喜欢的编程语言', key="name")
```

运行上述代码`streamlit run app.py` 即可在浏览器看到

![](/public/upload/python/streamlit_text_input.jpg)


## pydantic

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

## FastAPI

[三万字长文让你彻底掌握 FastAPI](https://mp.weixin.qq.com/s/b7-zb0FygFhiL6kfbNoazw)Python FastAPI是一个快速（高性能）的Web框架，用于构建基于Python的RESTful API，使用异步编程模型、WebSocket，支持类型检查和自动文档生成等功能，支持Swagger和JSON Schema规范，可以方便地与其他API工具进行集成。

```python
# main.py
from fastapi import FastAPI
# 创建一个FastAPI应用。
app = FastAPI()

# FastAPI使用装饰器来定义路由
@app.get(“/”)
async def root():
    return {“message”: “Hello, FastAPI!”}

# 可以使用查询参数、路径参数、请求体等来接收请求数据，并使用响应模型和状态码返回响应数据
@app.get(“/items/{item_id}”)
async def read_item(item_id: int, q: str = None):
    return {“item_id”: item_id, “q”: q}

@app.get("/girl/{user_id}")
async def read_info(user_id: str,request: Request):         
    # 查询参数
    query_params = request.query_params
    data = {"name": query_params.get("name"),
            "age": query_params.get("age"),
            "hobby": query_params.getlist("hobby")}
    # 实例化一个 Response 对象
    response = Response(
        
        orjson.dumps(data), # content，手动转成 json
        201,    # status_code，状态码
        {"Token": "xxx"}, # headers，响应头
        "application/json", # media_type，就是 HTML 中的 Content-Type，content 只是一坨字节流，需要告诉客户端响应类型这样客户端才能正确的解析
    )
    response.headers["ping"] = "pong" # 拿到 response 的时候，还可以单独对响应头和 cookie进行设置
    response.set_cookie("SessionID", "abc123456") # 设置 cookie 的话，通过 response.set_cookie。也可以通过 response.delete_cookie 删除 cookie
    return response

if __name__ == "__main__":
    # 启动服务"main:app" ，因为我们这个文件叫做 main.py，所以需要启动 main.py 里面的 app
    uvicorn.run("main:app", host="0.0.0.0", port=5555)
```

Response通过 Response 我们可以实现请求头、状态码、cookie 的自定义。内部接收如下参数：

1. content：返回的数据；
2. status_code：状态码；
3. headers：返回的响应头；
4. media_type：响应类型（就是响应头里面的 Content-Type，这里单独作为一个参数出现了，其实通过 headers 参数设置也是可以的）；
5. background：接收一个任务，Response 在返回之后会自动异步执行；

除了 Response 之外还有很多其它类型的响应，比如：

1. FileResponse：用于返回文件；
2. HTMLResponse：用于返回 HTML；
3. PlainTextResponse：用于返回纯文本；
4. JSONResponse：用于返回 JSON；
5. RedirectResponse：用于重定向；
6. StreamingResponse：用于返回二进制流；
它们都继承了 Response，只不过会自动帮你设置响应类型

```python
class HTMLResponse(Response):
    media_type = "text/html"
class PlainTextResponse(Response):
    media_type = "text/plain"
```
