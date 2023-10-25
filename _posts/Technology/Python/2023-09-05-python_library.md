---

layout: post
title: Python一些比较有意思的库
category: 技术
tags: Python
keywords: Python

---

* TOC
{:toc}

## 前言（未完成）

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

pydantic库是一种常用的用于数据接口schema定义与检查的库。PS： 有点像java的 apache commons 库
1. 所有基于pydantic的数据类型本质上都是一个BaseModel类
2. pydantic中的一些常用的基本类型 Dict, List, Sequence, Set, Tuple
3. 高级数据结构：Enum, Optional、Union
4. 可以使用validator和config方法来实现更为复杂的数据类型定义以及检查。
5. 使用field可以灵活地定义模型中的字段，指定字段类型、默认值，添加校验函数、文档字符串等

```python
from pydantic import BaseModel
class Person(BaseModel):
    name: str
# 直接传值，此时就不用定义 __init__ 了
p = Person(name="Tom") 
# 通过字典传入
p = {"name": "Tom"} 
p = Person(**p)
# 通过其他的实例化对象传入
    p2 = Person.copy(p) 
```

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
