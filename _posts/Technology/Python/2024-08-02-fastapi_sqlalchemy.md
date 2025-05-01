---

layout: post
title: fastapi+sqlalchemy进行项目开发
category: 技术
tags: Python
keywords: Python fastapi sqlalchemy

---

* TOC
{:toc}

## 简介

PS： fastapi 与Uvicorn 的关系有点像 springmvc 与tomcat的关系？

## FastAPI

[三万字长文让你彻底掌握 FastAPI](https://mp.weixin.qq.com/s/b7-zb0FygFhiL6kfbNoazw)Python FastAPI是一个快速（高性能）的Web框架/协程框架，用于构建基于Python的RESTful API，使用异步编程模型、WebSocket，支持类型检查和自动文档生成等功能，支持Swagger和JSON Schema规范，可以方便地与其他API工具进行集成。

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

FastAPI 最大的特点就是它使用了 Python 的类型注解，通过 Python 的类型声明，FastAPI 提供了数据校验的功能，当校验不通过的时候会清楚地指出没有通过的原因。通过 Python 的类型声明，FastAPI 提供了数据校验的功能，当校验不通过的时候会清楚地指出没有通过的原因。

任何一个请求都对应一个 Request 对象，请求的所有信息都在这个 Request 对象中。
1. 一种是体现在函数参数中，如果参数不对，FastAPI 会自动检测到，然后抛出预定义错误；
2. 另一种则是使用 Request 对象，此时请求相关的全部信息都会被封装到这个对象中，然后我们手动解析，当参数不合法时，可以自定义返回的错误信息，可控性更高。
1. 对于 POST、PUT 等类型的请求，我们必须要能够解析出请求体。在 FastAPI 中，请求体可以看成是 Model 对象。数据验证是通过 pydantic 实现的，我们需要从中导入 BaseModel，然后继承它。


既然有 Request，那么必然会有 Response，虽然我们之前都是直接返回一个字典，但 FastAPI 实际上会帮我们转成一个 Response 对象。通过 Response 我们可以实现请求头、状态码、cookie 的自定义。内部接收如下参数：

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

当我们在谈一个web框架的时候，一般会涉及以下几点：
1. 路由管理：APIRouter 类似于 Flask 的蓝图，可以更好地组织大型项目，app.include_router(router) 类似于 Flask register_blueprint
2. 错误处理： FastAPI 内部提供了一些异常类 HTTPException及其子类。定义完异常之后，还要定义一个 handler，将异常和 handler 绑定在一起，然后引发该异常的时候就会触发相应的 handler `@app.exception_handler(xxException)`。
3. 中间件：在请求进入视图函数/api handler之前，会先经过中间件（被称为请求中间件），，在里面我们可以对请求进行一些预处理，或者实现一个拦截器等等；同理当视图函数返回响应之后，也会经过中间件（被称为响应中间件），在里面我们也可以对响应进行一些润色。
    1. 自定义中间件，`@app.middleware("xx")`
    2. 内置的中间件 `app.add_middleware(xx)`
    3. CORS：随着前后端分离的流行，后端程序员和前端程序员的分工变得更加明确，后端只需要提供相应的接口、返回指定的 JSON 数据，剩下的交给前端去做。因此数据接入变得更加方便，但也涉及到了安全问题。所以浏览器为了安全起见，设置了同源策略，要求前端和后端必须是同源的。而协议、域名以及端口，只要有一个不同，那么就是不同源的。那么前端里面的 JavaScript 代码将无法和后端通信，此时我们就说出现了跨域。而 CORS 则是专门负责解决跨域的，让前后端即使不同源，也能进行数据访问。假设你的前端运行在 localhost:8080，并且尝试与 localhost:5555 进行通信。然后浏览器会向后端发送一个 HTTP OPTIONS 请求，后端会返回适当的 headers 来对这个源进行授权，浏览器判断前端所在的源是否被允许。所以后端必须有一个「允许的源」列表，如果前端对应的源是被允许的，浏览器才会允许前端向后端发请求，否则就会出现跨域失败。
4. 事件处理。FastAPI 支持在应用启动和关闭时执行一些特定的事件处理函数。
5. 测试与调试

    ```python
    from fastapi.testclient import TestClient
    from .main import app

    client = TestClient(app)

    def test_read_main():
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"message": "Hello World"}
    ```


### 依赖注入

依赖注入用于把一些可复用的逻辑抽离出来，减少代码重复。Depends参数/依赖的定义是一个 callable, 也就是说**Depends 可以接受任何可调用对象，如函数或类**，并自动解决其依赖关系。

依赖可以在三个地方添加：handler 函数参数，路径装饰器，全局 app 实例。如果在 handler 函数的 参数中添加，那么依赖的返回值会作为参数传递进去，就像其他参数一样。其他两种方式返回值都会被丢弃。PS： fastapi 在分析api handler（比如下面get_users）函数参数的时候，会将request 专门封装为一个RequestParams（具体名字忘了） 对象，其中包含Depends 成员list。 猜测如果Depends 管理的对象实现了`__enter__`和 `__exit__`（上下文管理器），那么在api handler干完活儿，fastapi 会触发RequestParams 的Depends成员的 `__exit__`的执行。从这个视角 说Depends 是搞依赖管理的，有点那味儿了。**对于一个web framework 来说，如果支持ioc的话，你可以在bean初始化和销毁时干一些活儿；也可以在请求到来和处理结束时干一些活儿（middleware），也可以在一个方法执行和结束时干一些活儿（装饰器）**。

```python
# 使用函数作为依赖
from fastapi import Depends

async def pagination(page: int, size: int):
    return {"page": page, "size": size}

@app.get("/users")
def get_users(pagination: dict=Depends(pagination)):
    users = user_model.get(**pagination)
    return users
```

接收到新的请求时，FastAPI 执行如下操作：
1. 用正确的参数调用依赖项函数
2. 获取函数返回的结果
3. 把函数返回的结果赋值给路径操作函数的参数（可以理解为把参数再加工一遍）
这样，只编写一次代码，FastAPI 就可以为多个路径操作共享这段代码。所以，依赖项的作用是：少写代码。Depends(callable) 
1. 如果callable 是一个函数，接收的参数和路径操作函数的参数一样。callable  函数不要参数也可以。
2. 如果 callable 是一个类，会将操作函数的参数赋值给它的成员。

### fastapi_sqlalchemy

常规使用 sqlalchemy 就是构建engine，获取session，之后就可以session.crud了。 session 的创建与销毁都在dao层做，sesion的获取和销毁用一个装饰器包一下。

```python
engine = create_engine(
    settings.SQLALCHEMY_DATABASE_URI,
    json_serializer=lambda obj: json.dumps(obj, ensure_ascii=False), echo=True
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
db = SessionLocal()
db.crud(...)
db.close()
```
fastapi与sqlalchemy 结合了之后，一般倾向于db/session 的生命周期与api handler 一致，进入api handler时创建好，api handler执行完毕后关闭/销毁。与 fastapi 结合更紧密的方式是 使用 DBSessionMiddleware 负责在api handler 开始时初始化一个db session，api handler 结束时关闭session，重点是将session 对象保存在了 ContextVar 里。fastapi_sqlalchemy.db 是一个DBSession 对象，db.session 也就是DBSession.session 方法负责从 ContextVar 取出session 并使用。所以最终效果是，请求可以通过 一个全局db变量，db.session 来获取并操作sqlalchemy.session，且因为 ContextVar 的支持，不会混用。 

```python
from fastapi import FastAPI
from fastapi_sqlalchemy import DBSessionMiddleware, db

app = FastAPI()
app.add_middleware(DBSessionMiddleware, db_url="sqlite:///./test.db")

# 模型定义
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True)

# API路由
@app.post("/users")
def create_user(username: str):
    user = User(username=username)
    db.session.add(user)
    db.session.commit()
    return {"message": "User created successfully"}
```
总结一下，操作db_client 要注意哪些事情
1. 有一个全局唯一入口对象db_client，可以直接在controller、service、dao  import 引用的方式获取db_client 操作po。
2. 操作db 有一个事务的问题，我们希望db操作完，自动执行db_client.commit()。 db_client 获取的位置越靠上层，这个事务的范围越大，毕竟api handler 直接对应业务的上的用户操作。所以一般将获取db_client 放在controller/api handler 那里，api handler 执行完成后自动commit。controller 将db_client 通过传参下传到service 和dao层。
3. 如果嫌弃2传参麻烦，老办法使用线程局部变量（实质是ContextVar），dao从ContextVar 取下db_client 干活。

### fastapi_pagination

```python
from fastapi import FastAPI
from fastapi_pagination import Page, paginate

from your_app.models import Item
from your_app.database import db

app = FastAPI()

@app.get("/items/")
async def get_items(page: int = 1, page_size: int = 10):
    query = db.query(Item)
    items = paginate(query, page, page_size)
    return Page(items=items, page=page, page_size=page_size, total=items.total)
```

### 后台任务

在 FastAPI 中使用后台任务，利用其依赖注入系统进行异步处理。

```python
from fastapi import BackgroundTasks, FastAPI

app = FastAPI()
def write_log(message: str):
    with open("log.txt", "a") as log_file:
        log_file.write(f"{message}\n")

@app.post("/log")
def log_message(message: str, background_tasks: BackgroundTasks):
    background_tasks.add_task(write_log, message)
    return {"message": "Log will be written in the background"}
```

## SQLAlchemy

SQLAlchemy 不具备连接数据库的能力，它连接数据库还是使用了驱动，同步驱动用 pymysql 异步驱动为 asyncmy。