---

layout: post
title: Python ioc
category: 技术
tags: Python
keywords: Python

---

* TOC
{:toc}

## 简介

控制反转（IoC）是一种软件设计模式，它改变了传统的程序控制流程。在传统编程中，程序显式控制对象的创建和依赖关系，而IoC则将这种控制权转移给外部容器或框架。这种模式能够显著提高代码的灵活性和可维护性。控制反转主要包含两个核心概念：
1. 依赖注入（DI）：一种实现IoC的具体方法，通过外部注入依赖而不是在类内部创建
2. 控制反转容器：负责管理对象的创建和生命周期。


```python
# 传统方式
class UserService:
    def __init__(self,xx_args):
        self.database = Database(xx_args)  # 硬编码依赖

    def get_user(self, user_id):
        return self.database.query(user_id)

# IoC方式
class UserService:
    def __init__(self, database):  # 依赖通过构造函数注入
        self.database = database

    def get_user(self, user_id):
        return self.database.query(user_id)

# 装饰器方式，原对象没有这个成员，给它一个成员
def inject_dependencies(**dependencies):
    def decorator(cls):
        original_init = cls.__init__
        def __init__(self, *args, **kwargs):
            for key, dependency in dependencies.items():
                setattr(self, key, dependency)
            original_init(self, *args, **kwargs)
        cls.__init__ = __init__
        return cls
    return decorator

@inject_dependencies(email_service=EmailService())
class UserNotifier:
    def notify(self, user, message):
        self.email_service.send_email(user.email, message)
```

PS： java spring 至少需要将 EmailService类作为UserNotifier类的成员，python 连这个声明都不需要了。

Python社区常用的容器包括：

1. dependency_injector
2. injector
3. pinject

## 控制反转

控制反转是框架和库的关键区别点。因为对于一个库来说，程序员使用的方式是主动的调用它，如下代码：

```python
import httpx
response = httpx.get("https://so1n.me")
print(response.status_code)
```
这段代码主动的调用httpx包的get方法发起一个请求以获取网站对应的状态码，由于这种调用方法属于开发者去主动调用库，所以属于正向的控制。而框架就不一样了，框架一般都会提供一些注册的方法将我们编写的代码注册到框架中，最后由框架来调用程序员编写的代码，如下例子：

```python
def demo(request: Request) -> PlainTextResponse:
    return PlainTextResponse(f"Hello {request.query_params.get('name', '')}!")

def create_app() -> Starlette:
    app: Starlette = Starlette()
    app.add_route("/", demo, methods=["GET"])
    return app

if __name__ == "__main__":
    import uvicorn

    app: Starlette = create_app()
    uvicorn.run(app)
```

这段代码中先是声明一个路由函数demo，这个路由函数是按照框架要求的方式编写的，这个要求是路由函数必须接收一个Request参数以及返回一个Response类；接着在实例化框架时通过add_route方法以path=/，method=GET的形式注册到框架中以及在调用uvicorn.run(app)的时候把控制权转移给了框架，并由框架在后续完成对demo路由函数的调用，这种调用方式属于反向控制。通过这种方式能公减少工程项目不同层次代码打耦合。

依赖注入框架则是一种根据对象的依赖关系的在运行时进行绑定的技术，通常它都会带有一个容器，这个容器托管着许多对象，并在运行时根据对象的依赖关系把对象传递给被控制的其它对象中。根据控制反转的思想，**把上层对象创建下层对象的权利和创建时机转移给第三方来控制**，仅保留上层对象对下层对象的使用权。在依赖层级比较深的时候能缓解开发者的心智负担。

```python
class Connection(object):
    def __init__(self, host: str, port: int, ssl: Any):  # <--
        self._host: str = host
        self._port: int = port
        self._ssl: Any = ssl

# Protocol 依赖Connection，就不得不在构造方法里带上创建Protocol 所需的参数
class Protocol(object):
    def __init__(self, host: str, port: int, ssl: Any):  # <--
        self._conn: Connection = Connection(host, port, ssl)  # <--

    def request(self, *args: Any, **kwargs: Any) -> Any:
        """发送请求并等待响应"""
        pass

class Client(object):
    def __init__(self, host: str, port: int, ssl: Any):  # <--
        self._protocol: Protocol = Protocol(host, port, Any)  # <--
```
一个类可能会被不同的类所依赖的，这意味着为一个基础类增减某些功能会导致其它依赖它的类也要进行修改
```python
class Connection(object):
    def __init__(self, host: str, port: int, ssl: Any):  # <--
        self._host: str = host
        self._port: int = port
        self._ssl: Any = ssl

class Protocol(object):
    def __init__(self, connection: Connection):  # <--
        self._conn: Connection = connection  # <--

    def request(self, *args: Any, **kwargs: Any) -> Any:
        """发送请求并等待响应"""
        pass
class Client(object):
    def __init__(self, protocol: Protocol):  # <--
        self._protocol: Protocol = protocol   # <--
```
这段代码经过变更后，每一层只接收自己需要依赖的对象。但管理依赖对象的生命周期以及对象的关系全靠手动编写代码，使其在运行时完成对象绑定的，如果项目中分了很多层或者依赖关系比较复杂的话，手动处理会比较麻烦，也不方便后续的迭代。 这时就需要通过依赖注入框架来帮忙自动整理依赖关系以及注入到需要的对象中，比如在使用dependency-injector这个依赖注入框架后，代码就可以变为如下:

```python
class Container(containers.DeclarativeContainer):
    config = providers.Configuration()
    conn = providers.Singleton(Connection,host=config.host,port=config.port)
    protocol = providers.Factory(Protocol,connection=Connection,)
    client = providers.Factory(Client, protocol=protocol,)
# 注入装饰器，可以自动的把对应的值注入到被装饰的函数中
@inject
def main(client: Client = Provide[Container.client]) -> None:
    ... # 对main方法来说，只提供依赖client对象即可
```

## dependency_injector 容器

dependency_injector框架主要构成
1. Providers
2. Overriding。	覆盖对象改变注入  
3. Configuration。
4. Resource
5. container。容器，里面定义了多个实例
6. Wiring。提供@inject装饰器，指示需要被装饰的函数

```python
from dependency_injector import containers, providers

# 配置容器
class AppContainer(containers.DeclarativeContainer):
    config = providers.Configuration()
    # 数据库服务，单例模式，全局就一个实例
    database = providers.Singleton(
        Database,
        connection_string=config.db.url
    )
    # 像个池子，需要的时候捞一个用
    db_pool = providers.Pool(
        Database，
        pool_size=5
    )
    # 业务服务，工厂模式，每次都整个新的
    user_service = providers.Factory(
        UserService,
        database=database
    )
    
# 应用程序启动
def main():
    container = AppContainer()
    container.config.from_yaml('config.yml')

    user_service = container.user_service()
    email_service = container.email_service()

    # 使用服务
    user = user_service.get_user(1)
    email_service.send_email(user.email, "欢迎使用我们的服务！")
```

有时候需要控制对象的生命周期
```
class Container(containers.DeclarativeContainer)：
    # 请求级别的作用域，每个请求一个新实例
    request = providers.Resource(
        RequestContext，
        timeout=30
    )
    # 上下文管理，用完自动关闭连接
    db_session = providers.ContextLocalSingleton(
        SessionFactory，
        cleanup=lambda s： s.close()
    )
```
有了ioc，new 一个对象这种事儿，不是没有了，而是集中在某一个位置了。对于java spring来说，是框架代劳了，对于dependency_injector来说，这把这块的代码集中到了xxContainer，用的时候通过container来取用。很多时候，当你能够干预一个obj的创建，就能塞很多私货。

## fastapi 中的依赖注入
1. 函数依赖。当一个路径操作函数声明了一个依赖项时，FastAPI 会在执行该函数之前自动调用依赖函数并注入结果。
    ```python
    from fastapi import FastAPI, Depends
    app = FastAPI()
    async def get_db():
        db = Database()
        try:
            yield db
        finally:
            db.close()
    @app.get("/users/{user_id}")
    async def read_user(user_id: int, db: Database = Depends(get_db)):
        return db.get_user(user_id)
    ```
2. 类依赖提供了更结构化的方式来组织依赖逻辑。通过实现 `__call__` 方法，类可以像函数一样被调用。这种方式特别适合需要在依赖之间共享状态的场景。
    ```python
    from fastapi import Depends
    class DatabaseDependency:
        def __init__(self):
            self.db = Database()

        async def __call__(self):
            try:
                yield self.db
            finally:
                self.db.close()

    db_dependency = DatabaseDependency()
    @app.get("/items/{item_id}")
    async def read_item(item_id: int, db: Database = Depends(db_dependency)):
        return db.get_item(item_id)
    ```
3. 子依赖允许我们创建依赖链。一个依赖可以依赖于其他依赖，FastAPI 会自动处理这些嵌套的依赖关系。这种方式有助于构建层次化的依赖结构。
    ```python
    async def get_db():
        return Database()
    async def get_user_service(db: Database = Depends(get_db)):
        return UserService(db)
    @app.get("/users/me")
    async def read_current_user(
        service: UserService = Depends(get_user_service)
    ):
        return service.get_current_user()
    ```

FastAPI 默认会缓存同一请求中的依赖结果。这意味着如果多个路径操作函数使用相同的依赖，该依赖只会被执行一次。这种机制可以提高性能，但在某些场景下可能需要禁用。

```python
# 使用 use_cache=False 禁用缓存
@app.get("/status")
async def get_status(
    service: Service = Depends(get_service, use_cache=False)
):
    return service.check_status()
```

## 为何在Python生态很少听说到依赖注入

[为何在Python生态很少听说到依赖注入](https://juejin.cn/post/7158485490650316831)
1. 大部分被依赖注入容器托管的对象都被要求是单例的，而Python的每个模块中的对象也都是单例的，这样一来实现工程项目就会比较方便。可以把Python的运行时环境认为是一个大的依赖注入容器。
2. 在静态语言中，编译期，装载期和运行时期都是严格分离的，无法在运行期执行装载期的工作，这样就需要依赖注入容器通过反射来进行处理，比如Java的Spring框架，而Python是一门动态语言，它的运行时环境可以认为是一个大的依赖注入容器，所以Python可以在运行时替换某个对象。 比如java 注解一般用到动态代理，B有一个IA成成员，你等运行时把b.a 换成a1 已经来不及了，所以只能框架帮忙在给b.a 赋值时就得b.a=a1。而这在python里不是个事儿。