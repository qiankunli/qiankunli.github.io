---

layout: post
title: typescript学习
category: 技术
tags: WEB
keywords:  typescript

---

## 简介（未完成）

JavaScript 作为一门拥有悠久历史的脚本语言，几乎无处不在，而 TypeScript 作为其超集，它们之间最核心的区别在于 静态类型系统（意味着变量的类型在程序中的任何时候都不能改变）。 TypeScript 代码需要先编译成 JavaScript 才能在浏览器中运行。

## 基础知识

TypeScript 的设计哲学可以总结为一句话：在不改变 JavaScript 运行行为的前提下，为其提供强大的静态分析能力。很大程度上是为了让编辑器（VS Code）更智能而存在的。
1. 类型擦除 (Type Erasure)。TS 的类型系统是完全独立于运行时的（不是给运行引擎看的指令），类型仅在编译时存在。一旦编译完成，所有的接口、类型声明都会像雾气一样消散，剩下的只有纯净的 JavaScript。虽然 Python 的 Type Hints 运行也不强制约束，但 Python 至少在运行时可以通过 `__annotations__` 查到类型信息。
2. 结构化类型 (Structural Typing)。TS 奉行“鸭子类型”。
3. 非侵入性 (Non-invasive)。TS 尽量不引入新的运行时语法（除了少数如 enum 和 namespace 的历史遗留）。如果你把一个 TS 文件的所有类型标注删掉，它应该是一个合法的 JS 文件。

### 变量

1. 声明变量有三种方式： const 、 let 和 var 。其中 var 是旧的声明方式，存在变量提升等问题，不推荐使用。常用的是const vs let
    1. const声明一个常量，一旦赋值就不能再修改引用，对于对象和数组，虽然引用不能修改，但对象的属性或数组的元素可以修改，必须在声明时初始化
    2. let 声明一个变量，值可以被修改，可以在声明时初始化，也可以稍后初始化。
2. export 关键字在 TypeScript（以及 ES 模块系统）中用于控制模块成员的 可见性 ，决定哪些成员可以被其他模块访问。

### 类型

1. 在 TypeScript 中，any 类型被称为 top type。所谓的 top type 可以理解为通用父类型，也就是能够包含所有值的类型。any 类型本质上是类型系统的一个逃生舱口，TypeScript 允许我们对 any 类型的值执行任何操作，而无需事先执行任何形式的检查。如果代码里使用了大量的 any，那 TypeScript 也就失去了意义，所以我们应该尽量避免使用 any 。
2. 为了解决 any 类型存在的安全隐患，在 TypeScript 3.0 时，引入一个新的 top type —— unknown 类型。同 any 一样，你也可以把任何值赋给 unknown 类型的变量。两者有啥区别呢？「any 类型：我不在乎它的类型，unknown 类型：我不知道它的类型。」你可以把它理解成类型安全的 any 类型。相比 any 类型，TypeScript 会对 unknown 类型的变量执行类型检查
3. interface只存在于编译阶段。当你把 TS 编译成 JS 后，interface 会被彻底抹除。它不占用任何运行时的内存。类似python中的TypedDict。
    1. Declaration Merging. 同名的 interface自动合并。
3. class是 JavaScript 的原生特性。编译后，class 依然存在于 JS 文件中。它既是类型，也是一个可以用来 new 的实体。
    1. 如果你不只是想描述数据，还想给数据绑定复杂的逻辑代码（如处理函数）。
    2. 如果你需要用到 instanceof（因为 interface 运行阶段不存在，没法 instanceof）。

### 包

在早期的 npm 中，包名是全局唯一的（先到先得）。比如你注册了 utils 这个名字，别人就不能用了。随着开发者越来越多，好名字都被占光了，而且还容易出现“冒充”或“命名冲突”。为了解决这个问题，npm 引入了 Scope（作用域），@ 符号表示这是一个 Scoped Package（作用域包），格式为 `@scope/package-name`。

## 运行

```
/my-app
├── package.json      # 项目管理配置, 运行方 npm / pnpm
├── tsconfig.json     # 规定 TS 语法检查的严格程度、转换成哪个版本的 JS, 运行方tsc(TypeScript 编译器)
├── node_modules/     # 存放：下载回来的代码（由 package.json 决定）
└── src/
    └── index.ts      
```

|python|nodejs|
|---|---|
|pyproject.toml|package.json|
|uv.lock|package-lock.json|
|Makefile|package.json 的 scripts|
|venv/lib/python3.x/site-packages|node_modules，每个项目一个，如使用npm则会跨项目重复下载依赖包，pnpm则会通过硬链接机制来优化这个问题|
|npm / pnpm, `npm run` |uv,`uv run`|
|`python main.py` 启动入口文件main.py|`node index.js`启动入口文件index.ts，在 index.ts 里定义一个 bootstrap() 或 main() 函数，并在文件底部调用它。|
|Python 会去 site-packages 找包|Node.js 会先看当前目录的 node_modules，找不到再往上一层找。|
|源码直接放根目录|源码必须放src文件夹里|
|Pydantic|`@sinclair/typebox`|

在你的 package.json 里，通常会看到这样的配置：
```
"scripts": {
  "dev": "tsx src/index.ts",        // 开发时直接运行
  "build": "tsc",                   // 打包成 JS
  "start": "node dist/index.js"     // 生产环境运行编译后的代码
}
```