---

layout: post
title: go interface及反射
category: 技术
tags: Go
keywords: Go interface reflect

---

## 前言

* TOC
{:toc}

interface，接口，restful接口，SQL 也是一种接口

go interface 及反射 代码都在runtime 包中。

## interface

[深度解密Go语言之关于 interface 的 10 个问题](https://mp.weixin.qq.com/s/EbxkBokYBajkCR-MazL0ZA)

鸭子类型，是动态编程语言的一种对象推断策略，它更关注对象能如何被使用，而不是对象的类型本身。Go 语言作为一门现代静态语言，是有后发优势的。它引入了动态语言的便利，同时又会进行静态语言的类型检查。

## 值接收者和指针接收者

方法带不带指针：`(p *Person)` refers to a pointer to the created instance of the Person struct. it is like using the keyword `this` in Java or `self` in Python when referring to the pointing object.
`(p Person)` is a copy of the value of Person ia passed to the function. any change that you make in  p if you pass it by value won't be reflected in source `p`.


    func (p *Person)GetFullName() string{
        return fmt.Println("%s %s",p.Name,p.Surname)
    }
    func (p Person)GetFullName() string{
        return fmt.Println("%s %s",p.Name,p.Surname)
    }

[深度解密Go语言之关于 interface 的 10 个问题](https://mp.weixin.qq.com/s/EbxkBokYBajkCR-MazL0ZA)如果实现了接收者是值类型的方法，会隐含地也实现了接收者是指针类型的方法。

是使用值接收者还是指针接收者，不是由该方法是否修改了调用者（也就是接收者）来决定，而是应该基于该类型的本质。

1. 如果类型具备“原始的本质”，也就是说它的成员都是由 Go 语言里内置的原始类型，如字符串，整型值等，那就定义值接收者类型的方法。像内置的引用类型，如 slice，map，interface，channel，这些类型比较特殊，声明他们的时候，实际上是创建了一个 header， 对于他们也是直接定义值接收者类型的方法。这样，调用函数时，是直接 copy 了这些类型的 header，而 header 本身就是为复制设计的。
2. 如果类型具备非原始的本质，不能被安全地复制，这种类型总是应该被共享，那就定义指针接收者的方法。比如 go 源码里的文件结构体（struct File）就不应该被复制，应该只有一份实体。

## interface 底层实现

![](/public/upload/go/go_interface_object.png)

[Go Data Structures: Interfaces](https://research.swtch.com/interfaces)Languages with methods typically fall into one of two camps: prepare tables for all the method calls statically (as in C++ and Java), or do a method lookup at each call (as in Smalltalk and its many imitators, JavaScript and Python included) and add fancy caching to make that call efficient. Go sits halfway between the two: it has method tables but computes them at run time. I don't know whether Go is the first language to use this technique, but it's certainly not a common one.

### eface 内部结构

```go
type Binary uint64
func main() {
	b := Binary(200)
	any := (interface{})(b)
	fmt.Println(any)
}
```
![](/public/upload/go/eface_fuzhi.png)

### iface 内部结构

```go
type Binary uint64
func (i Binary) String() string {
	return strconv.FormatUint(i.Get(), 10)
}
func (i Binary) Get() uint64 {
	return uint64(i)
}
func main() {
	b := Binary(200)
	any := Stringer(b)
	fmt.Println(any)
}
```

Interface values are represented as a two-word pair giving a pointer to information about the type stored in the interface and a pointer to the associated data. Assigning b to an interface value of type Stringer sets both words of the interface value.一个结构体实现了一个接口，把这个结构体变量赋值给这个接口变量，就是赋值这个接口变量里的俩指针，就完成了数据和实现的绑定。

![](/public/upload/go/iface_fuzhi.png)

Note that the itable corresponds to the interface type, not the dynamic type. In terms of our example, the itable for Stringer holding type Binary lists the methods used to satisfy Stringer, which is just String: Binary's other methods (Get) make no appearance in the itable.`itable(Stringer,Binary)` 的方法表只包含 String 方法不包含 Get 方法。

`any := Stringer(b)` 用伪代码表示 就是

    创建 iface struct for any
    创建 itab struct 
    tab := getSymAddr(`go.itab.main.Binary,main.Stringer`).(*itab)
    tab.inter = getSymAddr(`type.main.Stringer`).(*interfacetype)
    tab._type = getSymAddr(`type.main.Binary`).(*_type)
    tab.fun[0] = getSymAddr(`main.(*Binary).String`).(uintptr)

`any.String()` 相当于 `any.tab->fun[0]`


C++ 和 Go 在定义接口方式上的不同，也导致了底层实现上的不同。C++ 通过虚函数表来实现基类调用派生类的函数；而 Go 通过 itab 中的 fun 字段来**实现接口**变量调用实体类型的函数。C++ 中的虚函数表是在编译期生成的；而 Go 的 itab 中的 fun 字段是在运行期间动态生成的。

## 反射

[深度解密GO语言之反射](https://juejin.im/post/5cd0d6ed6fb9a0321556f618)反射的本质是程序在运行期探知对象的类型信息和内存结构（泛化一点说，就是我想知道某个指针对应的内存里有点什么），不用反射能行吗？可以的！使用汇编语言，直接和内层打交道，什么信息不能获取？但是，当编程迁移到高级语言上来之后，就不行了！就只能通过反射来达到此项技能。

![](/public/upload/go/go_reflect.jpeg)

reflect 包里定义了一个接口`reflect.Type`和一个结构体`reflect.Value`，它们提供很多函数来获取存储在接口里的类型信息。`reflect.Type` 主要提供关于类型相关的信息，所以它和 _type 关联比较紧密； `reflect.Value` 则结合 `_type` 和 data 两者，因此程序员可以获取甚至改变类型的值。

![](/public/upload/go/reflect_object.png)

### TypeOf

TypeOf 函数用来提取一个接口中值的类型信息。由于它的输入参数是一个空的 `interface{}`，调用此函数时，实参会先被转化为 `interface{}` 类型。这样，实参的类型信息、方法集、值信息都存储到 `interface{}` 变量里了。


```go
func TypeOf(i interface{}) Type{
    eface := *(*emptyInterface)(unsafe.Pointer(&i))
    return toType(eface.typ)
}
func toType(t *rtype) Type {
	if t == nil {
		return nil
	}
	return t
}
```

## ValueOf

reflect.Value 表示 interface{} 里存储的实际变量，它能提供实际变量的各种信息。相关的方法常常是需要结合类型信息和值信息。例如，如果要提取一个结构体的字段信息，那就需要用到 _type (具体到这里是指 structType) 类型持有的关于结构体的字段信息、偏移信息，以及 `*data` 所指向的内容 —— 结构体的实际值。

```go
func ValueOf(i interface{}) Value {
	if i == nil {
		return Value{}
	}
	escapes(i)
	return unpackEface(i)
}
func unpackEface(i interface{}) Value {
	e := (*emptyInterface)(unsafe.Pointer(&i))
	t := e.typ
	if t == nil {
		return Value{}
	}
	f := flag(t.Kind())
	if ifaceIndir(t) {
		f |= flagIndir
	}
	return Value{t, e.word, f}
}
```

通过 Type() 方法和 Interface() 方法可以打通 interface、 Type、 Value 三者。Type() 方法也可以返回变量的类型信息，与 reflect.TypeOf() 函数等价。Interface() 方法可以将 Value 还原成原来的 interface。

### 三大定律

**反射建立在类型系统之上**，下列struct 基本是一个东西，只是所在包有区别。

|类型系统相关struct|反射相关struct|
|---|---|
|eface|emptyInterface|
|_type|type|

1. Reflection goes from interface value to reflection object. 反射是一种检测存储在 interface 中的类型和值机制。这可以通过 TypeOf 函数和 ValueOf 函数得到。
2. Reflection goes from reflection object to interface value. 将 ValueOf 的返回值通过 Interface() 函数反向转变成 interface 变量。前两条就是说 **接口型变量（runtime中指向一个struct） 和 反射类型对象 可以相互转化**。
3. To modify a reflection object, the value must be settable.如果需要操作一个反射变量，那么它必须是可设置的。翻译一下就是：**如果想要操作原变量，反射变量 Value 必须要 hold 住原变量的地址才行**。

### 与java 对比

​**java中的反射，设计思路是，先类型后值**。意思是，无论如何，都是先找到属性和方法的描述，然后根据描述来获取属性的值、调用方法的执行。要进行这样的操作，入口都是由类的描述开始。golang设计思路为，值和类型划分的非常清晰，两条腿走路。

||java|go|
|---|---|---|
|获取对象的类型/表示|`getClass()`|`objType := reflect.TypeOf(obj)`|
|获取对象的值/表示|不支持|`objValue := reflect.ValueOf(obj)`|
|获取属性描述|`getClass().getField("fieldName")`|`objType.Field(index)`|
|获取属性的值|`field.get(obj)`|`objValue.Filed(index).Interface()`|
|获取方法的描述|`getClass().getMethod("methodName")`|`objType.Method(index)`|
|方法调用|`method.invoke(obj, args)`|`objValue.Method(index).Call(args)`|

在java中，通过类的描述，来获得method，由于该method是属于类级别的，所以，调用时，需要传入参数obj和args；而golang中，method是对象级别的，所以，调用时，不需要参数obj，只需要args。




