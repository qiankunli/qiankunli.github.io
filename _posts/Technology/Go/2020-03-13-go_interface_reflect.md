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

从编码的角度看：**Interfaces give programs structure**.Interfaces encourage design by composition.

You must do your best to understand what could change and use interfaces to decouple.

Go 的类型系统不太常见，而且非常简单。内建类型包括结构体、函数和接口。 任何实现了接口的方法的类型都可以成为实现了该接口。类型可以被隐式的从表达式中推导， 而且不需要被显式的指定。 有关接口的特殊处理以及隐式的类型推导使得 Go 看起来**像是**一种轻量级的动态类型语言。

运行时类型结构（类似于jvm的kclass？）

```go
type _type struct {
	size       uintptr
	ptrdata    uintptr // size of memory prefix holding all pointers
	hash       uint32
	tflag      tflag
	align      uint8
	fieldAlign uint8
	kind       uint8 // 类型
	// function for comparing objects of this type (ptr to object A, ptr to object B) -> ==?
	equal func(unsafe.Pointer, unsafe.Pointer) bool
	// gcdata stores the GC type data for the garbage collector. If the KindGCProg bit is set in kind, gcdata is a GC program. Otherwise it is a ptrmask bitmap. See mbitmap.go for details.
	gcdata    *byte
	str       nameOff
	ptrToThis typeOff
}
```

## interface

[深度解密Go语言之关于 interface 的 10 个问题](https://mp.weixin.qq.com/s/EbxkBokYBajkCR-MazL0ZA)

鸭子类型，是动态编程语言的一种对象推断策略，它更关注对象能如何被使用，而不是对象的类型本身。Go 语言作为一门现代静态语言，是有后发优势的。它引入了动态语言的便利，同时又会进行静态语言的类型检查。 [Go是如何判断实现了interface](https://mp.weixin.qq.com/s/qH9HDEelHGi96u-tkiOPdQ)

一个语言的类型系统 经常需要 一个“地位超然”的类型，可以表示任何类型，比如void* 或者 Object， 但真正在使用这个 类型的变量时，需要判断其真实类型，在类型转换后才能使用，所以会有类型断言的需求。

```go
func xx(p interface){
    if  v,ok := p.(string);ok{
        xxx
    }
    switch v:=p.(type){
        case int:
        case string:
    }
}
```

## 值接收者和指针接收者

方法带不带指针：`(p *Person)` refers to a pointer to the created instance of the Person struct. it is like using the keyword `this` in Java or `self` in Python when referring to the pointing object.
`(p Person)` is a copy of the value of Person ia passed to the function. any change that you make in  p if you pass it by value won't be reflected in source `p`.

结构体方法是要将接收器定义成值，还是指针。**这本质上与函数参数应该是值还是指针是同一个问题**。
```go
func (p *Person)GetFullName() string{
    return fmt.Println("%s %s",p.Name,p.Surname)
}
func (p Person)GetFullName() string{
    return fmt.Println("%s %s",p.Name,p.Surname)
}
func GetFullName(p *Person) string{
    return fmt.Println("%s %s",p.Name,p.Surname)
}
func GetFullName(p Person) string{
    return fmt.Println("%s %s",p.Name,p.Surname)
}
```

[深度解密Go语言之关于 interface 的 10 个问题](https://mp.weixin.qq.com/s/EbxkBokYBajkCR-MazL0ZA)如果实现了接收者是值类型的方法，会隐含地也实现了接收者是指针类型的方法。

是使用值接收者还是指针接收者，不是由该方法是否修改了调用者（也就是接收者）来决定，而是应该基于该类型的本质。

1. 如果类型具备“原始的本质”，也就是说它的成员都是由 Go 语言里内置的原始类型，如字符串，整型值等，那就定义值接收者类型的方法。像内置的引用类型，如 slice，map，interface，channel，这些类型比较特殊，声明他们的时候，实际上是创建了一个 header， 对于他们也是直接定义值接收者类型的方法。这样，调用函数时，是直接 copy 了这些类型的 header，而 header 本身就是为复制设计的。
2. 如果类型具备非原始的本质，不能被安全地复制，这种类型总是应该被共享，那就定义指针接收者的方法。比如 go 源码里的文件结构体（struct File）就不应该被复制，应该只有一份实体。

在一些框架代码中，会将指针接收者命名为 this，很有感觉

```go
func (this *Person)GetFullName() string{
    return fmt.Println("%s %s",this.Name,this.Surname)
}
```

[从栈上理解 Go 语言函数调用](https://mp.weixin.qq.com/s/-xn2i2depcN4uWT3wV63Pw)
1. 调用者 caller 会将参数值写入到栈上，调用函数 callee 实际上操作的是调用者 caller 栈帧上的参数值。
2. 在进行调用指针接收者(pointer receiver)方法调用的时候，实际上是先复制了结构体的指针到栈中，然后在方法调用中全都是基于指针的操作。

## interface 底层实现

Go 使用 iface 结构体表示包含方法的接口；使用 eface 结构体表示不包含任何方法的 interface{} 类型

![](/public/upload/go/go_interface_object.png)

[Go Data Structures: Interfaces](https://research.swtch.com/interfaces)Languages with methods typically fall into one of two camps: prepare tables for all the method calls statically (as in C++ and Java), or do a method lookup at each call (as in Smalltalk and its many imitators, JavaScript and Python included) and add fancy caching to make that call efficient. Go sits halfway between the two: it has method tables but computes them at run time. I don't know whether Go is the first language to use this technique, but it's certainly not a common one.

### interface{} 不是任意类型

[Go 语言设计与实现-接口](https://draveness.me/golang/docs/part2-foundation/ch04-basic/golang-interface/)

```go
package main
type TestStruct struct{}
func NilOrNot(v interface{}) bool {
	return v == nil
}
func main() {
	var s *TestStruct
	fmt.Println(s == nil)      // #=> true
	fmt.Println(NilOrNot(s))   // #=> false
}
```
出现上述现象的原因是 —— 调用 NilOrNot 函数时发生了隐式的类型转换，除了向方法传入参数之外，变量的赋值也会触发隐式类型转换。在类型转换时，`*TestStruct` 类型会转换成 `interface{}` 类型，转换后的变量（eface struct）不仅包含转换前的变量，还包含变量的类型信息 TestStruct，所以转换后的变量与 nil 不相等。

变量的赋值、向方法传入参数会触发隐式类型转换，类型转换的情况比较多：

1. 同一类型的转换，比如int64与int
2. 某类型与字符串的转换，这个有专门的包
3. 字符串与字符/short数组的转换，比如string与`[]uint8`等
4. 具体类型转换成接口类型。

类型断言是，一个大类型，比如`interface{}`，怀疑它可能是字符串，则可以`xxx.(string)`

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

reflect 包里定义了一个接口`reflect.Type`和一个结构体`reflect.Value`，它们提供很多函数来获取存储在接口里的类型信息，反射包中的所有方法基本都是围绕着 Type 和 Value 这两个类型设计的。`reflect.Type` 主要提供关于类型相关的信息，所以它和 _type 关联比较紧密； `reflect.Value` 则结合 `_type` 和 data 两者，因此程序员可以获取甚至改变类型的值。

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

### ValueOf

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

1. 按名字访问结构的成员 `reflect.ValueOf(e).FieldByName("Name")`
2. 按名字访问结构的方法 `reflect.ValueOf(e).methodByName("updateAge").Call(args)`

### 三大定律

**反射建立在类型系统之上**，以java 视角来表述的话，反射为程序提供了部分操作 jvm 数据的能力。

![](/public/upload/go/go_reflect.png)

1. Reflection goes from interface value to reflection object. 我们能将 Go 语言的 `interface{}` 变量转换成反射对象。为什么是从 `interface{}` 变量到反射对象？当我们执行 `reflect.ValueOf(1)` 时，虽然看起来是获取了基本类型 int 对应的反射类型，但是由于 `reflect.TypeOf`、`reflect.ValueOf` 两个方法的入参都是 `interface{}` 类型，所以在方法执行的过程中发生了类型转换。
2. Reflection goes from reflection object to interface value. 我们可以从反射对象可以获取 `interface{}` 变量(`Interface()` 方法)。
3. To modify a reflection object, the value must be settable.如果需要操作一个反射变量，那么它必须是可设置的。PS: 可设置 ==> 可以找到原变量地址 ==> go 是值传递 ==> `reflect.ValueOf(引用)` 反射变量 Value 必须要 hold 住原变量的地址才行


### 与java 对比

hotspot 内部c++对java 对象的表示

![](/public/upload/java/oop_kclass_model.png)

​**java中的反射，设计思路是，先类型后值**。意思是，无论如何，都是先找到属性和方法的描述，然后根据描述来获取属性的值、调用方法的执行。要进行这样的操作，入口都是由类的描述开始。

```java
Class cls = obj.getClass(); 
Constructor constructor = cls.getConstructor(); 
Method[] methods = cls.getDeclaredFields();
```

golang设计思路为，值和类型划分的非常清晰，两条腿走路。Go 没有类的概念，并且结构体只包含了已声明的字段。因此，我们需要借助“reflection”包来获得所需的信息

||java|go|
|---|---|---|
|获取对象的类型/表示|`getClass()`|`objType := reflect.TypeOf(obj)`|
|获取对象的值/表示|不支持|`objValue := reflect.ValueOf(obj)`|
|获取属性描述|`getClass().getField("fieldName")`|`objType.Field(index)`|
|获取属性的值|`field.get(obj)`|`objValue.Filed(index).Interface()`|
|获取方法的描述|`getClass().getMethod("methodName")`|`objType.Method(index)`|
|方法调用|`method.invoke(obj, args)`|`objValue.Method(index).Call(args)`|

在java中，通过类的描述，来获得method，由于该method是属于类级别的，所以，调用时，需要传入参数obj和args；而golang中，method是对象级别的，所以，调用时，不需要参数obj，只需要args。





