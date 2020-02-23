---

layout: post
title: 《编译原理之美》笔记——后端部分
category: 技术
tags: Basic
keywords:  Fundamentals Compiling

---

## 简介（持续更新）

* TOC
{:toc}

## 程序运行机制

### 程序和操作系统的关系

程序视角的堆栈结构 

![](/public/upload/basic/program_memory_view.jpg)

操作系统加载可执行文件到内存的，并且定位到代码区里程序的入口开始执行。操作系统看到的是一个个指令 和 一个个内存地址

![](/public/upload/basic/os_memory_view.jpg)

为什么会有一块动态内存区域？os 启动的时候， 一波三折， 不停的往内存加载更大的程序， 第一波占用的内存第二波就干别的用了。c 语言手动malloc 和 free，其实微观上，**汇编语言也是在字节层面上不停地malloc和free**。

os 提供systemcall malloc 内存，但没有提供 systemcall  malloc 一个heap 或 stack 出来。


### 汇编语言基础

计算机的处理器有很多不同的架构，比如 x86-64、ARM、Power 等，每种处理器的指令集都不相同，那也就意味着汇编语言不同。本文以x86-64 架构为例


    #include <stdio.h>
    int main(int argc, char* argv[]){
        printf("Hello %s!\n", "Richard");
        return 0;
    }

对应的汇编代码

    .section    __TEXT,__text,regular,pure_instructions
        .build_version macos, 10, 14    sdk_version 10, 14
        .globl  _main                   ## -- Begin function main
        .p2align    4, 0x90
    _main:                                  ## @main
        .cfi_startproc
    ## %bb.0:
        pushq   %rbp
        .cfi_def_cfa_offset 16
        .cfi_offset %rbp, -16
        movq    %rsp, %rbp
        .cfi_def_cfa_register %rbp
        leaq    L_.str(%rip), %rdi
        leaq    L_.str.1(%rip), %rsi
        xorl    %eax, %eax
        callq   _printf
        xorl    %eax, %eax
        popq    %rbp
        retq
        .cfi_endproc
                                            ## -- End function 
        .section    __TEXT,__cstring,cstring_literals
    L_.str:                                 ## @.str
        .asciz  "Hello %s!\n"

    L_.str.1:                               ## @.str.1
        .asciz  "Richard"

    .subsections_via_symbols

这段代码里有指令、伪指令、标签和注释四种元素，每个元素单独占一行。
1. 伪指令以“.”开头，末尾没有冒号“：”。伪指令是是辅助性的，不是真正的 CPU 指令，就是写给汇编器的，汇编器在生成目标文件时会用到这些信息。
2. 标签以冒号“:”结尾，用于对伪指令生成的数据或指令做标记。可以代表一段代码或者常量的地址，其他代码可以访问标签。可一开始，我们没法知道这个地址的具体值，**必须生成目标文件后，才能算出来**。所以，标签会简化汇编代码的编写。
3. 注释，以“#”号开头，这跟 C 语言中以 // 表示注释语句是一样的。

在代码中，助记符“movq”“xorl”中的“mov”和“xor”是指令，而“q”和“l”叫做后缀，表示操作数的位数。后缀一共有 b, w, l, q 四种，分别代表 8 位、16 位、32 位和 64 位。在指令中使用操作数，可以使用四种格式，它们分别是：立即数、寄存器、直接内存访问和间接内存访问。

|操作数格式|示例|含义|**指令除立即数外都是基于地址的**|
|---|---|---|---|
|立即数以$开头|`movl $40, %eax`|把 40 这个数字拷贝到 %eax 寄存器|
|直接内存访问 |`callq _printf`|调用printf函数|操作数是内存地址<br>_printf是一个函数入口的地址，<br>汇编器帮我们计算出程序装载在内存时，每个字面量和过程的地址|
|寄存器|`subq $8, %rsp`|把%rsp的值减8|**寄存器本身既是存储，也是存储地址**|
|间接内存访问带有括号|`movl $10, (%rsp)`|把`%rsp` 寄存器的值所指向的地址 对应的内存设为10|寄存器的值是内存地址|

### 过程调用和栈帧

    /*function-call1.c */
    #include <stdio.h>
    int fun1(int a, int b){
        int c = 10;
        return a+b+c;
    }
    int main(int argc, char *argv[]){
        printf("fun1: %d\n", fun1(1,2));
        return 0;
    } 

等价的汇编代码如下

    # function-call1-craft.s 函数调用和参数传递
        # 文本段,纯代码
        .section    __TEXT,__text,regular,pure_instructions

    _fun1:
        # 函数调用的序曲,设置栈指针
        pushq   %rbp           # 把调用者的栈帧底部地址保存起来   
        movq    %rsp, %rbp     # 把调用者的栈帧顶部地址,设置为本栈帧的底部

        subq    $4, %rsp       # 扩展栈

        movl    $10, -4(%rbp)  # 变量c赋值为10，也可以写成 movl $10, (%rsp)

        # 做加法
        movl    %edi, %eax     # 第一个参数放进%eax
        addl    %esi, %eax     # 把第二个参数加到%eax,%eax同时也是存放返回值的寄存器
        addl    -4(%rbp), %eax # 加上c的值

        addq    $4, %rsp       # 缩小栈

        # 函数调用的尾声,恢复栈指针为原来的值
        popq    %rbp           # 恢复调用者栈帧的底部数值
        retq                   # 返回

        .globl  _main          # .global伪指令让_main函数外部可见
    _main:                                  ## @main
        
        # 函数调用的序曲,设置栈指针
        pushq   %rbp           # 把调用者的栈帧底部地址保存起来  
        movq    %rsp, %rbp     # 把调用者的栈帧顶部地址,设置为本栈帧的底部
        
        # 设置第一个和第二个参数,分别为1和2
        movl    $1, %edi
        movl    $2, %esi

        callq   _fun1                # 调用函数

        # 为pritf设置参数
        leaq    L_.str(%rip), %rdi   # 第一个参数是字符串的地址
        movl    %eax, %esi           # 第二个参数是前一个参数的返回值

        callq   _printf              # 调用函数

        # 设置返回值。这句也常用 xorl %esi, %esi 这样的指令,都是置为零
        movl    $0, %eax
        
        # 函数调用的尾声,恢复栈指针为原来的值
        popq    %rbp         # 恢复调用者栈帧的底部数值
        retq                 # 返回

        # 文本段,保存字符串字面量                                  
        .section    __TEXT,__cstring,cstring_literals
    L_.str:                                 ## @.str
        .asciz  "Hello World! :%d \n"


C函数翻译成汇编代码，有这样的固定结构。就好像java的 synchronized 关键字 自动有entermonitor 和 exitmonitor 一样。 

    # 函数调用的序曲,设置栈指针
    pushq  %rbp        # 把调用者的栈帧底部地址保存起来  
    movq  %rsp, %rbp   # 把调用者的栈帧顶部地址，设置为本栈帧的底部

    ...

    # 函数调用的尾声,恢复栈指针为原来的值
    popq  %rbp         # 恢复调用者栈帧的底部数值

|函数调用涉及指令|等价指令|备注|
|---|---|---|
|`callq _fun1`|`pushq %rip`<br>`jmp _fun1`|保存下一条指令的地址，用于函数返回继续执行<br>跳转到函数_fun1|
|`pushq %rbp`|`subq $8, %rsp`<br>`movq %rbp, (%rsp)`|把%rsp的值减8，也就是栈增长8个字节，**从高地址向低地址增长**<br>把%rbp的值写到当前栈顶指示的内存位置|
|`popq %rbp`|`movq (%rsp), %rbp`<br>`addq $8, %rsp`|把栈顶位置的值恢复回%rbp，这是之前保存在栈里的值<br>把%rsp的值加8，也就是栈减少8个字节|
|`retq`|`popq %rip`<br>`jmp %rip`|恢复指令指针寄存器|

`pushq %rbp` 执行后情况

![](/public/upload/basic/pushq_rbp.jpg)

1. pushq 和 popq 虽然是单“参数”指令，但一个隐藏的“参数”就是 %rsp。
2. 通过移动 %rsp 指针来改变帧的大小。%rbp 和 %rsp 之间的空间就是当前栈帧。
3. 栈帧先进后出 （一个函数的相关 信息占用一帧）。或者栈帧二字 重点在帧上。%rbp 在函数调用时一次移动 一个栈帧的大小，**%rbp在整个函数执行期间是不变的**。使用 %rbp 前，要先保护起来，在栈帧内存一个备份。
4. 函数内部访问 栈帧 可以使用 `-4(%rbp)`表示地址，表示%rbp 寄存器存储的地址减去4的地址。说白了，**栈帧内可以基于 (%rbp) 随机访问**，`+4(%rsp)`效果类似。
5. **%rsp并不总是指向真实的栈顶**：在 X86-64 架构下，新的规范让程序可以访问栈顶之外 128 字节的内存，所以，我们甚至不需要通过改变 %rsp 来分配栈空间，而是直接用栈顶之外的空间。比如栈帧大小是16，即·`(%rbp)-(%rsp) = 16`，可以在代码中直接使用 内存地址`-32(%rbp)`。但如果函数内 还会调用 其它函数，为了pushq/popq 指令的正确性，编译器会为%rsp 设置正确的值使其 指向栈顶。
6. 除了callq/pushq/popq/retq  指令操作%rsp外，函数执行期间，可以mov (%rsp)使其指向栈顶一步到位，(%rsp)也可以和(%rbp)挨着一步不动，也可以随着变量的分配慢慢移动。
7. 函数调用 使用栈空间的大小 是编译时就可以确定的，但不是说编译时就分配好栈空间了，栈空间运行时动态分配与回收。这与全局变量、常量 编译时就确定好内存分配是不同的。

循环语句和 if 语句的秘密在于比较指令和**有条件跳转指令**（jmp是无条件跳转指令），它们都用到了 EFLAGS 寄存器。

## 生成代码

翻译 AST 自动生成这些汇编代码

### 遍历 AST 手工生成汇编代码

AST 的每个节点有各种属性，遍历节点时，针对节点属性类型做相应处理。**生成汇编代码的过程，基本上就是基于 AST 拼接字符串**

    case PlayScriptParser.ADD:
        // 为加法运算申请一个临时的存储位置，可以是寄存器和栈
        address = allocForExpression(ctx);
        bodyAsm.append("\tmovl\t").append(left).append(", ").append(address).append("\n");  //把左边节点拷贝到存储空间
        bodyAsm.append("\taddl\t").append(right).append(", ").append(address).append("\n");  //把右边节点加上去
        break;

翻译程序的入口generate

1. 生成一个.section 伪指令，表明这是一个放文本的代码段。
2. 遍历 AST 中的所有函数，调用 generateProcedure() 方法为每个函数生成一段汇编代码，再接着生成一个主程序的入口。
3. 在一个新的 section 中，声明一些全局的常量（字面量）

    public String generate() {
        StringBuffer sb = new StringBuffer();

        // 1.代码段的头
        sb.append("\t.section  __TEXT,__text,regular,pure_instructions\n");

        // 2.生成函数的代码
        for (Type type : at.types) {
            if (type instanceof Function) {
                Function function = (Function) type;
                FunctionDeclarationContext fdc = (FunctionDeclarationContext) function.ctx;
                visitFunctionDeclaration(fdc); // 遍历，代码生成到bodyAsm中了
                generateProcedure(function.name, sb);
            }
        }

        // 3.对主程序生成_main函数
        visitProg((ProgContext) at.ast);
        generateProcedure("main", sb);

        // 4.文本字面量
        sb.append("\n# 字符串字面量\n");
        sb.append("\t.section  __TEXT,__cstring,cstring_literals\n");
        for(int i = 0; i< stringLiterals.size(); i++){
            sb.append("L.str." + i + ":\n");
            sb.append("\t.asciz\t\"").append(stringLiterals.get(i)).append("\"\n");
        }

        // 5.重置全局的一些临时变量
        stringLiterals.clear();
        
        return sb.toString();
    }

generateProcedure() 方法把函数转换成汇编代码

1. 生成函数标签、序曲部分的代码、设置栈顶指针、保护寄存器原有的值等。
2. 接着是函数体，比如本地变量初始化、做加法运算等。
3. 最后是一系列收尾工作，包括恢复被保护的寄存器的值、恢复栈顶指针，以及尾声部分的代码。

    private void generateProcedure(String name, StringBuffer sb) {
        // 1.函数标签
        sb.append("\n## 过程:").append(name).append("\n");
        sb.append("\t.globl _").append(name).append("\n");
        sb.append("_").append(name).append(":\n");

        // 2.序曲
        sb.append("\n\t# 序曲\n");
        sb.append("\tpushq\t%rbp\n");
        sb.append("\tmovq\t%rsp, %rbp\n");

        // 3.设置栈顶
        // 16字节对齐
        if ((rspOffset % 16) != 0) {
            rspOffset = (rspOffset / 16 + 1) * 16;
        }
        sb.append("\n\t# 设置栈顶\n");
        sb.append("\tsubq\t$").append(rspOffset).append(", %rsp\n");

        // 4.保存用到的寄存器的值
        saveRegisters();

        // 5.函数体
        sb.append("\n\t# 过程体\n");
        sb.append(bodyAsm);

        // 6.恢复受保护的寄存器的值
        restoreRegisters();

        // 7.恢复栈顶
        sb.append("\n\t# 恢复栈顶\n");
        sb.append("\taddq\t$").append(rspOffset).append(", %rsp\n");

        // 8.如果是main函数，设置返回值为0
        if (name.equals("main")) {
            sb.append("\n\t# 返回值\n");
            sb.append("\txorl\t%eax, %eax\n");
        }

        // 9.尾声
        sb.append("\n\t# 尾声\n");
        sb.append("\tpopq\t%rbp\n");
        sb.append("\tretq\n");

        // 10.重置临时变量
        rspOffset = 0;
        localVars.clear();
        tempVars.clear();
        bodyAsm = new StringBuffer();
    }

### 二进制文件格式和链接

汇编器可以把每一个汇编文件都编译生成一个二进制的目标文件，或者叫做一个模块。在一个文件中调用另一个文件的函数时，并不知道函数的地址。所以，汇编器把这个任务推迟，交给链接器去解决。就好比你去饭店排队吃饭，首先要拿个号（函数的标签），但不知道具体坐哪桌。等叫到你的号的时候（链接过程），服务员才会给你安排一个确定的桌子（函数的地址）。

在 Linux 下，目标文件、共享对象文件、二进制文件，都是采用 ELF 格式。这些二进制文件的格式跟加载到内存中的程序的格式是很相似的。这样有什么好处呢？它可以迅速被操作系统读取，并加载到内存中去，加载速度越快，也就相当于程序的启动速度越快。

在 ELF 格式中，代码和数据也是分开的。这样做的好处是，程序的代码部分，可以在多个进程中共享，不需要在内存里放多份。放一份，然后映射到每个进程的代码区就行了。而数据部分，则是每个进程都不一样的，所以要为每个进程加载一份。

## 优化代码

前文通过 从 AST 生成汇编代码 的过程是比较机械的。

### IR中间表达式

IR 在高级语言和汇编语言的中间，与高级语言相比，IR 丢弃了大部分高级语言的语法特征和语义特征，比如循环语句、if 语句、作用域、面向对象等等，它更像高层次的汇编语言；而相比真正的汇编语言，它又不会有那么多琐碎的、与具体硬件相关的细节。


IR看做是一种高层次的汇编语言，主要体现在：
1. 它可以使用寄存器，但**寄存器的数量没有限制**；
2. 控制结构也跟汇编语言比较像，比如有跳转语句，分成多个程序块，用标签来标识程序块等；
3. 使用相当于汇编指令的操作码。这些操作码可以一对一地翻译成汇编代码，但有时一个操作码会对应多个汇编指令。

IR 格式

1. 三地址代码，学习用途
2. LLVM 的 IR，工业级，提供了一个 IR 生成的 API（应用编程接口）

    1. 静态单赋值（SSA），也就是每个变量（地址）最多被赋值一次，它这种特性有利于运行代码优化算法；
    2. 有更多的细节信息。比如整型变量的字长、内存对齐方式等等
    3. LLVM 汇编则带有一个类型系统。它能避免不安全的数据操作，并且有助于优化算法。
    4. 在 LLVM 汇编中可以声明全局变量、常量

![](/public/upload/basic/llvm_overview.png)

    int fun1(int a, int b){
        int c = 10;
        return a + b + c;
    }

前端工具 Clang生成 LLVM 的汇编码`clang -emit-llvm -S fun1.c -o fun1.ll`

LLVM优化后的编码`clang -emit-llvm -S -O2 fun1.c -o fun1.ll`

    define i32 @fun1(i32, i32) local_unnamed_addr #0 {
        %3 = add i32 %0, 10
        %4 = add i32 %3, %1
        ret i32 %4
    }

### 独立于机器的优化

### 依赖于机器的优化

## 其它

||线程上下文|函数上下文|
|---|---|
|组成|虚拟地址空间<br>寄存器<br>cpu上下文|函数栈帧|
|指令寄存器|%pc|%rip|
|栈寄存器|用户栈在用户切换内存空间时切换<br>内核栈current_task 指向当前的 task_struct<br>用户栈顶指针在内核栈顶部的 pt_regs 结构<br>内核栈顶指针%rsp|%rbp指向栈帧的底部<br>%rsp指向栈帧的顶部|
|调度方/切换方|操作系统|程序自己/编译器|