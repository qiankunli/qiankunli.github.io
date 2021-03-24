---

layout: post
title: 汇编语言
category: 技术
tags: Basic
keywords: 汇编语言

---

## 简介

* TOC
{:toc}

在计算机中，并不支持直接将数据在不同的内存之间传送，更不支持将数据从内存 直接传送到外部设备，例如磁盘或网络端口。cpu 唯一支持不同部件之间的直接数据传送只有寄存器到寄存器了。由于高级编程语言并不能直接操作寄存器，所有的数据传送指令以及针对寄存器的读写指令都被封装为面向变量的编程，而变量的存储介质是内存，因此高级编程语言中所有数据传送指令都必须经过寄存器的中转。

## 与机器对话——汇编语言基础

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

汇编代码其实比较简单，它所做的工作不外乎就是**把数据在内存和寄存器中搬来搬去或做一些基础的数学和逻辑运算**。

## 过程调用和栈帧

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
5. **%rsp并不总是指向真实的栈顶**：在 X86-64 架构下，新的规范让程序可以访问栈顶之外 128 字节的内存，所以，我们甚至不需要通过改变 %rsp 来分配栈空间，而是直接用栈顶之外的空间。比如栈帧大小是16，即·`(%rbp)-(%rsp) = 16`，可以在（只能在叶子函数的）代码中直接使用 内存地址`-32(%rbp)`。叶子函数是functions which do not call any other function.
6. 除了callq/pushq/popq/retq  指令操作%rsp外，函数执行期间，可以mov (%rsp)使其指向栈顶一步到位，(%rsp)也可以和(%rbp)挨着一步不动，也可以随着变量的分配慢慢移动。
7. 函数调用 使用栈空间的大小 是编译时就可以确定的，但不是说编译时就分配好栈空间了，栈空间运行时动态分配与回收。这与全局变量、常量 编译时就确定好内存分配是不同的。函数返回时，栈顶指针重新赋值了，栈顶外的内存就抛弃了，**这就是所谓的栈里的内存可以自动管理（在编译器和操作系统的配合下）。**所以说栈里声明的本地变量，它的生存期跟作用域是一致的。

循环语句和 if 语句的秘密在于比较指令和**有条件跳转指令**（jmp是无条件跳转指令），它们都用到了 EFLAGS 寄存器。







