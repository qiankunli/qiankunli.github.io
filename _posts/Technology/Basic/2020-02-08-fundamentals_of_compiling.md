---

layout: post
title: 《编译原理之美》笔记
category: 技术
tags: Basic
keywords:  Fundamentals Compiling

---

## 简介（持续更新）

* TOC
{:toc}

## 宏观

![](/public/upload/basic/compile_process.jpg)

前端（Front End）指的是编译器对程序代码的分析和理解过程，它通常只跟语言的语法有关，跟目标机器无关。

后端（Back End）”则是生成目标代码的过程，跟目标机器有关。

整理来说，**编译器就是 根据代码字符串生成AST，再根据AST生成目标代码的过程中**。 

## 前端

### 词法分析——将字符串转换为Token List

汉语里每个词之间是没有空格的，但我们会下意识地把句子里的词语正确地拆解出来。比如把“我学习编程”这个句子拆解成“我”“学习”“编程”，这个过程叫做“分词”。如果你要研发一款支持中文的全文检索引擎，需要有分词的功能。对于代码来说，需要将代码片段识别为关键字、标识符、操作符、数字字面量等，统称Token，可以通过构造有限自动机来实现。

词法分析器分析整个程序的字符串，当遇到不同的字符时，会驱使它迁移到不同的状态。例如，词法分析程序在扫描 age 的时候，处于“标识符”状态，等它遇到一个 > 符号，就切换到“比较操作符”的状态。词法分析过程，就是这样一个个状态迁移的过程。

![](/public/upload/basic/lexical_analyzer_process.jpg)

我们可以把所有常用的词法规则都用正则表达式描述出来，用现成工具生成词法分析器。

### 语法分析——将Token List转换为AST

语法分析就是在词法分析的基础上识别出**程序的语法结构**。这个结构是一个树状结构，是计算机容易理解和执行的。

一个程序就是一棵树，这棵树叫做抽象语法树（Abstract Syntax Tree，AST）。树的每个节点（子树）是一个语法单元，这个单元的构成规则就叫“语法”。每个节点还可以有下级节点。层层嵌套的树状结构，是我们对计算机程序的直观理解。**计算机语言总是一个结构套着另一个结构，大的程序套着子程序，子程序又可以包含子程序**。

![](/public/upload/basic/syntactic_analysis_ast.jpg)

AST的生成有很多现成的工具，比如 Yacc（或 GNU 的版本，Bison）、Antlr、JavaCC 等。[javascript-ast](https://resources.jointjs.com/demos/javascript-ast)提供javascript 语法树的可视化

### 语义分析

语义分析是要让计算机理解我们的真实意图，把一些模棱两可的地方消除掉。

1. 某个表达式的计算结果是什么数据类型？如果有数据类型不匹配的情况，是否要做自动转换？
2. 如果在一个代码块的内部和外部有相同名称的变量，我在执行的时候到底用哪个？
3. 在同一个作用域内，不允许有两个名称相同的变量，这是唯一性检查。

语义分析工作的某些成果，会作为属性**标注在抽象语法树上**。比如在 标识符节点和 字面量节点上标识它的数据类型是 int 型的。在AST上还可以标记很多属性，有些属性是在之前的两个阶段就被标注上了，比如所处的源代码行号，这一行的第几个字符。这样，在编译程序报错的时候，就可以比较清楚地了解出错的位置。

### 一个简单的解释器

![](/public/upload/basic/repl.jpg)

java类实现

![](/public/upload/basic/repl_object.png)

一个最简单版 脚本解释器实现

    public class SimpleScript {
        // 简单地用了一个 HashMap 作为变量存储区。在变量声明语句和赋值语句里，都可以修改这个变量存储区中的数据
        private HashMap<String, Integer> variables = new HashMap<String, Integer>();
        public static void main(String[] args) {
            SimpleScript script = new SimpleScript();
            SimpleParser parser = new SimpleParser();
            ...
            // 读取输入的一行
            String line = reader.readLine().trim();
            // 将输入经过词法分析器转为Token数组，再转换为AST
            ASTNode tree = parser.parse(scriptText);
            // 所谓解释执行，其实是对AST进行遍历计算
            script.evaluate(tree, "");
            ...
        }
        // 遍历AST，计算值
        private Integer evaluate(ASTNode node, String indent) throws Exception {
            switch (node.getType()) {
            case Programm://程序入口，根节点
            case Additive://加法表达式
                ASTNode child1 = node.getChildren().get(0);
                Integer value1 = evaluate(child1, indent + "\t");
                ASTNode child2 = node.getChildren().get(1);
                Integer value2 = evaluate(child2, indent + "\t");
                // 取出左右节点，递归求职，然后根据加法符号进行计算
                if (node.getText().equals("+")) {
                    result = value1 + value2;
                } else {
                    result = value1 - value2;
                }
                break;
            case Multiplicative://乘法表达式
            case IntLiteral://整型字面量
            case Identifier://标识符
            case AssignmentStmt://赋值语句
            case IntDeclaration://整型变量声明
            }
        }
    }
