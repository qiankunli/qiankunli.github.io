---

layout: post
title: 递归、动态规划和回溯
category: 技术
tags: Algorithm
keywords: backtracking

---

## 简介

笔者最近在和同事准备校招题的时候，有一道涉及到回溯的编程题，发现自己也不会，看了答案也没什么感觉，因此收集材料整理下。

本文涉及到两道题目：

1. 利用字符‘a’、‘b’ 、 ‘c’ 、‘d’ 、‘e’ 、‘f’、‘g’生成并输出所有可能的字符串（但字符不可重复使用，输出顺序无要求），比如：
“a”、“b”、“c”、“d”、“e”、“f”、“g”、“ab”、“ba”、“ac”、“ca”

2.   集合A的幂集是由集合A的所有子集所组成的的集合，如：```A=1，2，3```，则A的幂集```P（A）={1，2，3}，{1，2}，{1，3}，{1}，{2，3}，{2}，{3}，{ }```，求一个集合的幂集就是求一个集合的所有的子集。来自[回溯法求幂集](http://www.cnblogs.com/youxin/p/3219523.html) 

两个问题基本一样，主要是[回溯法求幂集](http://www.cnblogs.com/youxin/p/3219523.html) 对回溯法的一些思路表述的比较精彩。下文会混用两个问题。

## 重新理解递归

[写递归函数的正确思维方法](http://blog.csdn.net/vagrxie/article/details/8470798)基本要点：

1. 看到一个递归实现, 我们总是难免陷入不停的回溯验证之中（把变量的变化依次写出来）, 因为回溯就像反过来思考迭代, 这是我们习惯的思维方式, 但是其实无益于理解。数学归纳法才是理解的递归的方式，函数式编程也有这么点意思。
2. **递归并不是算法，它是和迭代对应的⼀种编程⽅法。只不过，我们通常借 助递归去分解问题⽽已**。对于`f(n) = max(f(n-1),f(n-2))` 可以用递归写，也可以用 用迭代 从f(0) f(1) 一直求到f(n)。**递归中如果存在重复计算**（称为重叠⼦问题），那就是使 ⽤记忆化递归（或动态规划）解题的强有⼒信号之⼀。可以看出动态规划 的核⼼就是使⽤记忆化的⼿段消除重复⼦问题的计算，如果这种重复⼦问 题的规模是指数或者更⾼规模，那么记忆化递归（或动态规划）带来的收 益会⾮常⼤。为了消除这种重复计算，我们可使⽤查表的⽅式。即⼀边递归⼀边使⽤ “记录表”（⽐如哈希表或者数组）记录我们已经计算过的情况，当下次再 次碰到的时候，如果之前已经计算了，那么直接返回即可，这样就避免了 重复计算。 
3. 如果你刚开始接触递归，⼀个简 单练习递归的⽅式是将你写的迭代全部改成递归形式。⽐如你写了⼀个程 序，功能是“将⼀个字符串逆序输出”，那么使⽤迭代将其写出来会⾮常容 易，那么你是否可以使⽤递归写出来呢？通过这样的练习，可以让你逐步 适应使⽤递归来写程序。

当我们处理递归问题时， 如何定义递归出口是非常重要的一步。递归出口是递归函数 可以直接处理的最简单子问题。一把有关树的DFS 问题，递归出口都是叶子节点或空节点，然后基于叶子节点/空节点 判断当前路径是否符合要求。

## 动态规划

动态规划和其它算法思想如递归、回溯、分治、贪心等方法都有一定的联系，**其背后的基本思想是枚举**，虽然看起来简单， 但如何涵盖所有可能，并尽可能重叠子问题的计算是一个难点。每一个动态规划问题，都可以被抽象为一个数学函数，这个函数的自变量集合就是题目的所有取值，值域就是题目要求的答案的所有可能。**我们的目的就是填充这个函数的内容**，使得给定自变量x 能够唯一映射到一个值y（当然自变量可能有多个，对应递归函数的参数可能有多个）。
4. 动态规划最重要的两个概念：最优⼦结构和⽆后效性。
    1. ⽆后效性决定了是否可使⽤动态规划来解决。即⼦问题的解⼀旦确定，就不再改变，不受在这之后、包含它的更⼤的问 题的求解决策影响。背包问题中选择是否拿第三件物品，不应该影 响是否拿前⾯的物品。⽐如题⽬规定了拿了第三件物品之后，第⼆件物品的价值就会变低或变⾼）。这种情况就不满⾜⽆后向性。
    2. 最优⼦结构决定了具体如何解决。
5. **动态规划的中⼼点是什么？那就是定义状态**。定义好了状态，就可以画出递归 树，聚焦最优⼦结构写转移⽅程就好了，因此我才说状态定义是动态规划 的核⼼，动态规划问题的状态确实不容易看出。⽐如⼀个字符串的状态，通常是 dp[i]表示字符串 s 以 i 结尾的 ....。 ⽐如两个字符串的状态，通常是 dp[i][j] 表 示字符串 s1 以 i 结尾，s2 以 j 结尾的 ....。 当你定义好了状态，剩下就三件事了：
    1. 临界条件； 比如⼀个⼈爬楼梯，每次只能爬 1 个或 2 个台阶，假设有 n 个台阶，那么这 个⼈有多少种不同的爬楼梯⽅法？如果我们⽤ f(n) 表示爬 n 级台阶有多少种⽅ 法的话，f(1) 与 f(2) 就是【边界】，`f(n) = f(n-1) + f(n-2)` 就是【状态转移公式】。
    2. 状态转移⽅程；动态规划中当前阶段的状态往往是上⼀阶段状态和上⼀阶段决策的结果。也就是说，如果给定了第 k 阶段的状态 s[k] 以及决策 choice(s[k])，则第k+1 阶段的状态 s[k+1] 也就完全确定，⽤公式表示就是：`s[k] +choice(s[k]) -> s[k+1]`， 这就是状态转移⽅程。需要注意的是 choice 可能 有多个，因此每个阶段的状态 s[k+1]也会有多个。**通过状态转移方程减小问题规模**。
        ![](/public/upload/algorithm/dynamic_programming_condition.png)
    3. 枚举状态。一些场景下，遍历顺序取决于状态转移⽅程。⽐如下面代码 我们就需要从左到右遍历，原因很简单，因为 dp[i] 依赖于 dp[i - 1]， 因此计算 dp[i] 的时候， dp[i - 1] 需要已经计算好了。如果是⼀维状态，那么我们使⽤⼀层循环可以搞定。如果是两维状态，那么我们使⽤两层循环可以搞定。PS：动态规划不一定非要递归实现，状态转移方程不一定都是f(n)和(n-1)的方程（实质跟枚举类似），尤其是对于一些不连续的问题，f(n)和f(n-1)不一定能建立推导关系
        ```
        for i in range(1, n + 1):
            dp[i] = dp[i - 1] + 1
        ```

如果从 状态转移方程的视角 看动态规划，状态转移方程也要在 for 循环里，也是迭代，只是迭代公式更复杂。
1. 线性动态规划，`s[k] +choice(s[k]) -> s[k+1]`
2. 区间动态规划，`f(i,j)=max{f(i,k)+f(k+1,j)+cost}`

## 以爬楼梯为例比较递归和动态规划

⼀个⼈爬楼梯，每次只能爬 1 个或 2 个台阶，假设有 n 个台阶，那么这 个⼈有多少种不同的爬楼梯⽅法？思路： 由于第 n 级台阶⼀定是从 n - 1 级台阶或者 n - 2 级台阶来的，因此到第 n级台阶的数⽬就是 到第 n - 1 级台阶的数⽬加上到第 n - 1 级台阶的数 ⽬ 。

爬楼梯问题的递归写法
```
function dp(n) {
    if (n === 1) return 1;
    if (n === 2) return 2;
    return dp(n - 1) + dp(n - 2);
}
```
爬楼梯问题的动态规划写法，查表，即db table(dynamic_programming)，⼀般我们写的 dp table，数组的索引通常对应记忆化递归的函数参数， 值对应递归函数的返回值。
```
function climbStairs(n) {
    if (n == 1) return 1;
    const dp = new Array(n);
    dp[0] = 1;
    dp[1] = 2;
    for (let i = 2; i < n; i++) {
        dp[i] = dp[i - 1] + dp[i - 2];
    }
    return dp[dp.length - 1];
}
```
滚动数组优化。爬楼梯我们并没有必要使⽤⼀维数组，⽽是借助两个变量来实现的，空间 复杂度是 O(1)
```
function climbStairs(n) {
    if (n === 1) return 1;
    if (n === 2) return 2;
    let a = 1;
    let b = 2;
    let temp;
    for (let i = 3; i <= n; i++) {
        temp = a + b;
        a = b;
        b = temp;
    }
    return temp;
}
```
他们的区别只不过是递归⽤调⽤栈枚举状态， ⽽动态规划使⽤迭 代枚举状态。**如果说递归是从问题的结果倒推，直到问题的规模缩⼩到寻常。 那么动态规划就是从寻常⼊⼿， 逐步扩⼤规模到最优⼦结构**。动态规划性能通常更好。 ⼀⽅⾯是递归的栈开销，⼀⽅⾯是滚动数 组的技巧。PS： 动态规划不一定写出来是递归。

## 回溯法

**回溯是 DFS 中的⼀种技巧**。回溯法采⽤ 试错 的思想，它尝试分步的去解 决⼀个问题。在分步解决问题的过程中，当它通过尝试发现现有的分步答 案不能得到有效的正确的解答的时候，它将取消上⼀步甚⾄是上⼏步的计 算，再通过其它的可能的分步解答再次尝试寻找问题的答案。 通俗上讲，回溯是⼀种⾛不通就回头的算法。回溯的本质是穷举所有可能，尽管有时候可以通过剪枝去除⼀些根本不可 能是答案的分⽀， 但是从本质上讲，仍然是⼀种暴⼒枚举算法。回溯法可以抽象为树形结构，并且是是⼀颗⾼度有限的树（N 叉树）。回溯法解决的都是在集合中查找⼦集，集合的⼤⼩就是树的叉树，递归的深 度，构成树的⾼度。

算法模板
```
const visited = {}
function dfs(i) {
    if (满⾜特定条件）{
        // 返回结果 or 退出搜索空间
    }
    visited[i] = true // 将当前状态标为已搜索
    dosomething(i) // 对i做⼀些操作
    for (根据i能到达的下个状态j) {  // for 循环⽤来枚举分割点
        if (!visited[j]) {      // 如果状态j没有被搜索过
            dfs(j)
        }
    }
    undo(i) // 恢复i 
}
```

**回溯的本质就是暴⼒枚举所有可能**。要注意的是，由于回溯通常结果集都 记录在回溯树的路径上，因此如果不进⾏撤销操作， 则可能在回溯后状 态不正确导致结果有差异， 因此需要在递归到底部往上冒泡的时候进⾏ 撤销状态。

回溯题⽬的另外⼀个考点是剪枝， 通过恰当地剪枝，可以有效减少时 间，剪枝在每道题的技巧都是不⼀样的， 不过⼀个简单的原则就是**避免根本不可能是答案的递归**。

### 求子集

其中“ab”=“ba”

迭代法：

```
// i,j 相当于头尾指针
for(i=0;i<len;i++){
    for(j=i;j<len;j++){
        print(str,i,j);
    }
}
```
	
位图法

设原集合为`<a,b,c,d>`，数组A的某次“加一”后的状态为`[1,0,1,1]`，则本次输出的子集为`<a,c,d>`

那么从数字0 ==> [0,0,0,0]，每次加1 ==> [0,0,0,1]，一直到[1,1,1,1]


回溯法

此处参考[回溯法求幂集](http://www.cnblogs.com/youxin/p/3219523.html)对回溯法的描述。

幂集中的每个元素是一个集合，它或是空集，或含集合A中一个元素，或含集合A中两个元素…… 或等于集合A。反之，从集合A 的每个元素来看，它只有两种状态：它或属幂集的无素集，或不属幂集的元素集。则求幂集p(A)的元素的过程可看成是依次对集合A中元素进行“取”或“舍”的过程，并且可以用一棵二叉树来表示过程中幂集元素的状态变化过程，树中的根结点表示幂集元素的初始状态（空集）；叶子结点表示它的终结状态，而第i层的分支结点，则表示已对集合A中前i-1个元素进行了取舍处理的当前状态（左分支表示取，右分支表示舍 ）。因此求幂集元素的过程即为先序遍历这棵状态树的过程

```
public class B {
    private String str = "abc";
    private int[] x = new int[3];
    private void print(int[] x) {
        for (int i = 0; i < x.length; i++) {
            if (x[i] == 1) {
                System.out.print(str.charAt(i));
            }
        }
        System.out.println();
    }
    public void backtrack(int i) {
            /*
            x[i] 本质也是个位图，等x[i]从0到len赋值完毕，即可根据位图打印字符串
            */
        if (i >= str.length()) {
                // 到达叶子节点
            print(x);
        } else {
            for (int isChoose = 0; isChoose <= 1; isChoose++) {
                x[i] = isChoose;
                backtrack(i + 1);
            }
            /*相当于
                x[i] = 0;
                backtrack(i + 1);
                x[i] = 1;
                backtrack(i + 1);
                这就有了二叉树分叉的效果
            */
        }
    }
    public static void main(String[] args) {
        new B().backtrack(0);
    }
}
```

对于代码：

```
backtrack{
    if (i >= str.length()) {
        print(x);
    }else{
        x[i] = 0;
        backtrack(i + 1);
        x[i] = 1;
        backtrack(i + 1);
    }
}
```
	
对于局部变量，递归函数每次运行时，都是全新的。**如果递归函数操作全局数组，则在递归的过程中，就天然的有了二叉树分叉的效果。** 此处,`{x0 ==> x2 ==> x2}` 代表一条路径。

![](/public/upload/algorithm/backtrack_1.png)

可以推断，表示一个三叉树的先序遍历，需要在同一层调用三次递归方法，此时x[i]就是另外的含义了。

```
x[i] = 0;
backtrack(i + 1);
x[i] = 1;
backtrack(i + 1);
x[i] = 2;
backtrack(i + 1);
```
	
	
回溯时，还有另外一种实现思路，即使用xc[i]直接存储子集

```
private String str = "abc";
private char[] xc = new char[3];
public void backtrack(int i) {
    System.out.println(new String(xc));
    if (i >= str.length()) {
        return;
    } else {
        xc[i] = str.charAt(i);
        backtrack(i + 1);
        xc[i] = 0;
        backtrack(i + 1);
    }
}
```
    

### 全排列

[回溯之子集树和排列树（子集和问题）](http://www.cnblogs.com/youxin/p/4316325.html)

用树描述一下所有解空间，**注意，分叉的数目是逐渐减少的。**    

逐步构建全排列

![](/public/upload/algorithm/backtrack_4.png)

代码如下

	private String str = "abc";
    private int[] x = new int[3];
    private char[] xc = new char[3];

    public void backtrack(int i) {
        if (i >= str.length()) {
            System.out.println(new String(xc));
        } else {
            for (int t = 0; t < str.length(); t++) {
            	   // 如果str.charAt(t)还未被选入到xc中
                if (x[t] == 0) {
                    x[t] = 1;		
                    xc[i] = str.charAt(t);
                    backtrack(i + 1);
                    x[t] = 0;
                    xc[i] = 0;	
                }
            }
        }
    }

此处x[t]即表示特定位置的字符是否已被加入到xc，如已经加入，则不递归。因此循环每次都是从0开始，通过if 判断也达到减少分支的效果。此处x数组的最终结果都是111，`{000 ==> 100 ==> 110 ==> 111}` 代表一条路径。


全排列的另一种解法

![](/public/upload/algorithm/backtrack_3.png)

	private String str = "abc";
    public void backtrack(String str, int i) {
        if (i >= str.length()) {
            System.out.println(str);
        } else {
        	  // 每个字符都有当第一个字符的机会，所以n个字符开n个分叉，然后分叉逐渐减少
            for (int t = i; t < str.length(); t++) {
                str = swap(str, t, i);
                backtrack(str,i + 1);
                str = swap(str, i, t);
            }
        }
    }

此处，**swap函数用以实现每个字符当字符串首字符的效果**，达到该效果也可以采用其它函数，比如

	// 将index 字符作为首字符
    private String firstChar(String str, int index) {
        StringBuilder sb = new StringBuilder(str);
        sb.deleteCharAt(index);
        sb.insert(0, str.charAt(index));
        return sb.toString();
    }
    // 将首字符放在index位置，作为firstChar的反操作
    private String _firstChar(String str, int index) {
        StringBuilder sb = new StringBuilder(str);
        sb.deleteCharAt(0);
        sb.insert(index, str.charAt(0));
        return sb.toString();
    }
    public void backtrack(String str, int i) {
        if (i >= str.length()) {
            System.out.println(str);
        } else {
            for (int t = i; t < str.length(); t++) {
                str = firstChar(str, t);
                backtrack(str, i + 1);
                str = _firstChar(str, t);
            }
        }
    } 

只是需要firstChar和_firstChar两个函数，不如swap来得简洁。但一简洁，导致很多博客通过交换来理解全排列，增加的理解的难度。

### 子集 + 全排列

即对于子集“ab”!=“ba”

第一感觉的方法：先求所有子集，再对所有子集全排列。

此外，对本文的全排列方法进行调整，即可在递归全排列的过程中，顺带输出子集。

	public void backtrack(int i) {
		 // 全排列过程中，此时xc还未构造完毕，相当于输出了子集
	 	 System.out.println(new String(xc));
        if (i >= str.length()) {
           return;
        } else {
            for (int t = 0; t < str.length(); t++) {
                if (x[t] == 0) {
                    x[t] = 1;		
                    xc[i] = str.charAt(t);
                    backtrack(i + 1);
                    x[t] = 0;
                    xc[i] = 0;	
                }
            }
        }
    }



## 小结

计算机并没有“大招”，关键是思维不要乱窜

直接看回溯的代码，通常会感觉莫名其妙，也非常容易遗忘。对于刷题的同学来说，则是非常大的记忆负担。其中的关节，便是建立一系列的抽象，来屏蔽一些实现细节。假设一个复杂系统要三层抽象，每一层都需要一些复杂算法。那么第三层的细节与第二层的细节一下子涌入脑海，人就容易懵逼。

