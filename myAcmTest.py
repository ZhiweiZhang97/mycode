# mytest
################################################################################
# 求解最大利益
import math

def MaxInterest():
    intR = input()
    intR = intR.split()
    intR = [float(i) for i in intR]
    maxv = -2000000000
    minv = intR[0]
    for R in range(1,len(intR)):
        maxv = max(maxv,intR[R] - minv)
        minv = min(minv,intR[R])

    print('最大价格差：' + str(maxv))

################################################################################
#排序算法
#1 插入排序
def InsertionSort(insertion,d=1):
    for i in range(d,len(insertion)):
        v = insertion[i]
        j = i - d
        while j >= 0 and insertion[j] > v:
            insertion[j+d] = insertion[j]
            j -= d
        insertion[j+d] = v
    return insertion

#2 冒泡排序
def BubbleSort(bubble):
    flag = 1
    while flag:
        flag = 0
        for i in range(len(bubble)-1,0,-1):
            if bubble[i] < bubble[i-1]:
                temp = bubble[i]
                bubble[i] = bubble[i-1]
                bubble[i-1] = temp
                flag =1
    return bubble

#3 选择排序
def SelectionSort(selection):
    for i in range(0,len(selection)):
        mini = i
        for j in range(i,len(selection)):
            if selection[j] < selection[mini]:
                mini = j
        temp = selection[i]
        selection[i] = selection[mini]
        selection[mini] = temp
    return selection

#4 希尔排序
def ShellSort(shell):
    d = [i for i in range(1,len(shell),3)]
    for  i in range(len(d)-1,-1,-1):
        InsertionSort(shell,d[i])
    return shell

#5 归并排序
def Merge(merge,left,mid,right):
    cnt = 0
    n1 = mid - left 
    n2 = right - mid
    L = []
    R = []
    for i in range(0,n1):
        L.append(merge[left + i])
    for i in range(0,n2):
        R.append(merge[mid + i])
    L.append(200000000)
    R.append(200000000)
    i = 0
    j = 0    
    for k in range(left,right):
        if L[i] <= R[j]:
            merge[k] = L[i]
            i += 1
        else:
            merge[k] = R[j]
            j += 1
            cnt += n1 - i
    return cnt
def MergeSort(merge,left,right):
    if left + 1 < right:
        mid = int((left + right) / 2)
        MergeSort(merge,left,mid)
        MergeSort(merge,mid,right)
        Merge(merge,left,mid,right)
    return merge

#6 快速排序
def Partition(sort,low,high):
    #分割
    pivot = sort[high]
    i = low - 1
    for j in range(low,high):
        if sort[j] <= pivot:
            i += 1
            temp = sort[i]
            sort[i] = sort[j]
            sort[j] = temp
    temp = sort[i+1]
    sort[i+1] = sort[high]
    sort[high] = temp
    return i + 1
def QuickSort(quick,low,high):
    if low < high:
        pivot = Partition(quick,low,high)
        QuickSort(quick,low,pivot - 1)
        QuickSort(quick,pivot + 1,high)
    return quick

#7 计数排序
def CountingSort(count):
    k = max(count)
    aux = [0 for i in range(0,k+1)]
    sort = [i for i in count]
    for i in range(0,len(sort)):
        aux[sort[i]] += 1
    for i in range(1,k+1):
        aux[i] = aux[i] + aux[i-1]
    for i in range(len(sort)-1,-1,-1):
        count[aux[sort[i]]-1] = sort[i]
        aux[sort[i]] -= 1
    return count

#8 逆序数
def Inversions(inver,left,right):
    if left + 1 < right:
        mid = int((left + right) / 2)
        v1 = Inversions(inver,left,mid)
        v2 = Inversions(inver,mid,right)
        v3 = Merge(inver,left,mid,right)
        return v1 + v2 + v3
    return 0

#9 最小成本排序
def MinimumCostSort(sort):
    s = min(sort)
    ans = 0
    T = [None for i in range(0,1000)]
    aux = [i for i in sort]
    sign = [False for i in range(0,len(sort))]
    ShellSort(aux)
    for i in range(0,len(sort)):
        T[aux[i]] = i
    for i in range(0,len(sort)):
        if sign[i]:
            continue
        cur = i
        S = 0
        m = 1000
        an = 0
        while True:
            sign[cur] = True
            an += 1
            v = sort[cur]
            m = min(m,v)
            S += v
            cur = T[v]
            if sign[cur]:
                break
        ans += min(S+(an-2)*m,m+S+(an+1)*s)
    return ans
    
## 排序算法是否稳定
#
def Sort():
    sort = input('请输入待排序元素:')
    sort = sort.split()
    sort = [int(i) for i in sort]
    SortName = ['插入','冒泡','选择','希尔','归并','快速','计数']
    SortAlgorithm = [InsertionSort,BubbleSort,SelectionSort,ShellSort,
                     MergeSort,QuickSort,CountingSort]
    i = 7 - 1
    inver = [i for i in sort]
    Minimum = [i for i in sort]
    n = MinimumCostSort(Minimum)
    print('排序最小成本：',n)
    m=Inversions(inver,0,len(inver))
    print('逆序数为：',m)
    SortAlgorithm[i](sort)
    print(SortName[i] + '排序结果为：' + str(sort))

################################################################################
#栈的应用：逆波兰
from Stack import stack

def RPN():
    s = input('请输入逆波兰表达式：')
    s = s.split()
    mystack = stack(len(s))
    for i in s:
        if i == '+':
            a = mystack.Pop()
            b = mystack.Pop()
            mystack.Push(float(a) + float(b))
        elif i == '-':
            b = mystack.Pop()
            a = mystack.Pop()
            mystack.Push(float(a) - float(b))
        elif i == '*':
            a = mystack.Pop()
            b = mystack.Pop()
            mystack.Push(float(a) * float(b))
        elif i == '/':
            b = mystack.Pop()
            a = mystack.Pop()
            mystack.Push(float(a) / float(b))
        else:
            mystack.Push(i)
    print('表达式计算结果：' + str(mystack.Pop()))

#队列的应用：任务调度
from Queue import Queue
  
def scheduling():
    r = input('输入任务数与时间片的整数：')
    r = [int(i) for i in r.split()]
    s = input('输入待调度任务：')
    s = s.split()
    elaps = 0
    Q = Queue(r[0]+1)
    a = [s[i] for i in range(0,len(s),2)]
    b = [int(s[i]) for i in range(1,len(s),2)]
    s = [[a[i],b[i]] for i in range(0,r[0])]
    for  i in s:
        Q.Enqueue(i)
    print('调度结果(任务 完成时间)：')
    while not(Q.isEmpty()):
        u = Q.Dequeue()
        c = min(r[1],u[1])
        u[1] -= c   # 计算剩余的所需时间
        elaps += c  # 累积已经过的时间
        # 如果处理尚未结束则重新添加至队列
        if u[1] > 0:
            Q.Enqueue(u)
        else:
            print(u[0] +' '+str(elaps))

#计算面积
#输入：用"/"和"\"代表地形断面图中的斜面，用"_"代表平地.在一行之内完成输入
#输出：第一行输出该地区积水处横截面的总面积A.
#      第二行从左至右按顺序输出积水处的数量k，
#      以及各积水处的横截面积Li(i=1,2,...k),相邻数据用空格隔开

from PyList import DoubleLinkList
def calculateArea():
    figure = input('请输入积水处图形：')
    S1 = stack(len(figure)+1)
    S2 = stack(len(figure)+1)
    count_s = 0
    L=[]
    for i in range(0,len(figure)):
        if figure[i] == '\\':
            S1.Push(i)
        elif figure[i] == '/' and S1.top > 0:
            j = S1.Pop()
            count_s += i - j
            a = i - j
            while S2.top > 0 and S2.Getstack()[0] > j:
                a += S2.Getstack()[1]
                S2.Pop()
            S2.Push([j,a])
    print(count_s)
    Q = [S2.Pop()[1] for i in range(0,S2.top)]
    print(len(Q),end = ' ')
    for i in range(0,len(Q)):
        print(Q[len(Q)-i-1],end = ' ')

################################################################################
#搜索
#1 线性搜索
def linearSearch(array,key):
    for i in range(0,len(array)):
        if array[i] == key:
            return i
    return 0

#2 二分搜索
def binarySearch(array,key):
    left = 0
    right = len(array)
    while left < right:
        mid = int((left + right) / 2)
        if array[mid] == key:
            return mid
        elif key < array[mid]:
            right = mid
        else:
            left = mid + 1
    return 0

#
def Search():
    array = input('输入待搜索数组：')
    array = [i for i in array.split()]
    key = input('请输入关键字数组：')
    count = 1
    for i in key:
        if binarySearch(array,i):
            count += 1
    print('两个数组共有' + str(count) + '个重复项')

#3 散列法
#实现一个能执行以下命令的简易"字典"
#insert str：向字典中添加字符串str
#find str：当前字典中包含str时输出yes，不包含时输出no
#输入：第一行中输入命令数n.随后n行按顺序输入n个命令.命令格式如上
#输出：对于各find命令输出yes或no，每个输出占一行
#限制：输入的字符串仅由"A" "C" "G" "T"四种字母构成
#      1 <= 字符串长度 <= 12     n <= 1000000
M = 1046527
H = [None for i in range(0,1046527)]
def getChar(ch):
    if ch == 'A':return 1
    elif ch == 'C':return 2
    elif ch == 'G':return 3
    elif ch == 'T':return 4
    else:return 0
def getKey(Str):
    sum = 0
    p = 1
    for i in range(0,len(Str)):
        sum += p*(getChar(Str[i]))
        p *= 5
    return sum
def h1(key):return key % M
def h2(key):return 1 + (key % (M - 1))
def find(Str):
    key = getKey(Str)
    i=0
    while True:
        h = (h1(key) + i * h2(key)) % M
        if H[h] == Str:return 1
        elif H[h] == None:return 0
    return 0
def insert(Str):
    key = getKey(Str)
    i = 0
    while True:
        h = (h1(key) + i * h2(key)) % M
        if H[h] == Str:return 1
        elif H[h] == None:
            H[h] = Str
            return 0
    return 0
def Dictionary():
    count = int(input('请输入命令数：'))
    command = [find,insert]
    print('输入命令')
    for i in range(0,count):
        com_input = input()
        com_input = [i for i in com_input.split()]
        if com_input[0] == 'insert':
            insert(com_input[1])
        elif com_input[0] =='find':
            if find(com_input[1]):
                print('yes')
            else:
                print('no')

#搜索的应用——计算最优解
#传送带依次送来了总量分别为wi(i=0,1,...,n-1)的n个货物.现在要将这些货物装到k辆
#卡车上.每辆卡车可装载的货物数大于等于0，但货物重量总和不得超过卡车的最大运载
#量P.所有卡车的最大运载量P一致
#输入整数n、k、wi，求出装载全部货物所需的最大运载量P的最小值
T = [None for i in range(0,100000)]
def check(P,n,k):
    i = 0
    for j in range(0,k):
        s = 0
        while s + T[i] <= P:
            s += T[i]
            i += 1
            if i == n:
                return n
    return i
def solve(n,k):
    left = 0
    right = 100000 * 10000
    while (right - left) >1:
        mid = int((left + right) / 2)
        v = check(mid,n,k)
        if v >= n:
            right = mid
        else:
            left = mid
    return right
def Allocation():
    Input = input('请输入货物数量与卡车数量：')
    Input = [int(i) for i in Input.split()]
    print('分别输入每件货物的总量：')
    for i in range(0,Input[0]):
        T[i] = int(input())
    ans = solve(Input[0],Input[1])
    print(ans)

################################################################################
#递归和分治法
#1 计算n的阶乘的递归函数
def factorial(n):
    if n == 1:return 1
    return n * factorial(n - 1)
#2 求最大值的算法
def findMaximum(A,l,r):
    m = int((l + r) / 2)
    if l == r - 1:return A[l]
    else:
        u = findMaximum(A,l,m) #递归求解前半部分的局部问题
        v = findMaximum(A,m,r) #递归求解后半部分的局部问题
        x = max(u,v)
    return x

#3 穷举搜索
#判断长度为n的数列A中任意几个元素相加是否能得到整数m
#若能输出yes，反之输出no
def ExhaustiveSearch():
    def ExhaustiveSolve(i,m,n):
        if m == 0:return 1
        if i >= n:return 0
        res =  ExhaustiveSolve(i+1,m,n) or ExhaustiveSolve(i+1,m-A[i],n)
        return res
    n = int(input())
    A = input()
    A = [int(i) for i in A.split()]
    q = int(input())
    for i in range(0,q):
        m = int(input())
        if ExhaustiveSolve(0,m,n):print('yes')
        else:print('no')

#4 科赫曲线
#输入整数n,输出科赫曲线的顶点坐标,设端点为(0,0)、(100,0)
def KochCurve():
    def koch(n,a,b):
        if n == 0:return
        s = []
        t = []
        u = []
        th = math.pi * 60 / 180
        s.append((2 * a[0] + 1 * b[0])/3.0)
        s.append((2 * a[1] + 1 * b[1])/3.0)
        t.append((1 * a[0] + 2 * b[0])/3.0)
        t.append((1 * a[1] + 2 * b[1])/3.0)
        u.append((t[0] - s[0]) * math.cos(th) - (t[1] - s[1]) * math.sin(th) + s[0])
        u.append((t[0] - s[0]) * math.sin(th) - (t[1] - s[1]) * math.cos(th) + s[1])
        koch(n-1,a,s)
        print(s[0],s[1])
        koch(n-1,s,u)
        print(u[0],u[1])
        koch(n-1,u,t)
        print(t[0],t[1])
        koch(n-1,t,b)
    n = int(input('请输入科赫曲线层数：'))
    a = [0,0]
    b = [100,0]
    print(a[0],a[1])
    koch(n,a,b)
    print(b[0],b[1])

################################################################################
#树
#树的遍历应用
#现有两个结点序列，分别是对同一个二叉树进行前序遍历和中序遍历的结果。
#请编写一个程序，输出二叉树按后序遍历时的结点序列
def get_PostOrder(PreOrder, InOrder, Post):
    if len(PreOrder) == 1:
        Post.append(PreOrder[0])
        return
    if len(PreOrder) == 0:
        return
    root = PreOrder[0]
    root_index = InOrder.index(root)
    get_PostOrder(PreOrder[1:root_index + 1], InOrder[:root_index], Post)
    get_PostOrder(PreOrder[root_index + 1:], InOrder[root_index + 1:], Post)
    Post.append(root)
    return Post

from Tree import BiTree
from Tree import BiSearchTree
def tree():
    print('二叉树...')
    B = BiTree()
    T = ['A','B','C','D','E','F','G','H','I','J','K']
    for i in range(len(T)):
        B.BiTreeInsert(T[i])
    B.PrintBiTree(B.root,0)
    print('层次遍历：',end = '')
    B.LevelOrder()
    print('前序遍历：',end = '')
    B.PreOrder(B.root)
    print()
    print('中序遍历：',end = '')
    B.InOrder(B.root)
    print()
    print('后序遍历：',end = '')
    B.PostOrder(B.root)
    print()
    print('二叉树的深度为：',B.TreeDepth(B.root))
    PreOrder = ['A','B','D','H','I','E','J','K','C','F','G']
    InOrder = ['H','D','I','B','J','E','K','A','F','C','G']
    Post = []
    get_PostOrder(PreOrder, InOrder, Post)
    print('后序遍历为：',Post)
    print('二叉搜索树...')
    Bs = BiSearchTree()
    T = [20,50,11,12,99,51,30,42,5,10]
    for i in T:
        Bs.Insert(i)
    Bs.PrintBiSearchTree(Bs.root,0)
    key = Bs.Search(12)
    minkey = Bs.get_Min()
    print('最小值为：',minkey.key)
    maxkey = Bs.get_Max()
    print('最大值为：',maxkey.key)
    print('==========')
    Bs.Delete(88)
    Bs.Delete(20)
    Bs.PrintBiSearchTree(Bs.root,0)
    minkey = Bs.get_Min()
    print('最小值为：',minkey.key)
    maxkey = Bs.get_Max()
    print('最大值为：',maxkey.key)
#堆
def maxHeapify(A,i,H):
    l = 2 * i + 1
    r = 2 * i + 2
    if l < H and A[l] > A[i]:
        largest = l
    else:
        largest = i
    if r < H and A[r] > A[largest]:
        largest = r
    if largest != i:
        temp = A[i]
        A[i] = A[largest]
        A[largest] = temp
        maxHeapify(A,largest,H)
def buildMaxHeap(A,H):
    for i in range(int(H/2)-1,-1,-1):
        maxHeapify(A,i,H)
def Heap():
    #H = int(input('输入堆的大小：'))
    #Input = input()
    #A = [int(i) for i in Input.split()]
    A = [4,1,3,2,16,9,10,14,8,7]
    H = len(A)
    B = BiTree()
    for i in A:
        B.BiTreeInsert(i)
    B.PrintBiTree(B.root,0)
    print('======')
    buildMaxHeap(A,H)
    B = BiTree()
    for i in A:
        B.BiTreeInsert(i)
    B.PrintBiTree(B.root,0)
    print(A)
#优先级队列
#Input:对优先级队列S输入多条命令。命令以insert K、extract、end的形式给出
#      每个命令占一行。这里的k代表插入的整数，end代表命令输入完毕
#Output:每执行一次extract命令，就输出一个从优先级队列S中取出的数，每个值占一行
INFTY = 999999999999
A = [None for i in range(0,10)]
H = 0
def pri_insert(A,key):
    global H
    H += 1
    A[H-1] = -INFTY
    heapIncreaseKey(A, H-1, key)

def heapIncreaseKey(A, i, key):
    if key < A[i]:
        return
    A[i] = key
    while i > 0 and A[math.ceil(i/2)-1] < A[i]:
        temp = A[i]
        A[i] = A[math.ceil(i/2)-1]
        A[math.ceil(i/2)-1] = temp
        i = math.ceil(i/2)-1
def heapExtractMax(A):
    global H
    if H < 0:
        return -INFTY
    Maxv = A[0]
    A[0] = A[H-1]
    H -= 1
    maxHeapify(A,0,H)
    return Maxv
def priority_queue():
    while True:
        Input = input()
        if Input == 'end':
            break
        Input = Input.split()
        if Input[0] == 'insert':
            pri_insert(A,int(Input[1]))
        if Input[0] == 'extract':
            Maxv = heapExtractMax(A)
            print(Maxv)
        else:
            print('Input Error')

################################################################################
#动态规划
'''
def dp_solve(i,m):
    if dp[i][m]:
        return dp[i][m]
    if m == 0:
        dp[i][m] = True
    elif i >= n:
        fp[i][m] = False
    elif dp_solve(i+1,m):
        dp[i][m] = True
    elif dp_solve(i+1,m-A[i]):
        dp[i][m] = True
    else:
        dp[i][m] = False
    return dp[i][m]
'''
#斐波那契数列
dp = [None for i in range(50)]
def fibonacci(n):
    #递归法
    if(n == 1 or n == 0):
        dp[n] = 1
        return 1
    if dp[n]:
        return dp[n]
    dp[n] = fibonacci(n-2) + fibonacci(n-1)
    return dp[n]
def makeFibonacci(n = 5):
    dp = [None for i in range(n+1)]
    dp[0] = dp[1] = 1
    for i in range(2,n+1):
        dp[i] = dp[i-2] + dp[i-1]
    print(dp)
#最长公共子序列
#Input:给定多组数据。第一行输入数组组数q。接下来的2*q行输入数据组，每组数据包含
#      X、Y共2个字符串，每个字符串占一行
#Output:输出每组X、Y的最长公共子序列Z的长度，每个长度占一行
import numpy as np
def lcatest():
    def LCA(X,Y):
        X = ' ' + X
        Y = ' ' + Y
        m = len(X)
        n = len(Y)
        dp = np.zeros((m,n))
        maxl = 0
        for i in range(1,m):
            for j in range(1,n):
                if X[i] == Y[j]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i][j-1],dp[i-1][j])
                maxl = max(maxl,dp[i][j])
        return maxl
    X = 'abcbdab'
    Y = 'bdcaba'
    maxl = LCA(X,Y)
    print('字符串',X,'和',Y,'的最长公共子序列的长度为：',str(maxl))
#矩阵链乘法
#Input:第一行输入矩阵数n。接下来n行输入矩阵Mi（i = 1...n）的维数r,c。其中r代表矩
#      阵的行数，c代表矩阵的列数，r、c均为整数，用空格隔开
#Output:输出最少次数，占一行
def matrixChainMultiplication():
    n = 6
    p = [30,35,35,15,15,5,10,10,20,20,25]
    m = np.zeros((n+1,n+1))
    for i in range(1,n+1):
        m[i][i] = 0
    for l in range(2,n+1):
        for i in range(1,n-l+2):
            j = i + l - 1
            m[i][j] = INFTY
            for k in range(i,j):
                m[i][j] = min(m[i][j], m[i][k] + m[k+1][j] + p[i-1] * p[k] * p[j])
    print(m[1][n-1])

################################################################################
#图
#请编写一个程序，将以邻接表形式给出的有向图G以邻接矩阵形式输出。G包含n( = |V|)个
#顶点，编号分别为1至n
#Input:第一行输入G的顶点数n，接下来的n行，按照下述格式输入各顶点u的邻接表Adj[u].
#      u k v1 v2 ... vk
#      其中u为顶点编号，k为u的度，v1 v2 ...vk为与u相邻的顶点编号
#Output:输出G的邻接矩阵。aij之间用一个空格隔开。
def adj_Graph():
    n = int(input('请输入图G的顶点数n：'))
    print('以邻接表表示法输入顶点编号：')
    count = 0
    adj = np.zeros((n,n))
    while(count < n):
        Input = [int(i) for i in input().split()]
        u,k = Input[0],Input[1]
        if k != 0:
            for i in range(2,len(Input)):
                v = Input[i]
                adj[u-1][v-1] = 1
        count += 1
    for i in range(n):
        for j in range(n):
            print(int(adj[i][j]),' ',end = '')
        print()
#
from Graph import ALGraph
def Graph():
    V = ['A','B','C','D','E']
    Graph = ALGraph(len(V))
    for i in V:
        Graph.addVertices(i)
    Graph.addEdge('A','B')
    Graph.addEdge('A','E')
    Graph.addEdge('B','A')
    Graph.addEdge('B','E')
    Graph.addEdge('B','C')
    Graph.addEdge('B','D')
    Graph.addEdge('C','B')
    Graph.addEdge('C','D')
    Graph.addEdge('D','B')
    Graph.addEdge('D','A')
    Graph.addEdge('D','E')
    Graph.addEdge('D','C')
    Graph.addEdge('E','D')
    Graph.addEdge('E','A')
    Graph.addEdge('E','B')
    Graph.addEdge('E','B')
    Graph.addEdge('F','B')
    Graph.printALGraph()
    print('广度优先搜索:',Graph.BFSTraverse())
    print('深度优先搜索:',Graph.DFSTraverse())
    if Graph.Adjacent('E','B'):
        print('存在边<E,B>')
    else:
        print('不存在边<E,B>')
    if Graph.Adjacent('F','B'):
        print('存在边<F,B>')
    else:
        print('不存在边<F,B>')
    print('与顶点A邻接的边为',Graph.Neighbors('A'))
    Graph.deleteEdge('B','E')
    print('------')
    Graph.printALGraph()
    Graph.deleteVertices('A')
    print('------')
    Graph.printALGraph()
    print('顶点B的第一个邻接点为',Graph.FirstNeighbor('B'))
    print('顶点C除了B之外下一个邻接点为',Graph.NextNeighbor('C','B'))
