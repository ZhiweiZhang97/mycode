#树
#二叉树
from Queue import Queue
class BiTreeNode():
    def __init__(self,elem=None,left=None,right=None):
        self.elem = elem
        self.lchild = left
        self.rchild = right
class BiTree():
    def __init__(self):
        self.root = BiTreeNode()
    def isEmpty(self):
        tree_node = self.root
        if tree_node.elem == None:
            return True
        else:
            return False
    def BiTreeInsert(self,elem):
        if isinstance(elem,BiTreeNode):
            node = elem
        else:
            node = BiTreeNode(elem)
        if self.isEmpty():
            self.root = node
        else:
            tree_node = self.root
            queue = Queue(20)
            queue.Enqueue(self.root)
            while not queue.isEmpty():
                tree_node = queue.Dequeue()
                if tree_node.lchild == None:
                    tree_node.lchild = node
                    return
                elif tree_node.rchild == None:
                    tree_node.rchild = node
                    return
                else:
                    if not tree_node.lchild == None:
                        queue.Enqueue(tree_node.lchild)
                    if not tree_node.rchild == None:
                        queue.Enqueue(tree_node.rchild)
    def BiTreeDelect(self,tree_node):
        #~~
        if self.isEmpty():return
        elif self.root == tree_node:
            self.root.lchild = None
            self.root.rchild = None
            self.root.elem = None
            return 
        else:
            queue = Queue(20)
            queue.Enqueue(self.root)
            while not queue.isEmpty():
                tree_node = queue.Dequeue()
                if tree_node == tree_node:
                   tree_node = None
                   return
                else:
                    if not tree_node.lchild == None:
                        queue.Enqueue(tree_node.lchild)
                    if not tree_node.rchild == None:
                        queue.Enqueue(tree_node.rchild)
    def LevelOrder(self):
        #层次遍历
        if self.isEmpty():
            return
        else:
            tree_node = self.root
            queue = Queue(20)
            queue.Enqueue(self.root)
            while not queue.isEmpty():
                tree_node = queue.Dequeue()
                print(tree_node.elem,' ',end = '')
                if not tree_node.lchild == None:
                    queue.Enqueue(tree_node.lchild)
                if not tree_node.rchild == None:
                    queue.Enqueue(tree_node.rchild)
        print()
    def PreOrder(self,BiTree_node):
        #前序遍历
        if BiTree_node == None:
            return
        print(BiTree_node.elem,' ',end = '')
        self.PreOrder(BiTree_node.lchild)
        self.PreOrder(BiTree_node.rchild)
    def InOrder(self,BiTree_node):
        #中序遍历
        if BiTree_node == None:
            return
        self.InOrder(BiTree_node.lchild)
        print(BiTree_node.elem,' ',end = '')
        self.InOrder(BiTree_node.rchild)
    def PostOrder(self,BiTree_node):
        #后序遍历
        if BiTree_node == None:
            return
        self.PostOrder(BiTree_node.lchild)
        self.PostOrder(BiTree_node.rchild)
        print(BiTree_node.elem,' ',end = '')
    def PrintBiTree(self,BiTree_node,n):
        #打印二叉树
        if BiTree_node == None:
            return
        self.PrintBiTree(BiTree_node.rchild,n+1)
        for i in range(0,n):
            print('  ',end = '')
        if n >= 0:
            print('---',end = '')
            print(BiTree_node.elem)
        self.PrintBiTree(BiTree_node.lchild,n+1)
    def TreeDepth(self,BiTree_node):
        if BiTree_node == None:
            return 0
        #elif BiTree_node.lchild == None and BiTree_node.rchild == None:
         #   return 1
        else:
            ldepth = self.TreeDepth(BiTree_node.lchild) + 1
            rdepth = self.TreeDepth(BiTree_node.rchild) + 1
            Depth = max(ldepth,rdepth) 
            return Depth

#二叉搜索树
class BiSearchTreeNode():
    def __init__(self,key):
        self.key = key
        self.left = None
        self.right = None
        self.parent = None
class BiSearchTree():
    def __init__(self):
        self.root = None
    def Insert(self,elem):
        node = BiSearchTreeNode(elem)
        cur= self.root
        parent = None
        while cur != None:
            if cur.key == elem:
                return -1
            parent = cur
            if elem < cur.key:
                cur = cur.left
            else:
                cur = cur.right
        node.parent = parent
        if parent == None:
            self.root = node
        elif elem < parent.key:
            parent.left = node
        else:
            parent.right = node
        
    def Search(self,elem):
        cur = self.root
        while cur != None:
            if cur.key == elem:
                return cur
            elif cur.key < elem:
                cur = cur.right
            else:
                cur = cur.left
        return None
    
    def Delete(self,elem):
        node = self.Search(elem)
        if node == None:
            return 'delete failed'
        parent = node.parent
        if node.left == None:
            if parent == None:
                self.root = node.right
                if node.right != None:
                    node.right.parent = None
            elif parent.left == node:
                parent.left = node.right
                if node.right != None:
                    node.right.parent = parent
            else:
                parent.right = node.right
                if node.right != None:
                    node.right.parent = parent
            return 'delete successfully'
        tmpNode = node.left
        while tmpNode.right != None:
            tmpNode = tmpNode.right
     
        tmpNode.right = node.right
        if node.right != None:
            node.right.father = tmpNode
     
        if parent == None:
            self.root = node.left
            node.left.parent = None
        elif parent.left == node:
            parent.left = node.left
            node.left.parent = parent
        else:
            parent.right = node.left
            node.left.parent = parent
        node = None
        return 'delete successfully'

    def get_Min(self):
        if self.root == None:
            return 'the tree is empty'
        else:
            node = self.root
            while(node != None):
                cur = node
                node = node.left
        return cur
    def get_Max(self):
        if self.root == None:
            return 'the tree is empty'
        else:
            node = self.root
            while(node != None):
                cur = node
                node = node.right
        return cur
    def PrintBiSearchTree(self,BiTree_node,n):
        #打印二叉树
        if BiTree_node == None:
            return
        self.PrintBiSearchTree(BiTree_node.right,n+1)
        for i in range(0,n):
            print('  ',end = '')
        if n >= 0:
            print('---',end = '')
            print(BiTree_node.key)
        self.PrintBiSearchTree(BiTree_node.left,n+1)

