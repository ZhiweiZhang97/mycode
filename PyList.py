#链表
################################################################################
#单链表
class SingleListNode():
    def __init__(self,Elem = None):
        self.elem = Elem
        self.next = None
class LinkList():
    def __init__(self):
        self.length = 0
        self.head = SingleListNode()
    def isEmpty(self):
        return self.length == 0
    def ListInsert(self,elem):
        if isinstance(elem,SingleListNode):
            node = elem
        else:
            node = SingleListNode(elem)
        '''
        if self.isEmpty():
            self.head = node
        else:
        '''
        p = self.head
        while p.next:
            p = p.next
        p.next = node
        self.length += 1
    def ListInsert_Index(self,elem,index):
        if index > self.length:
            return 'Error'
        if isinstance(elem, SingleListNode):
            node = elem
        else:
            node = SingleListNode(elem)
        '''
        if index == 0:
            node.next = self.head
            self.head = node
        else:
        '''
        p = self.head
        while index: #index-1
            p = p.next
            index -= 1
        node.next = p.next
        p.next = node
        self.length += 1
    def ListDelect(self,index):
        if not 0 <= index < self.length:
            return 'Error'
        '''
        if index == 0:
            self.head = self.head.next
        else:
        '''
        p = self.head
        while index: #index-1
            p = p.next
            index -= 1
        p.next = p.next.next
        self.length -= 1
    def ListNode_Update(self,elem,index):
        if not 0 <= index < self.length:
            return 'Error'
        '''
        if index == 0:
            self.head.elem = elem
        else:
        '''
        p = self.head.next # p = self.head
        while index:
            p = p.next
            index -= 1
        p.elem = elem
    def ListGet(self,index):
        if not 0 <= index < self.length:
            return 'Error'
        '''
        if index == 0:
            return self.head.elem
        '''
        p = self.head.next # p = self.head
        while index:
            p = p.next
            index -= 1
        return p.elem
    def ListLength(self):
        return self.length
    def ClearList(self):
        self.head = SingleListNode()
        self.length = 0
    def PrintList(self):
        if self.length == 0:
            return None
        else:
            p = self.head.next
            print('Head->',end = '')
            while p.next:
                print(p.elem,'-->',end = '',sep = '')
                p = p.next
            print(p.elem)

#双向链表
class DoubleListNode():
    def __init__(self,elem=None):
        self.elem = elem
        self.prev = None
        self.next = None
class DoubleLinkList():
    def __init__(self):
        self.head = DoubleListNode() 
        self.tail = DoubleListNode() 
        self.head.next = self.tail
        self.tail.prev = self.head
        self.length = 0
    def isEmpty(self):
        if self.head.next == self.tail and self.tail.prve == self.head:
            return True
        else:
            return False
    def ListInsert(self,elem):
        if isinstance(elem,DoubleListNode):
            node = elem
        else:
            node = DoubleListNode(elem)
        node.next = self.tail
        node.prev = self.tail.prev
        self.tail.prev.next = node
        self.tail.prev = node
        self.length += 1
    def ListGet(self,index):
        index = index if index >= 0 else self.length + index
        if index >= self.length or index < 0:
            return 'error'
        p = self.head.next
        while index:
            p = p.next
            index -= 1
        return p.elem
    def ListNode_Update(self,elem,index):
        if not 0 <= index < self.length:
            return 'Error'
        p = self.head.next
        while index:
            p = p.next
            index -= 1
        p.elem = elem
    def ListInsert_Index(self,elem,index):
        index = index if index >= 0 else self.length + index
        if index > self.length:
            return 'Error'
        if isinstance(elem, DoubleListNode):
            node = elem
        else:
            node = DoubleListNode(elem)
        if index == 0:
            node.next = self.head.next
            self.head.next.prev = node
            self.head.next = node
            node.prev = self.head
        else:
            p = self.head.next
            while index-1:
                p = p.next
                index -= 1
            node.next = p.next
            p.next.prev = node
            p.next = node
            node.prev = p
        self.length += 1
    def ListDelect(self,index):
        if not 0 <= index < self.length:
            return 'error'
        p = self.head
        while index:
            p = p.next
            index -= 1
        p.next.next.prev = p
        p.next = p.next.next
        self.length -= 1
    def ListLength(self):
        return self.length
    def ClearList(self):
        self.head.next = self.tail
        self.tail.prev = self.head
        self.length = 0
    def PrintList(self):
        if self.length == 0:
            return None
        else:
            p = self.head.next
            print('Head<==>',end = '')
            while p.next:
                print(p.elem,'<==>',end = '',sep = '')
                p = p.next
            print('Tail')

'''
if __name__ == '__main__':
    a = SingleListNode(1)
    b = SingleListNode(2)
    c = SingleListNode(3)
    d = SingleListNode(4)
    e = SingleListNode(5)
    s = LinkList()
    s.ListInsert(a)
    s.ListInsert(b)
    s.ListInsert(c)
    s.PrintList()
    s.ListInsert_Index(6,2)
    print(s.head.next.elem)
    print(s.head.next.next.elem)
    print(s.head.next.next.next.elem)
    s.ListInsert_Index(d,0)
    print(s.head.next.elem)
    print(s.head.next.next.elem)
    print(s.head.next.next.next.elem)
    s.ListDelect(0)
    print(s.head.next.elem)
    print(s.head.next.next.elem)
    print(s.head.next.next.next.elem)
    s.ListDelect(2)
    print(s.head.next.elem)
    print(s.head.next.next.elem)
    print(s.head.next.next.next.elem)
    s.ListNode_Update(0,1)
    print(s.head.next.elem)
    print(s.head.next.next.elem)
    print(s.head.next.next.next.elem)
    print(s.ListGet(0))
    print(s.ListGet(1))
    print(s.ListLength())
    s.PrintList()

if __name__ == '__main__':
    a = DoubleListNode(1)
    b = DoubleListNode(2)
    c = DoubleListNode(3)
    d = DoubleListNode(4)
    e = DoubleListNode(5)
    s = DoubleLinkList()
    s.ListInsert(a)
    s.ListInsert(b)
    s.ListInsert(c)
    s.PrintList()
    s.ListInsert_Index(6,2)
    print(s.head.next.elem)
    print(s.head.next.next.elem)
    print(s.head.next.next.next.elem)
    s.ListInsert_Index(d,0)
    print(s.head.next.elem)
    print(s.head.next.next.elem)
    print(s.head.next.next.next.elem)
    s.ListDelect(0)
    print(s.head.next.elem)
    print(s.head.next.next.elem)
    print(s.head.next.next.next.elem)
    s.ListDelect(2)
    print(s.head.next.elem)
    print(s.head.next.next.elem)
    print(s.head.next.next.next.elem)
    s.ListNode_Update(0,1)
    print(s.head.next.elem)
    print(s.head.next.next.elem)
    print(s.head.next.next.next.elem)
    print(s.ListGet(0))
    print(s.ListGet(1))
    print(s.ListLength())
    s.PrintList()
'''
