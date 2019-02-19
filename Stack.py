#栈
class stack(object):
    def __init__(self,size):
        self.size = size
        self.stack = []
        self.top = 0
    def Getstack(self):
        if self.top != 0:
            return self.stack[self.top-1]
        else:
            return None
    def Push(self,item):
        if self.stackFull():
            raise Exception('overflow!')
        else:
            self.stack.append(item)
            self.top += 1 
    def Pop(self):
        if self.isEmpty():
            raise Exception('underflow!')
        else:
            self.top -= 1
            return self.stack.pop()
    def isEmpty(self):
        if self.top == 0:
            return True
        else:
            return False
    def stackFull(self):
        if self.top == self.size:
            return True
        else:
            return False


