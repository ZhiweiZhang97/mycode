#队列
class Queue(object):
    def __init__(self,MaxSize):
        self.size = MaxSize
        self.queue = []
        self.rear = 0  #队尾指针
        self.front = 0 #队头指针
    def isEmpty(self):
        if self.rear == self.front:
            return True
        else:
            return False
    def queueFull(self):
        if ((self.rear+1)%self.size) == self.front:
            return True
        else:
            return False
    def Enqueue(self,item):
        if self.queueFull():
            raise Exception('overflow!')
        else:
            self.queue.insert(0,item)
            self.rear = (self.rear+1)%self.size
    def Dequeue(self):
        if self.isEmpty():
            raise Exception('underflow!')
        else:
            item = self.queue.pop()
            self.front = (self.front+1)%self.size
            return item
    def Getqueue(self):
        if self.isEmpty():
            raise Exception('underflow!')
        else:
            return self.queue[self.size-1-self.front-1]
