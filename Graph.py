# 图
#有向图 邻接矩阵存储
import numpy as np
class AdjMGraph():
    def __init__(self,MaxVertices = 100):
        self.Vertices = []
        self.edge = np.zeros((MaxVertices,MaxVertices),int)
        self.numOfEdges = 0
    def addVertices(self,elem):
        self.Vertices.append(elem)
    def addEdge(self,Ver1,Ver2,weight = 1):
        v1,v2 = self.Vertices.index(Ver1),self.Vertices.index(Ver2)
        self.edge[v1][v2] = weight
        #self.edge[v1][v2],self.edge[v2][v1] = weight,weight #无向图
        self.numOfEdges += 1
    def deleteEdge(self,Ver1,Ver2):
        v1,v2 = self.Vertices.index(Ver1),self.Vertices.index(Ver2)
        self.edge[v1][v2] = 0
        #self.edge[v1][v2],self.edge[v2][v1] = 0,0 #无向图
        self.numOfEdges -= 1
    def deleteVertices(self,elem):                      
        for v in range(len(self.Vertices)):
            if self.Vertices[v] == elem:
                self.edge = np.delete(self.edge,v,axis = 0)
                self.edge = np.delete(self.edge,v,axis = 1)
                self.Vertices.remove(elem)
                return
        print('不存在顶点',elem)
    def Adjacent(self,Ver1,Ver2):
        i,j = self.Vertices.index(Ver1),self.Vertices.index(Ver2)
        if self.edge[i][j] != 0:
        #if self.edge[i][j] != 0 and self.edge[j][i] != 0:
            return 1
        return 0
    def Neighbors(self,vertice):
        NeighborEdge = []
        i = self.Vertices.index(vertice)
        for j in range(len(self.Vertices)):
            if self.edge[i][j] != 0:
                NeighborEdge.append(vertice+'-->'+self.Vertices[j])
        return NeighborEdge
    def printAdjMGraph(self):
        for u in range(len(self.Vertices)):
            for v in range(len(self.Vertices)):
                print(Graph.edge[u][v],' ',end = '')
            print()
'''
if __name__ == '__main__':
    V = ['A','B','C','D','E']
    Graph = AdjMGraph(len(V))
    for i in V:
        Graph.addVertices(i)
    Graph.addEdge('A','B')
    Graph.addEdge('A','C')
    Graph.addEdge('A','D')
    Graph.addEdge('B','E')
    Graph.addEdge('C','D')
    if Graph.Adjacent('A','B'):
        print('存在边<A,B>')
    else:
        print('不存在边<A,B>')
    print('与顶点A邻接的边为',Graph.Neighbors('A'))
    print(Graph.Vertices)
    Graph.printAdjMGraph()
    Graph.deleteEdge('A','B')
    print()
    Graph.printAdjMGraph()
    Graph.deleteVertices('B')
    print(Graph.Vertices)
    Graph.printAdjMGraph()
'''
#邻接表存储
from Queue import Queue
class ArcNode():
    def __init__(self,adjvex,weight):
        self.adjvex = adjvex
        self.weight = weight
        self.next = None
class VNode():
    def __init__(self,elem):
        self.elem = elem
        self.first = None
class ALGraph():
    def __init__(self,MaxVertices = 100):
        self.vertices = []
        self.vexnum = 0
        self.arcnum = 0
    def locationOfVex(self,vertice):
        for i in range(len(self.vertices)):
            if self.vertices[i].elem == vertice:
                return i
        return -1
    def Adjacent(self,Ver1,Ver2):
        for i in range(len(self.vertices)):
            if self.vertices[i].elem == Ver1:
                p = self.vertices[i].first
                while p:
                    if p.adjvex == Ver2:
                        return 1
                    p = p.next
        return 0
    def Neighbors(self,vertice):
        NeighborsEdge = []
        for i in range(len(self.vertices)):
            if self.vertices[i].elem == vertice:
                p = self.vertices[i].first
                while p:
                    NeighborsEdge.append('<'+vertice+','+p.adjvex+'>')
                    p = p.next
                break
        return NeighborsEdge
    def FirstNeighbor(self,vertice):
        i = self.locationOfVex(vertice)
        if i != -1:
            if self.vertices[i].first:
                return self.vertices[i].first.adjvex
        return -1
    def NextNeighbor(self,vertice,Ver):
        i = self.locationOfVex(vertice)
        if i != -1:
            p = self.vertices[i].first
            while p:
                if p.adjvex == Ver:
                    break
                p = p.next
            if p.next:
                return p.next.adjvex
        return -1
    def addVertices(self,vertices):
        item = VNode(vertices)
        self.vertices.append(item)
        self.vexnum += 1
    def addEdge(self,Ver1,Ver2,weight = 1):
        item = ArcNode(Ver2,weight)
        if self.Adjacent(Ver1,Ver2):
            return
        i = self.locationOfVex(Ver1)
        if i != -1:
            if self.vertices[i].first == None:
                self.vertices[i].first = item
            else:
                p = self.vertices[i].first
                q = self.vertices[i].first.next
                while q:
                    q = q.next
                    p = p.next
                p.next = item
            self.arcnum += 1
            return
        print('不存在顶点',Ver1)
        return
    def deleteEdge(self,Ver1,Ver2):
        '''删除有向图边V1，V2'''
        if not self.Adjacent(Ver1,Ver2):
            #print('不存在',Ver1,'到',Ver2,'的边')
            return
        for i in range(len(self.vertices)):
            if self.vertices[i].elem == Ver1:
                if self.vertices[i].first.adjvex == Ver2:
                    self.vertices[i].first = self.vertices[i].first.next
                    self.arcnum -= 1
                    return
                else:
                    p = self.vertices[i].first
                    q = self.vertices[i].first.next
                    while q.adjvex != Ver2:
                        p = p.next
                        q = q.next
                    p.next = q.next
                    self.arcnum -= 1
                    return
    def deleteNEdge(self,Ver1,Ver2):
        '''删除无向图边V1，V2'''
        self.deleteEdge(Ver1,Ver2)
        self.deleteEdge(Ver2,Ver1)
    def deleteVertices(self,vertices):
        for i in range(len(self.vertices)):
            if self.vertices[i].elem == vertices:
                NeighborsEdge = self.Neighbors(vertices)
                self.arcnum -= len(NeighborsEdge)
                self.vertices[i] = vertices
                self.vertices.remove(vertices)
                self.vexnum -= 1
                break
        for i in range(len(self.vertices)):
            self.deleteEdge(self.vertices[i].elem,vertices)    
    def BFSTraverse(self):
        def BFS(v):
            visit.append(self.vertices[v].elem)
            visited[v] = True
            Q.Enqueue(v)
            while not Q.isEmpty():
                v = Q.Dequeue()
                location = self.locationOfVex(self.vertices[v].elem)
                vertices = []
                if location != -1:
                    p = self.vertices[location].first
                    while p:
                        vertices.append(p.adjvex)
                        p = p.next 
                for item in vertices:
                    w = self.locationOfVex(item)
                    if not visited[w]:
                        visit.append(item)
                        visited[w] = True
                        Q.Enqueue(w)
        Q = Queue(20)
        visited = [False for i in range(self.vexnum)]
        visit = []
        for i in range(self.vexnum):
            if not visited[i]:
                BFS(i)
        return visit
    def DFSTraverse(self):
        def DFS(v):
            visit.append(self.vertices[v].elem)
            visited[v] = True
            location = self.locationOfVex(self.vertices[v].elem)
            vertices = []
            if location != -1:
                p = self.vertices[location].first
                while p:
                    vertices.append(p.adjvex)
                    p = p.next
            for item in vertices:
                w = self.locationOfVex(item)
                if not visited[w]:
                    DFS(w)
        visited = [False for i in range(self.vexnum)]
        visit = []
        for i in range(self.vexnum):
            if not visited[i]:
                DFS(i)
        return visit
    def printALGraph(self):
        for i in range(len(self.vertices)):
            print(self.vertices[i].elem,'-->',end = '')
            q = self.vertices[i].first
            if q != None:
                p = self.vertices[i].first.next
                while p:
                    print(q.adjvex,'|',q.weight,'-->',end = '')
                    p = p.next
                    q = q.next
                print(q.adjvex,'|',q.weight)
            else:
                print('None')

'''
if __name__ == "__main__":
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
'''
