import numpy as np

class kdNode():
    def __init__(self, x, y):
        self.value = x
        self.dimension = y
        self.left = None
        self.right = None

def loadData():
    T = np.array([[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]])
    return T

def buildTree(T, Depth):
    if len(T) == 0:
        return None
    k = T.shape[1]
    T = T[T[:, int(Depth % k)].argsort()]
    mid = T.shape[0] // 2
    root = kdNode(T[mid], Depth % k)
    root.left = buildTree(T[:mid], Depth + 1)
    root.right = buildTree(T[mid + 1:], Depth + 1)
    if root.left != None :
        root.left.fa = root
    if root.right != None :
        root.right.fa = root
    return root

def showKdTree(root):
    queue = [root]
    level = 0
    while len(queue) != 0:
        temp = []
        print("第%d层结点:"%level)
        level += 1
        for i in range(len(queue)):
            print(queue[i].value)
            if queue[i].left != None :
                temp.append(queue[i].left)
            if queue[i].right != None:  
                temp.append(queue[i].right)
        queue = temp
    
def find_closet(point, x, min_dis, clost):
    if point == None :
        return 
    cur_distance = (sum((x[:] - point.value[:]) ** 2)) ** 0.5
    # print(cur_distance)
    if min_dis[0] < 0 or cur_distance < min_dis[0]:
        min_dis[0] = cur_distance
        for i in range(len(point.value)):   
            clost[i] = point.value[i]
    if x[point.dimension] <= point.value[point.dimension] :
        find_closet(point.left, x, min_dis, clost)
    else:
        find_closet(point.right, x, min_dis, clost)
    
    distance = abs(x[point.dimension] - point.value[point.dimension])
    if distance > min_dis[0] :
        return 
    else:
        if x[point.dimension] <= point.value[point.dimension] :
            find_closet(point.right, x, min_dis, clost)
        else:
            find_closet(point.left, x, min_dis, clost)

if __name__ == "__main__" :
    T = loadData()
    root = buildTree(T, 0)
    showKdTree(root)
    test = [3, 4.5]
    clost_point = np.copy(root.value)
    min_dis = np.array([-1.0])
    find_closet(root, test, min_dis, clost_point)
    print(test)
    print(clost_point)
    print(min_dis[0])