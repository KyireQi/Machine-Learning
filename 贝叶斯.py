import numpy as np

def loadData():
    X = np.array([[1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3], 
                ['S', 'M', 'M', 'S', 'S', 'S', 'M', 'M', 'L', 'L', 'L', 'M', 'M', 'L', 'L']])
    Y = np.array([-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1])
    return X, Y

def Bayes(X, Y, test, lab):
    print("lambda = %d"%lab)
    #y的取值集合
    set_Y = list(set(Y))
    #先验概率P(y)
    ans = {}
    Py = [(sum(Y == y) + lab) / (len(Y) + len(set_Y) * lab) for y in set_Y]
    for i in range(len(Py)):
        temp = Py[i]
        for j in range(len(test)):
            count = lab
            for k in range(len(X[j])):
                if X[j][k] == test[j] and Y[k] == set_Y[i]:
                    count += 1
            P = count / (sum(Y == set_Y[i]) + lab * len(set(X[j])))
            temp *= P
        ans[set_Y[i]] = temp
    print(ans)

if __name__ == "__main__" :
    X, Y = loadData()
    test = ['2', 'S']
    lab = 1
    Bayes(X, Y, test, lab)