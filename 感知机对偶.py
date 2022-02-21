import numpy as np

def loadData():
    X = np.mat([[3, 3], [4, 3], [1, 1]])
    Y = np.array([1, 1, -1])
    return X, Y

def Train(X, Y, iter = 15):
    N = X.shape[0]
    alpha = np.array([0 for i in range(N)])
    b = 0
    eta = 1
    G = [[0] * N for i in range(N)]
    for i in range(N) :
        for j in range(N):
            x = X[i] * X[j].T
            G[i][j] = int(x)
    for i in range(iter):
        print("第%d次迭代，alpha = "%i, alpha, "b = %d"%b)
        flag = 1
        for j in range(N):
            sum = np.sum(alpha * Y * np.asarray(G[j])) + b
            if sum * Y[j] <= 0:
                alpha[j] = alpha[j] + eta
                b = b + eta * Y[j]
                flag = 0
                break
        if flag == 1 :
            break
    return alpha, b

if __name__ == "__main__" :
    X, Y = loadData()
    alpha, b = Train(X, Y)
    w = Y * alpha * X
    print("最终结果:w = ", w, "b = %d"%b)