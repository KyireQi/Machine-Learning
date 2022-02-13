import numpy as np 

def loaddata():
    X = np.mat([[3, 3], [4, 3], [1, 1]])
    Y = np.array([1, 1, -1])
    return X, Y

def Train(X, Y):
    w = np.array([0 for i in range(X.shape[1])])
    b = 0
    eta = 1
    
    for i in range(15) : 
        print("第%d次迭代,w = "%i, w, "b = %d"%b)
        flag = 1
        for j in range(len(Y)):
            x = X[j]
            y = Y[j]
            if y * (w * x.T + b) <= 0:
                w = w + eta * y * x
                b = b + eta * y
                flag = 0
                break
        if flag :
            break
    return w, b

if __name__ == '__main__':
    X, Y = loaddata()
    w, b = Train(X, Y)
    print("Answer : " , w, b)