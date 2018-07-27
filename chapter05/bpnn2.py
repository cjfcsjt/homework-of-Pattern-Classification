import numpy as np
import mnist_reader as mr


# Inputs
np.random.seed(1)
train_images,num_images,num_rows,num_clos = mr.load_train_images()
train_images = train_images /256
#output
train_labels = mr.load_train_labels()
#resize input from 60000*28*28 ----> 60000*1*784
train_images.resize((num_images,784,1))

#zip train image and train label
zipper = np.array(list(zip(train_images, train_labels)))


test_images,num_images,num_rows,num_clos  = mr.load_test_images()
#归一化
test_images = test_images / 256
test_images.resize((num_images, 784, 1))
test_labels = mr.load_test_labels()

#zip test image and test label
zipper2 = np.array(list(zip(test_images, test_labels)))

#选择一个方案
scheme = 2

# Layers( weights, bias)
#1. two hidden layer size: 15 and 15
# NN Size: 784 x 50 x 50 x 10
if scheme == 1:
    size0 = 50  # Number of first hide layer neurons
    size1 = 50  # Number of second hide layer neurons

    W0 = 2 * np.random.random((784, size0)) - 1
    b0 = 0.1 * np.ones((size0,))
    W1 = 2 * np.random.random((size0, size1)) - 1
    b1 = 0.1 * np.ones((size1,))
    W2 = 2 * np.random.random((size1, 10)) - 1
    b2 = 0.1 * np.ones((10,))

#2. one hidden layer size : 30
# NN Size: 784 x 50 x 10
if scheme == 2:
    size0 = 50  # Number of first hide layer neurons
    W0 = 2 * np.random.random((784, size0)) - 1
    b0 = 0.1 * np.ones((size0,))
    W2 = 2 * np.random.random((size0, 10)) - 1
    b2 = 0.1 * np.ones((10,))

# Nonlinear functions
def sigmold(X, derive=False):
    if not derive:
        return 1 / (1 + np.exp(-X))
    else:
        return X * (1 - X)

def tanh(X, derive=False):
    if not derive:
        return np.tanh(X)
    else:
        return 1.0 - X ** 2


nonline = tanh
# 学习率
rate = 0.05



# 训练方案一： 2层hidden层
if scheme == 1:
    #train
    training_times = 10
    for i in range(training_times):
        print("次数" , i)
        cost = 0
        for p in zipper:
            # Layer1
            A0 = np.dot(np.array(p[0]).T, W0) + b0
            Z0 = nonline(A0)

            # Layer2
            A1 = np.dot(Z0, W1) + b1
            Z1 = nonline(A1)

            # Layer3
            A2 = np.dot(Z1, W2) + b2
            _y = Z2 = nonline(A2)
            #计算损失函数
            cost = _y - np.array(p[1]).T


            # Calc deltas 权值的更新
            delta_A2 = cost * nonline(Z2, derive=True)
            delta_b2 = delta_A2.sum(axis=0)
            delta_W2 = np.dot(Z1.T, delta_A2)

            delta_A1 = np.dot(delta_A2, W2.T) * nonline(Z1, derive=True)
            delta_b1 = delta_A1.sum(axis=0)
            delta_W1 = np.dot(Z0.T, delta_A1)

            delta_A0 = np.dot(delta_A1, W1.T) * nonline(Z0, derive=True)
            delta_b0 = delta_A0.sum(axis=0)
            delta_W0 = np.dot(np.array(p[0]), delta_A0)

            # Apply deltas 梯度下降
            W2 -= rate * delta_W2
            b2 -= rate * delta_b2
            W1 -= rate * delta_W1
            b1 -= rate * delta_b1
            W0 -= rate * delta_W0
            b0 -= rate * delta_b0
        print("Cost x:{}".format(np.mean(np.abs(cost))))
    else:
        # Print cost, weights, bias, last output
        print("Cost:{}".format(np.mean(np.abs(cost))))
        print(np.around(W0, 3), "W0")
        print(np.around(b0, 3), "b0")
        print(np.around(W1, 3), "W1")
        print(np.around(b1, 3), "b1")
        print(np.around(W2, 3), "W2")
        print(np.around(b2, 3), "b2")
        print(np.around(_y, 2), "_y")
    #test
    t = 0
    totalnum = 0
    for p in zipper2:
        totalnum +=1
        # Layer1
        A0 = np.dot(np.array(p[0]).T, W0) + b0
        Z0 = nonline(A0)

        # Layer2
        A1 = np.dot(Z0, W1) + b1
        Z1 = nonline(A1)

        # Layer3
        A2 = np.dot(Z1, W2) + b2
        _y = Z2 = nonline(A2)
        result = list(list(_y)[0])
        if int(p[1]) == result.index(max(result)):
            t += 1
    print("准确率",t/totalnum)


# 训练方案二： 1层hidden层
if scheme == 2:
    #train
    training_times = 10
    for i in range(training_times):
        print("次数" , i)
        cost = 0
        for p in zipper:
            # Layer1
            A0 = np.dot(np.array(p[0]).T, W0) + b0
            Z0 = nonline(A0)

            # Layer2
            A2 = np.dot(Z0, W2) + b2
            _y = Z2 = nonline(A2)
            #计算损失函数
            cost = _y - np.array(p[1]).T


            # Calc deltas 更新权值
            delta_A2 = cost * nonline(Z2, derive=True)
            delta_b2 = delta_A2.sum(axis=0)
            delta_W2 = np.dot(Z0.T, delta_A2)

            delta_A0 = np.dot(delta_A2, W2.T) * nonline(Z0, derive=True)
            delta_b0 = delta_A0.sum(axis=0)
            delta_W0 = np.dot(np.array(p[0]), delta_A0)

            # Apply deltas
            W2 -= rate * delta_W2
            b2 -= rate * delta_b2
            W0 -= rate * delta_W0
            b0 -= rate * delta_b0
        print("Cost x:{}".format(np.mean(np.abs(cost))))
    else:
        # Print cost, weights, bias, last output
        print("Cost:{}".format(np.mean(np.abs(cost))))
        print(np.around(W0, 3), "W0")
        print(np.around(b0, 3), "b0")
        print(np.around(W2, 3), "W2")
        print(np.around(b2, 3), "b2")
        print(np.around(_y, 2), "_y")
    #test
    t = 0
    totalnum = 0
    for p in zipper2:
        totalnum +=1
        # Layer1
        A0 = np.dot(np.array(p[0]).T, W0) + b0
        Z0 = nonline(A0)

        # Layer3
        A2 = np.dot(Z0, W2) + b2
        _y = Z2 = nonline(A2)
        result = list(list(_y)[0])
        if int(p[1]) == result.index(max(result)):
            t += 1
    print("准确率",t/totalnum)