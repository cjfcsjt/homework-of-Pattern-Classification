import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')

#基本梯度下降法 s为学习率 threshold为阈值
def BasicGradient(filename,s,threshold):
    #读取数据
    w1_x1_x2= []
    w3_x1_x2= []
    # get the data from the file
    with open(filename, 'r') as f:
        js = json.load(f)
        w1_x1_x2.append(js['w1']['x1'])
        w1_x1_x2.append(js['w1']['x2'])

        w3_x1_x2.append(js['w3']['x1'])
        w3_x1_x2.append(js['w3']['x2'])
        # transform list to matrix
        w1_x1_x2 = np.mat(w1_x1_x2)
        w3_x1_x2 = np.mat(w3_x1_x2)

    #w1和w3两类数据合并到一个数据中，并且w3中的数据全部取负值，为了后面判断操作的方便
    w_x1_x2 = np.hstack((w1_x1_x2,-w3_x1_x2))
    #获得行数
    row = np.size(w_x1_x2,1)

    #得到偏置节点
    bias = np.mat(np.ones(row))
    w_x1_x2 =np.vstack((w_x1_x2,bias))

    #定义初始权值、梯度和、准则函数结果数组
    weight = np.mat([1,1,1])
    grad = np.mat([1, 1, 1])
    normArray = []
    time =0
    #w1 >0 ; w3 <0
    #calculate batch of gradient
    while np.linalg.norm(grad)>threshold :
        grad = np.mat([0,0,0])
        norm = 0
        for i in range(0,row):
            if((weight*w_x1_x2[:,i])<=0):
                norm = norm + normfunction(w_x1_x2[:,i],weight)
                grad = grad + s*gradient(w_x1_x2[:,i],weight)
                print(grad)
        weight = weight - grad #更新权值
        normArray.append(norm) #记录一批后得到的准则函数值

    # 绘图
    d1 = pd.DataFrame(columns=['x', 'y'])
    d1['x'] = np.array(w1_x1_x2)[0]
    d1['y'] = np.array(w1_x1_x2)[1]
    d2 = pd.DataFrame(columns=['x', 'y'])
    d2['x'] = np.array(w3_x1_x2)[0]
    d2['y'] = np.array(w3_x1_x2)[1]
    x = np.linspace(-4, 0.01, 4)
    y = (weight[0,0]*x+weight[0,2])/-weight[0,1]
    # 分别画出scatter图，但是设置不同的颜色
    plt.scatter(d1['x'], d1['y'], color='blue', label='w1')
    plt.scatter(d2['x'], d2['y'], color='green', label='w2')
    plt.plot(x,y)
    # 设置图例
    plt.legend(loc=(1, 0))

    # 显示图片
    plt.show()

    #迭代次数和准则函数
    time = np.linspace(1, np.array(normArray).size,np.array(normArray).size)
    print("time",time)
    normValue = np.transpose(np.array(normArray))
    print("normvalue", normValue)
    plt.plot(time, normValue)

    # 显示图片
    plt.show()

    return time

#与基本梯度下降法代码基本相同，
def NewtonGradient(filename,s,threshold):
    w1_x1_x2= []
    w3_x1_x2= []
    # get the data from the file
    with open(filename, 'r') as f:
        js = json.load(f)
        w1_x1_x2.append(js['w1']['x1'])
        w1_x1_x2.append(js['w1']['x2'])

        w3_x1_x2.append(js['w3']['x1'])
        w3_x1_x2.append(js['w3']['x2'])
        # transform list to matrix
        w1_x1_x2 = np.mat(w1_x1_x2)
        w3_x1_x2 = np.mat(w3_x1_x2)

    w_x1_x2 = np.hstack((w1_x1_x2,-w3_x1_x2))
    row = np.size(w_x1_x2,1)

    bias = np.mat(np.ones(row))
    w_x1_x2 =np.vstack((w_x1_x2,bias))


    weight = np.mat([1,1,1])
    grad = np.mat([1, 1, 1])
    h = np.mat([[1, 0, 0],
               [0, 1, 0],
               [0, 0,1]])
    normArray = []
    time = 0
    #w1 >0 ; w3 <0
    #calculate batch of gradient
    while np.linalg.norm(grad*np.linalg.inv(h))>threshold :
        grad = np.mat([0,0,0])
        norm = 0
        h = np.mat([[0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0]])
        for i in range(0,row):
            if((weight*w_x1_x2[:,i])<=0):
                norm = norm + normfunction(w_x1_x2[:,i],weight)
                grad = grad + s*gradient(w_x1_x2[:,i],weight)
                #牛顿梯度下降法的特点，需要求荷森矩阵
                h = h + hsmatrix(w_x1_x2[:,i],weight)
                print("grad",grad)
        weight = weight - grad*np.linalg.inv(h)
        normArray.append(norm)
        time +=1
    # 绘图
    d1 = pd.DataFrame(columns=['x', 'y'])
    d1['x'] = np.array(w1_x1_x2)[0]
    print("x1",np.array(w1_x1_x2)[0])
    d1['y'] = np.array(w1_x1_x2)[1]
    print("x2", np.array(w1_x1_x2)[1])
    d2 = pd.DataFrame(columns=['x', 'y'])
    d2['x'] = np.array(w3_x1_x2)[0]
    d2['y'] = np.array(w3_x1_x2)[1]
    x = np.linspace(-4, 0.01, 4)
    y = (weight[0,0]*x+weight[0,2])/-weight[0,1]
    # 分别画出scatter图，但是设置不同的颜色
    plt.scatter(d1['x'], d1['y'], color='blue', label='w1')
    plt.scatter(d2['x'], d2['y'], color='green', label='w2')
    plt.plot(x,y)
    # 设置图例
    plt.legend(loc=(1, 0))

    # 显示图片
    plt.show()

    # 迭代次数和准则函数
    time = np.linspace(1, np.array(normArray).size, np.array(normArray).size)
    normValue = np.transpose(np.array(normArray))
    plt.plot(time, normValue[0][0])

    # 显示图片
    plt.show()

    return time



def gradient(sample,a):
    #牛顿法的梯度
    # ans = (a*sample)*np.transpose(sample)/pow(np.linalg.norm(sample),2)
    # 基本梯度下降法的梯度
    ans = (a * sample) * np.transpose(sample)
    # print("ans",ans)
    return ans

def normfunction(sample,a):
    # 牛顿梯度下降法的梯度
    # ans = pow(a*sample,2)/pow(np.linalg.norm(sample),2)
    # 基本梯度下降法的准则函数
    ans = pow(a * sample, 2)
    return ans

def hsmatrix(sample,a):
    #荷森矩阵的求解
    ans = sample*np.transpose(sample)/pow(np.linalg.norm(sample),2)
    return ans

if __name__ == '__main__':
    #基本梯度下降法
    times = BasicGradient("data_04.json",0.01,0.0001)

    #牛顿梯度下降法
    # times = NewtonGradient("data_04.json", 0.1, 0.0001)




    # 求收敛时间学习率曲线的代码
    # ttimes = []
    # x = np.linspace(0, 1, 100)
    # for i in x:
    #     times = NewtonGradient("data_04.json", i, 0.0001)
    #     ttimes.append(times.size)
    #
    # plt.plot(x, ttimes)
    # # 显示图片
    # plt.show()