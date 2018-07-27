import json
import numpy as np
import cmath
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#parzen函数，使用球形高斯函数作为窗函数
#s：样本训练数据   h：球半径   x：测试数据
#输出：测试数据的概率密度函数估计值
def parzenFun(s,h,x):
    #球形体积
    vn = 4*cmath.pi*pow(h,3)/3
    g=0
    #窗函数的叠加
    for i in range(0,np.size(s,1)):
        g = g + cmath.exp( float(-np.transpose(x-s[:,i])*(x-s[:,i]))/(2*h*h) )/vn
    g = g / np.size(s,1)
    return g

#利用parzen函数进行分类
#filename：样本文件  h1：球形半径
def parzenclassfication(filename,h1):
    w1_x1_x2_x3 = []
    w2_x1_x2_x3 = []
    w3_x1_x2_x3 = []
    # get the data from the file
    with open(filename, 'r') as f:
        js = json.load(f)
        w1_x1_x2_x3.append(js['w1']['x1'])
        w1_x1_x2_x3.append(js['w1']['x2'])
        w1_x1_x2_x3.append(js['w1']['x3'])

        w2_x1_x2_x3.append(js['w2']['x1'])
        w2_x1_x2_x3.append(js['w2']['x2'])
        w2_x1_x2_x3.append(js['w2']['x3'])

        w3_x1_x2_x3.append(js['w3']['x1'])
        w3_x1_x2_x3.append(js['w3']['x2'])
        w3_x1_x2_x3.append(js['w3']['x3'])
    # transform list to matrix
        w1_x1_x2_x3 = np.mat(w1_x1_x2_x3)
        w2_x1_x2_x3 = np.mat(w2_x1_x2_x3)
        w3_x1_x2_x3 = np.mat(w3_x1_x2_x3)


    # get the test data
    t1 = np.transpose(np.mat([0.5, 1.0, 0.0]));
    t2 = np.transpose(np.mat([0.31, 1.51, -0.50]));
    t3 = np.transpose(np.mat([-0.3, 0.44, -0.1]));


    #对测试数据1进行概率密度函数的预测
    p11 = parzenFun(w1_x1_x2_x3,h1,t1)
    p12 = parzenFun(w2_x1_x2_x3,h1,t1)
    p13 = parzenFun(w3_x1_x2_x3,h1,t1)
    #选择概率最大的作为测试数据1的类别
    parray1 = np.array([p11,p12,p13])
    print("测试数据1所在的类别为: ",np.argmax(parray1,axis=0)+1)

    # 对测试数据2进行概率密度函数的预测
    p21 = parzenFun(w1_x1_x2_x3, h1, t2)
    p22 = parzenFun(w2_x1_x2_x3, h1, t2)
    p23 = parzenFun(w3_x1_x2_x3, h1, t2)
    # 选择概率最大的作为测试数据2的类别
    parray2 = np.array([p21, p22, p23])
    print("测试数据2所在的类别为:", np.argmax(parray2, axis=0)+1)

    # 对测试数据3进行概率密度函数的预测
    p31 = parzenFun(w1_x1_x2_x3, h1, t3)
    p32 = parzenFun(w2_x1_x2_x3, h1, t3)
    p33 = parzenFun(w3_x1_x2_x3, h1, t3)
    # 选择概率最大的作为测试数据3的类别
    parray3 = np.array([p31, p32, p33])
    print("测试数据3所在的类别为:", np.argmax(parray3, axis=0)+1)

#k邻近法
#s：样本训练数据 kn：在区域内的样本数量 r：测试数据
def kneighbor(s,kn,r):
    #如果是一维数据
    if np.size(s,0) == 1:
        p = np.mat(np.transpose(np.zeros(np.size(r, 1))))
        distance = np.mat(np.transpose(np.zeros(np.size(s,1))))
        #测试数据的数量
        rSize = np.size(r,1)
        # 训练数据的数量
        sSize = np.size(s,1)

        for i in range(0,rSize) :
            for j in range(0,sSize):
                #对每一个测试数据，求他到训练数据中每一个点的欧式距离
                distance[:,j] = abs(s[:,j] - r[:,i])
            #对所求得对欧式距离进行排序
            distance.sort(axis = 1)
            # 从排好序对距离中按递增顺序选择kn个
            p[:,i] =  kn/rSize/(distance[:,kn-1])

    #如果是二维数据
    if np.size(s,0) == 2:
        p = np.mat(np.transpose(np.zeros(np.size(r, 1))))
        distance = np.mat(np.transpose(np.zeros(np.size(s,1))))
        rSize = np.size(r,1)
        sSize = np.size(s,1)

        for i in range(0,rSize) :
            for j in range(0,sSize):
                # 对每一个测试数据，求他到训练数据中每一个点的欧式距离
                distance[:,j] = np.linalg.norm(s[:,j]-r[:,i])
            # 对所求得对欧式距离进行排序
            distance.sort(axis = 1)
            # 从排好序对距离中按递增顺序选择kn个
            p[:,i] =  kn/rSize/(distance[:,kn-1])

    #如果是三维数据
    if np.size(s,0) == 3:
        p = np.mat(np.transpose(np.zeros(np.size(r, 1))))
        distance = np.mat(np.transpose(np.zeros(np.size(s,1))))
        rSize = np.size(r,1)
        sSize = np.size(s,1)
        for i in range(0,rSize) :
            for j in range(0,sSize):
                # 对每一个测试数据，求他到训练数据中每一个点的欧式距离
                distance[:,j] = np.linalg.norm(s[:,j]-r[:,i])
            # 对所求得对欧式距离进行排序
            distance.sort(axis  = 1)
            # 从排好序对距离中按递增顺序选择kn个
            p[:,i] =  kn/rSize/(distance[:,kn-1])

    return p


#对一维数据进行p（x）的估计
#filename：训练数据文件名  k：区域内的样本数量  n：测试数据的个数，随机数生成
def kneighbor1(filename,k,n):
    w3_x1 = []
    # get the data of x1 under w3
    with open(filename, 'r') as f:
        js = json.load(f)
        w3_x1 = js['w3']['x1']
    # transform list to matrix
    w3_x1 = np.mat(w3_x1)

    #【0-3】中生成数量为n的随机数
    r1 = np.transpose(np.mat(3*np.random.random_sample([n,1])))
    #k邻近估计概率密度
    p = kneighbor(w3_x1,k,r1)
    #绘图
    x = np.array(r1)[0]
    y = np.array(p)[0]
    plt.plot(x,y,'.')
    plt.show()

#对二维数据进行p（x）的估计
#filename：训练数据文件名  k：区域内的样本数量  n：测试数据的个数，随机数生成
def kneighbor2(filename,k,n):
    w2_x1_x2 = []
    # get the data of x1 under w1 and w2
    with open(filename, 'r') as f:
        js = json.load(f)
        w2_x1_x2.append(js['w2']['x1'])
        w2_x1_x2.append(js['w2']['x2'])
    # transform list to matrix
    w2_x1_x2 = np.mat(w2_x1_x2)

    # 【0-3】中生成数量为n的随机数
    r2 = np.transpose(np.mat(3*np.random.random_sample([n,2])))
    #k邻近估计概率密度
    p = kneighbor(w2_x1_x2,k,r2)
    #绘图
    x = np.array(r2)[0]
    y = np.array(r2)[1]
    z = np.array(p)[0]
    ax = plt.subplot(111, projection='3d')
    ax.scatter(x, y, z)
    ax.set_zlabel('p(x)',color = 'r')  # 坐标轴
    ax.set_ylabel('Y',color = 'b')
    ax.set_xlabel('X', color = 'g')
    plt.show()

#对三维数据进行p（x）的估计
#filename：训练数据文件名  k：区域内的样本数量
def kneighbor3(filename,k):
    # get the test data
    t1 = np.transpose(np.mat([-0.41, 0.82, 0.88]));
    t2 = np.transpose(np.mat([0.14, 0.72, 4.1]));
    t3 = np.transpose(np.mat([-0.81, 0.61, -0.38]));
    w1_x1_x2_x3 = []
    w2_x1_x2_x3 = []
    w3_x1_x2_x3 = []
    # get the data from the file
    with open(filename, 'r') as f:
        js = json.load(f)
        w1_x1_x2_x3.append(js['w1']['x1'])
        w1_x1_x2_x3.append(js['w1']['x2'])
        w1_x1_x2_x3.append(js['w1']['x3'])

        w2_x1_x2_x3.append(js['w2']['x1'])
        w2_x1_x2_x3.append(js['w2']['x2'])
        w2_x1_x2_x3.append(js['w2']['x3'])

        w3_x1_x2_x3.append(js['w3']['x1'])
        w3_x1_x2_x3.append(js['w3']['x2'])
        w3_x1_x2_x3.append(js['w3']['x3'])
        # transform list to matrix
        w1_x1_x2_x3 = np.mat(w1_x1_x2_x3)
        w2_x1_x2_x3 = np.mat(w2_x1_x2_x3)
        w3_x1_x2_x3 = np.mat(w3_x1_x2_x3)

    # k邻近估计概率密度
    p11 = kneighbor(w1_x1_x2_x3, k, t1)
    p21 = kneighbor(w2_x1_x2_x3, k, t1)
    p31 = kneighbor(w3_x1_x2_x3, k, t1)

    print("测试数据1在w1下的概率密度：", p11, "测试数据1在w2下的概率密度：", p21,"测试数据1在w3下的概率密度：", p31)
    # k邻近估计概率密度
    p12 = kneighbor(w1_x1_x2_x3, k, t2)
    p22 = kneighbor(w2_x1_x2_x3, k, t2)
    p32 = kneighbor(w3_x1_x2_x3, k, t2)
    print("测试数据1在w1下的概率密度：", p12, "测试数据1在w2下的概率密度：", p22, "测试数据1在w3下的概率密度：", p32)

    # k邻近估计概率密度
    p13 = kneighbor(w1_x1_x2_x3, k, t3)
    p23 = kneighbor(w2_x1_x2_x3, k, t3)
    p33 = kneighbor(w3_x1_x2_x3, k, t3)
    print("测试数据1在w1下的概率密度：", p13, "测试数据1在w2下的概率密度：", p23, "测试数据1在w3下的概率密度：", p33)


if __name__ == '__main__':
    # when h1 = 1
    print("使用parzen窗方法，在 h = 1时 ：")
    parzenclassfication("data_04.json",1)

    # when h1 = 0.1
    print("\n 使用parzen窗方法，在 h = 0.1时 ：")
    parzenclassfication("data_04.json", 0.1)

    print("\n使用k邻近方法，在 k=1 时 ：" )
    kneighbor3("data_04.json",1)


    print("\n使用k邻近方法，在 k=3 时 ：")
    kneighbor3("data_04.json", 3)


    print("\n使用k邻近方法，在 k=5 时 ：")
    kneighbor3("data_04.json", 5)

    # kneighbor1("data_04.json",1,100)
    # kneighbor1("data_04.json",3,100)
    # kneighbor1("data_04.json",5,100)
    # kneighbor2("data_04.json",1,100)
    # kneighbor2("data_04.json",3,100)
    # kneighbor2("data_04.json",5,100)



