import random
import numpy as np
import mnist_reader as mr
import cmath

from libsvm import svmutil
from libsvm import svm

# ****************** svm **********************
#读取训练数据（svm）
def loadtrainData3():
    #input
    train_images, num_images, num_rows, num_clos = mr.load_train_images()
    # output
    train_labels = mr.load_train_labels2()
    # resize input from 60000*28*28 ----> 60000*1*784
    train_images = train_images / 256
    train_images.resize((num_images, 1, 784))
    #zip train image and train label
    zipper = np.array(list(zip(train_images, train_labels)))

    return train_labels, train_images

#读取测试数据（svm）
def loadtestData3():
    test_images, num_images, num_rows, num_clos = mr.load_test_images()
    # 归一化
    test_images = test_images / 256
    test_images.resize((num_images, 1, 784))
    test_labels = mr.load_test_labels()

    # zip test image and test label
    zipper2 = np.array(list(zip(test_images, test_labels)))

    return test_labels,test_images

#svm数据处理函数，将数据格式转换为<class>   <index>:<data>,<index>:data,.........
def svmdataprocess(label,image):
    labels = []
    list = []
    images = []
    number = 0
    z = np.linspace(0, 783, 784)
    for i in z:
        list.append(int(i))
    # x = image[0]
    # a = dict(zip(list,x[0]))
    for i in image:
        a = dict(zip(list, i[0]))
        images.append(a)
        number +=1
        if number == 1000:
            break
    for i in label:
        labels.append(int(i))
        number -=1
        if number == 0:
            break
    return labels,images

#svm训练及测试
#输入参数为 训练标签，训练数据，测试标签，测试数据
#返回预测的标签
def svm(label,image,label2,image2):
    # 获得处理后的train data
    labels, images = svmdataprocess(label, image)
    # 获得处理后的test data
    labels2, images2 = svmdataprocess(label2, image2)
    # train
    prob = svmutil.svm_problem(labels, images)
    #参数设置
    # -t 核函数类型  0 -- linear(线性核)，1 -- polynomial(多项式核): 2 -- radial basis function(RBF,径向基核/高斯核): 3 -- sigmoid(S型核): 4 -- precomputed kernel(预计算核)：
    # -c 调整C-SVC, epsilon-SVR 和 nu-SVR中的Cost参数，默认为1
    # 是否估算正确概率,取值0 - 1，默认为0
    param = svmutil.svm_parameter('-t 0 -c 4 -b 1')
    #训练svm模型
    model = svmutil.svm_train(prob, param)
    # test
    p_label, p_acc, p_val = svmutil.svm_predict(labels2, images2, model)
    print(p_label)
    return p_label

#**************************** svm *****************************



#************************** parzen窗 **************************
# parzen函数，使用球形高斯函数作为窗函数
# # s：样本训练数据   h：球半径   x：测试数据
# # 输出：测试数据的概率密度函数估计值
def parzenFun(s, h, x):
    # 球形体积
    vn = 4 * cmath.pi * pow(h, 3) / 3
    g = 0
    # 窗函数的叠加
    #test
    b = s[0]
    c = x - s[0]
    d = np.transpose(c)
    e = np.dot(d,c)
    #test
    for i in range(0, len(s)):
        g = g + cmath.exp(float(-np.dot(d, c) / (2 * h * h)) / vn)
    g = g / np.size(s)
    return g


def parzen(zipper,zipper2,h):
    resultlist = []
    right = 0
    #0-9类的样本list
    list1 = [[] for i in range(10)]
    #训练集读取0-9的样本
    for i in zipper:
        number = int(i[1])
        list1[number].append(i[0])
    list2 = []
    listans = []
    #测试数据集
    for j in zipper2:
        list2.append(j[0])
        listans.append(j[1])
    predict = [[] for i in range(10)]
    #-9900可以删除 意义为测试数据集大小，-9900是为了快速验证程序正确性
    for k in range(0,len(list2)-9900):
        for s in range(0,10):
            predict[s] = parzenFun(list1[s],h,list2[k])
        #parray3 = np.array([predict[i] for i in range(0,10)])
        #print("测试数据3所在的类别为:", np.argmax(predict, axis=0),"实际为", listans[k])
        if np.argmax(predict, axis=0) == int(listans[k]):
            right +=1
        resultlist.append(np.argmax(predict, axis=0))
    print("parzen准确率",right/ (len(list2)-9900))
    print(resultlist)
    return resultlist

#***************** parzen窗 ********************


def loadtrainData2():
    #input
    train_images, num_images, num_rows, num_clos = mr.load_train_images()
    # output
    train_labels = mr.load_train_labels2()
    # resize input from 60000*28*28 ----> 60000*1*784
    train_images = train_images / 256
    train_images.resize((num_images, 784, 1))
    #zip train image and train label
    zipper = np.array(list(zip(train_images, train_labels)))

    return zipper

def loadtrainData():
    #input
    train_images, num_images, num_rows, num_clos = mr.load_train_images()
    # output
    train_labels = mr.load_train_labels()
    # resize input from 60000*28*28 ----> 60000*1*784
    train_images = train_images / 256
    train_images.resize((num_images, 784, 1))
    #zip train image and train label
    zipper = np.array(list(zip(train_images, train_labels)))

    return zipper

def loadtestData():
    test_images, num_images, num_rows, num_clos = mr.load_test_images()
    # 归一化
    test_images = test_images / 256
    test_images.resize((num_images, 784, 1))
    test_labels = mr.load_test_labels()

    # zip test image and test label
    zipper2 = np.array(list(zip(test_images, test_labels)))

    return zipper2
#********************  parzen窗    *****************


#**********************  bpnn    ***********************

def tanh(X, derive=False):
    if not derive:
        return np.tanh(X)
    else:
        return 1.0 - X ** 2

def bpnn(zipper,zipper2):

    #INITIALIZE

    # 1. two hidden layer size: 15 and 15
    # NN Size: 784 x 50 x 50 x 10
    size0 = 50  # Number of first hide layer neurons
    size1 = 50  # Number of second hide layer neurons

    W0 = 2 * np.random.random((784, size0)) - 1
    b0 = 0.1 * np.ones((size0,))
    W1 = 2 * np.random.random((size0, size1)) - 1
    b1 = 0.1 * np.ones((size1,))
    W2 = 2 * np.random.random((size1, 10)) - 1
    b2 = 0.1 * np.ones((10,))

    nonline = tanh
    # 学习率
    rate = 0.05
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
    resultlist = []
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
        resultlist.append(result.index(max(result)))
    print("bpnn准确率",t/totalnum)

    return resultlist

#********************* bpnn   **********************

#bagging方法中 随机从训练集中抽取相同大小的样本d
def generateSample(trainDATA):
    size = len(trainDATA)
    sample = []
    for i in range(0,size):
        rd = int(random.uniform(0,size))
        sample.append(trainDATA[rd])

    sample = np.array(sample)

    return sample

#随机从训练集中抽取相同大小的样本d
def generateSample2(label,image):
    size = len(label)
    samplelabel = []
    sampleimage = []
    for i in range(0,size):
        rd = int(random.uniform(0,size))
        sampleimage.append(image[rd])
        samplelabel.append(label[rd])


    return samplelabel,sampleimage

#bagging方法中最后的投票机制
def vote(result1,result2,result3,testlabel):
    result=[]
    for i in range(0,len(result1)):
        a = result1[i]
        b = result2[i]
        c = result3[i]
        if a==b==c: result.append(a)
        if a == b and a != c: result.append(a)
        if a == c and a != b: result.append(a)
        if b == c and a != c: result.append(b)
        if a!=b and a!=c and b!=c: result.append(b)
    #计算bagging方法得到的准确率
    number = 0
    for i in range(0,len(result)):
        if result[i] == testlabel[i]:
            number +=1
    print("bagging 方法准确率：",number/len(result))
    return result

if __name__ == '__main__':
    # 获取parzen窗模型得到的预测结果 h=1
    print("parzen窗方法：(时间较长)")
    traindata = loadtrainData2()
    testdata = loadtestData()
    sample1 = generateSample(traindata)
    parzenresult = parzen(sample1,testdata,1)

    print("bpnn方法：")
    #获取神经网络模型得到的预测结果
    traindata = loadtrainData()
    testdata = loadtestData()
    sample2 = generateSample(traindata)
    bpnnresult = bpnn(sample2,testdata)

    print("svm方法：")
    #获取svm模型得到的预测结果
    train_labels, train_images = loadtrainData3()
    testlabels, test_images = loadtestData3()
    train_labels, train_images = generateSample2(train_labels,train_images)
    svmresult = svm(train_labels, train_images, testlabels, test_images)

    #vote
    result = vote(parzenresult, bpnnresult, svmresult,testlabels)
    print("bagging方法得到的结果：",result)

