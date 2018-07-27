import json
import numpy as np

def question_01(filename):
    w1_x1 = []
    w1_x2 = []
    w1_x3 = []
    # get the data of one-dimension data x1 x2 x3under w1
    with open(filename, 'r') as f:
        js = json.load(f)
        w1_x1 = js['w1']['x1']
        w1_x2 = js['w1']['x2']
        w1_x3 = js['w1']['x3']
    #transform list to matrix
    w1_x1 = np.mat(w1_x1)
    w1_x2 = np.mat(w1_x2)
    w1_x3 = np.mat(w1_x3)


    # calculate sample mean
    mean1 = np.mean(w1_x1)
    mean2 = np.mean(w1_x2)
    mean3 = np.mean(w1_x3)

    sigma1 = np.cov(w1_x1)
    sigma2 = np.cov(w1_x2)
    sigma3 = np.cov(w1_x3)

    return mean1, mean2, mean3, sigma1, sigma2, sigma3

def question_02(filename):
    w1_x1_x2 = []
    w1_x1_x3 = []
    w1_x2_x3 = []
    # get the data of 2-dimension data under w1 
    with open(filename, 'r') as f:
        js = json.load(f)
        w1_x1_x2.append(js['w1']['x1'])
        w1_x1_x2.append(js['w1']['x2'])

        w1_x1_x3.append(js['w1']['x1'])
        w1_x1_x3.append(js['w1']['x3'])

        w1_x2_x3.append(js['w1']['x2'])
        w1_x2_x3.append(js['w1']['x3'])


    w1_x1_x2 = np.mat(w1_x1_x2)
    w1_x1_x3 = np.mat(w1_x1_x3)
    w1_x2_x3 = np.mat(w1_x2_x3)
    
    # calculate sample mean
    mean12 = np.mean(w1_x1_x2, 1)
    mean13 = np.mean(w1_x1_x3, 1)
    mean23 = np.mean(w1_x2_x3, 1)

    cov11 = np.cov(w1_x1_x2)
    cov13 = np.cov(w1_x1_x3)
    cov23 = np.cov(w1_x2_x3)

    return mean12,mean13,mean23,cov11,cov13,cov23

def question_03(filename):
    w1_x1_x2_x3 = []
    # get the data of 3-dimension data under w1 
    with open(filename, 'r') as f:
        js = json.load(f)
        w1_x1_x2_x3.append(js['w1']['x1'])
        w1_x1_x2_x3.append(js['w1']['x2'])
        w1_x1_x2_x3.append(js['w1']['x3'])


    w1_x1_x2_x3 = np.mat(w1_x1_x2_x3)

    mean123 = np.mean(w1_x1_x2_x3, 1)

    cov123 = np.cov(w1_x1_x2_x3)

    return mean123,cov123

def question_04(filename):
    w2_x1_x2_x3 = []
    w2_x1 = []
    w2_x2 = []
    w2_x3 = []
    # get the data of 3-dimension under w2
    with open(filename, 'r') as f:
        js = json.load(f)
        w2_x1_x2_x3.append(js['w2']['x1'])
        w2_x1_x2_x3.append(js['w2']['x2'])
        w2_x1_x2_x3.append(js['w3']['x3'])

        w2_x1 = js['w2']['x1']
        w2_x2 = js['w2']['x2']
        w2_x3 = js['w2']['x3']

    # transform list to matrix
    w2_x1 = np.mat(w2_x1)
    w2_x2 = np.mat(w2_x2)
    w2_x3 = np.mat(w2_x3)
    w2_x1_x2_x3 = np.mat(w2_x1_x2_x3)

    mean123 = np.mean(w2_x1_x2_x3, 1)

    sigma1 = np.cov(w2_x1)
    sigma2 = np.cov(w2_x2)
    sigma3 = np.cov(w2_x3)


    return mean123,sigma1,sigma2,sigma3

if __name__ == '__main__':
    mean1,mean2,mean3,sigma1,sigma2,sigma3 = question_01('data_02.json')
    print("question_01: \n", "mean1=",mean1,"\n mean2 = ",mean2, "\n mean3 = ", mean3, "\n sigma1 = ", sigma1 , "\n sigma2 =", sigma2 , "\n sigma3 = " , sigma3,"\n")

    mean1, mean2, mean3, sigma1, sigma2, sigma3 = question_02('data_02.json')
    print("question_02: \n ", "mean1=", mean1, "\n mean2 = ", mean2, "\n mean3 = ", mean3, "\n sigma1 = ", sigma1, "\n sigma2 =", sigma2,
          "\n sigma3 = ", sigma3,"\n")

    mean, sigma = question_03('data_02.json')
    print("question_03: \n " ,"mean=", mean, "sigma = ", sigma ,"\n")

    mean, sigma1, sigma2, sigma3 = question_04('data_02.json')
    print("question_04: \n", "mean=", mean, "\n sigma1 = ", sigma1,
          "\n sigma2 =", sigma2, "\n sigma3 = ", sigma3)

