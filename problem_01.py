import json
import numpy as np

CLASS1 = 1
CLASS2 = 2

#P(Wi|x) = P(x|Wi)*P(Wi) / P(x)
#P(x) = P(x|W1)*P(W1) + P(x|W2)*P(W2)
#P(W1) = 1/2 , P(W2) = 1/2 , P(W3) = 0
def classfication_01 (filename):
    w1_x1 = []
    w2_x1 = []
    #get the data of x1 under w1 and w2
    with open(filename, 'r') as f:
        js = json.load(f)
        w1_x1 = js['w1']['x1']
        w2_x1 = js['w2']['x1']
    #transform list to matrix
    w1_x1 = np.mat(w1_x1)
    w2_x1 = np.mat(w2_x1)

    #calculate sample mean
    mean1 = np.mean(w1_x1)
    mean2 = np.mean(w2_x1)

    #calculate sample cov
    cov1 = np.cov(w1_x1)
    cov2 = np.cov(w2_x1)

    #get the size of sample
    col1 = np.size(w1_x1, 1)
    col2 = np.size(w2_x1, 1)

    #ge† the prediction result
    result1 = np.zeros(col1)
    result2 = np.zeros(col2)

    #give an point of sample, get the class it belongs to
    # if P(W1|x) > P(W2|X) CLASS 1   else  CLASS 2
    for i in range(col1):
        if classfication_g(w1_x1[:,i],mean1,cov1,0.5) > classfication_g(w1_x1[:,i],mean2,cov2,0.5):
            result1[i] = CLASS1
        else:
            result1[i] = CLASS2
    for i in range(col2):
        if classfication_g(w2_x1[:,i],mean1,cov1,0.5) > classfication_g(w2_x1[:,i],mean2,cov2,0.5):
            result2[i] = CLASS1
        else:
            result2[i] = CLASS2

    return result1,result2

#the similar with classfication_01,  but the dimension is 2
def classfication_02 (filename):
    w1_x1_x2 = []
    w2_x1_x2 = []
    #get the data of x1,x2 under w1 and w2
    with open(filename, 'r') as f:
        js = json.load(f)
        w1_x1_x2.append(js['w1']['x1'])
        w1_x1_x2.append(js['w1']['x2'])

        w2_x1_x2.append(js['w2']['x1'])
        w2_x1_x2.append(js['w2']['x2'])
    w1_x1_x2 = np.mat(w1_x1_x2)
    w2_x1_x2 = np.mat(w2_x1_x2)
    mean1 = np.mean(w1_x1_x2,1)
    mean2 = np.mean(w2_x1_x2,1)
    cov1 = np.cov(w1_x1_x2)
    cov2 = np.cov(w2_x1_x2)
    col1 = np.size(w1_x1_x2,1)
    col2 = np.size(w2_x1_x2,1)
    result1 = np.zeros(col1)
    result2 = np.zeros(col2)
    for i in range(col1):
        if classfication_g(w1_x1_x2[:,i],mean1,cov1,0.5) > classfication_g(w1_x1_x2[:,i],mean2,cov2,0.5):
            result1[i] = CLASS1
        else:
            result1[i] = CLASS2
    for i in range(col2):
        if classfication_g(w2_x1_x2[:,i],mean1,cov1,0.5) > classfication_g(w2_x1_x2[:,i],mean2,cov2,0.5):
            result2[i] = CLASS1
        else:
            result2[i] = CLASS2

    return result1,result2

#similar with classfication_01, but the dimension is 3
def classfication_03 (filename):
    w1_x1_x2_x3 = []
    w2_x1_x2_x3 = []
    #get the data of x1,x2,x3 under w1 and w2
    with open(filename, 'r') as f:
        js = json.load(f)
        w1_x1_x2_x3.append(js['w1']['x1'])
        w1_x1_x2_x3.append(js['w1']['x2'])
        w1_x1_x2_x3.append(js['w1']['x3'])

        w2_x1_x2_x3.append(js['w2']['x1'])
        w2_x1_x2_x3.append(js['w2']['x2'])
        w2_x1_x2_x3.append(js['w2']['x3'])


    w1_x1_x2_x3 = np.mat(w1_x1_x2_x3)
    w2_x1_x2_x3 = np.mat(w2_x1_x2_x3)
    mean1 = np.mean(w1_x1_x2_x3,1)
    mean2 = np.mean(w2_x1_x2_x3,1)
    cov1 = np.cov(w1_x1_x2_x3)
    cov2 = np.cov(w2_x1_x2_x3)
    col1 = np.size(w1_x1_x2_x3,1)
    col2 = np.size(w2_x1_x2_x3,1)
    result1 = np.zeros(col1)
    result2 = np.zeros(col2)
    for i in range(col1):
        if classfication_g(w1_x1_x2_x3[:,i],mean1,cov1,0.5) > classfication_g(w1_x1_x2_x3[:,i],mean2,cov2,0.5):
            result1[i] = CLASS1
        else:
            result1[i] = CLASS2
    for i in range(col2):
        if classfication_g(w2_x1_x2_x3[:,i],mean1,cov1,0.5) > classfication_g(w2_x1_x2_x3[:,i],mean2,cov2,0.5):
            result2[i] = CLASS1
        else:
            result2[i] = CLASS2

    return result1, result2



#the
def classfication_g(data,u,sigma,pp):
# data - Input vector
# u - Mean
# sigma - Covariance matrix
# pp - Prior probability

#   case of 1 dimension
    if np.size(u) == 1:
        g = -0.5 * np.transpose(data-u) * (1.0/sigma) * (data - u) - np.size(u)/2.0*np.log(2*np.pi) - 0.5*np.log(np.abs(sigma)) + np.log(pp)
#   case of multi dimension
    else:
        g = (-0.5 * np.transpose(data - u) * np.linalg.inv(sigma) * (data - u)) - np.size(u) / 2.0 * np.log(2 * np.pi) - 0.5 * np.log(np.linalg.det(sigma)) + np.log(pp)

    return g

#  calculate the error
#  error = number of false classification / number of sample point
def classfication_error(array1, array2):
    err1 = np.size(np.where(array1 == CLASS2),1)
    err2 = np.size(np.where(array2 == CLASS1),1)
    return (err1+err2) / (np.size(array1)+np.size(array2))


if __name__ == '__main__':
    #change the function here to get different result
    result1, result2 = classfication_03('data_01.json')
    print("prediction of [x1] which is in w1 formerly \n",result1)
    print("prediction of [x2] which is in w2 formerly \n",result2)
    err = classfication_error(result1,result2)
    print("err:",err)