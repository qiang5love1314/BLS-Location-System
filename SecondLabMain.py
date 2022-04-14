import os
import numpy as np
import pandas as pd
import time
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
from pykalman import KalmanFilter
from scipy import signal
from numpy import random
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from scipy.stats import norm
import math
def show_accuracy(predictLabel, Label):
    Label = np.ravel(Label).tolist()
    predictLabel = predictLabel.tolist()
    count = 0
    for i in range(len(Label)):
        if Label[i] == predictLabel[i]:
            count += 1
    return (round(count / len(Label), 5))

def accuracyPre(predictions, labels):
    return np.mean(np.sqrt(np.sum((predictions-labels)**2,1)))

class scaler:
    def __init__(self):
        self._mean = 0
        self._std = 0

    def fit_transform(self, traindata):
        self._mean = traindata.mean(axis=0)
        self._std = traindata.std(axis=0)
        return (traindata - self._mean) / self._std

    def transform(self, testdata):
        return (testdata - self._mean) / self._std

def sigmoid(data):
    return 1.0 / (1 + np.exp(-data))

def tanh(data):
    return (np.exp(data) - np.exp(-data)) / (np.exp(data) + np.exp(-data))

def linear(data):
    return data

def tansig(x):
    return (2 / (1 + np.exp(-2 * x))) - 1

def pinv(A, reg):   #岭回归的正则化，暂未用
    return np.mat(reg * np.eye(A.shape[1]) + A.T.dot(A)).I.dot(A.T)

def shrinkage(a, b):  #参数压缩
    z = np.maximum(a - b, 0) - np.maximum(-a - b, 0)
    return z

def sparse_bls(A, b): #参数稀疏
    lam = 0.001
    itrs = 20
    AA = np.dot(A.T, A)
    m = A.shape[1]
    n = b.shape[1]
    wk = np.zeros([m, n], dtype='double')
    ok = np.zeros([m, n], dtype='double')
    uk = np.zeros([m, n], dtype='double')
    L1 = np.mat(AA + np.eye(m)).I
    L2 = np.dot(np.dot(L1, A.T), b)
    for i in range(itrs):
        tempc = ok - uk
        ck = L2 + np.dot(L1, tempc)
        ok = shrinkage(ck + uk, lam)
        uk += ck - ok
        wk = ok
    return wk

def bls_regression(train_x, train_y, test_x, test_y, s, C, NumFea, NumWin, NumEnhan):
    u = 0
    WF = list()
    for i in range(NumWin):
        random.seed(i + u)
        WeightFea = 2 * random.randn(train_x.shape[1] + 1, NumFea) - 1;
        WF.append(WeightFea)
    #    random.seed(100)
    WeightEnhan = 2 * random.randn(NumWin * NumFea + 1, NumEnhan) - 1;
    time_start = time.time()
    H1 = np.hstack([train_x, 0.1 * np.ones([train_x.shape[0], 1])]);
    y = np.zeros([train_x.shape[0], NumWin * NumFea])
    WFSparse = list()
    distOfMaxAndMin = np.zeros(NumWin)
    meanOfEachWindow = np.zeros(NumWin)
    for i in range(NumWin):
        WeightFea = WF[i]
        A1 = H1.dot(WeightFea)
        scaler1 = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(A1)
        A1 = scaler1.transform(A1)
        WeightFeaSparse = sparse_bls(A1, H1).T
        WFSparse.append(WeightFeaSparse)

        T1 = H1.dot(WeightFeaSparse)
        meanOfEachWindow[i] = T1.mean()
        distOfMaxAndMin[i] = T1.max() - T1.min()
        T1 = (T1 - meanOfEachWindow[i]) / distOfMaxAndMin[i]
        y[:, NumFea * i:NumFea * (i + 1)] = T1

    H2 = np.hstack([y, 0.1 * np.ones([y.shape[0], 1])])
    T2 = H2.dot(WeightEnhan)
    T2 = tanh(T2);
    T3 = np.hstack([y, T2])
    WeightTop = pinv(T3, C).dot(train_y)

    Training_time = time.time() - time_start
    print('Training has been finished!');
    print('The Total Training Time is : ', round(Training_time, 6), ' seconds')
    NetoutTrain = T3.dot(WeightTop)

    RMSE = np.sqrt((NetoutTrain - train_y).T * (NetoutTrain - train_y) / train_y.shape[0])
    MAPE = sum(abs(NetoutTrain - train_y)) / train_y.mean() / train_y.shape[0]
    train_ERR = RMSE
    train_MAPE = MAPE
    print('Training RMSE is : ', RMSE);
    print('Training MAPE is : ', MAPE)
    time_start = time.time()
    HH1 = np.hstack([test_x, 0.1 * np.ones([test_x.shape[0], 1])])
    yy1 = np.zeros([test_x.shape[0], NumWin * NumFea])
    for i in range(NumWin):
        WeightFeaSparse = WFSparse[i]
        TT1 = HH1.dot(WeightFeaSparse)
        TT1 = (TT1 - meanOfEachWindow[i]) / distOfMaxAndMin[i]
        yy1[:, NumFea * i:NumFea * (i + 1)] = TT1

    HH2 = np.hstack([yy1, 0.1 * np.ones([yy1.shape[0], 1])])
    TT2 = tanh(HH2.dot(WeightEnhan));
    TT3 = np.hstack([yy1, TT2])
    NetoutTest = TT3.dot(WeightTop)
    RMSE = np.sqrt((NetoutTest - test_y).T * (NetoutTest - test_y) / test_y.shape[0]);
    MAPE = sum(abs(NetoutTest - test_y)) / test_y.mean() / test_y.shape[0]
    test_ERR = RMSE
    test_MAPE = MAPE
    # %% Calculate the testing accuracy
    Testing_time = time.time() - time_start
    print('Testing has been finished!');
    print('The Total Testing Time is : ', round(Testing_time, 6), ' seconds');
    print('Testing RMSE is : ', RMSE)
    print('Testing MAPE is : ', MAPE)
    return test_ERR, test_MAPE, Testing_time, train_ERR, train_MAPE, Training_time, NetoutTrain, NetoutTest

def percentage(eigenValues, per):           #remain the 'percentage' of the data
    sortArray = np.sort(eigenValues)[-1::-1]
    arraySum = sum(sortArray)
    tempSum,num = 0,0
    for i in sortArray:
        tempSum += i
        num += 1
        if tempSum >= arraySum * per:
            return num

def zeroMean(data):                        #minus the mean value from the data
    meanValue = np.mean(data,axis = 0)
    meanData = data - meanValue
    return meanData,meanValue

def pca(data,per = 0.95):                  #the pca function
    data,mean = zeroMean(data)
    covariance = np.cov(data,rowvar = 0)
    eigenValues,eigenVectors = np.linalg.eig(np.mat(covariance))      #calculate the eigenvalue
    num = percentage(eigenValues,per)
    eigenVlueIndice = np.argsort(eigenValues)
    sel_eigenValInd = eigenVlueIndice[-1:-(num+1):-1]
    sel_eigenVectors = eigenVectors[:,sel_eigenValInd]
    lowerData = data*sel_eigenVectors      #the lower dimension data
    recondata = (lowerData*sel_eigenVectors.T)+mean
    return lowerData,recondata

def preProcessing(filePath):    #数据预处理包括：PCA(only reconstruct not reduce dimension)+Kalman+EM+Smoothed Filter
    c = loadmat(filePath)
    CSI = c['myData']
    AntennA, AntennB, AntennC = CSI[0][:][:], CSI[1][:][:], CSI[2][:][:]

    csiPCA = np.zeros((3, 30, 50), dtype=np.float)  # array size 3*30*50
    listAntenna = [AntennA, AntennB, AntennC]
    for i in range(len(listAntenna)):
        lowerData, reconData = pca(listAntenna[i], 0.95)  # 信息提取率95%
        csiPCA[i] = abs(reconData)
        print(len(lowerData[1]))

    b, a = signal.butter(5, 3 * 2 / 50, 'lowpass')  # set filter parameters
    for i in range(3):  # Kalman+EM process and smoothed filter
        for j in range(50):
            kf = KalmanFilter(transition_matrices=[1], observation_matrices=[1])
            measurements = np.asanyarray(csiPCA[i][:, j])
            kf = kf.em(measurements, n_iter=10)  # 10 has better performance
            (filtered_state_means, filtered_state_covariances) = kf.filter(measurements)
            # (smoothed_state_means, smoothed_state_covariances) = kf.smooth(filtered_state_means)
            # csiPCA[i][:, j] = list(smoothed_state_means)
            swap = filtered_state_means[:, 0]
            csiPCA[i][:, j] = signal.filtfilt(b, a, swap)
    return csiPCA

def rawCSI():
    xLabel = []
    yLabel = []

    for i in range(21):     #横坐标
        str = '%d' % (i + 1)
        xLabel.append(str)

    for j in range(23):     #纵坐标
        if(j<9):
            num=0
            str= '%d%d' % (num, j+1)
            yLabel.append(str)
        else:
            yLabel.append('%d'% (j+1) )

    originalCSI=np.zeros((252, 4500), dtype=np.float)
    count=0
    for i in range(21):
        for j in range(23):
            filePath = "D:\pythonWork\indoor Location\SwapData\coordinate" + xLabel[i] + yLabel[j] + ".mat"
            if (os.path.isfile(filePath)):
                c = loadmat(filePath)
                CSI = np.reshape(c['myData'], (1, 3 * 30 * 50))
                originalCSI[count,:]=CSI
                count+=1
    return originalCSI

def main():

    xLabel = []     #构建训练与测试集
    yLabel = []     #额外新增数据集该如何分，待定
    count = 0
    DataSet = []
    for i in range(21):     #横坐标
        str = '%d' % (i + 1)
        xLabel.append(str)

    for j in range(23):     #纵坐标
        if(j<9):
            num=0
            str= '%d%d' % (num, j+1)
            yLabel.append(str)
        else:
            yLabel.append('%d'% (j+1) )

    # for i in range(16):
    #     for j in range(11):
    #         # filePath = "D:\pythonWork\indoor Location\SwapData\coordinate" + xLabel[i] + yLabel[j] + ".mat"
    #         filePath = "D:\pythonWork\indoor Location\MeetingSwapData\coordinate" + xLabel[i] + yLabel[j] + ".mat"
    #         if (os.path.isfile(filePath)):
    #             count += 1
    #             csiPCA = preProcessing(filePath)
    #             DataSet.append(csiPCA)
    #             print(count)

    # savemat('myDataSetTest.mat',{'csiMatrix': DataSet})
    # savemat('MeetingRoomDataSet.mat', {'csiMatrix': DataSet})

    newName = []
    label = np.empty((0, 2), dtype=np.int) #制作标签或位置,二维标签
    for i in range(21):
        for j in range(23):
            filePath = "D:\pythonWork\indoor Location\SwapData\coordinate" + xLabel[i] + yLabel[j] + ".mat"
            name = xLabel[i] + yLabel[j]
            if (os.path.isfile(filePath)):
                newName.append(name)
                label = np.append(label, [[int(xLabel[i]), int(yLabel[j])]], axis=0)

    c = loadmat(r"/Users/zhuxiaoqiang/Desktop/IEEE Trans/Second paper/代码/myDataSetTest.mat")
    CSIDataSet = c['csiMatrix'] # 预处理已完成，运行时间长达1.5h，直接加载不再重新计算。数据统一规格3*30*50
    CSIDataSet = np.reshape(CSIDataSet,(252,3*30*50))
    originalCSI = rawCSI()
    X = Zscorenormalization(CSIDataSet[0:252, :])
    Y = label   #用于BLS, MLP, NN, LSTM, MultiOutput Regression
    traindata, testdata, trainlabel, testlabel = train_test_split(X, Y, test_size=0.2, random_state=10)
    print(len(traindata), len(trainlabel), len(testdata), len(testlabel))

    NumFea = 60
    NumWin = 3
    NumEnhan = 150
    s = 0.9  # shrink coefficient
    C = 2 ** -10  # Regularization coefficient
    test_ERR, test_MAPE, Testing_time, train_ERR, train_MAPE, Training_time,\
    NetoutTrain,NetoutTest = bls_regression(traindata, trainlabel, testdata, testlabel, s, C, NumFea, NumWin, NumEnhan)

    pointRowGaussSum = get_gaussian(originalCSI).sum(axis=1)
    normalGauss = pointRowGaussSum / sum(pointRowGaussSum)
    reconstructData = np.zeros((CSIDataSet.shape[0], CSIDataSet.shape[1]), dtype=np.float)
    target = np.zeros((CSIDataSet.shape[0], 2), dtype=np.float)
    for i in range(CSIDataSet.shape[0]):
        reconstructData[i] = originalCSI[i, :] * normalGauss[i]
        dataout = 1. / (1 + np.exp(-reconstructData))
        weightProbs = np.exp(-np.sqrt((originalCSI - dataout) ** 2).sum(axis=1) / (2 * 10 * np.var(originalCSI))) / 50
        probability = weightProbs / weightProbs.sum()
        target[i] = probability[i] * Y[i, :]
    kernelPosition = np.array([np.sum(target[:, 0]), np.sum(target[:, 1])]).reshape(1,2)

    index = []  #获取测试集的标签索引(正太分布的随机数)
    for i in range(len(testlabel)):
        index1 = np.where(Y[:,0] == testlabel[i][0])
        index2 = np.where(Y[:,1] == testlabel[i][1])
        similar = list(set(index1[0]).intersection(set(index2[0])))
        index.append(similar)

    for i in range (len(Y)):
        if normalGauss[i] <= np.mean(normalGauss):
            normalGauss[i] = 0.2
        else:
            normalGauss[i] = 0.8

    predictGauss = []
    for i in range (len(testlabel)):
        result = kernelPosition * normalGauss[index[i][0]] + (1 - normalGauss[index[i][0]]) * NetoutTest[i,:]
        predictGauss.append(np.array(result).flatten())
    predictGauss = np.array(predictGauss)

    result= np.asarray(NetoutTrain - trainlabel)
    error = np.asarray(predictGauss - testlabel)                          # Lab参数列表 60 3 150 0.001 20   meeting参数列表 30 3 30 0.001 10
    accuracy1 = np.mean(np.sqrt(result[:, 0] ** 2 + result[:, 1] ** 2))   # training time Lab 0.87865 s    meeting 0.226358 s
    accuracy2 = np.mean(np.sqrt(error[:, 0] ** 2 + error[:, 1] ** 2))     # testing time  Lab 0.005982 s   meeting 0.003033 s
    print('TrainSet location accuracy is ', accuracy1 * 50 / 100, 'm')    # Lab 2.7825494919432545 m    meeting 2.5322672209710784 m
    print('TestSet location accuracy is ',accuracy2 * 50 / 100, 'm')      # Lab 3.470633752181334 m     meeting 2.393443482419429 m
    # saveTestErrorMat(NetoutTest,testlabel,'BLS-WithoutFilter-Lab-Error')
    # plotCDF(predictGauss,testlabel)

    '------------对比实验---------------'
    # -----------MLP------------# 未数据标准化直接处理,2组数据已验证
    # from sklearn.neural_network import MLPRegressor
    # time_start = time.time()
    # clf = MLPRegressor(hidden_layer_sizes=(300,), solver='sgd',activation='tanh',random_state=2).fit(traindata,trainlabel)
    # Training_time = time.time() - time_start
    # predictions = clf.predict(testdata)
    # print('training time is ', round(Training_time, 6), 'seconds')  # Lab 4.580815 s            Meeting 5.669595 s
    # print(accuracyPre(predictions, testlabel) * 50 / 100, 'm')      # Lab 4.623876588022427 m   Meeting 2.7090642691747706 m
    # saveTestErrorMat(predictions, testlabel,'MLP-Meeting-Error')

    # -----------NN------------# 未数据标准化直接处理,2组数据已验证
    # from keras.layers import Dense
    # from keras import models
    # time_start = time.time()
    # model = models.Sequential()
    # model.add(Dense(176, input_dim= 4500, activation='sigmoid', use_bias=True))
    # model.add(Dense(128, activation='relu', use_bias=True))
    # model.add(Dense(2))
    # model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    # model.fit(traindata, trainlabel, epochs=100, batch_size=10, verbose=0)
    # Training_time = time.time() - time_start
    # predicts = model.predict(testdata)
    # acc = accuracyPre(predicts, testlabel)
    # print('training time is ', round(Training_time,6), 'seconds')   # Lab 47.679962 s  meeting  21.95028 s
    # # printLabAccuracy(acc)   # Lab 4.05714387923301 m
    # saveTestErrorMat(predicts, testlabel, 'NN-Meeting-Error')
    # printMeetingAccuracy(acc) # meeting 2.9650364283240807 m

    # -----------LSTM------------# 未数据标准化直接处理,2组数据已验证
    # from keras.models import Sequential
    # from keras.layers import Dense, Dropout
    # from keras.layers import Embedding
    # from keras.layers import LSTM
    # time_start = time.time()
    # model = Sequential()
    # model.add(Embedding(90, output_dim=176))
    # model.add(LSTM(128))
    # model.add(Dropout(0.5))
    # model.add(Dense(2, activation='tanh'))
    # model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    # model.fit(traindata, trainlabel, epochs=100, batch_size=10, verbose=0)
    # Training_time = time.time() - time_start
    # predicts = model.predict(testdata)
    # print('training time is ', round(Training_time,6), 'seconds')   # Lab 7706.123738 s         meeting 4904.169887 s
    # print(accuracyPre(predicts, testlabel) * 60/100,'m')            # Lab 7.478073975575391 m   meeting 5.537790304078949 m
    # saveTestErrorMat(predicts, testlabel, 'LSTM-Meeting-Error')

    # -----------MOR------------# 未数据标准化直接处理,2组数据已验证
    # from sklearn.multioutput import MultiOutputRegressor
    # from sklearn.ensemble import RandomForestRegressor
    # time_start = time.time()
    # clf = MultiOutputRegressor(RandomForestRegressor(n_estimators=100,max_depth=30,random_state=0)).fit(traindata,trainlabel)
    # Training_time = time.time() - time_start
    # predictions = clf.predict(testdata)
    # print('training time is ', round(Training_time, 6), 'seconds')  # Lab  40.883067 s            Meeting   23.095015 s
    # print(accuracyPre(predictions, testlabel) * 60 / 100, 'm')      # Lab  3.7198592188889053 m   Meeting   2.426021791633169 m
    # saveTestErrorMat(predictions, testlabel,'MLOP-Meeting-Error')

    # -----------Horus------------# 应添加进对比实验
    # from sklearn.gaussian_process import GaussianProcessRegressor
    # from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
    # time_start = time.time()
    # kernel = DotProduct() + WhiteKernel()
    # BGM = GaussianProcessRegressor(kernel=kernel, random_state=0).fit(traindata,trainlabel)
    # Training_time = time.time() - time_start
    # prediction = BGM.predict(testdata)
    # print('training time is ', round(Training_time, 6), 'seconds')  # Lab  0.134642 s             Meeting   0.149349 s
    # print(accuracyPre(prediction, testlabel) * 50 / 100, 'm')       # Lab  3.9004495792622267 m    Meeting   2.6208203986177607 m
    # saveTestErrorMat(prediction, testlabel, 'Hours-Lab-Error')

    # -----------RADAR------------# 应添加进对比实验
    # from sklearn.neighbors import KNeighborsRegressor
    # time_start = time.time()
    # KNN = KNeighborsRegressor(n_neighbors=2).fit(traindata,trainlabel)
    # Training_time = time.time() - time_start
    # prediction = KNN.predict(testdata)
    # print('training time is ', round(Training_time, 6), 'seconds')  # Lab  0.013958              Meeting   0.01193 s
    # print(accuracyPre(prediction, testlabel) * 50 / 100, 'm')       # Lab  3.8126913364161243 m    Meeting   3.0663604787899965 m
    # saveTestErrorMat(prediction, testlabel, 'RADAR-Lab-Error')

    # --------TMC'19  SWIM: Speed-aware WiFi-based Passive
    # Indoor Localization for Mobile Ship Environment  ------------#
    from sklearn.svm import SVC
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    time_start = time.time()
    svmRegre = make_pipeline(StandardScaler(), SVC(kernel='rbf', C=15.65874635, gamma='auto'))
    SWIMLabel = np.array(findIndexSWIM(Y, trainlabel)).flatten()
    svmRegre.fit(traindata, SWIMLabel)
    predictIndex = svmRegre.predict(testdata)
    prediction = np.array([Y[x] for x in predictIndex])
    Training_time = time.time() - time_start
    print('training time is ', round(Training_time, 6), 'seconds')  # Lab  0.990355 s
    print(accuracyPre(prediction, testlabel) * 50 / 100, 'm')       # Lab  4.5783174901400585 m
    saveTestErrorMat(prediction, testlabel, 'TMC19-Lab-Error')

    #---------IEEE'19 Bayes------------#
    # from sklearn import linear_model
    # time_start = time.time()
    # clf = linear_model.ARDRegression(n_iter=10, alpha_1=1e-01, lambda_1=1e-03)
    # SWIMLabel = np.array(findIndexSWIM(Y, trainlabel)).flatten()
    # clf.fit(traindata,SWIMLabel)
    # predictIndex = np.trunc(abs(clf.predict(testdata))).astype(int)
    # prediction = np.array([Y[x] for x in predictIndex])
    # Training_time = time.time() - time_start
    # print('training time is ', round(Training_time, 6), 'seconds')  # Lab  3.871753 s
    # print(accuracyPre(prediction, testlabel) * 50 / 100, 'm')       # Lab  4.558957363319246 m
    # saveTestErrorMat(prediction, testlabel, 'IeeeBayes19-Lab-Error')

    '----------舍弃的对比实验------------'
    # -----------Decision Tree------------# 未数据标准化直接处理,2组数据已验证
    # from sklearn.tree import DecisionTreeRegressor
    # from sklearn.linear_model import Ridge
    # time_start = time.time()
    # regressor = DecisionTreeRegressor( random_state=0)
    # # regressor = Ridge()
    # regressor.fit(traindata, trainlabel)
    # Training_time = time.time() - time_start
    # predictions = regressor.predict(testdata)
    # acc = accuracyPre(predictions, testlabel)
    # print('training time is ', round(Training_time, 6), 'seconds')  # Lab 0.462824 s   meeting 0.239877 s
    # printLabAccuracy(acc)       # Lab 3.890053316018748 m
    # saveTestErrorMat(predictions , testlabel, 'DT-Lab-Error')
    # printMeetingAccuracy(acc)   # meeting  m

    #-----------ELM------------# 未数据标准化直接处理,2组数据已验证
    # from elm import ELMRegressor
    # time_start = time.time()
    # ELM = ELMRegressor(n_hidden=90, alpha=0.6, rbf_width=2, activation_func='tanh',random_state=3)
    # ELM.fit(traindata, trainlabel)
    # Training_time = time.time() - time_start
    # predictions = ELM.predict(testdata)
    # acc = accuracyPre(predictions, testlabel)
    # print('training time is ', round(Training_time, 6), 'seconds')  # Lab 0.151548 s   meeting s
    # printLabAccuracy(acc)      # Lab 4.550552730259597 m
    # saveTestErrorMat(predictions, testlabel, 'ELM-Lab-Error')
    # printMeetingAccuracy(acc)  # meeting  m
    # saveTestErrorMat(predictions, testlabel, 'ELM-Meeting-Error')

def findIndexSWIM(label, pathPlan):
    index = []
    for i in range(len(pathPlan)):
        index1 = np.where(label[:, 0] == pathPlan[i][0])
        index2 = np.where(label[:, 1] == pathPlan[i][1])
        similar = list(set(index1[0]).intersection(set(index2[0])))
        index.append(similar)
    index = [x for x in index if x]  # 删除空元素
    return index

def saveTestErrorMat(predictions , testlabel, fileName):
    error = np.asarray(predictions - testlabel)
    sample = np.sqrt(error[:, 0] ** 2 + error[:, 1] ** 2) * 50 / 100
    savemat(fileName+'.mat', {'array': sample})

def plotCDF(prediction, testLabel):     #概率密度函数绘图
    import statsmodels.api as sm
    error = np.asarray(prediction - testLabel)
    sample = np.sqrt(error[:,0] ** 2 + error[:,1] ** 2) * 50 / 100
    ecdf = sm.distributions.ECDF(sample)
    x = np.linspace(min(sample), max(sample))
    y = ecdf(x)
    figure, ax = plt.subplots()
    plt.step(x, y, color = 'blue', marker ='.', label='BLSFL')

    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 15,
             }
    plt.xlabel('Mean Errors (m)',font2)
    plt.ylabel('CDF',font2)
    plt.grid(color="grey", linestyle=':', linewidth=0.5)
    plt.tick_params(labelsize=15)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    plt.legend(prop={'family': 'Times New Roman', 'size': 13})
    plt.show(dpi=500)
    #plt.savefig('figure.eps', bbox_inches = 'tight', dpi=500)

def printLabAccuracy(prediction):
    print(prediction*50/100,'m')

def printMeetingAccuracy(prediction):
    print(prediction*60/100,'m')

def get_gaussian(values):           #高斯分布
    mu = np.mean(values)
    sigma = np.std(values)
    y = (1 / (np.sqrt(2 * np.pi * np.power(sigma, 2)))) * \
        (np.power(np.e, -(np.power((values - mu), 2) / (2 * np.power(sigma, 2)))))
    return y

def Zscorenormalization(x):         #归一化
    x = ( x - np.mean(x)) / np.std(x)
    return x

def cos_sim(vector_a, vector_b):    #余弦相似度
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
    sim = 0.5 + 0.5 * cos
    return sim

if __name__ == '__main__':
    main()
    pass