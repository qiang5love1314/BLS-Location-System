# from plotMultiSub import rawCSI,get_gaussian
from SecondMeetingMain import rawCSI, get_gaussian
import os
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

def main():
    c = loadmat(r'/Users/zhuxiaoqiang/Desktop/IEEE Trans/Second paper/代码/MeetingRoomDataSet.mat')
    CSIDataSet = c['csiMatrix']
    CSIDataSet = np.reshape(CSIDataSet, (176, 3 * 30 * 50))
    originalCSI = rawCSI()
    # originalCSI=originalCSI[0:176,0:3000]

    xLabel = []  # 构建训练与测试集
    yLabel = []
    for i in range(16):     #横坐标
        str = '%d' % (i + 1)
        xLabel.append(str)
    for j in range(11):     #纵坐标
        if(j<9):
            num=0
            str= '%d%d' % (num, j+1)
            yLabel.append(str)
        else:
            yLabel.append('%d'% (j+1) )
    newName = []
    label = np.empty((0, 2), dtype=np.int)  # 制作标签或位置,二维标签
    for i in range(16):
        for j in range(11):
            # filePath = "D:\pythonWork\indoor Location\SwapData\coordinate" + xLabel[i] + yLabel[j] + ".mat"
            filePath = "D:\pythonWork\indoor Location\MeetingSwapData\coordinate" + xLabel[i] + yLabel[j] + ".mat"
            name = xLabel[i] + yLabel[j]
            if (os.path.isfile(filePath)):
                newName.append(name)
                label = np.append(label, [[int(xLabel[i]), int(yLabel[j])]], axis=0)

    Y = label
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
    kernelPosition = np.array([np.sum(target[:, 0]), np.sum(target[:, 1])]).reshape(1, 2)

    print(kernelPosition)
    figure, ax = plt.subplots()
    plt.plot(probability, color = 'blue')

    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 15,
             }
    plt.xlabel('Each sample', font2)
    plt.ylabel('Probability', font2)
    plt.grid(color="grey", linestyle=':', linewidth=0.5)
    plt.tick_params(labelsize=15)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    # plt.legend(prop={'family': 'Times New Roman', 'size': 12})
    # plt.savefig('Gauss_raw_CSI.png', bbox_inches='tight', dpi=500)
    # plt.show(dpi=500)

if __name__ == '__main__':
    main()
    pass