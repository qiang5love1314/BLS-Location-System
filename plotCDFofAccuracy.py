import statsmodels.api as sm
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

def read(path):
    sample = loadmat(path)
    sample = np.array(sample['array']).flatten()
    ecdf = sm.distributions.ECDF(sample)
    x = np.linspace(min(sample), max(sample))
    y = ecdf(x)
    return x, y

def main():
    '-----Lab and Meeting Room-----'
    x1, y1 = read('D:\pythonWork\indoor Location\BLSFL-Lab-Error.mat')
    x2, y2 = read('D:\pythonWork\indoor Location\MLP-Lab-Error.mat')
    x3, y3 = read(r'D:\pythonWork\indoor Location\Neural-Lab-Error.mat')
    x4, y4 = read('D:\pythonWork\indoor Location\LSTM-Lab-Error.mat')
    x5, y5 = read('D:\pythonWork\indoor Location\MLOP-Lab-Error.mat')
    x6, y6 = read('D:\pythonWork\indoor Location\Hours-Lab-Error.mat')
    x7, y7 = read('D:\pythonWork\indoor Location\RADAR-Lab-Error.mat')

    # x2, y2 = read('D:\pythonWork\indoor Location\BLSFL-Lab-Error.mat')
    # x3, y3 = read(r'D:\pythonWork\indoor Location\BLS-Meeting-Error.mat')
    # x4, y4 = read('D:\pythonWork\indoor Location\BLS-Lab-Error.mat')

    # x2, y2 = read('D:\pythonWork\indoor Location\BLSFL-WithoutFilter-Meeting-Error.mat')
    # x3, y3 = read(r'D:\pythonWork\indoor Location\BLS-WithoutFilter-Meeting-Error.mat')
    # x4, y4 = read('D:\pythonWork\indoor Location\BLSFL-Lab-Error.mat')
    # x5, y5 = read('D:\pythonWork\indoor Location\BLSFL-WithoutFilter-Lab-Error.mat')
    # x6, y6 = read('D:\pythonWork\indoor Location\BLS-WithoutFilter-Lab-Error.mat')

    # x2, y2 = read('D:\pythonWork\indoor Location\BLSFL-1Antenna-Meeting-Error.mat')
    # x3, y3 = read('D:\pythonWork\indoor Location\BLSFL-2Antenna-Meeting-Error.mat')
    # x4, y4 = read('D:\pythonWork\indoor Location\BLSFL-1and2Antenna-Meeting-Error.mat')
    # x5, y5 = read('D:\pythonWork\indoor Location\BLSFL-3Antenna-Meeting-Error.mat')

    sample = loadmat(r'D:\pythonWork\indoor Location\RADAR-Lab-Error.mat')
    sample = np.array(sample['array']).flatten()
    print(np.std(sample))

    figure, ax = plt.subplots()
    'comparison'
    plt.step(x1, y1, color = 'blue', marker ='.', label='BLS-Location')
    plt.step(x2, y2, color='green', marker='v', label='MLP')
    plt.step(x3, y3, color='red', marker='x', label='NN')
    # plt.step(x4, y4, color='c', marker='^', label='LSTM')
    plt.step(x5, y5, color='m', marker='p', label='MOR')
    plt.step(x6, y6, color='darkcyan', marker='D', label='HORUS')
    plt.step(x7, y7, color='brown', marker='*', label='RADAR')

    'kernel position'
    # plt.step(x1, y1, color='blue', marker='.', label='BLS-Location-LOS')
    # plt.step(x2, y2, color='green', marker='v', label='BLS-Location-NLOS')
    # plt.step(x3, y3, color='red', marker='x', label='BLS-LOS')
    # plt.step(x4, y4, color='c', marker='^', label='BLS-NLOS')

    'preprocessing'
    # plt.step(x1, y1, color = 'blue', marker ='.', label='BLS-Location-LOS')
    # plt.step(x2, y2, color='green', marker='v', label='BLS-Location without filter-LOS')
    # plt.step(x3, y3, color='red', marker='x', label='BLS without filter-LOS')
    # plt.step(x4, y4, color='c', marker='^', label='BLS-Location-NLOS')
    # plt.step(x5, y5, color='m', marker='p', label='BLS-Location without filter-NLOS')
    # plt.step(x6, y6, color='y', marker='d', label='BLS without Ffilter-NLOS')

    'different antenna'
    # plt.step(x1, y1, color='blue', marker='.', label='BLS-Location with all antennas')
    # plt.step(x2, y2, color='green', marker='v', label='BLS-Location with 1st antenna')
    # plt.step(x3, y3, color='red', marker='x', label='BLS-Location with 2nd antenna')
    # plt.step(x4, y4, color='c', marker='^', label='BLS-Location with 1st and 2nd antennas')

    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 15,
             }
    plt.xlabel('Estimated Errors (m)',font2)
    plt.ylabel('CDF',font2)
    plt.grid(color="grey", linestyle=':', linewidth=0.5)
    plt.tick_params(labelsize=15)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    plt.legend(prop={'family': 'Times New Roman', 'size': 12}, loc = 'lower right')
    plt.savefig('CDF_Lab_Accuracy.png', bbox_inches = 'tight', dpi=500)
    plt.show(dpi=500)

if __name__ == '__main__':
    main()
    pass