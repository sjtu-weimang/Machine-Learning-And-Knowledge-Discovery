import numpy as np
import matplotlib.pyplot as plt

"""
函数说明：解析文本数据

Parameters:
filename - 文件名
delim - 每一行不同特征数据之间的分隔方式，默认是tab键‘\t’

Returns:
j将float型数据值列表转化为矩阵返回

"""


def loadDataSet(filename, delim=','):
    fr = open(filename)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    datArr = [list(map(float, line[1:5])) for line in stringArr[1:]]
    tarArr = [line[-1] for line in stringArr[1:]]
    return np.mat(datArr), tarArr


"""
函数说明：PCA特征维度压缩函数

Parameters:
dataMat - 数据集数据
topNfeat - 需要保留的特征维度，即要压缩成的维度数，默认4096

Returns:
lowDDataMat - 压缩后的数据矩阵
reconMat - 压缩后的数据矩阵反构出原始数据矩阵

"""


def pca(dataMat, topNfeat=4096):
    # 求矩阵每一列的均值
    meanVals = np.mean(dataMat, axis=0)
    # 数据矩阵每一列特征减去该列特征均值
    meanRemoved = dataMat - meanVals
    # 计算协方差矩阵，处以n-1是为了得到协方差的无偏估计
    # cov(x, 0) = cov(x)除数是n-1(n为样本个数)
    # cov(x, 1)除数是n
    covMat = np.cov(meanRemoved, rowvar=0)
    # 计算协方差矩阵的特征值及对应的特征向量
    # 均保存在相应的矩阵中
    eigVals, eigVects = np.linalg.eig(np.mat(covMat))
    # sort():对特征值矩阵排序(由小到大)
    # argsort():对特征矩阵进行由小到大排序，返回对应排序后的索引
    eigValInd = np.argsort(eigVals)
    # 从排序后的矩阵最后一个开始自下而上选取最大的N个特征值，返回其对应的索引
    eigValInd = eigValInd[: -(topNfeat + 1): -1]
    # 将特征值最大的N个特征值对应索引的特征向量提取出来，组成压缩矩阵
    redEigVects = eigVects[:, eigValInd]
    # 将去除均值后的矩阵*压缩矩阵，转换到新的空间，使维度降低为N
    lowDDataMat = meanRemoved * redEigVects
    # 利用降维后的矩阵反构出原数据矩阵(用作测试，可跟未压缩的原矩阵比对)
    # 此处用转置和逆的结果一样redEigVects.I
    reconMat = (lowDDataMat * redEigVects.T) + meanVals
    # print(reconMat)
    # 返回压缩后的数据矩阵及该矩阵反构出原始数据矩阵
    return lowDDataMat, reconMat


def lda(data, target, n_dim):
    '''
    :param data: (n_samples, n_features)
    :param target: data class
    :param n_dim: target dimension
    :return: (n_samples, n_dims)
    '''

    clusters = np.unique(target)

    if n_dim > len(clusters) - 1:
        print("K is too much")
        print("please input again")
        exit(0)

    # within_class scatter matrix
    Sw = np.zeros((data.shape[1], data.shape[1]))
    for i in clusters:
        index = []
        for j in range(0, len(target)):
            if target[j] == i:
                index.append(j)
        for ind in index:
            datai = data[ind]
            datai = datai - datai.mean()
            Swi = np.mat(datai).T * np.mat(datai)
            Sw += Swi
    # between_class scatter matrix
    SB = np.zeros((data.shape[1], data.shape[1]))
    u = data.mean(0)  # 所有样本的平均值
    for i in clusters:
        for j in target:
            if j == i:
                Ni = data[target.index(j)].shape[0]
                ui = data[target.index(j)].mean(0)  # 某个类别的平均值
                SBi = Ni * np.mat(ui - u).T * np.mat(ui - u)
                SB += SBi
    S = np.linalg.inv(Sw) * SB
    eigVals, eigVects = np.linalg.eig(S)  # 求特征值，特征向量
    eigValInd = np.argsort(eigVals)
    eigValInd = eigValInd[:(-n_dim - 1):-1]
    w = eigVects[:, eigValInd]
    data_ndim = np.dot(data, w)
    return data_ndim


if __name__ == '__main__':
    data = loadDataSet('Iris.csv')
    dataMat = data[0]
    X = data[0]
    Y = data[1]

    data_1 = lda(X, Y, 2)
    label_mapping = {
        'Iris-setosa': 0,
        'Iris-versicolor': 1,
        'Iris-virginica': 2
    }
    # 将字符串标签转换为数值标签
    numeric_labels = [label_mapping[label] for label in Y]
    # 使用数值标签作为颜色编码
    #print(data_1[:30, 0])
    #print(data_1[:30, 1])
    fig = plt.figure()
    ay= fig.add_subplot(211)
    plt.subplots_adjust(left=0.08, right=0.95, top=0.9, bottom=0.1,wspace=0.1,hspace=0.3)
    ay.set_title("LDA")
    ay.scatter(list(data_1[:, 0]), list(data_1[:, 1]), c=numeric_labels)

    #fig.tight_layout()
    lowDmat, reconMat = pca(dataMat, 1)
    print('降维后的数据维度：', lowDmat.shape)
    print("重构后的数据维度：", reconMat.shape)  # 重构数据维度
    ax = fig.add_subplot(212)
    ax.set_title("PCA")
    ax.scatter(dataMat[:, 0].flatten().A[0], dataMat[:, 1].flatten().A[0], marker='^', s=10)
    ax.scatter(reconMat[:, 0].flatten().A[0], reconMat[:, 1].flatten().A[0], marker='o', s=10, c='red')
    plt.show()
