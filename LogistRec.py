import time
from numpy import *
import numpy as np
import sys
from scipy.optimize import minimize  # 选择minimize函数做梯度下降


# 定义sigmoid函数
def sigmoid(n):
    return 1.0 / (1 + exp(-n))


def getdata():
    # 读取数据集 并且把数据集分为测试集和训练集，奇数项为测试集，偶数项为训练集
    dataSet = []
    labels = []
    f = open('Iris.csv')
    flag = 1  # 设置flag来跳过第一行
    for line in f.readlines():
        if flag == 1:
            flag = 0
            continue
        str_data = line.strip().split(",")[1:]
        labels.append(str_data[-1])
        for k in range(len(str_data) - 1):
            str_data[k] = float(str_data[k])
        str_data.insert(0, 1.0)
        dataSet.append(str_data)
    train_data = []
    test_data = []
    train_label = []
    dataset = dataSet
    # 归一化处理优点：1.加快求解速度 2.可能提高精度
    for j in range(1, len(dataset[0]) - 1):
        l_sum = 0
        all_sum = 0
        for i in range(len(dataset)):
            l_sum += dataset[i][j]
        # 计算期望
        average = l_sum / len(dataset)
        # 计算标准差
        for i in range(len(dataset)):
            all_sum += pow(dataset[i][j] - average, 2)
        standard_deviation = math.sqrt(all_sum / (len(dataset) - 1))
        # 对每列数据进行归一化,减均值，除方差
        for i in range(len(dataset)):
            dataset[i][j] = (dataset[i][j] - average) / standard_deviation
    for i in range(len(dataSet)):
        if i % 2 == 0:
            train_data.append(dataSet[i])
            train_label.append(labels[i])
        else:
            test_data.append(dataSet[i])
    # 返回参数中，dataSet为全部数据，，train_data为训练集，test_data为数据集
    return dataSet, labels, train_data, test_data, train_label


def map_strings_to_numbers(string_list):
    mapping = {}
    number_list = []
    for string in string_list:
        if string not in mapping:
            mapping[string] = len(mapping)
        number_list.append(mapping[string])
    return number_list


data = getdata()
X = np.array(data[2])
X = np.delete(X, 5, 1)
Y = np.array(map_strings_to_numbers(data[4]))
X = X.astype(float)


class logistic_regression():

    def __init__(self):
        self.theta = None
        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))

    # 定义代价函数
    def regularized_cost(self, theta, X, y, l):
        thetaReg = theta[1:]
        first = (-y * np.log(self.sigmoid(X @ theta))) + (y - 1) * np.log(1 - self.sigmoid(X @ theta))
        reg = (thetaReg @ thetaReg) * l / (2 * len(X))
        return np.mean(first) + reg

    # 定义梯度
    def regularized_gradient(self, theta, X, y, l):
        thetaReg = theta[1:]
        first = (1 / len(X)) * X.T @ (self.sigmoid(X @ theta) - y)
        # 这里人为插入一维0，使得对theta_0不惩罚，方便计算
        reg = np.concatenate([np.array([0]), (l / len(X)) * thetaReg])
        return first + reg

        # 定义一对多分类器,使用梯度下降法
    def one_vs_all(self, X, y, l):
        all_theta = np.full((3, X.shape[1]), 0)  # (6, 1599)

        for i in range(0, 3):
            max_iter_num = sys.maxsize  # 最多迭代次数
            step = 0  # 当前已经迭代的次数
            pre_step = 0  # 上一次得到较好学习误差的迭代学习次数

            last_error_J = sys.maxsize  # 上一次得到较好学习误差的误差函数值
            threshold_value = 1e-6  # 定义在得到较好学习误差之后截止学习的阈值
            stay_threshold_times = 10  # 定义在得到较好学习误差之后截止学习之前的学习次数
            theta = np.ones(X.shape[1])
            y_i = np.array([1 if label == i else 0 for label in y])  # 将y转换成0和1，即是y类和不是y类
            for step in range(1, max_iter_num):
                loss = self.regularized_cost(theta, X, y_i, l)
                theta -= 0.01 * self.regularized_gradient(theta, X, y_i, l)
                # if step % 1000 == 0:
                # print("step %s: %.6f" % (step, loss))
                # 检测损失函数的变化值，提前结束迭代
                if loss < last_error_J - threshold_value:
                    last_error_J = loss
                    pre_step = step
                elif step - pre_step > stay_threshold_times:
                    print("loss%d:%.6f" % (i, loss))
                    break
            all_theta[i, :] = theta

        return all_theta

    '''
        # 使用TNC最优化函数

    def one_vs_all(self, X, y, l, K):
        all_theta = np.zeros((K, X.shape[1]))  # (6, 1599)

        for i in range(0, 3):
            theta = np.zeros(X.shape[1])
            y_i = np.array([1 if label == i else 0 for label in y])  # 将y转换成0和1，即是y类和不是y类
            ret = minimize(fun=self.regularized_cost, x0=theta, args=(X, y_i, l), method='TNC',
                           jac=self.regularized_gradient,
                           options={'disp': 0})
            all_theta[i, :] = ret.x

        return all_theta
        '''

    # 定义预测函数

    def predict_all(self, X, all_theta):
        h = self.sigmoid(X @ all_theta.T)  # 这里的all_theta需要转置
        # h为（5000，10） 每列是预测对应数字的概率
        h_argmax = np.argmax(h, axis=1)  # 取每列最大的概率的index加3就是所预测的类别
        h_argmax = h_argmax

        return h_argmax


model = logistic_regression()
Xl = np.insert(X, 0, 1, axis=1)  # (5000, 401) 添加偏置列
all_theta = model.one_vs_all(X, Y, 0.05)
# print(all_theta)
y_pred = model.predict_all(X, all_theta)
# print(y_pred)
accuracy = np.mean(y_pred == Y)
print('正确率 = {0}%'.format(accuracy * 100))
