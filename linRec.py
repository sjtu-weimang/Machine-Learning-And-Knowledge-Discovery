# 这是一个示例 Python 脚本。
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
from scipy.optimize import minimize  # 选择minimize函数做梯度下降


def linReg(x, y):
    # 设定超参数
    w0, w1, lrate = 1, 1, 0.01  # lrate代表学习率
    times = 5  # times表示迭代次数

    # 循环求模型的参数
    for i in range(times):
        # 输出每一轮运算过程中，w0、w1、1oss的变化过程：
        loss = ((w0 + w1 * x - y) ** 2).sum() / 2
        print('{:4}, w0:{:.8f}, w1:{:.8f}, loss:{:.8f}'.format(i + 1, w0, w1, loss))

        # 计算w0和W1方向上的偏导数，代入模型参数的更新公式
        d0 = (w0 + w1 * x - y).sum()
        d1 = (x * (w0 + w1 * x - y)).sum()
        # 更新w0和w1
        w0 = w0 - lrate * d0
        w1 = w1 - lrate * d1
    return w0, w1


dfr = pd.read_csv(r'winequality-red.csv', sep=';')  # dfr short for dataframe_red
dfw = pd.read_csv(r'winequality-white.csv', sep=';')  # dfw short for dataframe_white


def feature_label_split(pd_data):
    # 行数、列数
    row_cnt, column_cnt = pd_data.shape
    # 生成新的X、Y矩阵
    X = np.empty([row_cnt, column_cnt - 1])
    Y = np.empty([row_cnt, 1])
    for i in range(0, row_cnt):
        row_array = pd_data.iloc[i,]
        X[i] = np.array(row_array[0:-1])
        Y[i] = np.array(row_array[-1])
    return X, Y


X, Y = feature_label_split(dfr)


# 对数据进行归一化处理操作
def uniform_norm(X_in):
    X_max = X_in.max(axis=0)
    X_min = X_in.min(axis=0)
    X = (X_in - X_min) / (X_max - X_min)
    return X


# 对X矩阵进行归一化
unif_X = uniform_norm(X)


# 线性回归模型
class linear_regression:

    def __init__(self):
        self.theta = None

    def fit(self, train_X_in, train_Y, learning_rate=0.3, lamda=0.08, regularization="l2"):
        # 样本个数、样本的属性个数
        case_cnt, feature_cnt = train_X_in.shape
        # X矩阵添加X0向量,将截距向量合并
        train_X = np.c_[train_X_in, np.ones(case_cnt, )]
        # 初始化待调参数theta
        self.theta = np.zeros([feature_cnt + 1, 1])

        max_iter_num = sys.maxsize  # 最多迭代次数
        step = 0  # 当前已经迭代的次数
        pre_step = 0  # 上一次得到较好学习误差的迭代学习次数

        last_error_J = sys.maxsize  # 上一次得到较好学习误差的误差函数值
        threshold_value = 1e-6  # 定义在得到较好学习误差之后截止学习的阈值
        stay_threshold_times = 10  # 定义在得到较好学习误差之后截止学习之前的学习次数

        for step in range(0, max_iter_num):
            # 预测值
            pred = train_X.dot(self.theta)
            # 损失函数
            J_theta = sum((pred - train_Y) ** 2) / (2 * case_cnt) + lamda * sum(self.theta ** 2) / (2 * len(train_X))
            # 更新参数theta
            self.theta -= learning_rate * (lamda * self.theta + (train_X.T.dot(pred - train_Y)) / case_cnt)

            # 检测损失函数的变化值，提前结束迭代
            if J_theta < last_error_J - threshold_value:
                last_error_J = J_theta
                pre_step = step
            elif step - pre_step > stay_threshold_times:
                break

            # 定期打印，方便用户观察变化
            if step % 100 == 0:
                print("step %s: %.6f" % (step, J_theta))

    def predict(self, X_in):
        case_cnt = X_in.shape[0]
        X = np.c_[X_in, np.ones(case_cnt, )]
        pred = X.dot(self.theta)
        return pred


# 逻辑回归模型
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
    '''
    def one_vs_all(self, X, y, l, K):
        all_theta = np.full((6, X.shape[1]), 2)  # (6, 1599)

        for i in range(3, K + 3):
            max_iter_num = sys.maxsize  # 最多迭代次数
            step = 0  # 当前已经迭代的次数
            pre_step = 0  # 上一次得到较好学习误差的迭代学习次数

            last_error_J = sys.maxsize  # 上一次得到较好学习误差的误差函数值
            threshold_value = 1e-5  # 定义在得到较好学习误差之后截止学习的阈值
            stay_threshold_times = 10  # 定义在得到较好学习误差之后截止学习之前的学习次数
            theta = np.ones(X.shape[1])
            y_i = np.array([1 if label == i else 0 for label in y])  # 将y转换成0和1，即是y类和不是y类
            print(i)
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
                    print("loss:%.6f" % loss)
                    break
            print(i, "jest")
            all_theta[i - 3, :] = theta

        return all_theta
        '''

    # 使用TNC最优化函数
    def one_vs_all(self, X, y, l, K):
        all_theta = np.zeros((K, X.shape[1]))  # (6, 1599)

        for i in range(3, K + 3):
            theta = np.zeros(X.shape[1])
            y_i = np.array([1 if label == i else 0 for label in y])  # 将y转换成0和1，即是y类和不是y类
            ret = minimize(fun=self.regularized_cost, x0=theta, args=(X, y_i, l), method='TNC',
                           jac=self.regularized_gradient,
                           options={'disp': 0})
            all_theta[i - 3, :] = ret.x

        return all_theta

    # 定义预测函数
    def predict_all(self, X, all_theta):
        h = self.sigmoid(X @ all_theta.T)  # 这里的all_theta需要转置
        # h为（5000，10） 每列是预测对应数字的概率
        h_argmax = np.argmax(h, axis=1)  # 取每列最大的概率的index加3就是所预测的类别
        h_argmax = h_argmax + 3

        return h_argmax


# 线性回归模型训练
# 模型预测
model = linear_regression()
model.fit(unif_X, Y)

# 模型评估
test_pred = model.predict(unif_X)
test_pred_error = sum((test_pred - Y) ** 2) / (2 * unif_X.shape[0])
print("Predicted is ", test_pred)
print("Test error is %.6f" % test_pred_error)

# 逻辑回归模型训练
'''
model = logistic_regression()
Xl = np.insert(unif_X, 0, 1, axis=1)  # (5000, 401) 添加偏置列
all_theta = model.one_vs_all(X, Y, 10, 6)
# print(all_theta)
y_pred = model.predict_all(X, all_theta)
# print(y_pred)
accuracy = np.mean(y_pred == Y)
print('正确率 = {0}%'.format(accuracy * 100))

# print(model.theta)
# print(Y[50])
# print(model.predict(unif_X[50]))
'''
