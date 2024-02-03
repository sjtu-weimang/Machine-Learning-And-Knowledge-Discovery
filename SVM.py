import numpy as np
import matplotlib.pyplot as plt


class SMO:

    def __init__(self, X, y, C, kernel, tol, max_passes=4):
        self.X = X  # 样本特征 m*n m个样本 n个特征
        self.y = y  # 样本标签 m*1
        self.C = C  # 惩罚因子, 用于控制松弛变量的影响
        self.kernel = kernel  # 核函数
        self.tol = tol  # 容忍度
        self.max_passes = max_passes  # 最大迭代次数
        self.m, self.n = X.shape
        self.alpha = np.zeros(self.m)
        self.b = 0
        self.w = np.zeros(self.n)

    # 计算核函数
    def K(self, i, j):
        if self.kernel == 'linear':
            return np.dot(self.X[i].T, self.X[j])
        elif self.kernel == 'rbf':
            gamma = 0.5
            return np.exp(-gamma * np.linalg.norm(self.X[i] - self.X[j]) ** 2)

        else:
            raise ValueError('Invalid kernel specified')

    def predict(self, X):
        pred = np.zeros_like(X[:, 0])
        pred = np.dot(X_test, self.w) + self.b
        return np.sign(pred)

    def train(self):
        """
        训练模型
        :return:
        """
        passes = 0
        while passes < self.max_passes:
            num_changed_alphas = 0
            for i in range(self.m):

                E_i = 0
                for ii in range(self.m):
                    E_i += self.alpha[ii] * self.y[ii] * self.K(ii, i)
                E_i += self.b - self.y[i]
                # 检验样本x_i是否满足KKT条件
                if (self.y[i] * E_i < -self.tol and self.alpha[i] < self.C) or (
                        self.y[i] * E_i > self.tol and self.alpha[i] > 0):
                    # 随机选择样本x_j
                    j = np.random.choice(list(range(i)) + list(range(i + 1, self.m)), size=1)[0]
                    # 计算E_j, E_j = f(x_j) - y_j, f(x_j) = w^T * x_j + b
                    # E_j用于检验样本x_j是否满足KKT条件
                    E_j = 0
                    for jj in range(self.m):
                        E_j += self.alpha[jj] * self.y[jj] * self.K(jj, j)
                    E_j += self.b - self.y[j]

                    alpha_i_old = self.alpha[i].copy()
                    alpha_j_old = self.alpha[j].copy()

                    # L和H用于将alpha[j]调整到[0, C]之间
                    if self.y[i] != self.y[j]:
                        L = max(0, self.alpha[j] - self.alpha[i])
                        H = min(self.C, self.C + self.alpha[j] - self.alpha[i])
                    else:
                        L = max(0, self.alpha[i] + self.alpha[j] - self.C)
                        H = min(self.C, self.alpha[i] + self.alpha[j])

                    # 如果L == H，则不需要更新alpha[j]
                    if L == H:
                        continue

                    # eta: alpha[j]的最优修改量
                    eta = 2 * self.K(i, j) - self.K(i, i) - self.K(j, j)
                    # 如果eta >= 0, 则不需要更新alpha[j]
                    if eta >= 0:
                        continue

                    # 更新alpha[j]
                    self.alpha[j] -= (self.y[j] * (E_i - E_j)) / eta
                    # 根据取值范围修剪alpha[j]
                    self.alpha[j] = np.clip(self.alpha[j], L, H)

                    # 检查alpha[j]是否只有轻微改变，如果是则退出for循环
                    if abs(self.alpha[j] - alpha_j_old) < 1e-5:
                        continue

                    # 更新alpha[i]
                    self.alpha[i] += self.y[i] * self.y[j] * (alpha_j_old - self.alpha[j])

                    # 更新b1和b2
                    b1 = self.b - E_i - self.y[i] * (self.alpha[i] - alpha_i_old) * self.K(i, i) \
                         - self.y[j] * (self.alpha[j] - alpha_j_old) * self.K(i, j)
                    b2 = self.b - E_j - self.y[i] * (self.alpha[i] - alpha_i_old) * self.K(i, j) \
                         - self.y[j] * (self.alpha[j] - alpha_j_old) * self.K(j, j)

                    # 根据b1和b2更新b
                    if 0 < self.alpha[i] < self.C:
                        self.b = b1
                    elif 0 < self.alpha[j] < self.C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2

                    num_changed_alphas += 1

            if num_changed_alphas == 0:
                passes += 1
            else:
                passes = 0

        # 提取支持向量和对应的参数
        idx = self.alpha > 0  # 支持向量的索引
        # SVs = X[idx]
        selected_idx = np.where(idx)[0]
        SVs = X[selected_idx]
        SV_labels = y[selected_idx]
        SV_alphas = self.alpha[selected_idx]

        # 计算权重向量和截距
        self.w = np.sum(SV_alphas[:, None] * SV_labels[:, None] * SVs, axis=0)
        self.b = np.mean(SV_labels - np.dot(SVs, self.w))
        print("w", self.w)
        print("b", self.b)

    def score(self, X, y):
        predict = self.predict(X)
        print("predict", predict)
        print("target", y)
        return np.mean(predict == y)


def loadDataSet(filename, delim=','):
    fr = open(filename)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    datArr = [list(map(float, line[2:])) for line in stringArr[1:]]
    tarArr = [line[1] for line in stringArr[1:]]
    return np.mat(datArr), tarArr


def train_test_split(X, y, test_size=0.2, random_state=None):
    # 设置随机数种子
    if random_state is not None:
        np.random.seed(random_state)
    # 获取样本数量
    num_samples = X.shape[0]
    # 计算测试集的样本数量
    if isinstance(test_size, float):
        test_size = int(test_size * num_samples)
    # 打乱数据集的索引
    shuffled_indices = np.random.permutation(num_samples)
    # 根据测试集大小划分索引
    test_indices = shuffled_indices[:test_size]
    train_indices = shuffled_indices[test_size:]
    # 划分数据集
    X_train = X[train_indices]
    y_train = y[train_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]
    return X_train, X_test, y_train, y_test


def standardize(X):
    # 计算每个特征的均值和标准差
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    # 标准化每个特征
    X_std = (X - mean) / std
    return X_std


# 加载鸢尾花数据集
X, y = loadDataSet('data.csv')
X = np.array(X)
y = np.array(y)
y[y == 'M'] = 0
y[y == 'B'] = 1
y = y.astype(np.float64)
y[y == 0] -= 1
# print(y)
# 为了方便可视化，只取前两个特征，并且只取两类
# X = X[y < 2, :2]
# y = y[y < 2]
# # 分别画出类别 0 和 1 的点
plt.scatter(X[y != 1, 0], X[y != 1, 1], color='red')
plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue')
plt.show()

# 数据预处理，将特征进行标准化，并将数据划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=124)
X_train_std = standardize(X_train)

# 创建SVM对象并训练模型
svm = SMO(X_train_std, y_train, C=0.6, kernel='rbf', tol=0.001)
svm.train()

# 预测测试集的结果并计算准确率
X_test_std = standardize(X_test)
accuracy = svm.score(X_test_std, y_test)

print('正确率: {:.2%}'.format(accuracy))
