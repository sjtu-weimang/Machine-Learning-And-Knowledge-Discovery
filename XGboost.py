import numpy as np
import pandas as pd
import xgboost as xgb
from graphviz import Digraph


class Node(object):
    """结点
       leaf_value ： 记录叶子结点值
       split_feature ：特征i
       split_value ： 特征i的值
       left ： 左子树
       right ： 右子树
    """

    def __init__(self, leaf_value=None, split_feature=None, split_value=None, left=None, right=None):
        self.leaf_value = leaf_value
        self.split_feature = split_feature
        self.split_value = split_value
        self.left = left
        self.right = right

    def show(self):
        print(
            f'weight: {self.leaf_value}, split_feature: {self.split_feature}, split_value: {self.split_value}.')

    def visualize_tree(self, i):
        """
        递归查找绘制树
        """

        def add_nodes_edges(self, dot=None):
            if dot is None:
                dot = Digraph()
                dot.node(name=str(self),
                         label=f'{self.split_feature}<{self.split_value}')
            # Add nodes
            if self.left:
                if self.left.leaf_value:
                    dot.node(name=str(self.left),
                             label=f'leaf={self.left.leaf_value:.10f}')
                else:
                    dot.node(name=str(self.left),
                             label=f'{self.left.split_feature}<{self.left.split_value}')
                dot.edge(str(self), str(self.left))
                dot = add_nodes_edges(self.left, dot=dot)
            if self.right:
                if self.right.leaf_value:
                    dot.node(name=str(self.right),
                             label=f'leaf={self.right.leaf_value:.10f}')
                else:
                    dot.node(name=str(self.right),
                             label=f'{self.right.split_feature}<{self.right.split_value}')
                dot.edge(str(self), str(self.right))
                dot = add_nodes_edges(self.right, dot=dot)
            return dot

        dot = add_nodes_edges(self)
        dot.render('tree{}'.format(i), format='png', view=True)
        return dot


def log_loss_obj(preds, labels):
    preds = 1.0 / (1.0 + np.exp(-preds))
    grad = preds - labels
    hess = preds * (1.0 - preds)
    return grad, hess


def mse_obj(preds, labels):
    grad = labels - preds
    hess = np.ones_like(labels)
    return grad, hess


class XGB:
    def __init__(self, n_estimators=2, learning_rate=0.1, max_depth=3, min_samples_split=0, reg_lambda=1,
                 base_score=0.5, loss=log_loss_obj):
        # 学习控制参数
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.base_score = base_score
        # 树参数
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.loss = loss
        self.reg_lambda = reg_lambda

        self.trees = []
        self.feature_names = None

    def sigmoid_array(self, x):
        return 1 / (1 + np.exp(-x))

    def _predict(self, x, tree):
        # 循环终止条件：叶节点
        if tree.leaf_value is not None:
            return tree.leaf_value
        if x[tree.split_feature] < tree.split_value:
            return self._predict(x, tree.left)
        else:
            return self._predict(x, tree.right)

    def _build_tree(self, df, depth=1):
        df = df.copy()
        df['g'], df['h'] = self.loss(df.y_pred, df.y)
        G, H = df[['g', 'h']].sum()

        Gain_max = float('-inf')
        if df.shape[0] > self.min_samples_split and depth <= self.max_depth and df.y.nunique() > 1:
            # print(self.feature_names)
            for feature in self.feature_names:
                thresholds = sorted(set(df[feature]))
                # print(feature,thresholds)
                for thresh_value in thresholds[1:]:
                    left_instance = df[df[feature] < thresh_value]
                    right_instance = df[df[feature] >= thresh_value]
                    G_left, H_left = left_instance[['g', 'h']].sum()
                    G_right, H_right = right_instance[['g', 'h']].sum()

                    Gain = G_left ** 2 / (H_left + self.reg_lambda) + G_right ** 2 / \
                           (H_right + self.reg_lambda) - G ** 2 / (H + self.reg_lambda)
                    if Gain >= Gain_max:
                        Gain_max = Gain
                        split_feature = feature
                        split_value = thresh_value
                        left_data = left_instance
                        right_data = right_instance
                    # print(feature,'Gain:',Gain,'G-Left',G_left,'H-left',H_left,'G-Right',G_right,'H-right',H_right,'----',thresh_value)
            # print(Gain_max,split_feature,split_value)

            left = self._build_tree(left_data, depth + 1)
            right = self._build_tree(right_data, depth + 1)
            return Node(split_feature=split_feature, split_value=split_value, left=left, right=right)

        return Node(leaf_value=-G / (H + self.reg_lambda) * self.learning_rate)

    def fit(self, X, y):
        y_pred = -np.log((1 / self.base_score) - 1)
        df = pd.DataFrame(X)
        feature = df.columns.tolist()
        df['y'] = y
        self.feature_names = df.columns.tolist()[:-1]

        for i in range(self.n_estimators):
            df['y_pred'] = y_pred
            tree = self._build_tree(df)
            data_weight = df[feature].apply(self._predict, tree=tree, axis=1)
            y_pred += data_weight
            self.trees.append(tree)

    def predict(self, X):
        df = pd.DataFrame(X)
        feature = df.columns.tolist()
        y_pred = -np.log((1 / self.base_score) - 1)
        for tree in self.trees:
            df['y_pred'] = y_pred
            data_weight = df[feature].apply(self._predict, tree=tree, axis=1)
            y_pred += data_weight
        return self.sigmoid_array(y_pred)

    def __repr__(self):
        return 'XGBClassifier(' + ', '.join(f'{k}={v}' for k, v in self.__dict__.items() if not k.startswith('_')) + ')'


df = pd.read_csv('data.csv', index_col=0)
df = df.copy()
features = df.columns[1:25]
df['diagnosis'].replace({'M': 1, 'B': 0}, inplace=True)
df_train = df.iloc[:501, :]
df_test = df.iloc[501:, :]
model = XGB()
model.fit(df_train[features], df_train.diagnosis)
model.trees[1].visualize_tree(1)
model.trees[0].visualize_tree(0)

# 使用测试集预测
s = model.predict(df_test[features])
right = 0
all = len(df_test)
for index in s.index:
    if s.loc[index] > 0.5 and df.loc[index, 'diagnosis'] == 1:
        right += 1
    elif s.loc[index] < 0.5 and df.loc[index, 'diagnosis'] == 0:
        right += 1
print("预测准确率为：{}".format(right / all))

