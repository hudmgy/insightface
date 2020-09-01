from __future__ import division
import os
import time
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import random
import ipdb


def gen_training_data(coord):
    data_X = []
    data_Y = []
    num = coord.shape[0]

    if num >= 2:
        index = np.argsort(coord[:,0])
        coord = coord[index]
        index = list(range(num))
        for i in range(num):
            interv = random.sample(index, 2)
            interv.sort()
            p1, p2 = interv[0], interv[1]
            L2 = coord[p1] - coord[p2]
            L2 = L2.dot(L2)
            L2 = np.sqrt(L2)
            if L2 > 250:
                dv = np.hstack([coord[p1], coord[p2]])
                dn = p2 - p1
                data_X.append(dv)
                data_Y.append(dn)
    return data_X, data_Y


def load_dataset(folder):
    data_X = []
    data_Y = []
    for dirname, _, files in os.walk(folder):
        for fi in files:
            if fi.endswith('.npy'):
                coord = np.load(os.path.join(dirname, fi))
                dx, dy = gen_training_data(coord)
                data_X += dx
                data_Y += dy

    data_X = np.array(data_X)
    data_Y = np.array(data_Y)
    np.save('data_X.npy', data_X)
    np.save('data_Y.npy', data_Y)
    data_X[:,0] /= 256.0
    data_X[:,1] /= 144.0
    data_Y = data_Y / 10.0
    return data_X, data_Y


def svm_c(x_train, x_test, y_train, y_test):
    # rbf核函数，设置数据权重
    svc = SVC(kernel='rbf', class_weight='balanced',)
    c_range = np.logspace(-5, 15, 11, base=2)
    gamma_range = np.logspace(-9, 3, 13, base=2)
    # 网格搜索交叉验证的参数范围，cv=3,3折交叉
    param_grid = [{'kernel': ['rbf'], 'C': c_range, 'gamma': gamma_range}]
    grid = GridSearchCV(svc, param_grid, cv=3, n_jobs=-1)
    # 训练模型
    clf = grid.fit(x_train, y_train)
    # 计算测试集精度
    score = grid.score(x_test, y_test)
    print('精度为%s' % score)

if __name__ == '__main__':
    data_dir = '/home/ct/Data/queue/frames'
    data_X, data_Y = load_dataset(data_dir)
    train_num = len(data_Y) * 9 // 10
    train_data_X = data_X[:train_num]
    train_data_Y = data_Y[:train_num]
    test_data_X = data_X[train_num:]
    test_data_Y = data_Y[train_num:]

    # 初始化SVR
    svr = GridSearchCV(SVR(kernel='rbf', gamma=0.1), cv=5,
                       param_grid={"C": [1e0, 1e1, 1e2, 1e3],
                                   "gamma": np.logspace(-2, 2, 5)})
    ipdb.set_trace()
    # 训练
    t0 = time.time()
    svr.fit(train_data_X, train_data_Y)
    svr_fit = time.time() - t0

    #测试
    t0 = time.time()
    y_svr = svr.predict(test_data_X)
    svr_predict = time.time() - t0