import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from scipy.optimize import linear_sum_assignment

def acc(y_true, y_pred):
    """计算聚类准确率"""
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return sum([w[i, j] for i, j in zip(row_ind, col_ind)]) * 1.0 / y_pred.size

def nmi(y_true, y_pred):
    """计算标准化互信息"""
    return normalized_mutual_info_score(y_true, y_pred)

def ari(y_true, y_pred):
    """计算调整兰德指数"""
    return adjusted_rand_score(y_true, y_pred)