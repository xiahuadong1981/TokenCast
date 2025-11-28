import numpy as np  # 导入 NumPy 库，用于数值计算


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))  # 计算 RSE，相当于预测误差与数据自身波动的比值


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)  # 计算协方差的分子部分
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))  # 计算标准差乘积，做归一化
    return (u / d).mean(-1)  # 得到逐维相关系数后取平均，返回整体相关性


def MAE(pred, true):
    return np.mean(np.abs(pred - true))  # 计算 MAE：平均绝对误差


def MSE(pred, true):
    return np.mean((pred - true) ** 2)  # 计算 MSE：平均平方误差


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))  # 计算 RMSE：MSE 的平方根


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))  # 计算 MAPE：平均绝对百分比误差


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))  # 计算 MSPE：平均平方百分比误差


def metric(pred, true):
    mae = MAE(pred, true)  # 计算 MAE
    mse = MSE(pred, true)  # 计算 MSE
    rmse = RMSE(pred, true)  # 计算 RMSE
    mape = MAPE(pred, true)  # 计算 MAPE
    mspe = MSPE(pred, true)  # 计算 MSPE

    return mae, mse, rmse, mape, mspe  # 返回五种误差指标
