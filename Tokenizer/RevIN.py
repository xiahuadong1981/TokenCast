# code from https://github.com/ts-kim/RevIN, with minor modifications  # 原始代码来源，稍作修改

import torch  # 导入 PyTorch 张量库
import torch.nn as nn  # 导入神经网络模块

class RevIN(nn.Module):  # 定义 RevIN 类（Reversible Instance Normalization）
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        """
        :param num_features: the number of features or channels         # 输入特征或通道数
        :param eps: a value added for numerical stability                # 数值稳定项，避免除零
        :param affine: if True, RevIN has learnable affine parameters    # 是否使用可学习仿射参数
        """
        super(RevIN, self).__init__()  # 调用父类构造函数
        self.num_features = num_features  # 保存特征数量
        self.eps = eps  # 保存 eps
        self.affine = affine  # 是否使用仿射变换
        self.subtract_last = subtract_last  # 是否减去最后一个时间步的值（时序模型中常用）
        if self.affine:
            self._init_params()  # 初始化仿射参数

    def forward(self, x, mode:str):  # 前向传播，mode 表示是否归一化或反归一化
        if mode == 'norm':  # 执行归一化
            self._get_statistics(x)  # 计算均值/方差或 last/stdev
            x = self._normalize(x)  # 对输入执行归一化
        elif mode == 'denorm':  # 执行反归一化（推理阶段）
            x = self._denormalize(x)  # 恢复到原始尺度
        else: raise NotImplementedError  # 不支持其他模式
        return x  # 返回处理结果

    def _init_params(self):
        # initialize RevIN params: (C,)  # 初始化仿射参数，形状为 (通道数,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))  # 可学习缩放参数
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))   # 可学习偏置参数

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))  # 要求均值/方差的维度（除 batch 和 channel）
        if self.subtract_last:
            self.last = x[:,-1,:].unsqueeze(1)  # 使用最后一个时间步作为基准
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()  # 沿时序和特征维求均值，并 detach
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()  # 计算方差并取 sqrt 得到标准差

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last  # 减去最后一个时间步值
        else:
            x = x - self.mean  # 减去均值
        x = x / self.stdev  # 除以标准差（归一化）
        if self.affine:
            x = x * self.affine_weight  # 仿射缩放
            x = x + self.affine_bias    # 仿射平移
        return x  # 返回归一化结果

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias  # 移除仿射偏置
            x = x / (self.affine_weight + self.eps*self.eps)  # 恢复仿射缩放（避免除零）
        x = x * self.stdev  # 恢复原始标准差
        if self.subtract_last:
            x = x + self.last  # 恢复 last
        else:
            x = x + self.mean  # 恢复 mean
        return x  # 返回反归一化结果
