import torch  # 导入 PyTorch 主库
import torch.nn as nn  # 导入 PyTorch 的神经网络模块

class RevIN(nn.Module):  # 定义 RevIN 类，继承自 nn.Module
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):  # 初始化函数
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()  # 调用父类构造函数
        self.num_features = num_features  # 保存特征数量（通道数）
        self.eps = eps  # 数值稳定项，防止除零
        self.affine = affine  # 是否启用可学习仿射参数
        self.subtract_last = subtract_last  # 是否使用序列最后值代替均值
        if self.affine:  # 如果需要仿射变换
            self._init_params()  # 初始化仿射权重参数

    def forward(self, x, mode:str):  # 前向传播函数，mode 决定执行 norm 还是 denorm
        if mode == 'norm':  # 执行标准化
            self._get_statistics(x)  # 计算均值和标准差（或 last）
            x = self._normalize(x)  # 执行标准化
        elif mode == 'denorm':  # 执行反标准化
            x = self._denormalize(x)  # 执行反标准化
        else: raise NotImplementedError  # 其他模式不支持
        return x  # 返回结果

    def _init_params(self):  # 初始化可学习的仿射参数
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))  # 仿射缩放参数 γ，初始为 1
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))  # 仿射平移参数 β，初始为 0

    def _get_statistics(self, x):  # 计算统计量：均值(mean)/最后值(last) 和 标准差(stdev)
        dim2reduce = tuple(range(1, x.ndim-1))  # 要压缩的维度（除 batch 和 channel）
        if self.subtract_last:  # 是否使用序列最后一个值代替均值
            self.last = x[:,-1,:].unsqueeze(1)  # 取序列最后一时间步的值 (B,1,C)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()  # 计算均值并 detach
        self.stdev = torch.sqrt(  # 计算标准差
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps  # 加 eps 防止除零
        ).detach()  # detach 防止梯度回传

    def _normalize(self, x):  # 标准化操作
        if self.subtract_last:  # 如果使用 last 作为基准
            x = x - self.last  # 减去 last
        else:
            x = x - self.mean  # 减去均值
        x = x / self.stdev  # 除以标准差
        if self.affine:  # 如果启用仿射变换
            x = x * self.affine_weight  # 乘以可学习缩放参数 γ
            x = x + self.affine_bias  # 加上可学习平移参数 β
        return x  # 返回标准化结果

    def _denormalize(self, x):  # 反标准化操作
        if self.affine:  # 如果启用仿射参数
            x = x - self.affine_bias  # 先减去 β
            x = x / (self.affine_weight + self.eps*self.eps)  # 再除以 γ（加 eps 防止除零）
        x = x * self.stdev  # 乘回标准差
        if self.subtract_last:  # 恢复 last 或 mean
            x = x + self.last  # 加回 last
        else:
            x = x + self.mean  # 加回均值
        return x  # 返回反标准化结果
