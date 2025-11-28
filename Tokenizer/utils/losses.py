# This source code is provided for the purposes of scientific reproducibility             # 说明：此源码仅用于科研复现目的
# under the following limited license from Element AI Inc. The code is an                # 版权和授权信息
# implementation of the N-BEATS model (Oreshkin et al., N-BEATS: Neural basis            # 说明这是 N-BEATS 模型的实现
# expansion analysis for interpretable time series forecasting,                          # 模型论文链接
# https://arxiv.org/abs/1905.10437). The copyright to the source code is                 # 源码受版权保护
# licensed under the Creative Commons - Attribution-NonCommercial 4.0                    # 使用 CC BY-NC 4.0 非商业协议
# International license (CC BY-NC 4.0):                                                  # 协议链接
# https://creativecommons.org/licenses/by-nc/4.0/.  Any commercial use (whether         # 商业使用必须额外授权
# for the benefit of third parties or internally in production) requires an              #
# explicit license. The subject-matter of the N-BEATS model and associated               # N-BEATS 可能受专利保护
# materials are the property of Element AI Inc. and may be subject to patent             #
# protection. No license to patents is granted hereunder (whether express or             #
# implied). Copyright © 2020 Element AI Inc. All rights reserved.                        # 版权所有

"""
Loss functions for PyTorch.                                                             # 本文件用于定义 PyTorch 中的各种损失函数
"""

import torch as t                                                                        # 导入 PyTorch 并命名为 t
import torch.nn as nn                                                                    # 导入 PyTorch 神经网络模块
import numpy as np                                                                       # 导入 NumPy
import pdb                                                                               # 导入调试工具 pdb


def divide_no_nan(a, b):
    """
    a/b where the resulted NaN or Inf are replaced by 0.                                 # 实现安全除法，将 NaN 和 inf 替换为 0
    """
    result = a / b                                                                        # 普通除法
    result[result != result] = .0                                                         # 将 NaN（即 result != result）替换成 0
    result[result == np.inf] = .0                                                         # 将 +inf 替换成 0
    return result                                                                         # 返回安全结果


class mape_loss(nn.Module):
    def __init__(self):
        super(mape_loss, self).__init__()                                                 # 调用父类构造函数

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        MAPE loss as defined in: https://en.wikipedia.org/wiki/Mean_absolute_percentage_error

        :param forecast: Forecast values. Shape: batch, time                             # 预测值
        :param target: Target values. Shape: batch, time                                 # 真实值
        :param mask: 0/1 mask. Shape: batch, time                                        # 有效位置掩码
        :return: Loss value
        """
        weights = divide_no_nan(mask, target)                                             # 计算权重 = mask / target（避免除 0）
        return t.mean(t.abs((forecast - target) * weights))                               # MAPE = |预测-真实|/真实


class smape_loss(nn.Module):
    def __init__(self):
        super(smape_loss, self).__init__()                                                # 构造函数

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        sMAPE loss (对称平均绝对百分比误差)

        :param forecast: Forecast values. Shape: batch, time
        :param target: Target values. Shape: batch, time
        :param mask: 0/1 mask. Shape: batch, time
        :return: Loss value
        """
        return 200 * t.mean(                                                               # sMAPE * 200
            divide_no_nan(t.abs(forecast - target),                                        # 分子：|预测 - 实际|
                          t.abs(forecast.data) + t.abs(target.data))                       # 分母：|预测| + |实际|
            * mask                                                                         # 乘以 mask，忽略无效点
        )


class mase_loss(nn.Module):
    def __init__(self):
        super(mase_loss, self).__init__()                                                 # 构造函数

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        MASE 损失（Scaled Errors）

        :param insample: Insample values. Shape: batch, time_i                            # 历史数据，用于计算季节性尺度
        :param freq: Frequency value                                                      # 周期长度（如 24/7/12）
        :param forecast: Forecast values. Shape: batch, time_o
        :param target: Target values. Shape: batch, time_o
        :param mask: 0/1 mask. Shape: batch, time_o
        :return: Loss value
        """
        masep = t.mean(t.abs(insample[:, freq:] - insample[:, :-freq]), dim=1)            # 计算季节性误差平均值
        masked_masep_inv = divide_no_nan(mask, masep[:, None])                            # mask / masep（避免除 0）
        return t.mean(t.abs(target - forecast) * masked_masep_inv)                        # |预测-真实| / 季节性误差
