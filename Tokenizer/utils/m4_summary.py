```python
# This source code is provided for the purposes of scientific reproducibility             # 说明：此源码仅用于科研复现目的
# under the following limited license from Element AI Inc. The code is an                # 在 Element AI Inc. 的有限许可下发布
# implementation of the N-BEATS model (Oreshkin et al., N-BEATS: Neural basis            # 实现的是 N-BEATS 时间序列模型
# expansion analysis for interpretable time series forecasting,                          # 模型论文：可解释的时间序列预测
# https://arxiv.org/abs/1905.10437). The copyright to the source code is                 # 论文链接
# licensed under the Creative Commons - Attribution-NonCommercial 4.0                    # 源码在 CC BY-NC 4.0 协议下授权
# International license (CC BY-NC 4.0):                                                  # 非商业使用许可
# https://creativecommons.org/licenses/by-nc/4.0/.  Any commercial use (whether          # 商业使用需要单独授权
# for the benefit of third parties or internally in production) requires an              # 无论对内对外，只要商业用途都需授权
# explicit license. The subject-matter of the N-BEATS model and associated               # N-BEATS 模型及相关材料可能受专利保护
# materials are the property of Element AI Inc. and may be subject to patent             # 说明专利权归 Element AI 所有
# protection. No license to patents is granted hereunder (whether express or             # 本协议不授予任何专利许可
# implied). Copyright 2020 Element AI Inc. All rights reserved.                          # 版权声明

"""
M4 Summary                                                                             # 文档字符串：用于说明 M4 评估摘要
"""
from collections import OrderedDict                                                    # 从 collections 导入有序字典 OrderedDict

import numpy as np                                                                     # 导入 NumPy，用于数值计算
import pandas as pd                                                                    # 导入 Pandas，用于读写 CSV 等数据

from data_provider.m4 import M4Dataset                                                 # 从自定义模块导入 M4 数据集类
from data_provider.m4 import M4Meta                                                    # 从自定义模块导入 M4 元信息（比如季节性分组）
import os                                                                              # 导入 os，用于路径操作


def group_values(values, groups, group_name):                                          # 定义函数：根据分组名称提取对应时间序列并去掉 NaN
    return np.array([v[~np.isnan(v)] for v in values[groups == group_name]])          # 选出属于该组的序列，并去除其中的 NaN，最后转为数组


def mase(forecast, insample, outsample, frequency):                                   # 定义 MASE 评价指标（单条序列版本）
    return np.mean(np.abs(forecast - outsample)) / np.mean(                           # 返回：预测误差均值 / 季节性差分误差均值
        np.abs(insample[:-frequency] - insample[frequency:]))                         # 分母：历史序列按 frequency 做滞后差分的平均绝对值


def smape_2(forecast, target):                                                        # 定义 sMAPE（对称平均绝对百分比误差）的实现
    denom = np.abs(target) + np.abs(forecast)                                         # 分母：|真实值| + |预测值|
    # divide by 1.0 instead of 0.0, in case when denom is zero the enumerator will be 0.0 anyway.  # 说明：当分母为 0 时改成 1，避免除 0
    denom[denom == 0.0] = 1.0                                                         # 将分母为 0 的位置替换为 1.0
    return 200 * np.abs(forecast - target) / denom                                    # 返回 sMAPE = 200 * |预测-真实| / (|预测|+|真实|)


def mape(forecast, target):                                                           # 定义 MAPE（平均绝对百分比误差）的实现
    denom = np.abs(target)                                                            # 分母：|真实值|
    # divide by 1.0 instead of 0.0, in case when denom is zero the enumerator will be 0.0 anyway.  # 同样避免除 0
    denom[denom == 0.0] = 1.0                                                         # 将真实值为 0 的位置分母设置为 1.0
    return 100 * np.abs(forecast - target) / denom                                    # 返回 MAPE = 100 * |预测-真实| / |真实|


class M4Summary:                                                                      # 定义 M4Summary 类，用于对 M4 结果进行评估与汇总
    def __init__(self, file_path, root_path):                                         # 构造函数，传入文件路径和数据根目录
        self.file_path = file_path                                                    # 保存模型预测结果的前缀路径（不同组会拼接 group_name）
        self.training_set = M4Dataset.load(training=True, dataset_file=root_path)     # 加载 M4 训练集
        self.test_set = M4Dataset.load(training=False, dataset_file=root_path)        # 加载 M4 测试集
        self.naive_path = os.path.join(root_path, 'submission-Naive2.csv')            # Naive2 基线提交文件路径（官方提供的基准）

    def evaluate(self):                                                               # 定义评估方法，对模型在 M4 测试集上的表现进行打分
        """
        Evaluate forecasts using M4 test dataset.

        :param forecast: Forecasts. Shape: timeseries, time.
        :return: sMAPE and OWA grouped by seasonal patterns.
        """
        grouped_owa = OrderedDict()                                                   # 有序字典，用来保存按组计算的 OWA 指标

        naive2_forecasts = pd.read_csv(self.naive_path).values[:, 1:].astype(np.float32)  # 读取 Naive2 提交文件，去掉第一列 ID，转为 float32
        naive2_forecasts = np.array([v[~np.isnan(v)] for v in naive2_forecasts])      # 每条序列去除 NaN，并重新打包为数组

        model_mases = {}                                                              # 保存模型按组的 MASE
        naive2_smapes = {}                                                            # 保存 Naive2 按组的 sMAPE
        naive2_mases = {}                                                             # 保存 Naive2 按组的 MASE
        grouped_smapes = {}                                                           # 保存模型按组的 sMAPE
        grouped_mapes = {}                                                            # 保存模型按组的 MAPE
        for group_name in M4Meta.seasonal_patterns:                                   # 遍历所有季节性分组（Yearly/Quarterly/Monthly/...）
            file_name = self.file_path + group_name + "_forecast.csv"                # 当前组对应的模型预测文件路径
            if os.path.exists(file_name):                                            # 如果该组的预测文件存在
                model_forecast = pd.read_csv(file_name).values                       # 读取预测文件为 ndarray（每行一条时间序列）

            naive2_forecast = group_values(naive2_forecasts,                         # 针对当前组，从 Naive2 预测中取出同组序列并去掉 NaN
                                           self.test_set.groups, group_name)
            target = group_values(self.test_set.values,                              # 从测试集真实值中取出同组序列并去掉 NaN
                                   self.test_set.groups, group_name)
            # all timeseries within group have same frequency                         # 注释：同一组内所有序列的频率（周期）相同
            frequency = self.training_set.frequencies[self.test_set.groups == group_name][0]  # 取该组的频率值（例如 12、4、24 等）
            insample = group_values(self.training_set.values,                        # 从训练集取出该组对应的 insample 序列（历史部分）
                                   self.test_set.groups, group_name)

            model_mases[group_name] = np.mean([                                      # 计算当前组下模型的平均 MASE
                mase(forecast=model_forecast[i],                                     # 对每一条序列计算 MASE
                     insample=insample[i],
                     outsample=target[i],
                     frequency=frequency) for i in range(len(model_forecast))])      # 遍历所有序列求平均

            naive2_mases[group_name] = np.mean([                                     # 计算当前组下 Naive2 的平均 MASE
                mase(forecast=naive2_forecast[i],                                    # 使用 Naive2 的预测
                     insample=insample[i],
                     outsample=target[i],
                     frequency=frequency) for i in range(len(model_forecast))])      # 注意长度以 model_forecast 为基准

            naive2_smapes[group_name] = np.mean(smape_2(naive2_forecast, target))    # 计算当前组 Naive2 的平均 sMAPE
            grouped_smapes[group_name] = np.mean(                                    # 计算当前组模型的平均 sMAPE
                smape_2(forecast=model_forecast, target=target))
            grouped_mapes[group_name] = np.mean(                                     # 计算当前组模型的平均 MAPE
                mape(forecast=model_forecast, target=target))

        grouped_smapes = self.summarize_groups(grouped_smapes)                       # 按 M4 规则重新汇总 sMAPE（包括 Others、Average）
        grouped_mapes = self.summarize_groups(grouped_mapes)                         # 按规则汇总 MAPE
        grouped_model_mases = self.summarize_groups(model_mases)                     # 按规则汇总模型的 MASE
        grouped_naive2_smapes = self.summarize_groups(naive2_smapes)                 # 按规则汇总 Naive2 的 sMAPE
        grouped_naive2_mases = self.summarize_groups(naive2_mases)                   # 按规则汇总 Naive2 的 MASE
        for k in grouped_model_mases.keys():                                         # 遍历所有分组（Yearly/Quarterly/.../Average）
            grouped_owa[k] = (grouped_model_mases[k] / grouped_naive2_mases[k] +     # OWA = (模型 MASE / Naive2 MASE
                              grouped_smapes[k] / grouped_naive2_smapes[k]) / 2      #      + 模型 sMAPE / Naive2 sMAPE) / 2

        def round_all(d):                                                            # 内部函数：把字典中的数值全部四舍五入到 3 位小数
            return dict(map(lambda kv: (kv[0], np.round(kv[1], 3)), d.items()))      # 返回新的字典，值被 np.round(x, 3) 处理

        return (                                                                     # 返回四类评价结果（均为按组汇总后的字典）
            round_all(grouped_smapes),                                               # 1) sMAPE
            round_all(grouped_owa),                                                  # 2) OWA
            round_all(grouped_mapes),                                                # 3) MAPE
            round_all(grouped_model_mases))                                          # 4) 模型 MASE

    def summarize_groups(self, scores):                                              # 按 M4 比赛规则对各组分数做重新汇总
        """
        Re-group scores respecting M4 rules.
        :param scores: Scores per group.
        :return: Grouped scores.
        """
        scores_summary = OrderedDict()                                               # 保存汇总后的分数字典（保持顺序）

        def group_count(group_name):                                                 # 内部函数：计算某个组中包含多少条时间序列
            return len(np.where(self.test_set.groups == group_name)[0])              # 在 test_set.groups 中统计等于 group_name 的个数

        weighted_score = {}                                                          # 保存各组加权分数（分数 * 个数）
        for g in ['Yearly', 'Quarterly', 'Monthly']:                                 # 先处理三大主类：年、季、月
            weighted_score[g] = scores[g] * group_count(g)                           # 加权分数 = 该组分数 * 该组样本数量
            scores_summary[g] = scores[g]                                            # 直接把该组的平均分存入汇总结果

        others_score = 0                                                             # Others 组的加权总分初始化为 0
        others_count = 0                                                             # Others 组的总个数初始化为 0
        for g in ['Weekly', 'Daily', 'Hourly']:                                      # 将周、日、小时归为 Others 组
            others_score += scores[g] * group_count(g)                               # 累加各自的加权分数
            others_count += group_count(g)                                           # 累加各自的样本数量
        weighted_score['Others'] = others_score                                      # Others 的总加权分数
        scores_summary['Others'] = others_score / others_count                       # Others 的平均分 = 总加权分数 / 总个数

        average = np.sum(list(weighted_score.values())) / len(self.test_set.groups)  # 全局平均分：所有加权分数和 / 总序列数量
        scores_summary['Average'] = average                                          # 记录到汇总结果中

        return scores_summary                                                        # 返回包含各组和平均分的有序字典
```
