# From: gluonts/src/gluonts/time_feature/_base.py  # 文件来源说明
# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").          # 版权相关说明
# You may not use this file except in compliance with the License.         # 不可脱离协议使用
# A copy of the License is located at                                      # 协议位置
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed # 分发协议
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either # 无担保条款
# express or implied. See the License for the specific language governing   # 使用限制
# permissions and limitations under the License.

from typing import List  # 导入 List 类型提示

import numpy as np       # 导入 numpy
import pandas as pd      # 导入 pandas
from pandas.tseries import offsets  # 时间偏移相关类，例如 MonthEnd 等
from pandas.tseries.frequencies import to_offset  # 将字符串频率转为 offset 对象


class TimeFeature:
    def __init__(self):
        pass  # 空初始化，无成员变量

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        pass  # 子类需要实现，可调用对象

    def __repr__(self):
        return self.__class__.__name__ + "()"  # 返回类名() 用于打印


class SecondOfMinute(TimeFeature):
    """Minute of hour encoded as value between [-0.5, 0.5]"""  # 秒归一化为 [-0.5, 0.5]

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.second / 59.0 - 0.5  # 当前秒除以最大59并平移到中心对称区间


class MinuteOfHour(TimeFeature):
    """Minute of hour encoded as value between [-0.5, 0.5]"""  # 分钟归一化

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.minute / 59.0 - 0.5  # 归一化分钟


class HourOfDay(TimeFeature):
    """Hour of day encoded as value between [-0.5, 0.5]"""  # 小时归一化

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.hour / 23.0 - 0.5  # 归一化小时


class DayOfWeek(TimeFeature):
    """Hour of day encoded as value between [-0.5, 0.5]"""  # 星期归一化

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.dayofweek / 6.0 - 0.5  # 周一=0，周日=6，归一化


class DayOfMonth(TimeFeature):
    """Day of month encoded as value between [-0.5, 0.5]"""  # 月内日期归一化

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.day - 1) / 30.0 - 0.5  # 日期 1~31 → 归一化到中心


class DayOfYear(TimeFeature):
    """Day of year encoded as value between [-0.5, 0.5]"""  # 年内日期归一化

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.dayofyear - 1) / 365.0 - 0.5  # 一年 1~365 → [-0.5,0.5]


class MonthOfYear(TimeFeature):
    """Month of year encoded as value between [-0.5, 0.5]"""  # 月份归一化

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.month - 1) / 11.0 - 0.5  # 0~11 → [-0.5,0.5]


class WeekOfYear(TimeFeature):
    """Week of year encoded as value between [-0.5, 0.5]"""  # 周序号归一化

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.isocalendar().week - 1) / 52.0 - 0.5  # ISO周 → 归一化


def time_features_from_frequency_str(freq_str: str) -> List[TimeFeature]:
    """
    Returns a list of time features that will be appropriate for the given frequency string.
    Parameters
    ----------
    freq_str
        Frequency string of the form [multiple][granularity] such as "12H", "5min", "1D" etc.
    """  # 根据频率字符串，选择对应的时间特征列表

    features_by_offsets = {
        offsets.YearEnd: [],  # 年度数据不需要时间特征
        offsets.QuarterEnd: [MonthOfYear],  # 季度特征：月份
        offsets.MonthEnd: [MonthOfYear],    # 月度特征：月份
        offsets.Week: [DayOfMonth, WeekOfYear],  # 每周：使用日和周
        offsets.Day: [DayOfWeek, DayOfMonth, DayOfYear],  # 每日：星期、日、年内日
        offsets.BusinessDay: [DayOfWeek, DayOfMonth, DayOfYear],  # 工作日同上
        offsets.Hour: [HourOfDay, DayOfWeek, DayOfMonth, DayOfYear],  # 小时级
        offsets.Minute: [  # 分钟级特征
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
        offsets.Second: [  # 秒级特征
            SecondOfMinute,
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
    }

    offset = to_offset(freq_str)  # 转换字符串频率 → pandas offset 对象

    for offset_type, feature_classes in features_by_offsets.items():  # 遍历偏移类型
        if isinstance(offset, offset_type):  # 判断频率属于哪个类型
            return [cls() for cls in feature_classes]  # 实例化对应特征类

    supported_freq_msg = f"""
    Unsupported frequency {freq_str}
    The following frequencies are supported:
        Y   - yearly
            alias: A
        M   - monthly
        W   - weekly
        D   - daily
        B   - business days
        H   - hourly
        T   - minutely
            alias: min
        S   - secondly
    """  # 未支持频率时的错误提示
    raise RuntimeError(supported_freq_msg)  # 抛出错误


def time_features(dates, freq='h'):
    return np.vstack([feat(dates) for feat in time_features_from_frequency_str(freq)])  # 生成所有时间特征并按行堆叠成矩阵
