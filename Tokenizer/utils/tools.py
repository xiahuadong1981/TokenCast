import os                                              # 导入操作系统库，用于创建目录、路径操作
import numpy as np                                     # 导入 NumPy，用于数值计算
import torch                                           # 导入 PyTorch
import matplotlib.pyplot as plt                        # 导入 matplotlib 绘图库
import pandas as pd                                    # 导入 pandas，用于数据处理
from collections.abc import Iterable                   # 导入 Iterable，用于判断是否可迭代
from sklearn.decomposition import PCA                  # 导入 PCA，用于降维

plt.switch_backend('agg')                              # 设置 Matplotlib 使用无界面后端（服务器绘图）

def plot_token_distribution(train_tokens: torch.Tensor, test_tokens: torch.Tensor, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)               # 创建保存目录（存在则忽略）

    _train_tokens = train_tokens.flatten()             # 将训练 token 拉成一维
    _test_tokens = test_tokens.flatten()               # 将测试 token 拉成一维

    # 使用 np.unique 获取数组中每个元素的出现次数
    train_uni_elements, train_cnts_elements = np.unique(_train_tokens, return_counts=True)  # 统计训练 token 频率
    test_uni_elements, test_cnts_elements = np.unique(_test_tokens, return_counts=True)     # 统计测试 token 频率

    plt.clf()                                          # 清空画布

    # 绘制 Groundtruth 的 Token 分布
    plt.bar(train_uni_elements, train_cnts_elements, label='Train')          # 柱状图
    plt.xlabel('Token ID')                               # X 轴标签
    plt.ylabel('Token Count')                            # Y 轴标签
    plt.title('Token Distribution')                      # 标题
    plt.legend()                                         # 图例
    plt.savefig(os.path.join(save_dir, 'train_token_distribution.png'))      # 保存图片
    
    plt.clf()                                            # 清空画布

    # 绘制 Prediction 的 Token 分布
    plt.bar(test_uni_elements, test_cnts_elements, label='Test')             # 柱状图
    plt.xlabel('Token ID')
    plt.ylabel('Token Count')
    plt.title('Token Distribution')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'test_token_distribution.png'))

    plt.clf()

    # 绘制 Groundtruth 和 Prediction 的 Token 分布
    plt.bar(train_uni_elements, train_cnts_elements, label='Train')          # 绘制训练 token 分布
    plt.bar(test_uni_elements, test_cnts_elements, label='Test')             # 绘制测试 token 分布
    plt.xlabel('Token ID')
    plt.ylabel('Token Count')
    plt.title('Token Distribution')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'train_test_token_distribution.png'))

    plt.clf()

def plot_PCA(train_ids, X, save_path, max_token_num):
    # calculate the frequency of each token
    train_tokens = train_ids.flatten()                                       # 展平 token
    train_uni_elements, train_cnts_elements = np.unique(train_tokens, return_counts=True)  # 统计频率
    train_cnts = np.zeros((max_token_num, ))                                 # 初始化频率数组
    train_cnts[train_uni_elements] = train_cnts_elements                     # 将统计值填入

    mask = np.where(train_cnts > 0)                                          # 只取出现过的 token
    X = X[mask]                                                              # 根据 mask 过滤 X
    train_cnts = train_cnts[mask]                                            # 过滤频率

    print(train_cnts)                                                        # 输出 token 频率

    pca = PCA(n_components=2)                                                # 进行 PCA 降到 2 维
    X_r = pca.fit_transform(X)                                               # PCA 降维结果

    X, y = X_r[:, 0], X_r[:, 1]                                              # 两个主成分
    weights = train_cnts                                                     # 点大小或颜色权重

    scatter = plt.scatter(X, y, c=weights, cmap='hot')                       # 绘制散点图
    plt.legend(loc='best', shadow=False, scatterpoints=1)                    # 图例
    plt.title('PCA with weights')                                            # 标题

    # 添加颜色条
    plt.colorbar(scatter)                                                    # 加 colorbar

    plt.savefig(save_path)                                                   # 保存图片

    exit(0)                                                                  # 退出程序

def statistic_freqs(train_ids):
    train_tokens = train_ids.flatten()                                       # 展平 token
    train_uni_elements, train_cnts_elements = np.unique(train_tokens, return_counts=True)  # 统计频率

    total_nums = len(train_tokens)                                           # token 总数
    statis_list = [10,5,2,1.5,1.2,1,0.8,0.7,0.6,0.5,0.2,0.1]                 # 要统计的百分比区间

    for statis in statis_list:
        board = total_nums * (statis / 100.)                                 # 计算阈值
        print(f'Freqs large than {statis}%: {np.sum(train_cnts_elements >= board)}')  # 输出个数
        
    return 

def plot_token_distribution_with_stratify(train_tokens: torch.Tensor, test_tokens: torch.Tensor,
                                          save_dir: str, max_token_num=255, freq=True):

    os.makedirs(save_dir, exist_ok=True)                                     # 创建目录

    _train_tokens = train_tokens.flatten()                                   # 展平训练 token
    _test_tokens = test_tokens.flatten()                                     # 展平测试 token

    # 使用 np.unique 获取数组中每个元素的出现次数
    train_uni_elements, train_cnts_elements = np.unique(_train_tokens, return_counts=True)  # 训练计数
    test_uni_elements, test_cnts_elements = np.unique(_test_tokens, return_counts=True)     # 测试计数

    if freq:
        train_cnts_elements = train_cnts_elements / len(_train_tokens)       # 转为频率
        test_cnts_elements = test_cnts_elements / len(_test_tokens)          # 转为频率

    plt.clf()

    # Groundtruth token 分布
    plt.bar(train_uni_elements, train_cnts_elements, label='Train')
    plt.xlabel('Token ID')
    plt.ylabel('Token Count')
    plt.title('Token Distribution')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'train_token_distribution.png'))

    plt.clf()

    # Prediction token 分布
    plt.bar(test_uni_elements, test_cnts_elements, label='Test')
    plt.xlabel('Token ID')
    plt.ylabel('Token Count')
    plt.title('Token Distribution')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'test_token_distribution.png'))

    plt.clf()

    # Groundtruth 和 Prediction token 分布（对齐 max_token_num）
    train_cnts = np.zeros((max_token_num, ))                                 # 初始化数组
    train_cnts[train_uni_elements] = train_cnts_elements                     # 填入训练频率

    test_cnts = np.zeros((max_token_num, ))                                  # 初始化数组
    test_cnts[test_uni_elements] = test_cnts_elements                        # 填入测试频率

    data1, data2 = train_cnts, test_cnts                                     # 简化变量

    data_low = [min(d1, d2) for d1, d2 in zip(data1, data2)]                # 每个 token 两者较小部分
    data_high = [max(d1, d2) for d1, d2 in zip(data1, data2)]              # 较大部分

    colors_low = ['blue' if d1 < d2 else 'orange' for d1, d2 in zip(data1, data2)]   # 测试低值颜色
    colors_high = ['orange' if d1 < d2 else 'blue' for d1, d2 in zip(data1, data2)]  # 训练高值颜色

    x = np.arange(len(data1))                                               # X 坐标

    # 绘制柱状图（堆叠显示差异）
    plt.bar(x, data_low, color=colors_low, label='Test')                    # 下半部分
    plt.bar(x, data_high, bottom=data_low, color=colors_high, label='Train')# 上半部分

    plt.xlabel('Token ID')
    plt.ylabel('Token Count')
    plt.title('Token Distribution')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'train_test_token_distribution.png'))

    plt.clf()


def clever_format(nums, format="%.2f"):
    if not isinstance(nums, Iterable):                                      # 不是可迭代则包装成列表
        nums = [nums]
    clever_nums = []

    for num in nums:
        if num > 1e12:
            clever_nums.append(format % (num / 1e12) + "T")                # 转为万亿
        elif num > 1e9:
            clever_nums.append(format % (num / 1e9) + "G")                 # 转为十亿
        elif num > 1e6:
            clever_nums.append(format % (num / 1e6) + "M")                 # 转为百万
        elif num > 1e3:
            clever_nums.append(format % (num / 1e3) + "K")                 # 转为千
        else:
            clever_nums.append(format % num + "B")                         # 保留原值加 B

    clever_nums = clever_nums[0] if len(clever_nums) == 1 else (*clever_nums,)   # 单值返回标量

    return clever_nums


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}     # 每 epoch 衰减 0.5
    elif args.lradj == 'type2':
        lr_adjust = {                                                             # 自定义学习率表
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    if epoch in lr_adjust.keys():                                                 # 若当前 epoch 需要调整
        lr = lr_adjust[epoch]                                                     # 获取学习率
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr                                                # 更新学习率
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience                # 容忍没有提升的 epoch 数
        self.verbose = verbose                  # 是否打印信息
        self.counter = 0                        # 连续未提升计数
        self.best_score = None                  # 最佳分数
        self.early_stop = False                 # 是否提前停止
        self.val_loss_min = np.Inf              # 最小验证损失
        self.delta = delta                      # 最小改善幅度

    def __call__(self, val_loss, model, path):
        score = -val_loss                       # 以负 loss 作为“分数”
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)          # 保存第一次模型
        elif score < self.best_score + self.delta:               # 没提升
            self.counter += 1                                    # 计数 +1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:                    # 连续超过 patience
                self.early_stop = True                           # 提前停止
        else:
            self.best_score = score                              # 记录最佳
            self.save_checkpoint(val_loss, model, path)          # 保存模型
            self.counter = 0                                     # 重置计数

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint_96.pth')      # 保存权重
        self.val_loss_min = val_loss                                          # 更新最小 loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get                    # 支持 a.b 写法访问字典
    __setattr__ = dict.__setitem__           # 支持 a.b = c 写法
    __delattr__ = dict.__delitem__           # 支持 del a.b


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean                     # 均值
        self.std = std                       # 标准差

    def transform(self, data):
        return (data - self.mean) / self.std  # 标准化

    def inverse_transform(self, data):
        return (data * self.std) + self.mean  # 反标准化


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()                              # 新建图
    plt.plot(true, label='GroundTruth', linewidth=2)           # 画真实值
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)       # 画预测值
    plt.legend()                               # 图例
    plt.savefig(name, bbox_inches='tight')      # 保存图像


def adjustment(gt, pred):
    anomaly_state = False                      # 当前是否处于异常段
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:  # 异常段开始
            anomaly_state = True
            for j in range(i, 0, -1):                          # 向前扩展
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):                        # 向后扩展
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False                              # 异常段结束
        if anomaly_state:
            pred[i] = 1                                        # 异常段全置 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)                           # 计算准确率


def plot_and_save_reconstruction(model, test_loader, save_path, dims_to_plot=None):
    model.eval()                                                # 设置为 eval 模式
    test_data_iter = iter(test_loader)                          # 获取迭代器
    batch_x, batch_y, _, _ = next(test_data_iter)               # 取一个 batch

    sample_index = 0                                            # 取第一个样本
    sample_data_x = batch_x[sample_index].unsqueeze(0).float().to(next(model.parameters()).device)  # 移动到设备
    sample_data_y = batch_y[sample_index].unsqueeze(0).float().to(next(model.parameters()).device)

    with torch.no_grad():                                       # 禁用梯度
        reconstructed, _, _ = model(sample_data_x, sample_data_y)  # 做重建

    original_data = sample_data_y.squeeze(0).cpu().numpy()       # 原始数据
    reconstructed_data = reconstructed.squeeze(0)[-sample_data_y.size(1):].cpu().numpy()  # 重建后的数据

    # 自动判断是否为单变量时间序列
    if original_data.ndim == 1 or (original_data.ndim == 2 and original_data.shape[1] == 1):
        original_data = original_data.squeeze()                  # 压缩形状
        reconstructed_data = reconstructed_data.squeeze()

        plt.figure(figsize=(12, 4))
        plt.plot(original_data, label='Original')                # 画原始数据
        plt.plot(reconstructed_data, label='Reconstructed')      # 画重建数据
        plt.title("Single-variable Reconstruction Comparison")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'single_variable_reconstruction.pdf'))
        plt.close()
    else:
        # 多变量情况：默认绘制前几个维度
        num_dims = original_data.shape[1]                        # 数据维度数
        if dims_to_plot is None:
            dims_to_plot = list(range(min(4, num_dims)))         # 默认绘 4 个维度

        fig, axes = plt.subplots(len(dims_to_plot), 1, figsize=(12, len(dims_to_plot) * 2))
        for idx, dim in enumerate(dims_to_plot):
            axes[idx].plot(original_data[:, dim], label=f'Original Dim {dim}')   # 原始
            axes[idx].plot(reconstructed_data[:, dim], label=f'Recon Dim {dim}') # 重建
            axes[idx].set_title(f"Data Comparison - Dim {dim}")
            axes[idx].legend()

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'multi_variable_dim_comparison.pdf'))
        plt.close()

        # 总体重建图（多变量）
        fig, ax = plt.subplots(2, 1, figsize=(12, 8))
        for dim in dims_to_plot:
            ax[0].plot(original_data[:, dim], label=f'Dim {dim}')          # 原始
            ax[1].plot(reconstructed_data[:, dim], label=f'Dim {dim}')      # 重建
        ax[0].set_title("Original Data - Selected Dims")
        ax[1].set_title("Reconstructed Data - Selected Dims")
        ax[0].legend()
        ax[1].legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'multi_variable_total_reconstruction.pdf'))
        plt.close()
