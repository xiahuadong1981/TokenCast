
import json  # 导入 JSON 处理模块
import torch  # 导入 PyTorch
import random  # 导入随机数模块
import os  # 操作系统接口
import numpy as np  # 数值计算模块
import warnings  # 警告控制模块

warnings.filterwarnings('ignore')  # 忽略所有警告
from data_provider.data_factory import data_provider  # 数据提供器
from args import args  # 配置参数
from process import Trainer  # 训练流程类
from models.VQVAE import VQVAE  # VQVAE 模型
from models.W_SimVQ import W_SimVQ  # SimVQ 模型
from models.W_SimVQ_CNN import W_SimVQ_CNN  # CNN 版本 SimVQ
from models.W_InstructTimeVQ import W_InstructTimeVQ  # 时间序列 Instruct VQ
from models.ResidualVQ_tcn_enc import VQVAE as ResidualVQ  # 残差 VQ 模型
import torch.utils.data as Data  # PyTorch 数据工具

def seed_everything(seed):  # 固定所有随机种子
    random.seed(seed)  # 固定 random 模块的随机种子
    os.environ["PYTHONHASHSEED"] = str(seed)  # 固定 Python 哈希种子
    np.random.seed(seed)  # 固定 numpy 随机数
    torch.manual_seed(seed)  # 固定 CPU 上的 PyTorch 随机数
    torch.cuda.manual_seed(seed)  # 固定当前 GPU 的随机数
    torch.cuda.manual_seed_all(seed)  # 固定所有 GPU 的随机数
    torch.backends.cudnn.deterministic = True  # 确保 CUDA 算法可复现
    torch.backends.cudnn.benchmark = False  # 禁用 benchmark 提升复现性
    torch.backends.cudnn.enabled = True  # 启用 cuDNN

def get_data(flag):  # 加载数据集
    data_set, data_loader = data_provider(args, flag)  # 通过 data_provider 获取数据
    return data_set, data_loader  # 返回 dataset 与 dataloader

def main():  # 主函数
    seed_everything(seed=2024)  # 固定随机种子为 2024

    # train_dataset = Dataset(device=args.device, mode='train', args=args)  # 旧方法：构造训练数据
    # train_loader = Data.DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)  # 旧方法：训练 loader
    # test_dataset = Dataset(device=args.device, mode='test', args=args)  # 旧方法：构造测试集
    # test_loader = Data.DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)  # 旧方法：测试 loader
    train_data, train_loader = get_data(flag='train')  # 获取训练集
    vali_data, vali_loader = get_data(flag='val')  # 获取验证集
    test_data, test_loader = get_data(flag='test')  # 获取测试集

    print('dataset initial ends')  # 数据初始化完成提示

    # model = VQVAE(...)  # 旧方法：原版 VQ-VAE
    # model = W_VQVAE(args)  # 旧方法：另一个 VQVA 模型

    if args.vq_model == 'SimVQ':  # 若选择 SimVQ
        model = W_SimVQ(args)  # 初始化 SimVQ 模型
    elif args.vq_model == 'VanillaVQ':  # 原始 VQ 模型
        model = W_InstructTimeVQ(args)  # 初始化 InstructTimeVQ
    elif args.vq_model == 'SimVQ_CNN':  # CNN 版 SimVQ
        model = W_SimVQ_CNN(args)  # 初始化 CNN VQ 模型
    elif args.vq_model == 'ResidualVQ':  # 残差量化模型
        model = ResidualVQ(args)  # 初始化残差 VQ
    else:
        raise ValueError('Invalid VQ model name')  # 无效的模型名报错

    print('model initial ends')  # 模型初始化完成提示

    trainer = Trainer(args, model, train_loader, vali_loader, test_loader, verbose=True)  # 创建 Trainer 实例
    print('trainer initial ends')  # Trainer 初始化完成提示

    if args.is_training:  # 判断是否进行训练
        trainer.train()  # 执行训练流程

    trainer.test()  # 执行测试流程

if __name__ == '__main__':  # 入口判断
    main()  # 启动主程序

