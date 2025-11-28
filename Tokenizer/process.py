import os  # 操作系统路径、文件操作
import time  # 时间测量
import torch  # PyTorch 主库
import pickle  # 用于序列化存储权重/字典
import torch.nn as nn  # 神经网络组件

import numpy as np  # 数值计算

from tqdm import tqdm  # 进度条工具
from loss import MSE  # 自定义 MSE 损失类
from torch.optim.lr_scheduler import LambdaLR  # 学习率调度器

from utils.tools import plot_token_distribution, plot_token_distribution_with_stratify  # Token分布绘图
from utils.tools import plot_and_save_reconstruction, plot_PCA, statistic_freqs  # 重建/可视化工具

class Trainer():  # 训练器类
    def __init__(self, args, model, train_loader, vali_loader, test_loader, verbose=False):  # 初始化函数
        self.args = args  # 训练参数
        self.verbose = verbose  # 是否打印训练信息
        self.device = args.device  # 设备（CPU/GPU）
        self.print_process(self.device)  # 打印设备信息
        self.model = model.to(torch.device(self.device))  # 将模型移动到设备上

        self.train_loader = train_loader  # 训练集 DataLoader
        self.vali_loader = vali_loader  # 验证集 DataLoader
        self.test_loader = test_loader  # 测试集 DataLoader
        
        self.lr_decay = args.lr_decay_rate  # 学习率衰减因子
        self.lr_decay_steps = args.lr_decay_steps  # 衰减步数
        self.weight_decay = args.weight_decay  # 权重衰减
        self.model_name = self.model.get_name()  # 获取模型名称
        self.print_process(self.model_name)  # 打印模型名称

        self.cr = MSE(self.model)  # 定义损失函数类（包含 MSE + latent loss）

        self.num_epoch = args.num_epoch  # 最大训练轮次
        self.eval_per_steps = args.eval_per_steps  # 多少 step 评估一次
        self.save_path = args.save_path  # 模型保存路径
        
        if args.load_path is not None:  # 若指定加载路径
            self.load_path = args.load_path  # 使用指定路径
        else:
            self.load_path = args.save_path  # 否则使用保存路径
        
        if self.num_epoch:  # 如果需要训练
            self.result_file = open(self.save_path + '/result.txt', 'w')  # 创建结果文件
            self.result_file.close()  # 关闭文件

        self.step = 0  # 全局 step 计数
        self.best_metric = 1e9  # 初始化最优指标
        self.metric = 'reconst_mse'  # 选择评估指标（重构误差）


    def train(self):  # 总训练入口
        self.print_process('\n######### Start Training #########')  # 打印训练开始
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr, weight_decay=self.weight_decay)  # 优化器
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lambda step: self.lr_decay ** step, verbose=self.verbose)  # 学习率调度
        
        for epoch in range(self.num_epoch):  # 逐 epoch 训练
            loss_epoch, time_cost = self._train_one_epoch(epoch)  # 训练一个 epoch
            
            self.result_file = open(self.save_path + '/result.txt', 'a+')  # 打开结果文件
            self.print_process(
                'Basic Model train epoch:{0}, loss:{1:.6f}, training_time:{2:.6f}'.format(epoch + 1, loss_epoch, time_cost))  # 打印日志
            
            print('Basic Model train epoch:{0}, loss:{1:.6f}, training_time:{2:.6f}'.format(
                epoch + 1, loss_epoch, time_cost), file=self.result_file)  # 写入日志
            
            self.result_file.close()  # 关闭结果文件
        
        self.print_process(self.best_metric)  # 打印最佳指标
        
        return self.best_metric  # 返回最佳值

      
    def _eval(self, epoch):  # 评估模型
        metric_dict = {}  # 存储 train/valid/test 三组指标
        
        for key in ['train', 'valid', 'test']:  # 依次评估三个数据集
            if key == 'train': data_loader = self.train_loader  # 训练集
            elif key == 'valid': data_loader = self.vali_loader  # 验证集
            elif key == 'test': data_loader = self.test_loader  # 测试集
            
            _metric = self.eval_model_vqvae(data_loader)  # 获取评估指标
            metric_dict[key] = _metric  # 保存
            
            print(f'{key}: ', end='')  # 打印当前类型
            self.print_process(_metric)  # 输出指标
            
        print('\n')  # 换行
        
        metric = metric_dict['valid']  # 选择验证集指标
        self.result_file = open(self.save_path + '/result.txt', 'a+')  # 打开结果文件
        
        if not self.args.eval_per_epoch:  # 按 step 评估
            print('step{0}'.format(self.step), file=self.result_file)
        else:
            print('epoch{0}'.format(epoch), file=self.result_file)  # 按 epoch 评估
            
        print(metric, file=self.result_file)  # 保存指标
        self.result_file.close()  # 关闭结果文件
        
        print(self.metric, metric[self.metric], self.best_metric)  # 打印当前指标和历史最佳
        
        if metric[self.metric] < self.best_metric:  # 如果更优
            self.model.eval()  # 切换 eval
            torch.save(self.model.state_dict(), self.save_path + '/model.pkl')  # 保存模型权重
            self.result_file = open(self.save_path + '/result.txt', 'a+')  # 写结果文件
            
            if not self.args.eval_per_epoch:
                print('best model saved at step{0}'.format(self.step))  # 打印信息
            else:
                print('best model saved at epoch{0}'.format(epoch))  # 打印信息
            
            self.result_file.close()  # 关闭文件
            self.best_metric = metric[self.metric]  # 更新最佳值
            
        self.model.train()  # 切回 train 模式

        
    def _get_all_ids(self, data_loader):  # 从给定数据集里获取所有 token id，并统计重构损失
        mse = nn.MSELoss()  # 使用 MSE 作为重构误差
        total_recon_loss = 0.0  # 累积重构误差
        total_batches = 0  # 累积 batch 数

        # get test token distribution and calculate mse  # 获取 token 分布并计算 MSE
        ids = []  # 用于收集所有 token id
        with torch.no_grad():  # 评估阶段不计算梯度
            for idx, (batch_x, batch_y, _, _) in enumerate(data_loader):  # 遍历所有 batch
                batch_y = batch_y[:, -self.args.pred_len:, :].float().to(self.args.device)  # 取预测窗口部分并送到设备
                seqs_x = batch_x.float().to(self.args.device)  # 输入序列送到设备
                out_x, _, id_x = self.model(seqs_x, batch_y)  # 前向传播，得到重构和离散 id
                
                ids.append(id_x.flatten())  # 展平 token id 收集
                seqs_x = torch.cat([seqs_x, batch_y], dim=1)  # 拼接完整目标序列（历史+预测）
                recon_loss = mse(out_x, seqs_x)  # 计算重构误差

                total_recon_loss += recon_loss.item()  # 累加重构误差
                total_batches += 1  # 累加 batch 数
        
        ids = torch.cat(ids).cpu().numpy()  # 将所有 id 拼接并转为 numpy
        return ids  # 返回所有 token id

    def _train_one_epoch(self, epoch):  # 训练单个 epoch
        t0 = time.perf_counter()  # 记录起始时间
        self.model.train()  # 切换到训练模式
        tqdm_dataloader = tqdm(self.train_loader) if self.verbose else self.train_loader  # 若 verbose 则显示进度条
        loss_sum = 0  # 累积 loss
        
        for idx, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(tqdm_dataloader):  # 遍历训练集
            self.optimizer.zero_grad()  # 梯度清零
            batch_y = batch_y[:, -self.args.pred_len:, :].float().to(self.args.device)  # 取预测部分送到设备
            loss = self.cr.compute(batch_x.float().to(self.args.device), batch_y)  # 计算损失（重构+latent）
            loss_sum += loss.item()  # 累积损失

            loss.backward()  # 反向传播
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)  # 梯度裁剪，防止梯度爆炸
            self.optimizer.step()  # 更新参数

            self.step += 1  # 全局 step +1
            if self.step % self.lr_decay_steps == 0:  # 到达学习率衰减间隔
                self.scheduler.step()  # 执行学习率衰减
                
            if (self.step % self.eval_per_steps == 0) and (self.args.eval_per_epoch == False):  # 若按 step 评估
                self._eval(epoch)  # 调用评估
        
        if self.args.eval_per_epoch:  # 若按 epoch 评估
            self._eval(epoch)  # 评估当前 epoch
            
            # plot the distribution of train and test tokens  # 绘制训练/测试 token 分布
            train_ids = self._get_all_ids(self.train_loader)  # 获取训练集 token id
            test_ids = self._get_all_ids(self.test_loader)  # 获取测试集 token id
            
            plot_path = os.path.join(self.load_path, 'token_distribution_epoch{}'.format(epoch))  # 分布图保存路径
            
            plot_token_distribution_with_stratify(  # 绘制分层 token 分布
                train_ids,
                test_ids,
                save_dir=plot_path,
                max_token_num=self.args.n_embed,
                freq=True
            )

        return loss_sum / len(self.train_loader), time.perf_counter() - t0  # 返回平均 loss 和本 epoch 耗时

    def eval_model_vqvae(self, data_loader):  # 在给定数据集上评估 VQ-VAE
        self.model.eval()  # 切换到评估模式
        tqdm_data_loader = tqdm(data_loader) if self.verbose else data_loader  # 决定是否显示进度条
        metrics = {'reconst_mse': 0, 'latent_mse': 0}  # 初始化指标字典

        with torch.no_grad():  # 不计算梯度
            for idx, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(tqdm_data_loader):  # 遍历数据
                batch_y = batch_y[:, -self.args.pred_len:, :].float().to(self.args.device)  # 取预测部分
                loss_dict = self.cr.compute(batch_x.float().to(self.args.device), batch_y, details=True)  # 详细计算损失
                metrics['reconst_mse'] += loss_dict['recon_loss']  # 累加重构误差
                metrics['latent_mse'] += loss_dict['latent_loss']  # 累加 latent 误差
                
        metrics['reconst_mse'] /= len(data_loader)  # 取重构误差平均
        metrics['latent_mse'] /= len(data_loader)  # 取 latent 误差平均
        
        return metrics  # 返回指标字典
    
    def print_process(self, *x):  # 控制打印输出
        if self.verbose:  # 仅在 verbose=True 时打印
            print(*x)  # 打印参数

    def test(self):  # 测试入口
        self.print_process('\n######### Start Testing #########')  # 打印测试开始
        
        state_dict = torch.load(os.path.join(self.load_path, 'model.pkl'), map_location='cpu')  # 加载保存的模型参数
        self.model.load_state_dict(state_dict)  # 将参数加载到模型
        self.model.eval()  # 切换 eval 模式
        
        # plot the low-dimensional representation of the code book  # 预留：绘制 codebook 的低维可视化
        
        # get reconst mse  # 计算重构 MSE
        mse = nn.MSELoss()  # 定义 MSE 损失
        total_recon_loss = 0.0  # 累积重构误差（未真正使用最终结果）
        total_batches = 0  # 累积 batch 数
        mae = nn.L1Loss()  # 定义 MAE 损失

        # get train token distribution  # 计算训练集 token 分布
        train_ids = []  # 收集训练集离散 token id
        with torch.no_grad():  # 测试阶段不反传
            for idx, (batch_x, batch_y, _, _) in enumerate(self.train_loader):  # 遍历训练集
                batch_y = batch_y[:, -self.args.pred_len:, :].float().to(self.args.device)  # 取预测部分
                seqs_x = batch_x.float().to(self.args.device)  # 输入送到设备
                _, _, id_x = self.model(seqs_x, batch_y)  # 前向传播，获取 token id
                
                train_ids.append(id_x.flatten())  # 展平并存储
                
        if False:  # 该分支目前不会执行（调试/备用功能）
            train_ids = torch.cat(train_ids).cpu().numpy()  # 拼接训练 token id
            train_tokens = train_ids.flatten()  # 展平
            train_uni_elements, train_cnts_elements = np.unique(train_tokens, return_counts=True)  # 统计唯一 token 及频次
            
            statistic_freqs(train_tokens)  # 打印频率统计
            
            total_nums = len(train_tokens)  # token 总数
            statis = 0  # 阈值比例（当前为 0）
            board = total_nums * (statis / 100.)  # 频次阈值
            
            elect_index = np.where(train_cnts_elements >= board)  # 选出频次超过阈值的索引
            elect_ids = train_uni_elements[elect_index]  # 对应 token id
            self.model.elect_codebook(elect_ids, statis)  # 在模型中筛选 codebook
            
        # get test token distribution and calculate mse  # 计算测试集 token 分布和误差
        test_ids = []  # 收集测试集 token id
        total_mse_input = 0.0  # 输入部分 MSE 累积
        total_mae_input = 0.0  # 输入部分 MAE 累积
        total_mse_output = 0.0  # 输出部分 MSE 累积
        total_mae_output = 0.0  # 输出部分 MAE 累积

        total_batches = 0  # 重置 batch 计数

        with torch.no_grad():  # 测试阶段不反传
            for idx, (batch_x, batch_y, _, _) in enumerate(self.test_loader):  # 遍历测试集
                batch_y = batch_y[:, -self.args.pred_len:, :].float().to(self.args.device)  # 取预测部分
                seqs_x = batch_x.float().to(self.args.device)  # 输入送到设备
                out_x, _, id_x = self.model(seqs_x, batch_y)  # 前向传播，得到重构和 token id
                
                test_ids.append(id_x.flatten())  # 收集 token id
                
                # 拼接 ground truth  # 构造完整目标序列
                full_target = torch.cat([seqs_x, batch_y], dim=1)  # [历史 + 预测]
                # 对应输出长度  # 确定输入长度
                input_len = seqs_x.size(1)  # 输入序列长度
                total_len = full_target.size(1)  # 总长度（未直接使用）

                # 分别计算 MSE/MAE  # 输入部分误差
                mse_input = mse(out_x[:, :input_len, :], full_target[:, :input_len, :])  # 输入区间 MSE
                mae_input = mae(out_x[:, :input_len, :], full_target[:, :input_len, :])  # 输入区间 MAE

                mse_output = mse(out_x[:, input_len:, :], full_target[:, input_len:, :])  # 输出区间 MSE
                mae_output = mae(out_x[:, input_len:, :], full_target[:, input_len:, :])  # 输出区间 MAE

                # 累加  # 累加误差
                total_mse_input += mse_input.item()  # 累积输入部分 MSE
                total_mae_input += mae_input.item()  # 累积输入部分 MAE
                total_mse_output += mse_output.item()  # 累积输出部分 MSE
                total_mae_output += mae_output.item()  # 累积输出部分 MAE

                total_batches += 1  # 累加 batch 数

        # 平均值  # 计算平均误差
        avg_mse_input = total_mse_input / total_batches  # 输入部分平均 MSE
        avg_mae_input = total_mae_input / total_batches  # 输入部分平均 MAE
        avg_mse_output = total_mse_output / total_batches  # 输出部分平均 MSE
        avg_mae_output = total_mae_output / total_batches  # 输出部分平均 MAE

        print('[Input Part]  MSE: {:.6f}, MAE: {:.6f}'.format(avg_mse_input, avg_mae_input))  # 打印输入误差
        print('[Output Part] MSE: {:.6f}, MAE: {:.6f}'.format(avg_mse_output, avg_mae_output))  # 打印输出误差
                        
        # plot the distribution of train and test tokens  # 绘制 train/test token 分布
        
        plot_path = os.path.join(self.load_path, 'token_distribution')  # token 分布图保存目录
        
        test_ids = torch.cat(test_ids).cpu().numpy()  # 拼接测试 token id
        
        # print the statistics of the token distribution  # 打印 token 分布统计
        # 
        
        codebook_plot_path = os.path.join(self.load_path, 'codebook_with_used_freqs.png')  # codebook 可视化图路径
        # codebook = self.model.get_codebook_weight()  # 获取 codebook 权重（已注释）
        # plot_PCA(train_ids, codebook, codebook_plot_path, max_token_num=self.args.n_embed)  # PCA 可视化 codebook
        
        # exit(0)  # 调试用提前退出
        train_ids = self._get_all_ids(self.train_loader)  # 重新获取训练集所有 token id
        
        statistic_freqs(train_ids.flatten())  # 打印训练集 token 频率统计
        # test_ids = self._get_all_ids(self.test_loader)  # 可选：重新获取测试 token
        plot_token_distribution_with_stratify(  # 绘制 train/test token 分布
            train_ids,
            test_ids,
            save_dir=plot_path,
            max_token_num=self.args.n_embed
        )
        
        # count the frequence of train tokens  # 统计训练 token 的频率
        freq = np.bincount(train_ids, minlength=self.args.n_embed)  # 每个 token 出现次数
        fixed_freq = np.where(freq > 0, freq, 1e-7)  # 将 0 次出现替换为极小值避免除零
        
        print(len(freq))  # 打印 token 频率数组长度
        
        n_classes = len(set(train_ids))  # 实际使用到的 token 个数
        weight = len(train_ids) / (n_classes * fixed_freq)  # 按频率反比计算权重
        scale = 5.0  # 控制最终平均权重尺度（可调）
        # weight = weight / np.mean(weight)  # 归一化为均值为 1（已注释）
        weight = weight * scale  # 放大权重
        
        mask = freq > 0  # 标记哪些 token 被使用
        train_tokens = train_ids.flatten()  # 展平训练 token
        train_uni_elements, train_cnts_elements = \
            np.unique(train_tokens, return_counts=True)  # 统计唯一 token 及其计数
            
        weight_dict = {  # 保存权重相关信息
            'weight': weight,  # 每个 token 的权重
            'mask': mask,  # 使用掩码
            'train_uni_elements': train_uni_elements,  # 训练中出现的 token
            'train_cnts_elements': train_cnts_elements,  # 对应出现次数
            'total_nums': len(train_ids)  # token 总数
        }
        
        print("Successfully save weight.pkl")  # 打印提示信息
        
        save_w_path = os.path.join(self.load_path, 'weight.pkl')  # 权重保存路径
        pickle.dump(weight_dict, open(save_w_path, 'wb'))  # 将权重字典保存到 pkl 文件
        plot_path = os.path.join(self.load_path, 'reconstruction')  # 重建图像保存目录
        os.makedirs(plot_path, exist_ok=True)  # 若目录不存在则创建
        
        plot_and_save_reconstruction(self.model, self.test_loader, plot_path)  # 绘制并保存重建结果
        print("Images have been saved.")  # 提示重建图保存完成
        
        exit(0)  # 直接退出程序（后续代码不会执行）
        
        # Just calculate the minimun weight from existing tokens  # 下面是预留调试代码
        
        # print((freq > 0).shape)  # 查看 mask 形状
        
        # real_min_weight = np.min(weight, where=(freq > 0), initial=np.inf)  # 有效 token 的最小权重
        # max_weight = real_min_weight * 20  # 最大权重上限
        # weight = np.clip(weight, a_min=None, a_max=max_weight)  # 限制权重最大值
        
        # print("#### Weight Statistics: ####")  # 打印权重统计
        # print(weight.shape, max(weight), min(weight)) # min:0.11 max: 647  # 权重范围
        
        print("#### Token Distribution Analysis ####")  # 打印 token 分布分析
        print("Training Set: Used token is {}, Total token is {}".format(len(set(train_ids)), self.args.n_embed))  # 训练集中使用的 token 数
        print("Test Set: Used token is {}, Total token is {}".format(len(set(test_ids)), self.args.n_embed))  # 测试集中使用的 token 数

        avg_recon_loss = total_recon_loss / total_batches  # 平均重构误差（当前逻辑中 total_recon_loss 未更新）
        
        print('reconstruct loss(mse) on test dataset: {:.6f}\n'.format(avg_recon_loss))  # 打印重构 MSE


        

            
