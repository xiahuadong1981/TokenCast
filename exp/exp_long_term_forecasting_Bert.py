import os                                              # 文件与路径操作
import time                                            # 计时与时间工具
import torch                                           # PyTorch 主库
import torch.nn as nn                                  # 神经网络模块
import warnings                                        # 警告控制
import numpy as np                                     # 数值计算库
from tqdm import tqdm                                  # 进度条显示
from torch import optim                                # 优化器
import sys                                             # 与解释器交互
import pickle                                          # 序列化/反序列化
from data_provider.data_factory import data_provider   # 数据集/Loader 构建函数
from exp.exp_basic import Exp_Basic                    # 实验基础类（加速、DDP 等）
from utils.tools import EarlyStopping, adjust_learning_rate, visual, clever_format, plot_token_distribution_with_stratify, get_cosine_schedule_with_warmup  # 工具函数与调度器
from utils.metrics import metric, token_metric         # 评价指标计算
from models.Model4F import Model                       # 预测模型（未直接使用）
from ecg_tokenizer.model_v1 import W_SimVQ             # SimVQ VQ-VAE 版本
from ecg_tokenizer.Sim_VQ_CNN import W_SimVQ_CNN       # CNN 版 SimVQ
from ecg_tokenizer.W_SimVQ_CNN_double import W_SimVQ_CNN_double         # 双通道 CNN VQ
from ecg_tokenizer.W_SimVQ_CNN_double_token import W_SimVQ_CNN_double_token  # 双 token CNN VQ
from ecg_tokenizer.ResidualVQ_tcn_enc import VQVAE as ResidualVQ        # 残差 VQ-VAE
from ecg_tokenizer.TimeSeriesPromptGenerator import TimeSeriesPromptGenerator  # 时间序列 Prompt 生成器（未使用）
from torch.nn.utils.rnn import pad_sequence           # 序列 padding 工具（未使用）
from peft import PeftModel                            # PEFT/LoRA 适配器加载
warnings.filterwarnings('ignore')                     # 忽略所有警告输出


class Exp_Long_Term_Forecast_Bert_v4(Exp_Basic):      # 长期预测实验类，继承基础实验类
    def __init__(self, args):                         # 构造函数
        super().__init__(args)                        # 调用父类初始化（Accelerate、模型字典等）

    def _build_model(self):                           # 构建模型与 VQ 模块
        if self.args.VQ_type == 'SimVQ':              # 选择 SimVQ
            vq_model = W_SimVQ(self.args)             # 实例化 VQ 模型
        elif self.args.VQ_type == 'SimVQ_CNN':        # 选择 CNN 版 SimVQ
            vq_model = W_SimVQ_CNN(self.args)
        elif self.args.VQ_type == 'W_SimVQ_CNN_double':  # 双通道 CNN VQ
            vq_model = W_SimVQ_CNN_double(self.args)
        elif self.args.VQ_type == 'SimVQ_CNN_double_token':  # 双 token VQ
            vq_model = W_SimVQ_CNN_double_token(self.args)
        elif self.args.VQ_type == 'ResidualVQ':       # 残差 VQ-VAE
            vq_model = ResidualVQ(self.args)
        else:                                         # 类型不支持
            raise ValueError(f"VQ type@ {self.args.VQ_type} not supported!")  # 抛出错误

        vqvae_state_dict = torch.load(os.path.join(self.args.vqvae_model_path, 'model.pkl'), map_location="cpu")  # 加载 VQ-VAE 权重
        vq_model.load_state_dict(vqvae_state_dict, strict=False)  # 加载参数（宽松匹配）
        weight_dict = pickle.load(open(os.path.join(self.args.vqvae_model_path, 'weight.pkl'), 'rb'))  # 加载权重字典（如 token 权重）
        self.args.elected_n_embed = self.args.n_embed  # 记录使用的词表大小

        model = self.model_dict[self.args.model].Model(self.args).float()  # 通过 model_dict 创建主模型并转 float
        if not self.args.zero and self.args.pretrained_model:              # 如果不是从零训练且提供了预训练模型
            if self.accelerator.is_local_main_process:                     # 只在主进程打印
                print(f"Loading pretrained model from {self.args.pretrained_model}")
            state_dict = torch.load(self.args.pretrained_model, map_location='cpu')  # 加载预训练权重
            model.load_state_dict(state_dict, strict=False)                # 宽松加载
            if self.accelerator.is_local_main_process:
                print('Model loaded successfully.')

        weight, mask = weight_dict['weight'], weight_dict['mask']          # 分类权重和掩码
        real_min_weight = np.min(weight, where=(mask == True), initial=np.inf)  # 仅在 mask==True 范围内取最小权重
        max_weight = real_min_weight * self.args.max_mpls                  # 设置最大权重阈值
        classification_weight = None                                       # 初始化分类权重
        if self.args.max_mpls > 0:                                         # 若启用阈值裁剪
            weight = np.clip(weight, a_min=None, a_max=max_weight)         # 裁剪上限
            classification_weight = torch.tensor(weight, dtype=torch.float)  # 转为 tensor
            if self.accelerator.is_local_main_process:
                print(f'Classification weight loaded: shape {classification_weight.shape}, min {real_min_weight}, max {max_weight}')

        vq_model = vq_model.to(self.device)                                # 将 VQ 模型移动到当前设备
        for p in vq_model.parameters():                                    # 冻结 VQ 参数
            p.requires_grad = False
        self.vq_model = vq_model                                           # 保存到实例

        # model = self.accelerator.prepare(model)                          # 若需要可用 accelerate 包装模型
        self.model = model                                                 # 保存主模型引用

        return model, vq_model, classification_weight                      # 返回模型、VQ 模型和分类权重
    
    
    import torch                                                           # 冗余导入（可以删）

    def build_input_and_label(self, batch_x, batch_y, start_date, end_date, is_train=True):  # 构建大模型输入与标签
        """
        构建符合详细格式的输入和标签，动态接收起止日期。
        """
        # 元数据部分仍然是硬编码的
        series_metadata = {                                                # 时间序列元信息
            "source": "FRED-MD (Federal Reserve Economic Data - Monthly)", # 数据来源描述
            "name": "Industrial Production Index",                         # 指标名称
            "id": "INDPRO",                                                # 指标 ID
            "category": "Output and Income",                               # 类别
            "transformation": "Logarithmic first difference",              # 变换方式
            "semantic_meaning": "The provided tokens represent the monthly percentage growth rate of industrial production."  # 语义说明
        }

        # ... (从 VQ-VAE 获取 tokens 到 定义 tokenizer 和特殊标记的部分，与之前完全相同)
        # 1. 从 VQ-VAE 获取离散 tokens
        tokens = self.vq_model.get_code(batch_x, batch_y)                  # 利用 VQ 模型将序列编码为 token

        # 自动获取 token 数量
        input_token_count = self.args.seq_len // self.args.wave_length     # 输入 token 数（按窗口长度划分）
        output_token_count = self.args.pred_len // self.args.wave_length   # 输出 token 数

        output_tokens = tokens[:, -output_token_count:]                    # 预测部分 token
        input_tokens = tokens[:, :-output_token_count]                     # 历史部分 token

        # 添加词表偏移量
        input_tokens = input_tokens + self.model.original_len              # 时间序列 token 映射到扩展词表区间
        output_tokens = output_tokens + self.model.original_len

        # 准备 tokenizer 和特殊标记
        tokenizer = self.model.text_tokenizer                              # 文本 tokenizer
        device = batch_x.device                                            # 当前设备
        batch_size = input_tokens.size(0)                                  # batch 大小

        ts_start_id = tokenizer.convert_tokens_to_ids("<TS_START>")        # 时间序列起始标记 ID
        ts_end_id = tokenizer.convert_tokens_to_ids("<TS_END>")            # 时间序列结束标记 ID
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")          # 对话结束标记 ID
        
        ts_start = torch.full((batch_size, 1), ts_start_id, dtype=torch.long, device=device)  # <TS_START> 批量张量
        ts_end = torch.full((batch_size, 1), ts_end_id, dtype=torch.long, device=device)      # <TS_END>
        im_end = torch.full((batch_size, 1), im_end_id, dtype=torch.long, device=device)      # <|im_end|>

        input_tokens_with_markers = torch.cat([ts_start, input_tokens, ts_end], dim=1)        # 包装输入 token 序列
        if is_train:                                                    # 训练模式才构建输出 token 序列
            output_tokens_with_markers = torch.cat([ts_start, output_tokens, ts_end], dim=1)

        def encode_and_repeat(text):                                   # 文本编码并按 batch 复制
            ids = tokenizer(text, return_tensors="pt", padding=False, add_special_tokens=False)["input_ids"].to(device)  # 编码为 ID
            return ids.repeat(batch_size, 1) if ids.shape[0] == 1 else ids   # 若只有一行则 repeat 到 batch

        # 2. 构建 System Prompt
        system_prompt_text = (                                        # system 角色提示词
            "<|im_start|>system\n"
            "You are an expert econometrician and time series forecaster. Your task is to analyze the provided "
            "macroeconomic data and context to produce the most likely forecast. Pay close attention to all "
            "metadata, especially the transformation and statistical properties.\n<|im_end|>\n"
        )
        system_ids = encode_and_repeat(system_prompt_text)             # 编码 system prompt

        # 3. 构建 User Prompt, 现在包含动态日期
        input_mean = batch_x.mean().item()                             # 输入均值
        input_std = batch_x.std().item()                               # 输入标准差

        user_prompt_prefix_text = f"""<|im_start|>user
    Your primary task is to perform time series forecasting. You must return only the predicted time series tokens, enclosed strictly between <TS_START> and <TS_END> markers.

    ### Time Series Metadata ###
    - **Source**: {series_metadata['source']}
    - **Series Name**: {series_metadata['name']}
    - **Series ID**: {series_metadata['id']}
    - **Category**: {series_metadata['category']}
    - **Transformation Applied**: {series_metadata['transformation']}
    - **Semantic Meaning**: {series_metadata['semantic_meaning']}

    ### Statistical Properties of the Input Data ###
    - **Input Window Start Date**: {start_date}
    - **Input Window End Date**: {end_date}
    - **Input Window Mean (of transformed data)**: {input_mean:.4f}
    - **Input Window Std. Dev. (of transformed data)**: {input_std:.4f}

    Based on the metadata, statistical properties, economic context, and the historical tokens provided below, predict the next {output_token_count} tokens.
    """                                                              # user 提示词（含元数据与统计量）
        user_prompt_prefix_ids = encode_and_repeat(user_prompt_prefix_text)  # 编码 user 提示词前缀

        user_prompt_suffix_text = f"\nThe tokens capture historical trends in the {series_metadata['id']} growth rate.\n<|im_end|>\n"  # user 提示尾部
        user_prompt_suffix_ids = encode_and_repeat(user_prompt_suffix_text)  # 编码尾部

        # 4. 构建 Assistant Prompt 的起始部分
        assistant_start_ids = encode_and_repeat("<|im_start|>assistant\n")   # assistant 起始标记

        # 5. 组合最终的输入和标签 (这部分逻辑不变)
        if is_train:                                                         # 训练模式：构建带 label 的输入
            input_ids = torch.cat([
                system_ids, user_prompt_prefix_ids, input_tokens_with_markers,
                user_prompt_suffix_ids, assistant_start_ids, output_tokens_with_markers, im_end
            ], dim=1)                                                        # 拼接完整对话+TS token
            labels = torch.full_like(input_ids, -100)                        # 初始化 label 全为 ignore_index
            start_of_label = (                                               # 预测区间起点位置
                system_ids.shape[1] + user_prompt_prefix_ids.shape[1] +
                input_tokens_with_markers.shape[1] + user_prompt_suffix_ids.shape[1] +
                assistant_start_ids.shape[1]
            )
            end_of_label = start_of_label + output_tokens_with_markers.shape[1] + im_end.shape[1]  # 预测区间终点
            labels[:, start_of_label:end_of_label] = input_ids[:, start_of_label:end_of_label]     # 仅预测区间参与 loss
            return input_ids, labels                                          # 返回输入与标签
        else:                                                                 # 推理模式
            input_ids = torch.cat([
                system_ids, user_prompt_prefix_ids, input_tokens_with_markers,
                user_prompt_suffix_ids, assistant_start_ids
            ], dim=1)                                                         # 不包含输出 token
            output_tokens_original = output_tokens - self.model.original_len  # 还原 VQ token 原索引
            input_tokens_original = input_tokens - self.model.original_len
            return input_ids, output_tokens_original, input_tokens_original, 0, output_tokens_original.shape[1]  # 返回及 token 长度信息

    def _print_trainable_parameters(self, model):                             # 打印可训练参数统计信息
        """Print statistics about model parameters, including trainable vs frozen counts, memory, and device usage."""
        freeze_params = 0                                                    # 冻结参数数量
        trainable_params = 0                                                 # 可训练参数数量
        trainable_param_list = []                                            # 可训练参数列表
        total_size_bytes = 0                                                 # 可训练参数字节数
        device_counter = {}                                                  # 不同设备上张量数

        # Collect parameter statistics
        for name, param in model.named_parameters():                         # 遍历所有参数
            param_device = str(param.device)                                 # 参数所在设备
            param_dtype = param.dtype                                        # 参数数据类型
            param_size = param.nelement() * param.element_size()             # 参数字节大小

            if param.requires_grad:                                          # 可训练参数
                trainable_params += param.nelement()
                total_size_bytes += param_size
                trainable_param_list.append((name, param_device, param_dtype))
                device_counter[param_device] = device_counter.get(param_device, 0) + 1
            else:                                                            # 冻结参数
                freeze_params += param.nelement()

        total_params = trainable_params + freeze_params                      # 总参数量

        def format_size(num_bytes):                                          # 字节数人类可读格式
            for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
                if num_bytes < 1024.0:
                    return f"{num_bytes:.2f}{unit}"
                num_bytes /= 1024.0
            return f"{num_bytes:.2f}PB"

        if self.accelerator.is_local_main_process:                           # 仅主进程打印
            print('=' * 60)
            print('Model Parameter Statistics:')
            print(f'Trainable parameters: {trainable_params:,}')
            print(f'Frozen parameters:    {freeze_params:,}')
            print(f'Total parameters:     {total_params:,}')
            print(f'Trainable ratio:      {(trainable_params / total_params) * 100:.2f}%')
            print(f'Estimated trainable parameter size: {format_size(total_size_bytes)}')

            print('\nDevice distribution of trainable parameters:')
            for device, count in device_counter.items():
                print(f'- {device}: {count} tensors')

            print('\nTrainable parameter names (first 10):')
            for name, device, dtype in trainable_param_list[:10]:
                print(f'- {name} (device: {device}, dtype: {str(dtype)})')

            if len(trainable_param_list) > 10:
                print(f'... and {len(trainable_param_list) - 10} more parameters')
            print('=' * 60)

    def _get_data(self, flag, data=None):                                    # 获取数据集与 DataLoader
        """Get data loader for training, validation, or testing."""
        data_set, data_loader = data_provider(self.args, flag, data)         # 调用 data_provider
        return data_set, data_loader

    def _select_optimizer(self, lr):                                         # 构建优化器
        """Create an optimizer for the model."""
        return optim.Adam(self.model.parameters(), lr=lr, weight_decay=self.args.weight_decay)  # Adam 优化器

    def _select_criterion(self):                                             # 选择损失函数
        """Create a loss function."""
        return nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1)   # 交叉熵 + label smoothing
    
    def pretrain(self, setting):                                             # 预训练过程（与 train 类似）
        """Train the model with distributed multi-GPU support using HuggingFace Accelerate."""
        loaders = {k: self._get_data(flag=k) for k in ['train', 'val', 'test']}  # 分别获取 train/val/test 数据
        train_data, train_loader = loaders['train']
        vali_data, vali_loader = loaders['val']
        test_data, test_loader = loaders['test']
        criterion = self._select_criterion()                                 # 损失函数
        self._print_trainable_parameters(self.model)                         # 打印参数统计
        path = os.path.join(self.args.checkpoints, setting)                  # checkpoint 目录
        if self.accelerator.is_local_main_process:
            os.makedirs(path, exist_ok=True)                                 # 创建目录

        model_optim = self._select_optimizer(lr=self.args.learning_rate)     # 优化器
        train_steps = len(train_loader)                                      # 每个 epoch 的迭代次数
        time_now = time.time()                                               # 当前时间
        scheduler = get_cosine_schedule_with_warmup(                         # 余弦 + warmup 学习率调度
            model_optim,
            warmup_epochs=getattr(self.args, 'warmup_epochs', 2) * train_steps,
            total_epochs=self.args.train_epochs * train_steps
        )
        def test_callback():                                                 # EarlyStopping 时调用的测试回调
            mse, mae = self.test(setting, test=1, save_root=self.args.checkpoints)
            self.accelerator.print(f"[✅] Test after saving best model | MSE: {mse:.3f}, MAE: {mae:.3f}")

        early_stopping = EarlyStopping(accelerator=self.accelerator, patience=self.args.patience,test_fn=test_callback)  # 提前停止器
        best_model_path = os.path.join(path, 'checkpoint.pth')               # 最佳模型路径
        
        self.accelerator.init_trackers(setting)                              # 初始化日志跟踪器
        train_loader, vali_loader, test_loader = self.accelerator.prepare(train_loader, vali_loader, test_loader)  # 包装 DataLoader
        self.model, model_optim, scheduler = self.accelerator.prepare(self.model, model_optim, scheduler)           # 包装模型与优化器
        accumulation_steps = getattr(self.args, 'accumulation_steps', 1)     # 梯度累积步数
        iter_verbose = 100                                                   # 多少 iter 打一次日志

        for epoch in range(self.args.train_epochs):                          # 训练多个 epoch
            self.model.train()
            train_loss, iter_count = [], 0
            epoch_time = time.time()

            all_ts_correct = []                                              # TS token 正确数列表
            all_ts_total = []                                                # TS token 总数列表
            all_text_correct = []                                            # 文本 token 正确数
            all_text_total = []                                              # 文本 token 总数
            all_correct = []                                                 # 总正确数
            all_total = []                                                   # 总 token 数

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, start_dates, end_dates) in enumerate(train_loader):  # 迭代训练集
                iter_count += 1
                batch_x, batch_y = batch_x.float(), batch_y.float()         # 转 float
                batch_y = batch_y[:, -self.args.pred_len:, :]               # 只取预测部分

                ts_ids, labels = self.build_input_and_label(
                    batch_x,
                    batch_y,
                    start_date=start_dates[0],                               # 使用本 batch 第一个样本的起始日期
                    end_date=end_dates[0],                                   # 使用终止日期
                    is_train=True
                )
                ts_ids = ts_ids.to(self.device)
                labels = labels.to(self.device)

                inputs = {
                    'text_ids': None,
                    'ts_ids': ts_ids,
                    'labels': ts_ids                                         # 注意：这里 labels 实际传 ts_ids 给模型
                }
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():                          # 混合精度
                        outputs = self.model(inputs)
                        loss = outputs.loss
                else:
                    outputs = self.model(inputs)
                    loss = outputs.loss

                loss = loss.mean() / accumulation_steps                     # 除以累积步数
                train_loss.append(loss.item() * accumulation_steps)         # 保存原始 loss

                if hasattr(outputs, 'logits'):                              # 若模型返回 logits
                    logits = outputs.logits
                    preds = torch.argmax(logits, dim=-1)                    # 取 argmax 预测

                    # === Shifted Accuracy Calculation ===
                    shifted_preds = preds[:, :-1]                            # 右移预测
                    shifted_labels = labels[:, 1:]                           # 左移标签

                    valid_mask = shifted_labels != -100                      # 有效标签 mask
                    ts_mask = valid_mask & (shifted_labels >= self.model.original_len)  # TS 区域 mask
                    text_mask = valid_mask & (shifted_labels < self.model.original_len) # 文本区域 mask

                    ts_correct = ((shifted_preds == shifted_labels) & ts_mask).int()    # TS 正确
                    text_correct = ((shifted_preds == shifted_labels) & text_mask).int()# 文本正确
                    total_correct = ((shifted_preds == shifted_labels) & valid_mask).int() # 总正确

                    ts_total = ts_mask.int()                                 # TS 总数 mask
                    text_total = text_mask.int()                             # 文本总数 mask
                    total_token = valid_mask.int()                           # 总 token 数 mask

                    ts_correct = self.accelerator.gather_for_metrics(ts_correct).sum()   # 聚合多卡统计
                    ts_total = self.accelerator.gather_for_metrics(ts_total).sum()
                    text_correct = self.accelerator.gather_for_metrics(text_correct).sum()
                    text_total = self.accelerator.gather_for_metrics(text_total).sum()
                    total_correct = self.accelerator.gather_for_metrics(total_correct).sum()
                    total_token = self.accelerator.gather_for_metrics(total_token).sum()

                    all_ts_correct.append(ts_correct)
                    all_ts_total.append(ts_total)
                    all_text_correct.append(text_correct)
                    all_text_total.append(text_total)
                    all_correct.append(total_correct)
                    all_total.append(total_token)
                
                # if i ==2:
                #     break
                self.accelerator.backward(loss)                             # 反向传播
                if (i + 1) % accumulation_steps == 0:                       # 梯度累积到步数后更新
                    model_optim.step()
                    scheduler.step()
                    model_optim.zero_grad()
                if (i + 1) % iter_verbose == 0:                             # 打印训练日志
                    ts_acc = torch.stack(all_ts_correct).sum().item() / max(torch.stack(all_ts_total).sum().item(), 1)
                    text_acc = torch.stack(all_text_correct).sum().item() / max(torch.stack(all_text_total).sum().item(), 1)
                    total_acc = torch.stack(all_correct).sum().item() / max(torch.stack(all_total).sum().item(), 1)
                    self.accelerator.print(
                        f"\titers: {i + 1}, epoch: {epoch + 1} | loss: {loss.item() * accumulation_steps:.7f} | "
                        f"text_acc: {text_acc * 100:.2f}% | ts_acc: {ts_acc * 100:.2f}% | total_acc: {total_acc * 100:.2f}%")
                    speed = (time.time() - time_now) / iter_count           # 平均单 iter 时间
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)  # 估计剩余时间
                    self.accelerator.print(f"\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s")
                    iter_count = 0
                    time_now = time.time()
                    step = epoch * train_steps + i                          # 当前全局 step
                    self.accelerator.log(
                        {"train_loss": loss.item() * accumulation_steps,
                        "learning_rate": scheduler.get_last_lr()[0]},
                        step=step
                    )
            train_loss = np.average(train_loss)                             # epoch 平均 loss
            ts_acc = torch.stack(all_ts_correct).sum().item() / max(torch.stack(all_ts_total).sum().item(), 1)
            text_acc = torch.stack(all_text_correct).sum().item() / max(torch.stack(all_text_total).sum().item(), 1)
            total_acc = torch.stack(all_correct).sum().item() / max(torch.stack(all_total).sum().item(), 1)
            self.accelerator.print(f"Epoch {epoch + 1} completed in {time.time() - epoch_time:.2f}s")  # 打印 epoch 耗时

            vali_loss = self.vali(vali_data, vali_loader, criterion)        # 验证集 loss
            test_loss = self.vali(test_data, test_loader, criterion)        # 测试集 loss（做监控）

            self.accelerator.print(
                f"Epoch: {epoch + 1}, Steps: {train_steps} | Train Loss: {train_loss:.7f} "
                f"Vali Loss: {vali_loss:.7f} Test Loss: {test_loss:.7f} "
                f"Text Acc: {text_acc * 100:.2f}% TS Acc: {ts_acc * 100:.2f}% Total Acc: {total_acc * 100:.2f}%"
            )
            self.accelerator.log(
                {"epoch": epoch,
                "train_loss_avg": train_loss,
                "val_loss": vali_loss,
                "test_loss": test_loss,
                "train_ts_acc": ts_acc,
                "train_text_acc": text_acc,
                "train_total_acc": total_acc},
                step=epoch
            )
            early_stopping(vali_loss, self.model, path)                     # 调用 EarlyStopping
            if early_stopping.early_stop:                                   # 如果触发提前停止
                self.accelerator.print("Early stopping")
                break
        if self.accelerator.is_local_main_process and os.path.exists(best_model_path):  # 若存在最佳模型
            model_state = torch.load(best_model_path, map_location=self.device)
            unwrapped_model = self.accelerator.unwrap_model(self.model)     # 解包模型
            unwrapped_model.load_state_dict(model_state)                    # 加载最佳权重
        self.accelerator.end_training()                                     # 结束训练（清理 tracker 等）
        return self.model                                                   # 返回训练好的模型

    def vali(self, vali_data, vali_loader, criterion):                      # 验证函数
        """Run validation and return average loss (print total/text/ts token accuracy)."""
        self.model.eval()                                                   # eval 模式
        total_loss = []                                                     # loss 列表

        all_ts_correct, all_ts_total = [], []                               # TS 统计
        all_text_correct, all_text_total = [], []                           # 文本统计
        all_correct, all_total = [], []                                     # 总体统计

        with torch.no_grad():                                               # 不计算梯度
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, start_dates, end_dates) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_y = batch_y[:, -self.args.pred_len:, :]              # 取预测部分

                ts_ids, labels = self.build_input_and_label(
                    batch_x,
                    batch_y,
                    start_date=start_dates[0],                              # 使用第一个样本的日期
                    end_date=end_dates[0],
                    is_train=True
                )
                ts_ids = ts_ids.to(self.device)
                labels = labels.to(self.device)

                inputs = {
                    'text_ids': None,
                    'ts_ids': ts_ids,
                    'labels': labels
                }

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(inputs)
                else:
                    outputs = self.model(inputs)

                loss = outputs.loss.mean()                                 # 平均 loss
                total_loss.append(loss.detach().cpu().item())
                # if i==2:
                #     break

                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                    preds = torch.argmax(logits, dim=-1)

                    # Shift predictions and labels
                    shifted_preds = preds[:, :-1].contiguous()             # 右移预测
                    shifted_labels = labels[:, 1:].contiguous()            # 左移标签

                    valid_mask = shifted_labels != -100                    # 有效位置
                    ts_mask = valid_mask & (shifted_labels >= self.model.original_len)   # TS mask
                    text_mask = valid_mask & (shifted_labels < self.model.original_len)  # 文本 mask
                    total_mask = valid_mask                                # 总 mask

                    ts_correct = ((shifted_preds == shifted_labels) & ts_mask).int()
                    text_correct = ((shifted_preds == shifted_labels) & text_mask).int()
                    total_correct = ((shifted_preds == shifted_labels) & total_mask).int()

                    ts_total = ts_mask.int()
                    text_total = text_mask.int()
                    total_token = total_mask.int()

                    ts_correct = self.accelerator.gather_for_metrics(ts_correct).sum()
                    ts_total = self.accelerator.gather_for_metrics(ts_total).sum()
                    text_correct = self.accelerator.gather_for_metrics(text_correct).sum()
                    text_total = self.accelerator.gather_for_metrics(text_total).sum()
                    total_correct = self.accelerator.gather_for_metrics(total_correct).sum()
                    total_token = self.accelerator.gather_for_metrics(total_token).sum()

                    all_ts_correct.append(ts_correct)
                    all_ts_total.append(ts_total)
                    all_text_correct.append(text_correct)
                    all_text_total.append(text_total)
                    all_correct.append(total_correct)
                    all_total.append(total_token)

        self.model.train()                                                 # 还原为 train 模式

        ts_acc = torch.stack(all_ts_correct).sum().item() / max(torch.stack(all_ts_total).sum().item(), 1)
        text_acc = torch.stack(all_text_correct).sum().item() / max(torch.stack(all_text_total).sum().item(), 1)
        total_acc = torch.stack(all_correct).sum().item() / max(torch.stack(all_total).sum().item(), 1)

        self.accelerator.print(
            f"[Validation] Loss: {np.mean(total_loss):.6f} | Total Acc: {total_acc * 100:.2f}% "
            f"(Text: {text_acc * 100:.2f}% | TS: {ts_acc * 100:.2f}%)"
        )

        return np.mean(total_loss)                                         # 返回平均 loss

    def train(self, setting, test=0):                                      # 主训练接口（与 pretrain 相似）
        """Train the model with distributed multi-GPU support using HuggingFace Accelerate."""

        loaders = {k: self._get_data(flag=k) for k in ['train', 'val', 'test']}  # 获取数据
        train_data, train_loader = loaders['train']
        vali_data, vali_loader = loaders['val']
        test_data, test_loader = loaders['test']

        criterion = self._select_criterion()                               # 损失函数
        self._print_trainable_parameters(self.model)                       # 打印参数统计

        path = os.path.join(self.args.checkpoints, setting)                # checkpoint 路径
        if self.accelerator.is_local_main_process:
            os.makedirs(path, exist_ok=True)

        model_optim = self._select_optimizer(lr=self.args.learning_rate)   # 优化器
        train_steps = len(train_loader)                                    # 每个 epoch 的迭代数
        time_now = time.time()                                             # 当前时间

        scheduler = get_cosine_schedule_with_warmup(                       # 学习率调度
            model_optim,
            warmup_epochs=getattr(self.args, 'warmup_epochs', 2) * train_steps,
            total_epochs=self.args.train_epochs * train_steps
        )

        def test_callback():                                               # EarlyStopping 触发时的回调
            mse, mae = self.test(setting, test=1, save_root=self.args.checkpoints)
            self.accelerator.print(f"[✅] Test after saving best model | MSE: {mse:.3f}, MAE: {mae:.3f}")

        early_stopping = EarlyStopping(accelerator=self.accelerator, patience=self.args.patience,test_fn=test_callback)  # 提前停止
        best_model_path = os.path.join(path, 'checkpoint.pth')             # 最佳模型路径
        # if os.path.exists(best_model_path):
        #     self.accelerator.print("Resuming from last checkpoint.")
        #     self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))

        self.accelerator.init_trackers(setting)                            # 初始化 tracker
        train_loader, vali_loader, test_loader = self.accelerator.prepare(train_loader, vali_loader, test_loader)  # 包装 DataLoader
        self.model, model_optim, scheduler = self.accelerator.prepare(self.model, model_optim, scheduler)         # 包装模型与优化器

        accumulation_steps = getattr(self.args, 'accumulation_steps', 1)   # 梯度累积步数
        iter_verbose = 100                                                 # 日志输出间隔

        for epoch in range(self.args.train_epochs):                        # 训练 loop
            self.model.train()
            train_loss, iter_count = [], 0
            epoch_time = time.time()

            all_ts_correct = []
            all_ts_total = []
            all_text_correct = []
            all_text_total = []
            all_correct = []
            all_total = []

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, start_dates, end_dates) in enumerate(train_loader):  # 遍历 batch
                iter_count += 1

                batch_x, batch_y = batch_x.float(), batch_y.float()
                batch_y = batch_y[:, -self.args.pred_len:, :]

                ts_ids, labels = self.build_input_and_label(
                    batch_x,
                    batch_y,
                    start_date=start_dates[0],                               # 使用第一个样本日期
                    end_date=end_dates[0],
                    is_train=True
                )
                ts_ids = ts_ids.to(self.device)
                labels = labels.to(self.device)

                inputs = {
                    'text_ids': None,
                    'ts_ids': ts_ids,
                    'labels': labels
                }

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(inputs)
                        loss = outputs.loss
                else:
                    outputs = self.model(inputs)
                    loss = outputs.loss

                loss = loss.mean() / accumulation_steps
                train_loss.append(loss.item() * accumulation_steps)

                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                    preds = torch.argmax(logits, dim=-1)

                    # === Shifted Accuracy Calculation ===
                    shifted_preds = preds[:, :-1]
                    shifted_labels = labels[:, 1:]

                    valid_mask = shifted_labels != -100
                    ts_mask = valid_mask & (shifted_labels >= self.model.original_len)
                    text_mask = valid_mask & (shifted_labels < self.model.original_len)

                    ts_correct = ((shifted_preds == shifted_labels) & ts_mask).int()
                    text_correct = ((shifted_preds == shifted_labels) & text_mask).int()
                    total_correct = ((shifted_preds == shifted_labels) & valid_mask).int()

                    ts_total = ts_mask.int()
                    text_total = text_mask.int()
                    total_token = valid_mask.int()

                    ts_correct = self.accelerator.gather_for_metrics(ts_correct).sum()
                    ts_total = self.accelerator.gather_for_metrics(ts_total).sum()
                    text_correct = self.accelerator.gather_for_metrics(text_correct).sum()
                    text_total = self.accelerator.gather_for_metrics(text_total).sum()
                    total_correct = self.accelerator.gather_for_metrics(total_correct).sum()
                    total_token = self.accelerator.gather_for_metrics(total_token).sum()

                    all_ts_correct.append(ts_correct)
                    all_ts_total.append(ts_total)
                    all_text_correct.append(text_correct)
                    all_text_total.append(text_total)
                    all_correct.append(total_correct)
                    all_total.append(total_token)
                
                # if i ==2:
                #     break

                self.accelerator.backward(loss)
                if (i + 1) % accumulation_steps == 0:
                    model_optim.step()
                    scheduler.step()
                    model_optim.zero_grad()

                if (i + 1) % iter_verbose == 0:
                    ts_acc = torch.stack(all_ts_correct).sum().item() / max(torch.stack(all_ts_total).sum().item(), 1)
                    text_acc = torch.stack(all_text_correct).sum().item() / max(torch.stack(all_text_total).sum().item(), 1)
                    total_acc = torch.stack(all_correct).sum().item() / max(torch.stack(all_total).sum().item(), 1)
                    self.accelerator.print(
                        f"\titers: {i + 1}, epoch: {epoch + 1} | loss: {loss.item() * accumulation_steps:.7f} | "
                        f"text_acc: {text_acc * 100:.2f}% | ts_acc: {ts_acc * 100:.2f}% | total_acc: {total_acc * 100:.2f}%")
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    self.accelerator.print(f"\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s")
                    iter_count = 0
                    time_now = time.time()

                    step = epoch * train_steps + i
                    self.accelerator.log(
                        {"train_loss": loss.item() * accumulation_steps,
                        "learning_rate": scheduler.get_last_lr()[0]},
                        step=step
                    )

            train_loss = np.average(train_loss)
            ts_acc = torch.stack(all_ts_correct).sum().item() / max(torch.stack(all_ts_total).sum().item(), 1)
            text_acc = torch.stack(all_text_correct).sum().item() / max(torch.stack(all_text_total).sum().item(), 1)
            total_acc = torch.stack(all_correct).sum().item() / max(torch.stack(all_total).sum().item(), 1)

            self.accelerator.print(f"Epoch {epoch + 1} completed in {time.time() - epoch_time:.2f}s")

            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            self.accelerator.print(
                f"Epoch: {epoch + 1}, Steps: {train_steps} | Train Loss: {train_loss:.7f} "
                f"Vali Loss: {vali_loss:.7f} Test Loss: {test_loss:.7f} "
                f"Text Acc: {text_acc * 100:.2f}% TS Acc: {ts_acc * 100:.2f}% Total Acc: {total_acc * 100:.2f}%"
            )

            self.accelerator.log(
                {"epoch": epoch,
                "train_loss_avg": train_loss,
                "val_loss": vali_loss,
                "test_loss": test_loss,
                "train_ts_acc": ts_acc,
                "train_text_acc": text_acc,
                "train_total_acc": total_acc},
                step=epoch
            )

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                self.accelerator.print("Early stopping")
                break

        if self.accelerator.is_local_main_process and os.path.exists(best_model_path):
            self.accelerator.print(f"Loading best model from {best_model_path}")
            model_state = torch.load(best_model_path, map_location=self.device)
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            unwrapped_model.load_state_dict(model_state)

        self.accelerator.end_training()
        return self.model

    @torch.no_grad()
    def decode_ts(self,input_ids, output_ids, B):                            # 解码时间序列 tokens 为实际数值
        """Decode time series tokens back to values."""
        # num_t = 2
        B_C, n_nt = output_ids.shape                                       # 输出 token 的 batch*通道 与长度
        # output_ids = torch.reshape(output_ids, (-1, num_t))
        device = input_ids.device                                          # 使用输入同设备
        input_tokens = torch.cat([input_ids, output_ids.to(device)], dim=1)# 拼接输入 + 输出 tokens
        # Decode tokens
        input_tokens = input_tokens.to(self.vq_model.device if hasattr(self.vq_model, 'device') else self.device)  # 放到 VQ 模型设备

        decode_ts = self.vq_model.decode_ids(input_tokens).squeeze()       # 通过 VQ 解码为时间序列
        if decode_ts.ndim == 2:
            decode_ts = decode_ts.unsqueeze(0)                             # 保证 B 维存在
        if self.args.chan_indep:                                           # 若通道独立
            decode_ts = torch.reshape(decode_ts, (B_C, -1))                # 先展平
            decode_ts = torch.reshape(decode_ts, (B, -1, decode_ts.shape[-1]))  # 再按 B,L,C 重排
            decode_ts = decode_ts.permute(0, 2, 1)                         # 维度变为 B,C,L
        
        # Apply revin if used
        B, L, C = decode_ts.shape                                          # 记录形状
        if self.vq_model.revin == True:                                    # 若使用 ReVIN 层
            decode_ts = self.vq_model.revin_layer(decode_ts, 'denorm')     # 执行反标准化
        
        return decode_ts[:, -self.args.pred_len:, :]                       # 仅返回预测长度部分

    def process_output_tokens(self, output_tokens):                         # 从输出中分离文本 token 与 TS token
        """Process output tokens to separate text and time series tokens based on special tokens."""
        batch_size = output_tokens.shape[0]                                 # batch 大小
        
        # Get special token IDs
        ts_start_id = self.model.text_tokenizer.convert_tokens_to_ids('<TS_START>')  # TS 起始 ID
        ts_end_id = self.model.text_tokenizer.convert_tokens_to_ids('<TS_END>')      # TS 结束 ID
        
        # Initialize lists to store tokens
        text_tokens_list = []                                              # 每条样本的文本 token 列表
        ts_tokens_list = []                                                # 每条样本的 TS token 列表
        
        # Process each sequence in the batch
        for i in range(batch_size):                                        # 逐样本处理
            seq = output_tokens[i]
            
            # Find special token positions
            ts_start_pos = torch.where(seq == ts_start_id)[0]              # 找到 TS_START 位置
            ts_end_pos = torch.where(seq == ts_end_id)[0]                  # 找到 TS_END 位置
            
            if len(ts_start_pos) == 0 or len(ts_end_pos) == 0:             # 若缺失特殊标记
                self.accelerator.print(f"Warning: Missing special tokens in sequence {i}")
                continue
                
            # Extract text tokens (before first TS_START)
            text_tokens = seq[:ts_start_pos[0]]                            # 起始前为文本 token
            
            # Extract time series tokens (between TS_START and TS_END)
            ts_tokens = seq[ts_start_pos[0]+1:ts_end_pos[0]]               # 两标记之间为 TS token
            
            text_tokens_list.append(text_tokens)
            ts_tokens_list.append(ts_tokens)
        
        # Stack the tokens back into batch
        ts_tokens = torch.stack(ts_tokens_list)                            # 堆叠 TS token
        
        ts_tokens = ts_tokens - self.model.original_len                    # 把 TS token 映射回 VQ 原索引
        
        return text_tokens_list, ts_tokens                                 # 返回文本与 TS token
    
    @torch.no_grad()
    def test_func(self, setting, test=0):                                  # 实际执行测试的函数（生成+解码）
        test_data, test_loader = self._get_data(flag='test')               # 获取测试数据

        if test:
            self.accelerator.print("Loading model...")
            self.model.load_state_dict(torch.load(self.args.pretrained_model, map_location=self.device))  # 加载预训练模型

        self.model.eval()
        preds, trues, inputx = [], [], []                                  # 存储预测/真实/输入
        output_tokens_list, gt_tokens_list = [], []                        # 存储 token

        folder_path = os.path.join('./test_results2', setting)             # 测试结果保存目录
        if self.accelerator.is_local_main_process:
            os.makedirs(folder_path, exist_ok=True)

        test_loader = self.accelerator.prepare(test_loader)                # 包装 Dataloader
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, start_dates, end_dates) in enumerate(
            tqdm(test_loader, desc="Testing", disable=not self.accelerator.is_local_main_process, file=sys.stderr)
        ):                                                                 # tqdm 显示测试进度
            # if i < 167:
            #     continue  # ✅ 只执行第 137 个 batch
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)
            batch_y = batch_y[:, -self.args.pred_len:, :]                  # 取预测部分

            B = batch_x.shape[0]                                           # batch 大小
            ts_ids, gt_tokens, input_tokens, text_tokens_len,ts_token_len = self.build_input_and_label(
                batch_x,
                batch_y,
                start_date=start_dates[0],                                 # 使用第一条样本的日期
                end_date=end_dates[0],
                is_train=False
            )

            inputs = {
                'text_ids': None,
                'ts_ids': ts_ids.to(self.device),
                'labels': gt_tokens.to(self.device),
            }

            output_tokens = self.model.gen_ts(inputs, text_tokens_len,ts_token_len)  # 通过模型生成 TS token
            gt_tokens_list.append(gt_tokens)                               # 收集 GT token

            text_tokens, ts_tokens = self.process_output_tokens(output_tokens)  # 分离文本与 TS tokens
            
            output_tokens_list.append(ts_tokens)                           # 收集输出 TS token
            
            outputs = self.decode_ts(input_tokens.to(self.device),ts_tokens, B=B)  # 将 token 解码为时间序列

            pred = outputs.detach().cpu().numpy()                          # 转 numpy
            true = batch_y.detach().cpu().numpy()
            preds.append(pred)
            trues.append(true)
            inputx.append(batch_x.detach().cpu().numpy())
            
            if i % 5 == 0 and self.accelerator.is_local_main_process:      # 每 5 个 batch 画一次图
                input_np = batch_x.detach().cpu().numpy()
                gt = np.concatenate((input_np[0, :, -1], true[0, :, -1]), axis=0)   # 拼接 ground truth 曲线
                pd = np.concatenate((input_np[0, :, -1], pred[0, :, -1]), axis=0)   # 拼接预测曲线
                visual(gt, pd, os.path.join(folder_path, f"{i}.pdf"))      # 保存对比图
            
            # if i == 2:
            #     break

        output_tokens_list = torch.cat(output_tokens_list, dim=0)          # 合并所有预测 TS token
        gt_tokens_list = torch.cat(gt_tokens_list, dim=0)                  # 合并所有 GT token
        return preds, trues, inputx, output_tokens_list, gt_tokens_list    # 返回所有结果

    def test(self, setting, test=0, save_root='checkpoints'):              # 完整 test 入口（含 token 分布与指标）
        _, _ = self._get_data(flag='train')                                # 触发 data_provider 初始化
        test_data, test_loader = self._get_data(flag='test')

        if test:
            self.accelerator.print("Loading model...")

            model_path = os.path.join(save_root, setting)                  # 模型路径
            lora_adapter_path = os.path.join(model_path, "lora_adapter")   # LoRA 适配器路径
            state_dict_path = os.path.join(model_path, "checkpoint.pth")   # 普通权重路径
            print(state_dict_path)

            unwrapped_model = self.accelerator.unwrap_model(self.model)    # 获取原始模型

            if os.path.exists(lora_adapter_path):
                # ✅ LoRA adapter exists，加载 adapter
                self.accelerator.print(f"Loading LoRA adapter from {lora_adapter_path}")
                self.model = PeftModel.from_pretrained(unwrapped_model, lora_adapter_path)  # 用 LoRA 包装模型
            elif os.path.exists(state_dict_path):
                # ✅ 加载普通 state_dict
                self.accelerator.print(f"Loading full model state_dict from {state_dict_path}")
                model_state = torch.load(state_dict_path, map_location=self.device)
                unwrapped_model.load_state_dict(model_state)
            else:
                raise FileNotFoundError(f"No checkpoint found in {model_path}")  # 找不到任何 checkpoint

        self.model.eval()
        preds, trues, inputx, output_tokens, gt_tokens = self.test_func(setting=setting)  # 执行测试流程

        # Plot token distribution
        plot_token_distribution_with_stratify(
            gt_tokens, output_tokens,
            save_dir=os.path.join(save_root, setting),
            max_token_num=self.args.elected_n_embed,
            dataset='test'
        )                                                                    # 绘制 token 分布对比图

        token_metric_dict = token_metric(output_tokens, gt_tokens)          # 计算 token 级指标
        self.accelerator.print("Token Metric:")
        self.accelerator.print(token_metric_dict)

        preds = np.concatenate(preds, axis=0)                               # 拼接所有预测
        trues = np.concatenate(trues, axis=0)
        inputx = np.concatenate(inputx, axis=0)

        mae, mse, rmse, mape, mspe = metric(preds, trues)                   # 计算数值级指标
        self.accelerator.print(f"mse: {mse}, mae: {mae}")

        if self.accelerator.is_local_main_process:                          # 仅主进程写日志文件
            with open("result.txt", 'a') as f:
                f.write(f"{setting}\n")
                f.write(f"mse: {mse}, mae: {mae}\n\n")

        return mse, mae                                                     # 返回主要指标

    def test_single_sample_overfit(self, setting, num_epochs=150):         # 单样本过拟合测试函数
        print('\n################# Single Sample Overfit Test #################')
        self.device = self.args.gpu                                        # 这里直接使用 gpu id（有点粗暴）
        self.model.to(self.device)                                         # 模型移到 GPU
        self.vq_model.to(self.device)                                      # VQ 模型移到 GPU

        # 取单个样本
        train_data, train_loader = self._get_data(flag='train')            # 获取训练数据
        single_batch = next(iter(train_loader))                            # 取第一个 batch
        batch_x, batch_y, _, _ = single_batch                              # 解包
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y[:, -self.args.pred_len:, :].float().to(self.device)

        # 构造输入与标签
        ts_ids, labels = self.build_input_and_label(batch_x, batch_y, is_train=True)  # 构建训练输入与 label
        inputs = {
            'text_ids': None,
            'ts_ids': ts_ids.to(self.device),
            'labels': labels.to(self.device)
        }

        model_optim = self._select_optimizer(lr=0.001)                     # 使用较大学习率
        criterion = self._select_criterion()                               # 损失函数（未直接用）

        print(f"Training on single sample for {num_epochs} epochs...")
        for epoch in range(num_epochs):                                    # 在单样本上反复训练
            self.model.train()
            model_optim.zero_grad()
            outputs = self.model(inputs)                                   # 前向
            loss = outputs.loss.mean()
            loss.backward()
            model_optim.step()

            if (epoch + 1) % 10 == 0:                                      # 每 10 个 epoch 打一次日志
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.6f}")

                # Token-level Accuracy with SHIFT
                with torch.no_grad():
                    logits = outputs.logits                                # (B, T, V)
                    preds = torch.argmax(logits, dim=-1)                   # (B, T)

                    shifted_preds = preds[:, :-1]
                    shifted_labels = labels[:, 1:]

                    valid_mask = shifted_labels != -100
                    ts_mask = valid_mask & (shifted_labels >= self.model.original_len)
                    text_mask = valid_mask & (shifted_labels < self.model.original_len)

                    ts_acc = ((shifted_preds == shifted_labels) & ts_mask).sum().item() / max(ts_mask.sum().item(), 1)
                    text_acc = ((shifted_preds == shifted_labels) & text_mask).sum().item() / max(text_mask.sum().item(), 1)
                    total_acc = ((shifted_preds == shifted_labels) & valid_mask).sum().item() / max(valid_mask.sum().item(), 1)

                    print(f"Token Acc | Text: {text_acc*100:.2f}%, TS: {ts_acc*100:.2f}%, Total: {total_acc*100:.2f}%")

        # 推理评估
        test_ts_ids, test_labels, out_token_shape = self.build_input_and_label(batch_x, batch_y, is_train=False)  # 构建推理输入
        inputs = {
            'text_ids': None,
            'ts_ids': test_ts_ids.to(self.device),
            'labels': test_labels.to(self.device),
        }

        self.model.eval()
        with torch.no_grad():
            output_tokens = self.model.gen_ts(inputs, out_token_shape)     # 生成 token
            text_tokens, ts_tokens = self.process_output_tokens(output_tokens)  # 分离文本/TS token

            B = batch_x.shape[0]
            decoded_outputs = self.decode_ts(ts_tokens, B=B)               # 解码时间序列
            decoded_text = self.model.text_tokenizer.batch_decode(
                text_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            print(decoded_text)                                            # 打印生成的文本部分

            decoded_outputs = decoded_outputs.detach().cpu().numpy()
            batch_y = batch_y.detach().cpu().numpy()
            
            print("ts_tokens shape:", ts_tokens.shape)
            print("gt_tokens shape:", test_labels.shape)
            token_metric_dict = token_metric(ts_tokens, test_labels)       # token 指标
            print("\nToken Metrics:")
            print(token_metric_dict)

            mae, mse, rmse, mape, mspe = metric(decoded_outputs, batch_y)  # 数值重建指标
            print("\nReconstruction Metrics:")
            print(f"MSE: {mse:.6f}")
            print(f"MAE: {mae:.6f}")
            print(f"RMSE: {rmse:.6f}")

            folder_path = './test_results/' + setting + '/'                # 结果保存路径
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            input_data = batch_x.detach().cpu().numpy()
            gt = np.concatenate((input_data[0, :, -1], batch_y[0, :, -1]), axis=0)            # ground truth 曲线
            pd = np.concatenate((input_data[0, :, -1], decoded_outputs[0, :, -1]), axis=0)    # 预测曲线
            visual(gt, pd, os.path.join(folder_path, 'single_sample_overfit.pdf'))            # 画对比图

            plot_token_distribution_with_stratify(
                test_labels, ts_tokens,
                save_dir=folder_path,
                max_token_num=self.args.elected_n_embed,
                dataset='single_sample_overfit'
            )                                            # 单样本 token 分布对比

        return mse, mae                                 # 返回重建误差
