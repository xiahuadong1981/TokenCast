import torch                           # 导入 PyTorch 主包
from torch import nn                   # 从 PyTorch 中导入神经网络模块 nn
from torch.nn import functional as F   # 导入函数式接口 F（如激活函数、卷积等）
from torch.nn.init import xavier_normal_, constant_  # 导入权重初始化方法 xavier_normal_ 和 constant_
# from layers.Transformer_EncDec import Encoder, EncoderLayer  # 预留：Transformer 编码器相关模块（当前未使用）
# from layers.SelfAttention_Family import FullAttention, AttentionLayer  # 预留：自注意力模块（当前未使用）
# from vector_quantize_pytorch import ResidualVQ  # 预留：Residual VQ 量化模块（当前未使用）
from models.RevIN import RevIN         # 从自定义模型中导入 RevIN（可逆归一化）模块
from einops import rearrange           # 导入 einops 的 rearrange，用于灵活变换张量
import torch                           # 再次导入 torch（重复导入，一般可省略）
from torch import nn                   # 再次导入 nn（重复导入）
from torch.nn.utils import weight_norm # 导入 weight_norm，为卷积层添加权重归一化
from models.CasualTRM import CasualTRM # 导入自定义的因果 Transformer 模块 CasualTRM

import torch                           # 第三次导入 torch（重复导入）
import torch.nn as nn                  # 以别名 nn 导入 torch.nn
import torch.nn.functional as F        # 以别名 F 导入函数式接口
import numpy as np                     # 导入 NumPy，用于数值计算（如 log）
from einops import rearrange           # 再次导入 rearrange（重复导入）

class Quantize(nn.Module):             # 定义量化模块 Quantize，继承自 nn.Module
    def __init__(self, dim, n_embed,configs, beta=0.25, eps=1e-5):  # dim: 向量维度; n_embed: codebook 大小
        super().__init__()             # 初始化父类 nn.Module
        self.dim = dim                 # 保存特征维度
        self.n_embed = n_embed         # 保存嵌入数量（codebook 大小）
        self.beta = beta               # commitment loss 的权重系数
        self.entropy_penalty = configs.entropy_penalty  # 熵正则项的权重
        self.entropy_temp = configs.entropy_temp        # 熵正则温度参数
        self.eps = eps                 # 数值稳定用的极小值

        self.embedding = nn.Embedding(n_embed, dim)     # 定义 codebook，形状 [n_embed, dim]
        nn.init.normal_(self.embedding.weight, mean=0.0, std=dim ** -0.5)  # 正态初始化 codebook 权重
        self.embedding_proj = nn.Linear(dim, dim)       # 用线性层对 codebook 做投影

    def forward(self, input):          # 前向传播 input: [B, T, C]
        B, T, C = input.shape          # 读取 batch 大小 B、时间步数 T、通道数 C
        flatten = input.reshape(-1, C) # 展平成 [B*T, C]，便于与 codebook 计算距离

        # if self.training:
        #     flatten = flatten + 0.01 * torch.randn_like(flatten)  # 训练时可加入微小噪声做稳定（已注释）

        # codebook projection
        codebook = self.embedding_proj(self.embedding.weight)  # [n_embed, dim]，对嵌入向量做一次线性投影

        # compute distance
        d = torch.sum(flatten ** 2, dim=1, keepdim=True) + \  # x^2 部分，形状 [B*T, 1]
            torch.sum(codebook ** 2, dim=1) - 2 * torch.matmul(flatten, codebook.t())  # c^2 - 2 x·c，得到 [B*T, n_embed]

        # soft assignment (for encoder entropy loss)
        logits = -d / self.entropy_temp               # 距离转为负数并除以温度，得到似然 logits
        probs = F.softmax(logits, dim=-1)             # 对 codebook 维度做 softmax，得到每个 token 相对于所有 code 的概率
        soft_entropy = -torch.sum(probs * torch.log(probs + self.eps), dim=-1).mean()  # 计算平均软熵（信息熵）
        max_entropy = np.log(self.n_embed)            # 理论最大熵（均匀分布时）
        norm_soft_entropy = soft_entropy / max_entropy  # 归一化软熵，范围约 [0,1]
        soft_entropy_loss = self.entropy_penalty * (1.0 - norm_soft_entropy)  # 熵正则损失（鼓励更均匀的使用）

        # hard assignment
        indices = torch.argmax(probs, dim=-1)  # [B*T]，选取概率最大的 codebook 索引（硬分配）
        z_q = F.embedding(indices, codebook).view(B, T, C)  # 通过索引查找 codebook，并还原为 [B,T,C]

        # commitment and embedding loss
        diff_loss = F.mse_loss(z_q.detach(), input)       # 让 codebook 靠近 encoder 输出的损失（更新 codebook）
        commit_loss = F.mse_loss(z_q, input.detach())     # 让 encoder 输出靠近 codebook 的损失（更新 encoder）
        vq_loss = diff_loss + self.beta * commit_loss     # 组合得到 VQ 的主损失

        # additional hard token usage entropy (non-differentiable, optional)
        with torch.no_grad():                             # 使用硬 one-hot 统计 token 使用情况（不反传）
            one_hot = F.one_hot(indices, num_classes=self.n_embed).float()  # [B*T, n_embed]，one-hot 编码
            avg_probs = one_hot.mean(dim=0) + self.eps   # 求每个 code 被使用的平均频率
            token_usage_entropy = -torch.sum(avg_probs * torch.log(avg_probs))  # 对使用频率计算熵
            token_usage_max = torch.log(torch.tensor(self.n_embed, dtype=token_usage_entropy.dtype, device=token_usage_entropy.device))  # 最大熵
            norm_token_usage_entropy = token_usage_entropy / token_usage_max  # 归一化熵
            token_entropy_loss = self.entropy_penalty * (1.0 - norm_token_usage_entropy)  # token 使用的熵正则

        # total loss (only soft_entropy_loss is differentiable w.r.t. encoder)
        total_loss = vq_loss + soft_entropy_loss + token_entropy_loss  # 总损失 = VQ + 软熵正则 + 硬熵正则

        # straight-through estimator
        z_q = input + (z_q - input).detach()             # 直通估计：前向用量化结果，反向对 input 传梯度

        return z_q, total_loss, indices                  # 返回量化后的特征、总损失和离散索引


    def embed_code(self, embed_id):                      # 根据 code 索引返回对应 embedding 的接口
        embedding = F.embedding(embed_id, self.embed.transpose(0, 1))  # NOTE: 这里使用 self.embed 可能是笔误
        return self.embedding_proj(embedding)            # 将 embedding 通过同一个线性投影层


# TCN from tsai
class Chomp1d(nn.Module):                               # 定义 Chomp1d，用于裁剪 1D 卷积后多出来的 padding
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()                 # 调用父类初始化
        self.chomp_size = chomp_size                    # 需要裁剪的长度

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()  # 在时间维度上裁剪掉 chomp_size，保持张量连续


class TemporalBlock(nn.Module):                         # 定义 TCN 的基础残差块 TemporalBlock
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()           # 调用父类初始化
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))  # 第一层卷积并加权重归一化
        self.chomp1 = Chomp1d(padding)                  # 第一层卷积后的裁剪，去除因 padding 引入的未来信息
        self.relu1 = nn.ReLU()                          # 第一层 ReLU 激活
        self.dropout1 = nn.Dropout(dropout)             # 第一层 Dropout

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))  # 第二层卷积
        self.chomp2 = Chomp1d(padding)                  # 第二层卷积后的裁剪
        self.relu2 = nn.ReLU()                          # 第二层 ReLU
        self.dropout2 = nn.Dropout(dropout)             # 第二层 Dropout

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)  # 将两层串联成一个子网络
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None  # 若维度不同，用 1x1 卷积做残差映射
        self.relu = nn.ReLU()                           # 残差输出后的 ReLU
        self.init_weights()                             # 初始化权重

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)         # 第一层卷积权重正态初始化
        self.conv2.weight.data.normal_(0, 0.01)         # 第二层卷积权重正态初始化
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)  # 下采样卷积权重初始化

    def forward(self, x):
        out = self.net(x)                               # 通过两层卷积 + ReLU + Dropout 的子网络
        res = x if self.downsample is None else self.downsample(x)  # 残差分支，通道不匹配时通过 1x1 卷积调整
        return self.relu(out + res)                     # 残差相加并激活


class TemporalConvNet(nn.Module):                       # 定义多层堆叠的 TCN 网络
    def __init__(self, channel_in, num_channels, kernel_size=3, dropout=0.2):
        super(TemporalConvNet, self).__init__()         # 调用父类初始化

        layers = []                                     # 存放多个 TemporalBlock
        num_levels = len(num_channels)                  # 层数 = num_channels 的长度
        for i in range(num_levels):
            dilation_size = 1 ** i                      # 当前代码中膨胀系数恒为 1（一般 TCN 会用 2**i）
            in_channels = channel_in if i == 0 else num_channels[i - 1]  # 输入通道
            out_channels = num_channels[i]              # 输出通道
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]  # 构造一个 TemporalBlock 加入列表

        self.network = nn.Sequential(*layers)           # 将所有 block 串联成一个顺序网络

    def forward(self, x):
        return self.network(x)                          # 直接调用顺序网络的前向


class Encoder(nn.Module):                               # 定义编码器，使用 TCN 作为特征抽取
    def __init__(self, chan_indep,channel_in, hidden_dim, block_num=3, kernel_size=3, dropout=0.2):
        super().__init__()                              # 调用父类初始化
        self.chan_indep = chan_indep                    # 是否 channel 独立处理（每个特征单独过 TCN）

        self.TCN = TemporalConvNet(channel_in, [hidden_dim]*block_num, kernel_size=kernel_size, dropout=dropout)  # 堆叠 block_num 个 TCN Block
        
    def forward(self, x):
        x = x.permute(0, 2, 1)                          # [B, T, C] → [B, C, T] 以匹配 Conv1d 输入格式
        if self.chan_indep:                             # 若 channel 独立处理
            x = x.reshape(-1, x.shape[-1]).unsqueeze(1) # 将每个变量单独拉直为 [B*C, 1, T]
        x = self.TCN(x)                                 # 经过 TCN 提取时序特征
        x = x.permute(0, 2, 1)                          # 输出 [B, T, C] 形式
        return x
    
class Decoder(nn.Module):                               # 定义解码器，采用 CasualTRM + MLP 输出
    def __init__(self, patch_len, enc_in, hidden_dim, n_heads=4, block_num=3, dropout=0.2):
        super().__init__()                              # 调用父类初始化
        self.decoder = CasualTRM(dim=hidden_dim, d_ff=hidden_dim*4,
                                 n_heads=n_heads, n_layers=block_num, dropout=dropout)  # 因果 Transformer 解码器

        # 将每个 token 特征维度通过 conv1d 从 1→通道
        # self.conv1d = nn.Conv1d(in_channels=1, out_channels=enc_in, kernel_size=1)  # 之前的 conv 解码结构（已注释）

        # # 非线性映射
        # self.activation = nn.ReLU()                      # ReLU 激活（旧结构）

        # 映射到 patch_len
        self.linear = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),       # 先将维度扩展到 2 倍
            nn.GELU(),                                   # GELU 非线性激活
            nn.Linear(hidden_dim * 2, patch_len * enc_in)  # 映射到 patch_len * enc_in（预测多个时间步）
        )

        self.patch_len = patch_len                      # 每个 patch 的时间长度
        self.enc_in = enc_in                            # 原始特征维度（变量数）

    def forward(self, x):
        """
        x: [B, T, D]  → D = hidden_dim
        output: [B, patch_len * enc_in, 1] 或 reshape 后为 [B, pred_len, enc_in]
        """
        B, T, D = x.shape                               # 获取 batch、长度、隐藏维度
        x, _ = self.decoder(x)                          # 通过因果 Transformer 解码，输出序列特征
        x = self.linear(x)                              # 线性映射为 [B, T, patch_len * enc_in]
        x = x.view(B, T * self.patch_len, self.enc_in)  # 把 T*patch_len 合成 pred_len 维度，得到 [B, pred_len, enc_in]
        # x = x.reshape(B * T, 1, D)                     # 旧解码路径：先 reshape 再 conv1d
        # x = self.conv1d(x)                             # 旧：通过 conv1d 映射到 enc_in 通道
        # x = self.activation(x)                         # 旧：激活
       
        # x = self.linear(x)                             # 旧：线性映射到 patch_len
        # x = x.transpose(1, 2)                          # 旧：调整维度顺序
        # x = x.reshape(B, T * self.patch_len,self.enc_in)  # 旧：恢复到 [B, pred_len, enc_in]

        return x                                        # 返回预测序列

class VQVAE(nn.Module):                                 # 定义整体 VQ-VAE 模型（时序版）
    def __init__(
            self,
            configs                                      # 配置对象，包含各种超参数
    ):
        super().__init__()                              # 调用父类初始化

        self.seq_len = configs.seq_len                  # 输入序列长度
        self.pred_len = configs.pred_len                # 预测序列长度
        total_len = self.seq_len + self.pred_len        # 总长度（lookback + prediction）
        hidden_dim = configs.d_model                    # 隐藏维度
        n_embed = configs.n_embed                       # codebook 大小
        # codebook_num = configs.codebook_num           # 预留：多 codebook 数量（未使用）
        block_num = configs.block_num                   # 编码器 / 解码器 block 数
        # kernel_size = configs.kernel_size_vqvae       # 预留：TCN kernel size（未使用）
        # dropout = configs.dropout_vqvae               # 预留：TCN dropout（未使用）
        self.patch_len = configs.wave_length            # patch 长度（一次量化/预测的时间步数）
        # n_heads = configs.n_heads_vqvae               # 预留：多头注意力头数（未使用）
        # d_layers = configs.d_layers_vqvae             # 预留：解码层数（未使用）
        
        self.revin = configs.revin                      # 是否使用 RevIN 做归一化
        affine = configs.affine                         # RevIN 是否使用仿射变换
        subtract_last = configs.subtract_last           # RevIN 是否减去最后一个值
        # Channel independent
        self.chan_indep = configs.chan_indep            # 是否按通道独立编码

        enc_in = configs.enc_in if configs.chan_indep == 0 else 1  # 若通道独立，则输入通道设为 1
        
        data_shape = (total_len, enc_in)                # 预留：数据形状信息（当前未实际使用）
        
        self.enc = Encoder(self.chan_indep,enc_in, hidden_dim, block_num)  # 构造 TCN 编码器
        wave_patch = (self.patch_len, hidden_dim)       # 量化输入的卷积 kernel 大小（时间长度 × 通道）
        self.quantize_input = nn.Conv2d(1, hidden_dim, kernel_size=wave_patch, stride=wave_patch)  # Conv2d 提取 patch 特征
        self.quantize = Quantize(hidden_dim, n_embed,configs)  # VQ 量化模块
        self.dec = Decoder(self.patch_len,enc_in, hidden_dim)  # Transformer 解码模块
        
        if self.revin:
            self.revin_layer = RevIN(enc_in, affine=affine, subtract_last=subtract_last)  # 若启用 RevIN，则构造 RevIN 层

    def forward(self, x,y):                             # 前向传播，x 为 lookback，y 为待预测段
        if self.revin:
            x_look_back = self.revin_layer(x, 'norm')   # 对历史序列做 RevIN 归一化
            x_pred = self.revin_layer._normalize(y)     # 对预测部分也做相同归一化
            x = torch.cat([x_look_back, x_pred], dim=1) # 在时间维上拼接得到总序列
        # x= x.permute(0,2,1)                           # 旧代码：调整维度（已注释）
        # x = torch                                      # 残留无效代码（已注释）


        # print(f'input:{input.shape}')                 # 调试信息（已注释）
        
        # x = x.unfold(1, self.patch_len, self.patch_len) # 旧 patch 划分写法（已注释）
        # x = x.permute(0, 1, 3, 2)                     # 旧：调整 patch 维度顺序
        # patch_num = x.shape[1]                        # 旧：patch 数量
        # x = x.reshape(-1, x.shape[-2], x.shape[-1])   # 旧：[bs * patch_num x patch_len x n_vars]
        n_var = x.shape[-1]                             # 变量数（特征维度）
        B = x.shape[0]                                  # batch 大小
        enc = self.enc(x)                               # 通过 TCN 编码器得到 [B,T,H]
        enc = enc.unsqueeze(1)                          # 增加一维变为 [B,1,T,H]，以匹配 Conv2d 输入
        # print(f'endcoder_output:{enc.shape}')         # 调试输出（已注释）
        quant = self.quantize_input(enc).squeeze(-1).transpose(1, 2)  # Conv2d 提取 patch → [B,patch_num,H]
        # print(quant.shape)                            # 调试输出（已注释）

        # quant = quant.reshape(-1, patch_num, quant.shape[2])  # 旧 reshape 方式（已注释）

        # print(f'quantize_input:{quant.shape}')        # 调试输出（已注释）
        quant, diff,ids  = self.quantize(quant)         # 量化：获得量化后的特征、损失以及 code 索引
        # print(f'quantize_output:{quant.shape}')       # 调试输出（已注释）
        # print(f'dec_input:{quant.shape}')             # 调试输出（已注释）
        # print(f'ids:{ids.shape}')                     # 调试输出（已注释）

        dec = self.dec(quant)                           # 使用解码器从量化表示恢复到时间序列
        if self.chan_indep:                             # 若通道独立
            dec = dec.permute(0,2,1)                    # [B,T,C] → [B,C,T]
            dec = dec.reshape(-1, n_var, dec.shape[-1]) # 合并 batch 与通道
            dec = dec.permute(0,2,1)                    # 恢复为 [B*T, T, C] 形式
        # dec = dec.permute(0,2,1)                      # 旧代码（已注释）
        if self.revin:
            dec = self.revin_layer(dec, 'denorm')       # 使用 RevIN 将结果从归一化空间还原

        #print(f'dec_output:{dec.shape}')               # 调试输出（已注释）

        return dec, diff, ids                           # 返回解码结果、VQ 损失和离散 token 索引

    def get_name(self):
        return 'r_vqvae'                                # 返回模型名称标识

    def get_embedding(self, x):                         # 获取量化后的 embedding 表示（编码部分）
        enc = self.enc(x)                               # 编码
        enc = enc.unsqueeze(1)                          # [B,T,H] → [B,1,T,H]
        quant = self.quantize_input(enc).squeeze(-1).transpose(1, 2)  # Conv2d→[B,patch_num,H]
        quant, ids, diff = self.quantize(quant)         # 量化
        return quant                                    # 返回量化嵌入

    def get_ids(self, x):                               # 获取输入序列对应的离散 code 索引
        if self.revin:
            x = self.revin_layer(x, 'norm')             # 若启用 RevIN，先归一化
        enc = self.enc(x)                               # 编码
        enc = enc.unsqueeze(1)                          # [B,1,T,H]
        quant = self.quantize_input(enc).squeeze(-1).transpose(1, 2)  # Conv2d→[B,patch_num,H]
        quant, ids, diff = self.quantize(quant)         # 量化
        return ids                                      # 返回离散索引

    def decode_from_ids(self, look_back, ids):
        """ 
        lookback: [bs x seq_len x 1]
        ids: [bs x pred_num x codebook_num]
        """
        if self.revin:
            look_back = self.revin_layer(look_back, 'norm')  # 预测前对历史序列做归一化
        # print(look_back.shape)                         # 调试输出（已注释）
        enc = self.enc(look_back)                       # 对历史序列编码
        enc = enc.unsqueeze(1)                          # [B,T,H]→[B,1,T,H]
        
        quant = self.quantize_input(enc).squeeze(-1).transpose(1, 2)  # Conv2d→[B,N,H]
        quant, _, _ = self.quantize(quant)              # 根据历史先得到一个 quant 表示（未使用 ids）

        B, N, Q = ids.shape                             # B: batch, N: pred_num, Q: codebook_num
        ids_t = ids.permute(2, 0, 1).reshape(Q, -1)     # 转为 [Q, B*N]，方便批量索引
        quant_ids = torch.arange(Q, device=ids.device).unsqueeze(1).expand(Q, B*N)  # 形成每个 codebook 的索引行
        # 查找所有 embedding
        embeddings = self.quantize.codebooks[quant_ids, ids_t]  # (Q, B*N, D)  从多 codebook 中取出对应 embedding
        embeddings = embeddings.permute(1, 0, 2).reshape(B, N, Q, -1)  # 重排维度为 [B,N,Q,D]
        embedding = embeddings.sum(dim=-2)              # 在 codebook 维度上求和，得到最终 embedding

        # [bs x pred_num x embedding_size]
        quant = torch.cat([quant, embedding], dim=1)    # 将历史 quant 和预测 embedding 拼在一起
        dec = self.dec(quant)                           # 解码得到完整序列

        if self.revin:
            dec = self.revin_layer(dec, 'denorm')       # RevIN 反归一化
        return dec[:, look_back.shape[1]:, :]           # 只取预测部分 [B, pred_len, C]
        
    def decode(self, quant):                            # 仅根据量化特征解码
        dec = self.dec(quant)                           # 调用解码器
        return dec                                      # 返回解码结果
