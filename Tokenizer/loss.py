import torch.nn as nn  # 导入 PyTorch 的神经网络模块
from args import args  # 导入训练配置参数
import torch           # 导入 PyTorch 主库

class MSE:  # 定义一个 MSE 损失封装类
    def __init__(self, model, latent_loss_weight=0.25,dist_penalty_weight=0.25, dist_penalty_topk=1):
        self.model = model                               # 保存模型，用于前向计算
        self.latent_loss_weight = latent_loss_weight     # 隐变量损失权重
        self.dist_penalty_weight = dist_penalty_weight   # 代码本距离惩罚项权重
        self.dist_penalty_topk = dist_penalty_topk       # 距离惩罚只计算 top-k 最近的距离
        self.mse = nn.MSELoss()                          # MSE Loss 实例化

    def compute(self, batch,batch_y, details=False,dist_penalty=False):
        seqs = torch.cat([batch,batch_y],dim=1)          # 拼接输入与目标序列作为重构目标
        out, latent_loss, _ = self.model(batch,batch_y)  # 通过模型得到输出与 latent loss
        recon_loss = self.mse(out, seqs)                 # 计算重构误差 MSE
        latent_loss = latent_loss.mean()                 # 隐变量损失取平均
        loss = recon_loss + self.latent_loss_weight * latent_loss  # 总损失 = 重构 + 隐变量损失

        if dist_penalty:                                 # 若启用距离惩罚，则执行以下计算
            codebook_weight = self.model.quantize.embedding.weight  # 取出 VQ 代码本权重
            _detach_weight = codebook_weight.detach()                # 创建不参与梯度的副本

            dist = torch.cdist(codebook_weight, _detach_weight, p=2) # 计算代码本向量间的 pairwise 距离
            diagonal_matrix = torch.eye(dist.shape[0]) * 1e9         # 构造对角矩阵（非常大，用来屏蔽自身）

            dist = dist + diagonal_matrix.to(dist.device)            # 将自身距离置为无穷大（防止选到自己）

            min_dist, _ = torch.topk(dist, self.dist_penalty_topk, dim=-1, largest=False)  # 取最小 k 个距离
            min_dist = 1. / (min_dist + 1e-5)                       # 距离越小罚得越重：取倒数

            dist_penalty_value = min_dist.mean()                    # 距离惩罚取平均

            loss = loss + self.dist_penalty_weight * dist_penalty_value  # 加入到总损失中

        if details:  # 若需要返回详细信息
            return {
                'loss': loss,             # 总损失
                'recon_loss': recon_loss, # 重构损失
                'latent_loss': latent_loss # 隐变量损失
            }
        else:
            return loss  # 否则只返回总损失
