import torch  # 导入 PyTorch 主库
import torch.nn as nn  # 导入神经网络模块


class CE:  # 定义 CE（交叉熵）损失类
    def __init__(self, model):
        self.model = model                              # 保存模型，用于前向计算
        self.ce = nn.CrossEntropyLoss()                 # 标准交叉熵损失
        self.ce_pretrain = nn.CrossEntropyLoss(ignore_index=0)  # 预训练用 CE，忽略索引为 0 的标签

    def compute(self, batch):
        seqs, labels = batch                            # 解包输入序列和标签
        outputs = self.model(seqs)  # B * N             # 模型输出类别分布
        labels = labels.view(-1).long()                 # 将标签展平为一维
        loss = self.ce(outputs, labels)                 # 计算交叉熵损失
        return loss                                     # 返回损失


class Align:  # 定义对齐损失类（mask 对齐）
    def __init__(self):
        self.mse = nn.MSELoss(reduction='mean')         # 用均值 MSE 计算对齐损失
        self.ce = nn.CrossEntropyLoss()                 # 备用交叉熵损失（未使用）

    def compute(self, rep_mask, rep_mask_prediction):
        align_loss = self.mse(rep_mask, rep_mask_prediction)  # 计算掩码特征与预测的 MSE
        return align_loss                                     # 返回对齐损失


class Reconstruct:  # 定义重构损失类（token 重建）
    def __init__(self):
        self.ce = nn.CrossEntropyLoss(label_smoothing=0.2)    # 使用标签平滑的交叉熵损失

    def compute(self, token_prediction_prob, tokens):
        hits = torch.sum(torch.argmax(token_prediction_prob, dim=-1) == tokens)  # 预测命中数量
        NDCG10 = recalls_and_ndcgs_for_ks(                                      # 计算 NDCG@10
            token_prediction_prob.view(-1, token_prediction_prob.shape[-1]),
            tokens.reshape(-1, 1),
            10
        )
        reconstruct_loss = self.ce(                                              # 重构损失
            token_prediction_prob.view(-1, token_prediction_prob.shape[-1]),
            tokens.view(-1)
        )
        return reconstruct_loss, hits, NDCG10                                     # 返回三项指标


def recalls_and_ndcgs_for_ks(scores, answers, k):
    answers = answers.tolist()                                   # 转为 Python list
    labels = torch.zeros_like(scores).to(scores.device)          # 初始化 one-hot 标签
    for i in range(len(answers)):                                 # 构造 one-hot 分布
        labels[i][answers[i]] = 1
    answer_count = labels.sum(1)                                  # 每个样本的答案数（一般为 1）

    labels_float = labels.float()                                 # 转为浮点数
    rank = (-scores).argsort(dim=1)                               # 根据得分降序排序得到排名
    cut = rank[:, :k]                                             # 取前 k 个预测
    hits = labels_float.gather(1, cut)                            # 提取命中的标签位置

    position = torch.arange(2, 2 + k)                             # 位置从 2 开始（DCG 公式）
    weights = 1 / torch.log2(position.float())                    # DCG 权重
    dcg = (hits * weights.to(hits.device)).sum(1)                 # 计算 DCG
    idcg = torch.Tensor([weights[:min(int(n), k)].sum()           # 理想 DCG（按真实答案数）
                         for n in answer_count]).to(dcg.device)
    ndcg = (dcg / idcg).mean()                                    # 平均 NDCG
    ndcg = ndcg.cpu().item()                                      # 转为 Python float
    return ndcg                                                   # 返回 NDCG@k


import torch  # 再次导入 torch（重复但无影响）
from tqdm import tqdm  # 导入 tqdm 进度条库
from sklearn.pipeline import make_pipeline  # 导入 sklearn pipeline
from sklearn.preprocessing import StandardScaler  # 标准化工具
from sklearn.linear_model import LogisticRegression  # 逻辑回归分类器


def get_rep_with_label(model, dataloader):
    reps = []                                          # 存储特征表示
    labels = []                                        # 存储标签
    with torch.no_grad():                              # 不计算梯度
        for batch in tqdm(dataloader):                 # 遍历数据集
            seq, label = batch                         # 解包输入
            seq = seq.to(model.device)                 # 放到模型设备上
            labels += label.cpu().numpy().tolist()     # 保存标签
            rep = model(seq)                           # 获取模型的表示向量
            reps += rep.cpu().numpy().tolist()         # 保存表示
    return reps, labels                                # 返回特征与标签


def fit_lr(features, y):
    pipe = make_pipeline(                              # 组合标准化 + 逻辑回归
        StandardScaler(),                              # 标准化特征
        LogisticRegression(
            random_state=3407,                         # 固定随机种子
            max_iter=1000000,                          # 超长迭代，保证收敛
            multi_class='ovr'                          # one-vs-rest 多分类
        )
    )
    pipe.fit(features, y)                              # 拟合模型
    return pipe                                        # 返回训练好的逻辑回归模型
