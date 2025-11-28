import argparse  # 导入命令行参数解析库
import os        # 导入操作系统路径、文件等工具
import json      # 导入 JSON 读写库

parser = argparse.ArgumentParser()  # 创建一个参数解析器实例
# basic config
parser.add_argument("--is_training", type=int, default=1)  # 是否为训练模式
parser.add_argument("--save_path", type=str, default=None)  # 保存模型的目录
parser.add_argument("--load_path", type=str, default=None)  # 加载模型的目录

# dataset
parser.add_argument('--data', type=str, required=False, default='ETTh1', help='dataset type')  # 数据集名称
parser.add_argument('--freq', type=str, default='h',  # 时间特征的频率编码
                        help='freq for time features encoding, options:[s, t, h, d, b, w, m] ...')
parser.add_argument('--features', type=str, default='M',  # 预测类型（多变量/单变量）
                        help='forecasting task, options:[M, S, MS]')

# dataloader
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')  # 目标列名称
parser.add_argument('--token_len', type=int, default=336, help='input sequence length')  # Token 输入序列长度
parser.add_argument('--percent', type=int, default=100)  # 用于数据比例（如百分比采样）

# seq_len
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')  # 输入序列长度
parser.add_argument('--label_len', type=int, default=0, help='label sequence length')  # 预测辅助长度
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')  # 输出预测长度

parser.add_argument('--embed', type=str, default='timeF',  # 时间特征 embedding 方式
                        help='time features encoding, options:[timeF, fixed, learned]')

parser.add_argument('--root_path', type=str, default='./dataset/ETT-small', help='root path of the data file')  # 数据根目录
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')  # 数据文件
parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')  # 季节模式设置（M4用）

parser.add_argument("--train_batch_size", type=int, default=32)  # 训练批大小
parser.add_argument("--test_batch_size", type=int, default=64)   # 测试批大小

parser.add_argument("--num_workers", type=int, default=0)  # dataloader 线程数
parser.add_argument('--use_fullmodel', type=int, default=0, help='use full model or just encoder')  # 是否使用全模型
parser.add_argument('--use_closedllm', type=int, default=0, help='use closedllm or not')  # 是否使用闭源大模型
parser.add_argument('--text_len', type=int, default=4)  # 文本 token 长度

# model args
parser.add_argument("--d_model", type=int, default=64)  # 模型隐藏维度
parser.add_argument("--enc_in", type=int, default=21)   # 输入维度数量（变量数）
parser.add_argument("--dropout", type=float, default=0.2)  # dropout 比例
parser.add_argument("--block_num", type=int, default=2)  # 模型 block 数量

parser.add_argument("--n_embed", type=int, default=256)   # VQ embedding 维度
parser.add_argument("--wave_length", type=int, default=7)  # 波长编码长度
parser.add_argument("--chan_indep", type=int, default=0, help="independent channels")  # 是否独立通道编码

parser.add_argument("--vq_model", type=str, default='SimVQ', help='options:[SimVQ, VanillaVQ, SimVQ_CNN]')  # VQ 模型类型

# Revin
parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')  # 是否启用 RevIN
parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')  # RevIN 的 affine 设置
parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')  # RevIN 的减法方式

# train args
parser.add_argument("--lr", type=float, default=0.0001)  # 学习率
parser.add_argument("--lr_decay_rate", type=float, default=0.99)  # 学习率衰减率
parser.add_argument("--lr_decay_steps", type=int, default=300)  # 学习率衰减步长
parser.add_argument("--weight_decay", type=float, default=1e-5)  # 权重衰减
parser.add_argument("--num_epoch", type=int, default=60)  # 训练轮数
parser.add_argument("--eval_per_steps", type=int, default=300)  # 每多少步评估一次
parser.add_argument("--device", type=str, default="cuda:0")  # 使用的设备

parser.add_argument("--eval_per_epoch", action="store_true")  # 是否按 epoch 评估
parser.add_argument('--multi_dataset', action='store_true', help='Enable multi-dataset joint training')  # 多数据集联合训练
parser.add_argument('--entropy_penalty', type=float, default=0.1, help='Penalty weight for entropy regularization')  # 熵惩罚权重
parser.add_argument('--entropy_temp', type=float, default=0.5, help='Temperature for soft assignment in VQ')  # Soft assignment 温度

args = parser.parse_args()  # 解析命令行参数

# Train_data,Test_data = load_ETT(... )  # 示例：加载数据（被注释掉）

args.dataset = args.data_path.split('.')[0]  # 从文件名中提取数据集名称（不含扩展名）

vq_setting = "unfreeze_codebook"  # VQ 代码本设置方式
if args.save_path is None:  # 若未设置 save_path，则自动生成
    path_str = 'checkpoints//{}_{}_dm{}_dr{}_emb{}_wl{}_bl{}_{}_{}'.format(
            args.dataset,
            args.token_len,
            args.d_model,
            args.dropout,
            args.n_embed,
            args.wave_length,
            args.block_num,
            args.vq_model,
            vq_setting
            )
    
    args.save_path = path_str  # 设置 save_path
    
if not os.path.exists(args.save_path):  # 若目录不存在则创建
    print("Creating save dir: {}".format(args.save_path))  # 打印提示
    os.makedirs(args.save_path)  # 创建目录

with open(args.save_path + "/args.json", "w") as f:  # 将所有参数保存为 JSON 文件
    tmp = args.__dict__  # 获取 args 的字典形式
    json.dump(tmp, f, indent=1)  # 写入 JSON 文件
    print(args)  # 打印参数
