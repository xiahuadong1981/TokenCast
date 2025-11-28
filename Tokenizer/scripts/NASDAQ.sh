#!/bin/bash  # 指定脚本使用 bash 解释器运行

# 创建统一的父目录以及子目录
mkdir -p ./exp_NASDAQ/logs/Tokenizer  # 创建日志保存目录（若不存在则递归创建）
mkdir -p ./exp_NASDAQ/checkpoints/Tokenizer  # 创建模型 checkpoint 保存目录

# 环境设置
export CUDA_VISIBLE_DEVICES="3"  # 指定使用 GPU 3
export CUDA_HOME=/usr/local/cuda  # CUDA 安装根目录
export PATH=$CUDA_HOME/bin:$PATH  # 将 CUDA 的 bin 路径加入系统 PATH
source ~/.bashrc  # 加载用户环境，确保 CUDA 等配置生效

# 固定参数
root_path_name='./datasets'  # 数据集根目录路径
data_path_name=NASDAQ.csv  # 数据文件名
model_id_name=NASDAQ  # 模型 ID，用于目录结构
data_name=NASDAQ  # 数据名称（传给 main.py）

wave_length=2  # 波长超参数（模型相关）
seq_len=36  # 输入序列长度（脚本中未直接使用）
token_len=96  # token 长度，用于 VQ tokenizer
pred_len=60  # 预测长度（未来步数）
vq_model='ResidualVQ'  # 使用的 VQ 模型（ResidualVQ）
block_num=2  # 模型 block 数量

# 可变参数列表
n_embeds=(64)  # embedding 维度列表（可用于超参搜索）
d_models=(64)  # Transformer 隐层维度列表（可用于超参搜索）

# 多参数组合实验
for d_model in "${d_models[@]}"; do  # 遍历 d_model 参数
    for n_embed in "${n_embeds[@]}"; do  # 遍历 n_embed 参数

                tag="${data_name}_emb${n_embed}_d${d_model}_wl${wave_length}_bl${block_num}"  # 生成本次实验的唯一标签名
                save_path="./exp_NASDAQ/checkpoints/${tag}"  # 模型保存目录
                log_file="./exp_NASDAQ/logs/${tag}.log"  # 日志文件路径

                mkdir -p $save_path  # 创建模型保存目录（避免不存在导致错误）

                python -u main.py \  # 无缓存输出方式运行 main.py
                    --is_training 1 \  # 启用训练模式
                    --vq_model $vq_model \  # 指定 VQ 模型种类
                    --root_path $root_path_name \  # 数据集根路径
                    --data_path $data_path_name \  # 数据文件名
                    --data $data_name \  # 数据名称，决定数据集类型
                    --wave_length $wave_length \  # 波长超参数
                    --features M \  # 使用多变量特征（M = multivariate）
                    --token_len $token_len \  # token 长度
                    --n_embed $n_embed \  # embedding 维度
                    --chan_indep 0 \  # 是否通道独立（0 = 否）
                    --enc_in 5 \  # 输入特征维度（NASDAQ 有 5 列）
                    --pred_len $pred_len \  # 预测长度
                    --d_model $d_model \  # Transformer 隐藏维度 d_model
                    --block_num $block_num \  # block 数量
                    --dropout 0.2 \  # dropout 比例
                    --num_epoch 30 \  # 训练轮数
                    --eval_per_epoch \  # 每个 epoch 进行验证
                    --train_batch_size 16 \  # 训练 batch 大小
                    --test_batch_size 16 \  # 测试 batch 大小
                    --save_path $save_path \  # 模型保存目录
                    --lr 0.0001 \  # 学习率
                    > $log_file  # 将输出写入日志文件
    done  # 结束 n_embed 循环
done  # 结束 d_model 循环
