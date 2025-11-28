#!/bin/bash  # 指定使用 bash 解释器

# 创建统一的父目录以及子目录
mkdir -p ./exp_results/logs/Tokenizer  # 创建日志目录（若不存在则递归创建）
mkdir -p ./exp_results/checkpoints/Tokenizer  # 创建模型检查点目录

# 环境设置
export CUDA_VISIBLE_DEVICES="1"  # 指定使用 GPU 1
export CUDA_HOME=/usr/local/cuda  # 设置 CUDA 主目录路径
export PATH=$CUDA_HOME/bin:$PATH  # 将 CUDA 的 bin 路径加入系统 PATH
source ~/.bashrc  # 加载用户环境变量，使设置生效

# 固定参数
root_path_name='./datasets'  # 数据集根目录
data_path_name=Covid-19.csv  # 数据文件名称
model_id_name=Covid-19  # 模型 ID 名称（未使用，但保留）
data_name=Covid-19  # 数据名称，传入 main.py

wave_length=2  # 波长参数（任务相关超参）
seq_len=36  # 输入序列长度（未在脚本中用到）
token_len=96  # token 长度，用于 VQ tokenizer
pred_len=60  # 预测长度
vq_model='ResidualVQ'  # VQ 模型类别
block_num=1  # 模型 block 数量

# 可变参数列表
n_embeds=(64)  # embedding 数量列表，可循环选用
d_models=(64)  # d_model 列表，可循环选用

# 多参数组合实验
for d_model in "${d_models[@]}"; do  # 遍历 d_model
    for n_embed in "${n_embeds[@]}"; do  # 遍历 n_embed
                tag="${data_name}_emb${n_embed}_d${d_model}_wl${wave_length}_bl${block_num}"  # 实验标签名
                save_path="./exp_results/checkpoints/${tag}"  # 保存模型路径
                log_file="./exp_results/logs/${tag}.log"  # 日志文件路径
                mkdir -p $save_path  # 创建模型保存目录
                python -u main.py \  # 无缓存输出运行 main.py
                    --is_training 1 \  # 训练模式
                    --vq_model $vq_model \  # 指定使用的 VQ 模型
                    --root_path $root_path_name \  # 数据根目录
                    --data_path $data_path_name \  # 数据文件名
                    --data $data_name \  # 数据集名称
                    --wave_length $wave_length \  # 波长超参
                    --features M \  # 多变量特征（M=multivariate）
                    --token_len $token_len \  # token 长度
                    --n_embed $n_embed \  # embedding 数量
                    --chan_indep 0 \  # 是否独立处理通道（0=否）
                    --enc_in 948 \  # 输入通道数（具体数据维度）
                    --pred_len $pred_len \  # 预测长度
                    --d_model $d_model \  # Transformer 隐层维度
                    --block_num $block_num \  # block 数量
                    --dropout 0.2 \  # dropout 比例
                    --num_epoch 30 \  # 训练总 epoch 数
                    --eval_per_epoch \  # 每个 epoch 后评估
                    --train_batch_size 4 \  # 训练 batch size
                    --test_batch_size 4 \  # 测试 batch size
                    --save_path $save_path \  # 模型保存路径
                    --lr 0.0001 \  # 学习率
                    > $log_file  # 将输出写入日志文件

            done  # 结束 n_embed 循环
        done  # 结束 d_model 循环
    done  # 多余的 done，多一个
done  # 多余的 done，再多一个
