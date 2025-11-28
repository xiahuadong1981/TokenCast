#!/bin/bash  # 指定脚本由 bash 解释器执行

# ==== 主目录配置 ====
output_base="exp_FRED_MD"  # 🧩 统一父目录，用于保存所有实验结果

# ==== 基本配置 ====
root_path_name=/data/tinyy/first/CrossTimeNet/dataset  # 数据根目录
data_path_name=FRED-MD.csv  # 数据文件名
model_id_name=FRED-MD  # 模型标识名
data_name=FRED-MD  # 数据集名称（传入 main.py）

wave_length=4  # 波长超参数（任务相关）
seq_len=36  # 输入序列长度（脚本中未直接使用）
token_len=96  # token 的长度，用于 VQ tokenizer
pred_len=60  # 预测长度
vq_model='ResidualVQ'  # 使用的 VQ 模型（ResidualVQ）
block_num=2  # 模型 block 数量

# ==== 搜索空间 ====
n_embeds=(64 128 256)  # embedding 维度搜索列表
d_models=(96 128)  # d_model 搜索列表
entropy_penalties=(0.5)  # 熵惩罚系数搜索列表
entropy_temps=(0.5)  # 熵温度超参数搜索列表

for n_embed in "${n_embeds[@]}"; do  # 遍历 embedding 维度
for d_model in "${d_models[@]}"; do  # 遍历模型隐藏维度
for entropy_penalty in "${entropy_penalties[@]}"; do  # 遍历熵惩罚系数
for entropy_temp in "${entropy_temps[@]}"; do  # 遍历熵温度系数

# 🏷️ 唯一标识每组参数
tag="emb${n_embed}_d${d_model}_ep${entropy_penalty}_et${entropy_temp}_wl${wave_length}_bl${block_num}_${vq_model}"  # 组合所有超参数生成唯一标签

# 🗂️ 构建日志和模型保存路径
log_dir="${output_base}/logs/${model_id_name}/${tag}"  # 日志保存目录（未创建）
ckpt_dir="${output_base}/checkpoints/${model_id_name}/${tag}"  # 模型 checkpoint 保存目录
# mkdir -p "$log_dir"  # 日志目录创建被注释掉了
mkdir -p "$ckpt_dir"  # 创建 checkpoint 目录

log_file="${log_dir}.log"  # 日志文件路径（未创建目录情况下可能不存在）


python -u main.py \  # 不使用输出缓存执行 main.py
    --is_training 1 \  # 启用训练模式
    --vq_model $vq_model \  # 指定 VQ 模型类型
    --root_path $root_path_name \  # 数据集根路径
    --data_path $data_path_name \  # 数据文件
    --data $data_name \  # 数据集名称
    --wave_length $wave_length \  # 波长参数
    --features M \  # 多变量特征
    --token_len $token_len \  # token 长度
    --n_embed $n_embed \  # embedding 维度
    --chan_indep 0 \  # 通道是否独立（0 = 否）
    --enc_in 107 \  # 输入通道维度
    --pred_len $pred_len \  # 预测长度
    --d_model $d_model \  # 模型隐藏维度
    --block_num $block_num \  # block 数量
    --dropout 0.2 \  # dropout 比例
    --num_epoch 30 \  # 训练 epoch 数
    --entropy_penalty $entropy_penalty \  # 熵惩罚超参数
    --entropy_temp $entropy_temp \  # 熵温度系数
    --eval_per_epoch \  # 每个 epoch 验证一次
    --train_batch_size 4 \  # 训练批大小
    --test_batch_size 4 \  # 测试批大小
    --lr 0.0001 \  # 初始学习率
    --save_path $ckpt_dir \  # 模型保存路径
    > $log_file  # 将所有输出写入日志文件（若目录不存在会失败）

done  # 结束 entropy_temp 循环
done  # 结束 entropy_penalty 循环
done  # 结束 d_model 循环
done  # 结束 n_embed 循环
