#!/bin/bash  # 指定脚本使用 bash 解释器执行

# ========== 环境设置 ==========
export CUDA_VISIBLE_DEVICES="3"  # 使用 GPU 3
export CUDA_HOME=/usr/local/cuda  # CUDA 安装目录
export PATH=$CUDA_HOME/bin:$PATH  # 将 CUDA 的 bin 加入 PATH
source ~/.bashrc  # 加载用户环境配置，使环境变量生效


# ========== 数据与基本配置 ==========
root_path_name='./datasets'  # 数据集根目录
data_path_name=CzeLan.csv  # 数据文件名
model_id_name=CzeLan  # 模型 ID
data_name=CzeLan  # 数据集名称（传入 main.py）

wave_length=4  # 波长参数（任务相关超参）
seq_len=96  # 输入序列长度
token_len=96  # token 长度（VQ 输入长度）
pred_len=192  # 预测步数
vq_model='ResidualVQ'  # 选择的 VQ 模型（可选 VanillaVQ / SimVQ）
block_num=2  # 模型的 block 数量
enc_in=11  # 输入变量维度（通道数）
train_batch_size=256  # 训练 batch size
test_batch_size=256  # 测试 batch size
num_epoch=30  # 总训练 epoch 数
lr=0.001  # 学习率

n_embeds=(64 128 256 512)  # embedding 列表，用于网格搜索
d_models=(64)  # d_model 列表，用于网格搜索

# ========== 日志与检查点目录结构 ==========
BASE_LOG_DIR=$model_id_name/logs  # 日志保存目录
BASE_CKPT_DIR=$model_id_name/checkpoints  # 模型 checkpoint 保存目录
mkdir -p $BASE_LOG_DIR  # 创建日志目录（若不存在则创建）
mkdir -p $BASE_CKPT_DIR  # 创建 checkpoint 目录

# ========== 启动网格搜索训练 ==========
for n_embed in "${n_embeds[@]}"; do  # 遍历所有 embedding 数量
for d_model in "${d_models[@]}"; do  # 遍历所有 d_model 配置

    # 保存路径（用于模型、日志）
    combo_name="emb${n_embed}_d${d_model}_wl${wave_length}_bl${block_num}_${vq_model}"  # 实验组合名
    save_path="${BASE_CKPT_DIR}/${combo_name}"  # checkpoint 保存路径
    log_file="${BASE_LOG_DIR}/${combo_name}.log"  # 日志文件路径
    mkdir -p "$save_path"  # 创建保存目录

    echo "🚀 Running config: $combo_name"  # 输出当前运行的配置
    
    python -u main.py \  # 无缓存输出执行 main.py
        --is_training 1 \  # 启用训练模式
        --vq_model $vq_model \  # 指定使用的 VQ 模型
        --root_path $root_path_name \  # 数据根路径
        --data_path $data_path_name \  # 数据文件名
        --data $data_name \  # 数据集名称
        --wave_length $wave_length \  # 波长超参
        --features M \  # 多变量特征（M=multivariate）
        --token_len $token_len \  # token 长度
        --n_embed $n_embed \  # embedding 数量
        --chan_indep 0 \  # 是否通道独立处理（0=否）
        --enc_in $enc_in \  # 输入通道数
        --pred_len $pred_len \  # 预测长度
        --d_model $d_model \  # Transformer 中的隐藏维度 d_model
        --block_num $block_num \  # 模型 block 数量
        --dropout 0.2 \  # dropout 比例
        --num_epoch $num_epoch \  # 总训练 epoch 数
        --eval_per_epoch \  # 每个 epoch 评估一次
        --train_batch_size $train_batch_size \  # 训练 batch size
        --test_batch_size $test_batch_size \  # 测试 batch size
        --save_path "$save_path" \  # checkpoint 保存路径
        --lr $lr \  # 学习率
        > "$log_file"  # 将所有输出写入日志文件
        done  # 结束 d_model 循环
done  # 结束 n_embed 循环
