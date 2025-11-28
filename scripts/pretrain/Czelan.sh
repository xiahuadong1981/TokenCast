#!/bin/bash    # 指定脚本使用 Bash 解释器

# ========== 实验日志与模型保存目录 ==========
LOG_DIR="log_vq_sft"    # 日志文件夹名称
mkdir -p "$LOG_DIR"     # 若不存在则创建日志目录

BASE_CHECKPOINT_DIR="checkpoints_vq_sft"   # 模型 checkpoint 根目录
mkdir -p "$BASE_CHECKPOINT_DIR"            # 若不存在则创建 checkpoint 目录

# ========== 你可以自由修改的变量 ==========
LEARNING_RATES=(1e-5)   # 学习率列表（可多组实验）
EPOCHS_LIST=(2)         # epoch 列表（可多组实验）
VQ_TYPE="ResidualVQ"    # VQ 类型（如 ResidualVQ 或其他 VQ 模型类型）
FROZEN=0                # 是否冻结除输出层/embedding 外的所有参数（1=冻结，0=不冻结）

# 固定参数
PRED_LEN=24             # 预测长度（未来预测多少步）
SEQ_LEN=96              # 输入序列长度
Token_LEN=16            # Tokenizer 生成的 token 序列长度
D_MODEL=64              # Transformer 模型的 d_model 维度
N_EMBED=256             # embedding/codebook 维度

BATCH_SIZE=4            # batch 大小
ELECT_RATE=1            # elect_rate（用于 elect token 的策略）
PRETRAIN_LR=1e-3        # 预训练学习率
DEVICES="0,2"           # 指定使用 GPU 0 和 2

# ========== 显式指定你想用的 GPU ==========
export CUDA_VISIBLE_DEVICES=$DEVICES   # 设置可见 GPU，仅使用选定的 GPU

# ========== 启动实验 ==========
for LR in "${LEARNING_RATES[@]}"; do      # 遍历学习率组合
    for EPOCHS in "${EPOCHS_LIST[@]}"; do # 遍历 epoch 组合

        VQVAE_PATH="./TSTokenizer/checkpoints/CzeLan_96_dm64_dr0.2_emb256_wl4_bl2_ResidualVQ_unfreeze_codebook"   # VQ-VAE tokenizer 模型路径
                
        CHECKPOINT_DIR="${BASE_CHECKPOINT_DIR}/pred_${PRED_LEN}_seq_${SEQ_LEN}/lr_${LR}_ep_${EPOCHS}_vq_${VQ_TYPE}_chat_mask_64_pretrain_frozen_${FROZEN}"  # 当前实验的 checkpoint 保存路径

        LOG_FILE="$LOG_DIR/experiment_pred_${PRED_LEN}_seq${SEQ_LEN}_lr_${LR}_ep_${EPOCHS}_$(date +'%Y%m%d_%H%M%S')_pretrain_frozen_${FROZEN}.log"   # 每次实验生成独立的日志文件

        echo "🔹 Running experiment with lr=$LR, epochs=$EPOCHS, frozen=$FROZEN on GPUs $DEVICES"   # 输出当前实验参数

        accelerate launch \                       # 使用 accelerate 启动多 GPU 训练
            --multi_gpu \                         # 启用多 GPU
            --num_processes 2 \                   # 使用两个进程（与 GPU 数对应）
            --main_process_port 29600 \           # 主进程通信端口
            run.py \                              # 主训练脚本
            --is_training 1 \                     # 启用训练模式
            --pretrain 1 \                        # 开启预训练阶段
            --shuffle 0 \                         # 数据是否 shuffle（0=否）
            --batch_size "$BATCH_SIZE" \          # batch 大小
            --data CzeLan \                       # 数据集名称
            --root_path "/data/tinyy/first/CrossTimeNet/dataset" \   # 数据集根路径
            --data_path "CzeLan.csv" \            # 数据文件名称
            --pred_len "$PRED_LEN" \              # 预测长度
            --seq_len "$SEQ_LEN" \                # 输入序列长度
            --token_len "$Token_LEN" \            # token 序列长度
            --n_embed "$N_EMBED" \                # embedding 维度
            --d_model "$D_MODEL" \                # Transformer 的隐藏维度
            --learning_rate "$LR" \               # 学习率
            --weight_decay 0 \                    # 权重衰减
            --model "qwen4ts" \                   # 模型名称（自定义 qwen4ts）
            --task_name "long_term_forecast_bert_v4" \   # 任务名称
            --vqvae_model_path "$VQVAE_PATH" \    # 指定 VQ-VAE 模型路径
            --dropout 0.1 \                       # dropout 比例
            --chan_indep 0 \                      # 是否通道独立（0=否）
            --enc_in 11 \                         # 输入特征维度
            --feat_dim 11 \                       # 特征维度
            --local_model_path "/data/tinyy/first/CrossTimeNet/2-models/Qwen2.5-0.5B" \  # LLM 本地路径
            --pretrained_model "checkpoints_pretrain/pred_720_seq_512/lr_1e-5_ep_10_vq_ResidualVQ_chat_mask_64_pretrain_frozen_0_0.5/long_term_forecast_bert_v4_ETTh1_qwen4ts_720_ResidualVQ/checkpoint.pth" \  # 预训练模型路径
            --frozen "$FROZEN" \                  # 是否冻结部分参数
            --zero 1 \                            # zero-shot 参数？自定义
            --layers 1 \                          # 模型层数
            --params 1 \                          # 调整模型参数规模
            --wave_length 4 \                     # wave embedding 相关参数
            --checkpoints "$CHECKPOINT_DIR" \     # checkpoint 保存路径
            --seed 42 \                           # 随机种子
            --init_method "word" \                # embedding 初始化方式
            --train_epochs "$EPOCHS" \            # 训练 epoch 数
            --pretrain_lr "$PRETRAIN_LR" \        # 预训练学习率
            --use_multi_gpu \                     # 启用多 GPU
            --elect_rate "$ELECT_RATE" \          # elect rate
            --VQ_type "$VQ_TYPE" \                # VQ 类型
            --accumulation_steps 4 \              # 梯度累积步数
            --test 0 \                            # 是否只测试（0=训练）
            > "$LOG_FILE" 2>&1                    # 将输出与错误写入日志文件
    done
done
