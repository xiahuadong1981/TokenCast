#!/bin/bash   # 指定使用 bash 解释器执行脚本

# ========== 实验日志与模型保存目录 ==========
LOG_DIR="log_vq_sft"   # 训练日志输出目录
mkdir -p "$LOG_DIR"    # 若目录不存在则创建

BASE_CHECKPOINT_DIR="checkpoints_vq_sft"   # 模型检查点保存目录
mkdir -p "$BASE_CHECKPOINT_DIR"            # 若目录不存在则创建

# ========== 你可以自由修改的变量 ==========
LEARNING_RATES=(1e-5)     # 学习率列表，可循环实验多个学习率
EPOCHS_LIST=(10)          # 训练轮数列表，可循环实验多个 epoch 数
VQ_TYPE="ResidualVQ"      # 使用的 VQ 类型
FROZEN=0                  # 是否冻结 encoder（除输出层和嵌入层外）1=冻结, 0=不冻结

# 固定参数
PRED_LEN=24        # 预测长度
SEQ_LEN=96         # 输入序列长度
Token_LEN=16       # Token 序列长度
D_MODEL=64         # 模型隐藏维度 d_model
N_EMBED=256        # embedding 维度

BATCH_SIZE=4       # batch 大小
ELECT_RATE=1       # elect 率（可能为 token 选取比例）
PRETRAIN_LR=1e-3   # 预训练阶段的学习率
DEVICES="0,2"      # 指定使用 GPU 0 和 GPU 2

# ========== 显式指定你想用的 GPU ==========
export CUDA_VISIBLE_DEVICES=$DEVICES   # 设置当前脚本中可见 GPU

# ========== 启动实验 ==========
for LR in "${LEARNING_RATES[@]}"; do     # 遍历学习率列表
    for EPOCHS in "${EPOCHS_LIST[@]}"; do    # 遍历 epoch 列表

        VQVAE_PATH="./TSTokenizer/checkpoints/CzeLan_96_dm64_dr0.2_emb256_wl4_bl2_ResidualVQ_unfreeze_codebook"   # 已训练好的 VQ-VAE 模型路径
                
        CHECKPOINT_DIR="${BASE_CHECKPOINT_DIR}/pred_${PRED_LEN}_seq_${SEQ_LEN}/lr_${LR}_ep_${EPOCHS}_vq_${VQ_TYPE}_chat_mask_64_pretrain_frozen_${FROZEN}"   # 动态生成 checkpoint 路径
        LOG_FILE="$LOG_DIR/experiment_pred_${PRED_LEN}_seq${SEQ_LEN}_lr_${LR}_ep_${EPOCHS}_$(date +'%Y%m%d_%H%M%S')_pretrain_frozen_${FROZEN}.log"   # 日志文件路径（带时间戳）

        echo "🔹 Running experiment with lr=$LR, epochs=$EPOCHS, frozen=$FROZEN on GPUs $DEVICES"   # 打印当前实验配置

        accelerate launch \                       # 使用 accelerate 启动分布式训练
            --multi_gpu \                         # 开启多 GPU
            --num_processes 2 \                   # 总共启 2 个训练进程
            --main_process_port 29600 \           # 主进程通讯端口
            run.py \                              # 启动的训练脚本
            --is_training 1 \                     # 是否训练模式
            --pretrain 0 \                        # 是否进行预训练（0=否）
            --shuffle 0 \                         # 数据是否 shuffle（0=否）
            --batch_size "$BATCH_SIZE" \          # batch size
            --data CzeLan \                       # 数据集名称
            --root_path "/data/tinyy/first/CrossTimeNet/dataset" \   # 数据集路径
            --data_path "CzeLan.csv" \            # 数据文件
            --pred_len "$PRED_LEN" \              # 预测长度
            --seq_len "$SEQ_LEN" \                # 输入序列长度
            --token_len "$Token_LEN" \            # token 序列长度
            --n_embed "$N_EMBED" \                # embedding 大小
            --d_model "$D_MODEL" \                # 模型 d_model
            --learning_rate "$LR" \               # 学习率
            --weight_decay 0 \                    # 权重衰减
            --model "qwen4ts" \                   # 选择模型结构
            --task_name "long_term_forecast_bert_v4" \   # 任务名称
            --vqvae_model_path "$VQVAE_PATH" \    # VQ-VAE 模型路径
            --dropout 0.1 \                       # dropout 比例
            --chan_indep 0 \                      # 通道是否独立
            --enc_in 11 \                         # 输入特征维度
            --feat_dim 11 \                       # 特征维度
            --local_model_path "/data/tinyy/first/CrossTimeNet/2-models/Qwen2.5-0.5B" \   # 本地基础模型路径
            --pretrained_model "checkpoints_pretrain/pred_720_seq_512/lr_1e-5_ep_10_vq_ResidualVQ_chat_mask_64_pretrain_frozen_0_0.5/long_term_forecast_bert_v4_ETTh1_qwen4ts_720_ResidualVQ/checkpoint.pth" \   # 预训练模型路径
            --frozen "$FROZEN" \                  # 是否冻结模型
            --zero 1 \                            # 是否启用 zero 技术
            --layers 0 \                          # 使用多少层（0=默认？）
            --params 1 \                          # 参数量设置
            --wave_length 4 \                     # 波长参数（用于时间编码）
            --checkpoints "$CHECKPOINT_DIR" \     # checkpoint 输出路径
            --seed 42 \                           # 随机种子
            --init_method "word" \                # 初始化方式
            --train_epochs "$EPOCHS" \            # 训练轮数
            --pretrain_lr "$PRETRAIN_LR" \        # 预训练 lr
            --use_multi_gpu \                     # 使用多 GPU
            --elect_rate "$ELECT_RATE" \          # elect 率
            --VQ_type "$VQ_TYPE" \                # VQ 类型
            --accumulation_steps 4 \              # 梯度累积步数
            --test 0 \                            # 是否测试模式（0=否）
            > "$LOG_FILE" 2>&1                    # 将所有输出重定向到日志文件
    done
done
