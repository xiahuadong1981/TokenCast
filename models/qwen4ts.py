import torch  # 导入 PyTorch 主包
import torch.nn as nn  # 从 PyTorch 导入神经网络模块
from transformers import (  # 从 transformers 库中导入所需类
    AutoModelForCausalLM,  # 自动加载用于因果语言建模的预训练模型
    AutoTokenizer,  # 自动加载对应的分词器
    AutoConfig,  # 自动加载模型配置
    LogitsProcessorList, LogitsProcessor  # 生成时用于修改 logits 的处理器基类及列表容器
)
import random  # Python 内置随机数库
from peft import get_peft_model, LoraConfig, TaskType  # PEFT：加载 LoRA 配置并包装模型以支持参数高效微调

import torch  # 再次导入 torch（重复导入不会出错但可以省略）
import torch.nn as nn  # 再次导入 nn（同上）
import torch.nn.functional as F  # 导入函数式接口，一般用于损失函数等


class FocalLoss(nn.Module):  # 定义 Focal Loss 损失函数类，继承 nn.Module
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean', ignore_index=-100):  # 初始化超参数
        super(FocalLoss, self).__init__()  # 调用父类构造函数
        self.alpha = alpha  # 类别不均衡时的缩放系数
        self.gamma = gamma  # Focal Loss 中的聚焦因子 γ
        self.reduction = reduction  # 损失聚合方式：'mean'、'sum' 或 'none'
        self.ignore_index = ignore_index  # 指定忽略的标签值（用于 padding）

    def forward(self, inputs, targets, position_weights=None):  # 前向计算接口
        """
        inputs: (B, L, C) or (B, C) logits
        targets: (B, L) or (B,) with class indices, may include -100 for ignore
        position_weights: (B, L) or (B,) or None
        """
        if inputs.dim() == 3:  # 若输入为序列形式 (B, L, C)
            B, L, C = inputs.shape  # 解析 batch、序列长度和类别数
            inputs = inputs.reshape(B * L, C)  # 展平成 (B*L, C)，方便计算交叉熵
            targets = targets.reshape(B * L)  # 标签同样展平
            if position_weights is not None:  # 若有位置权重
                position_weights = position_weights.reshape(B * L)  # 同样展平
        else:  # 若输入为普通分类 (B, C)
            B, C = inputs.shape  # 解析 batch 和类别数
            targets = targets.reshape(B)  # 标签展平为 (B,)
            if position_weights is not None:  # 若有位置权重
                position_weights = position_weights.reshape(B)  # 展平

        valid_mask = (targets != self.ignore_index).float()  # 计算有效样本掩码，忽略 ignore_index 标签
        # Compute cross entropy (no reduction)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', ignore_index=self.ignore_index)  # (N,) 逐样本交叉熵损失，不做聚合
        pt = torch.exp(-ce_loss)  # p_t = exp(-CE)，即真实类别的预测概率
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss  # (N,) 计算 Focal Loss：α(1-p)^γ * CE

        # Apply optional weights
        if position_weights is not None:  # 如果提供了位置权重
            focal_loss = focal_loss * position_weights  # 对每个位置乘以对应权重

        # Reduction
        if self.reduction == 'mean':  # 若指定做平均
            denom = (valid_mask * (position_weights if position_weights is not None else 1.0)).sum()  # 有效样本加权计数作为分母
            return focal_loss.sum() / (denom + 1e-8)  # 防止除零
        elif self.reduction == 'sum':  # 若指定求和
            return focal_loss.sum()  # 返回总损失
        else:  # 'none'
            if inputs.dim() == 2:  # 对于 (B, C) 情况
                return focal_loss.view(B)  # 恢复到 (B,)
            else:  # 对于序列 (B, L, C) 情况
                return focal_loss.view(B, L)  # 恢复到 (B, L)


class TsTokenFormatController(LogitsProcessor):  # 自定义 logits 处理器，用于强约束时序 token 的生成格式
    def __init__(self, ts_token_range, ts_start_token_id, ts_end_token_id, ts_start_pos, ts_len):  # 初始化约束参数
        self.ts_token_start, self.ts_token_end = ts_token_range  # 时序 token 的 id 范围 [start, end)
        self.ts_start_token_id = ts_start_token_id  # <TS_START> 的 token id
        self.ts_end_token_id = ts_end_token_id  # <TS_END> 的 token id
        self.ts_start_pos = ts_start_pos  # 时序片段在生成序列中的起始位置（包含 <TS_START>）
        self.ts_end_pos = ts_start_pos + 1 + ts_len  # 时序片段结束位置（<TS_START> + ts_len 个时序 token 之后，位置等于写 <TS_END> 的位置）

    def __call__(self, input_ids, scores):  # 每一步生成时会被调用，修改 scores 后返回
        cur_len = input_ids.shape[1]  # 当前已生成的 token 长度（不含本步待采样 token）
       
        mask = torch.full_like(scores, float("-inf"))  # 初始化一个全为 -inf 的 mask，用来禁止不允许的 token

        if cur_len == self.ts_start_pos:  # 当到达时序片段的起始位置时
            mask[:, self.ts_start_token_id] = scores[:, self.ts_start_token_id]  # 仅允许生成 <TS_START> 这个 token
            return mask  # 返回被约束后的 logits

        elif self.ts_start_pos < cur_len < self.ts_end_pos:  # 在 <TS_START> 和 <TS_END> 之间的位置，生成的是时序 token
            mask[:, self.ts_token_start:self.ts_token_end] = scores[:, self.ts_token_start:self.ts_token_end]  # 仅允许在时序 token 范围内采样

            # 下面这行 topk 只是例子/调试，结果没有被使用
            topk = torch.topk(mask[:, self.ts_token_start:self.ts_token_end], k=5, dim=-1)  # 取前 5 个最大 logits（未实际用到）

            return mask  # 返回约束后的 logits

        elif cur_len == self.ts_end_pos:  # 当到达时序片段结束位置时
            mask[:, self.ts_end_token_id] = scores[:, self.ts_end_token_id]  # 仅允许生成 <TS_END> token
            return mask  # 返回约束后的 logits
        else:
            return scores  # 其他位置不做任何约束，返回原始 logits


class Model(nn.Module):  # 主模型类，包装 Qwen 语言模型并扩展时序 token 能力
    def __init__(self, configs):  # 初始化，configs 为自定义配置对象
        super(Model, self).__init__()  # 调用父类构造函数
        self.configs = configs  # 保存配置
        config = AutoConfig.from_pretrained(self.configs.local_model_path)  # 从本地路径加载模型配置
        self.d_model = config.hidden_size  # 模型隐藏层维度
        self.text_tokenizer = AutoTokenizer.from_pretrained(self.configs.local_model_path)  # 从本地路径加载分词器
        self.text_tokenizer.pad_token = self.text_tokenizer.eos_token  # 将 pad_token 设置为 eos_token，避免 pad 为空

        # 添加时序特殊token
        special_tokens_dict = {  # 定义需要添加的额外特殊 token
        'additional_special_tokens': ['<TS_START>', '<TS_END>']  # 用于标记时序片段的开始和结束
        }

        self.text_tokenizer.add_special_tokens(special_tokens_dict)  # 向 tokenizer 中注册这些特殊 token
        # self.text_tokenizer.apply_chat_template  # 预留：可选择应用 chat 模板（当前未使用）
        
        self.n_embed = self.configs.elected_n_embed  # 为时序 token 预留的 embedding 数量
        # 初始化Qwen模型
        self.model = self._initialize_model(config)  # 根据配置初始化 Qwen 模型
        # 初始化嵌入层
        self._initialize_embedding_layer()  # 扩展 embedding 以容纳时序 token 和特殊 token

        self._initialize_output_layer(config)  # 构建并替换输出层，使其与扩展后的 embedding 对齐

        if self.configs.layers:  # 若指定只训练部分层（分层微调）
            num_layers = len(self.model.model.layers)  # 获取 Transformer 总层数
            print(f"Qwen2.5 共有 {num_layers} 层 Transformer")  # 打印层数信息
            for param in self.model.model.parameters():  # 先将 backbone 所有参数冻结
                param.requires_grad = False
            n_unfreeze = self.configs.n_layers  # 从配置中读取需要解冻的层数（通常为末尾若干层）
            print(n_unfreeze)  # 打印解冻层数

            for i in range(num_layers - n_unfreeze, num_layers):  # 解冻最后 n_unfreeze 层 Transformer
                for param in self.model.model.layers[i].parameters():
                    param.requires_grad = True  # 允许这些层参与训练
            
            # ✅ Step 3: 解冻 embedding 层
            for param in self.model.model.embed_tokens.parameters():  # 解冻词嵌入层
                param.requires_grad = True

            # ✅ Step 4: 解冻输出层（lm_head）
            for param in self.model.lm_head.parameters():  # 解冻语言模型头部（输出层）
                param.requires_grad = True

        if self.configs.frozen:  # 如果配置要求整体冻结模型
            # 全部冻结
            for param in self.parameters():  # 先冻结当前模型所有参数
                param.requires_grad = False

            # 解冻嵌入层
            for param in self.model.model.embed_tokens.parameters():  # 仅解冻 embedding 层
                param.requires_grad = True

            # 解冻输出层
            for param in self.model.lm_head.parameters():  # 仅解冻输出层
                param.requires_grad = True

        # 如果启用 LoRA
        if self.configs.use_lora:
            print("🔧 Applying LoRA to model...")  # 打印提示信息
            lora_config = LoraConfig(  # 构造 LoRA 配置
                r=8,  # LoRA rank
                lora_alpha=32,  # LoRA 缩放因子
                lora_dropout=0.1,  # LoRA dropout 概率
                bias="none",  # 不对 bias 使用 LoRA
                task_type=TaskType.CAUSAL_LM,  # 因为是自回归语言模型任务
                target_modules=["q_proj", "v_proj"]  # 指定应用 LoRA 的子模块名称（Q/K/V 中的 Q/V）
            )
            self.model = get_peft_model(self.model, lora_config)  # 将原始模型包装为 LoRA 模型
            

    def _initialize_model(self, config):  # 内部方法：根据配置初始化模型
        if self.configs.params:  # 如果需要从预训练权重加载
            return AutoModelForCausalLM.from_pretrained(
                self.configs.local_model_path,  # 本地模型路径
                output_attentions=True,  # 在前向中输出注意力
                output_hidden_states=True,  # 在前向中输出各层隐藏状态
                trust_remote_code=True  # 信任远程自定义模型代码
            )
        else:  # 否则仅根据 config 从头初始化模型参数
            return AutoModelForCausalLM.from_config(config, trust_remote_code=True)

    def _initialize_embedding_layer(self, use_normal_dist=True):  # 内部方法：扩展并初始化 embedding 层
        original_weight = self.model.model.embed_tokens.weight  # 原始词嵌入权重矩阵 (V, d)
        self.original_len = len(original_weight)  # 原始词表大小 V

        # 🔸 获取 special token 数量
        special_tokens_len = len(self.text_tokenizer.additional_special_tokens)  # 额外特殊 token 的个数

        if use_normal_dist:  # 若采用高斯分布进行初始化
            mu = torch.mean(original_weight, dim=0)  # 计算原 embedding 的均值向量 μ
            n = original_weight.size()[0]  # 词表大小 n
            sigma = ((original_weight - mu).T @ (original_weight - mu)) / n  # 简单估计协方差矩阵 Σ
            dist = torch.distributions.multivariate_normal.MultivariateNormal(mu, covariance_matrix=1e-5*sigma)  # 构造多元高斯分布（协方差缩小 1e-5）

            ts_weight = torch.stack([dist.sample() for _ in range(self.n_embed)], dim=0)  # 为时序 token 采样 n_embed 个 embedding 向量
            special_tokens_weight = torch.stack([dist.sample() for _ in range(special_tokens_len)], dim=0)  # 为 special tokens 采样 embedding 向量
        else:  # 若采用从原 embedding 中随机采样的方式初始化
            random.seed(self.configs.seed)  # 固定随机种子，保证可复现
            sample_indices = random.sample(range(len(original_weight)), self.n_embed)  # 随机选择 n_embed 个索引
            ts_weight = original_weight[sample_indices]  # 选取对应的 embedding 作为时序 token 的权重

            special_indices = random.sample(range(len(original_weight)), special_tokens_len)  # 为 special tokens 随机选择若干索引
            special_tokens_weight = original_weight[special_indices]  # 直接复制对应 embedding

        # 🔸 扩展词表
        total_vocab_size = self.original_len + self.n_embed + special_tokens_len  # 扩容后的词表总大小
        self.model.resize_token_embeddings(total_vocab_size)  # 调整模型 embedding 和输出层的词表大小

        # 🔸 赋值新嵌入
        start_idx = self.original_len  # 时序 token 的起始索引
        end_idx = start_idx + self.n_embed  # 时序 token 的结束索引
        self.model.model.embed_tokens.weight.data[start_idx:end_idx] = ts_weight  # 将时序 token 的权重写入 embedding

        start_idx = end_idx  # special token 的起始索引
        end_idx = start_idx + special_tokens_len  # special token 的结束索引
        self.model.model.embed_tokens.weight.data[start_idx:end_idx] = special_tokens_weight  # 将 special token 的权重写入 embedding

        # 🔸 保存 embedding 权重以供输出层使用
        self.embedding_weight = self.model.model.embed_tokens.weight  # 保存共享的 embedding 权重引用


    def _initialize_output_layer(self, config):  # 内部方法：创建并替换输出层
        # 创建输出层，与embedding layer共享权重
        output_layer = nn.Linear(config.hidden_size, self.embedding_weight.size(0), bias=False)  # 线性层输出维度等于词表大小
        # 使用embedding layer的权重初始化输出层
        output_layer.weight.data = self.embedding_weight.data  # 直接共享 embedding 权重
        
        # 替换Qwen模型的输出层
        self.model.set_output_embeddings(output_layer)  # 将线性层注册为模型的输出 embeddings
        self.model.lm_head.weight = self.model.model.embed_tokens.weight  # 确保 lm_head 与 embed_tokens 权重绑定（权重共享）
        
    def forward(self, inputs):     # 前向传播接口，接收自定义 inputs 字典
        text_ids, input_ids, labels = inputs['text_ids'], inputs['ts_ids'], inputs['labels']  # 解析输入中的文本 id、时序 id 以及标签
        device = input_ids.device  # 获取当前张量所在设备

        # 构造 attention mask
        attention_mask = torch.ones(input_ids.shape[0], input_ids.shape[1], dtype=torch.float32, device=device)  # 简单地为非 padding 输入构造全 1 的注意力 mask（此处未考虑 padding）

        # 构造新 token 的位置权重（例如加大新引入 token 的损失权重）
        new_token_weight = self.configs.new_token_weight if hasattr(self.configs, 'new_token_weight') else 1  # 若配置中存在 new_token_weight，则使用它，否则默认为 1
        orig_token_weight = 1  # 原始词表 token 的权重为 1
        position_weights = torch.where(labels >= self.original_len, new_token_weight, orig_token_weight)  # 如果 label 的 id 超过原词表长度，认为是新 token，使用 new_token_weight
        ts_start_id = self.text_tokenizer.convert_tokens_to_ids("<TS_START>")  # 获取 <TS_START> 的 id（当前未直接使用）
        ts_end_id = self.text_tokenizer.convert_tokens_to_ids("<TS_END>")  # 获取 <TS_END> 的 id（当前未直接使用）

        # 🚀 正确地传 input_ids，别用 inputs_embeds！
        outputs = self.model(  # 调用底层 Qwen 模型进行前向传播
            input_ids=input_ids,  # 输入 token id 序列
            labels=labels,  # 语言模型训练标签序列（shifted 内部处理）
            attention_mask=attention_mask,  # 注意力 mask
            output_hidden_states=True  # 输出隐藏状态（便于调试或后续使用）
        )

        loss_fn = FocalLoss(alpha=1.0, gamma=2.0, reduction='mean')  # 构造 FocalLoss 损失函数
        loss = loss_fn(outputs.logits[..., :-1, :], labels[..., 1:], position_weights[..., 1:])  # 使用 teacher-forcing 方式手动对齐 logits 与标签（右移一个位置）
        outputs.loss = loss  # 将自定义的 loss 写回 outputs 对象，方便外部统一访问
        
        return outputs  # 返回包含 loss、logits 等信息的输出对象


    def gen_ts(self, inputs, text_token_len=112, ts_token_len=12):  # 根据输入生成时序 token 序列
        tokenizer = self.text_tokenizer  # 使用已扩展的 tokenizer
        device = next(self.model.parameters()).device  # 获取模型所在设备
        original_len = self.original_len  # 原始词表长度
        n_ts_token = self.n_embed  # 时序 token 个数
        ts_token_range = (original_len, original_len + n_ts_token)  # 时序 token 在新词表中的 id 范围

        input_ids = inputs['ts_ids']  # 获取输入序列（这里 ts_ids 实际上包含了文本 + 时序位置模板）

        device = next(self.model.parameters()).device  # 再次确定设备（可省略）
        attention_mask = (input_ids != tokenizer.pad_token_id).long().to(device)  # 根据 pad_token_id 构造 0/1 注意力 mask
        input_ids = input_ids.to(device)  # 将 input_ids 移到模型设备

        ts_end_token_id = tokenizer.convert_tokens_to_ids("<TS_END>")  # 获取 <TS_END> id
        ts_start_token_id = tokenizer.convert_tokens_to_ids("<TS_START>")  # 获取 <TS_START> id
        max_len = text_token_len + ts_token_len + 2  # 最大新生成 token 数（文本 + TS_START/END + ts_token_len 的上界）

        logits_processor = LogitsProcessorList([  # 构建 logits 处理器列表
            TsTokenFormatController(
                ts_token_range=ts_token_range,     # 假设时序token id是这个范围
                ts_start_token_id=ts_start_token_id,          # <TS_START>
                ts_end_token_id=ts_end_token_id,            # <TS_END>
                ts_start_pos=text_token_len + input_ids.shape[1],                  # 文本 token 为前 text_token_len 个；这里的写法需结合具体拼接方式理解
                ts_len=ts_token_len                     # 时序 token 输出长度
            )
        ])

        generated = self.model.generate(  # 使用 huggingface generate 接口进行自回归生成
            input_ids=input_ids,  # 初始输入序列（prompt）
            attention_mask=attention_mask,  # 对应的 attention mask
            max_new_tokens=max_len,  # 限制生成的新 token 数量
            eos_token_id=tokenizer.convert_tokens_to_ids("<|im_end|>"),  # 指定终止 token id
            pad_token_id=tokenizer.pad_token_id,  # 指定 padding token id
            return_dict_in_generate=True,  # 返回字典形式结果，包含 sequences、scores 等
            logits_processor=logits_processor  # 在生成过程中注入自定义 logits 约束
        )

        return generated.sequences[:, input_ids.shape[1]:]  # 只返回新生成的部分（去掉原始 prompt 部分）


    @staticmethod
    def init_weights_kaiming(m):  # 静态方法：对线性层使用 Kaiming 初始化
        if isinstance(m, nn.Linear):  # 判断模块类型是否为 Linear
            nn.init.kaiming_normal_(m.weight, nonlinearity="leaky_relu")  # 使用 Kaiming 正态初始化权重，适配 leaky_relu
            m.bias.data.fill_(0.01)  # 将 bias 初始化为常数 0.01
