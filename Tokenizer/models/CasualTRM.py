import torch                                # 导入 PyTorch 主库
from torch import nn                         # 导入神经网络模块
# from layers.Transformer_EncDec import Encoder, EncoderLayer   # 可选：旧版 Transformer 编解码器
import torch.nn.functional as F              # 导入函数式 API：如 softmax、relu

HAS_FLASH = False                            # 是否启用 Flash Attention（根据环境）

# Rotary Positional Embedding（旋转位置编码）
def build_rope_cache(max_seq_len, head_dim, device):            # 构建 RoPE 缓存
    # head_dim must be even for RoPE
    half_dim = head_dim // 2                                   # 一半的维度（偶数）
    freq_seq = torch.arange(half_dim, dtype=torch.float32, device=device)  # 频率序列
    inv_freq = 1.0 / (10000 ** (freq_seq / half_dim))          # 逆频率
    t = torch.arange(max_seq_len, dtype=torch.float32, device=device)      # 时间步
    freqs = torch.outer(t, inv_freq)                           # 外积生成频率矩阵
    emb = torch.cat((freqs, freqs), dim=-1)                    # 拼接得到完整维度
    cos = emb.cos()[None, :, None, :]                          # 计算 cos 部分
    sin = emb.sin()[None, :, None, :]                          # 计算 sin 部分
    return cos, sin                                            # 返回 cos/sin 缓存

def apply_rope(x, cos, sin):                                   # 对 Q/K 应用 RoPE
    # x: [B, seq_len, H, head_dim]
    # Split even/odd dims
    x_even = x[..., ::2]                                       # 偶数维
    x_odd = x[..., 1::2]                                       # 奇数维
    cos_even = cos[..., ::2]                                   # cos 偶数维
    sin_even = sin[..., ::2]                                   # sin 偶数维
    # Rotate
    x_even_rot = x_even * cos_even - x_odd * sin_even          # 旋转公式一
    x_odd_rot = x_even * sin_even + x_odd * cos_even           # 旋转公式二
    # Interleave back
    x_rot = torch.stack([x_even_rot, x_odd_rot], dim=-1)       # 合并奇偶
    x_rot = x_rot.flatten(-2)                                  # 展开成 head_dim
    return x_rot                                               # 返回旋转后的向量

class AttentionLayer(nn.Module):                               # 自注意力层
    def __init__(self, dim, n_heads, dropout=0.1,
                 use_flash=True, use_rope=True, max_seq_len=2048):
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError(f"Embedding dim {dim} must be divisible by n_heads {n_heads}")  # 检查维度
        head_dim = dim // n_heads                             # 每头的维度
        if use_rope and (head_dim % 2 != 0):
            raise ValueError(f"head_dim ({head_dim}) must be even for RoPE")  # RoPE 需偶数维度
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5                          # √d 缩放
        self.use_flash = use_flash and HAS_FLASH               # 是否真的用 flash attention
        self.use_rope = use_rope                               # 是否启用 RoPE
        self.max_seq_len = max_seq_len

        self.qkv_proj = nn.Linear(dim, dim * 3)                # QKV投影
        self.out_proj = nn.Linear(dim, dim)                    # 输出投影
        self.out_dropout = nn.Dropout(dropout)                 # 输出 dropout
        self.dropout = dropout

        # 注册 RoPE 缓存
        self.register_buffer('cos_cache', None, persistent=False)  # cos缓存
        self.register_buffer('sin_cache', None, persistent=False)  # sin缓存

    def _update_rope_cache(self, seq_len, device):              # 更新 RoPE 缓存
        if self.cos_cache is None or self.cos_cache.shape[1] < seq_len:
            cos, sin = build_rope_cache(self.max_seq_len,
                                        self.head_dim,
                                        device)                # 构建缓存
            self.cos_cache = cos                                # 更新缓存
            self.sin_cache = sin

    def forward(self, x, past_kv=None, attention_mask=None, use_cache=False):    # 前向传播
        """
        x: [B, T, dim]                                          # 输入 embedding
        past_kv: Tuple(k, v) or None                            # KV 缓存
        attention_mask: BoolTensor [B, T]                       # mask
        use_cache: 是否返回新 KV
        """
        B, T, _ = x.shape
        H, Dh = self.n_heads, self.head_dim
        device = x.device

        qkv = self.qkv_proj(x).view(B, T, 3, H, Dh)            # QKV 投影
        q, k, v = qkv.unbind(2)                                # 拆分成三份

        # RoPE if enabled
        if self.use_rope:
            past_len = past_kv[0].shape[1] if past_kv is not None else 0  # 之前长度
            total_len = past_len + T                                     # 总长度
            self._update_rope_cache(total_len, device)                   # 更新缓存
            cos_all = self.cos_cache[:, :total_len, :, :]               # 截取 cos
            sin_all = self.sin_cache[:, :total_len, :, :]               # 截取 sin
            # apply to k and q
            k = apply_rope(k, cos_all[:, :k.shape[1]], sin_all[:, :k.shape[1]])  # 对 k
            q = apply_rope(q,
                           cos_all[:, past_len:past_len+T],              # 对 q
                           sin_all[:, past_len:past_len+T])

        # concatenate past K/V
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=1)                  # KV 拼接
            v = torch.cat([past_v, v], dim=1)

        # attention
        if self.use_flash:
            out = flash_attn_func(                             # 使用 Flash Attention
                q, k, v,
                dropout_p=self.dropout if self.training else 0.0,
                causal=True
            )
        else:
            q_ = q.permute(0, 2, 1, 3)                         # 调整维度
            k_ = k.permute(0, 2, 1, 3)
            v_ = v.permute(0, 2, 1, 3)

            scores = (q_ @ k_.transpose(-2, -1)) * self.scale  # 注意力分数
            # causal mask only in non-cache
            if not use_cache:
                key_len = k.shape[1]                           # K 的总长度
                causal_mask = torch.arange(key_len, device=device)[None, :] <= (
                    torch.arange(T, device=device)[:, None]
                )                                              # 构建下三角 mask
                scores = scores.masked_fill(~causal_mask[None, None, :, :], float('-inf'))
            # padding mask
            if attention_mask is not None and not use_cache:
                mask2 = attention_mask[:, None, None, :].expand(B, 1, T, k.shape[1])  # padding mask
                scores = scores.masked_fill(~mask2, float('-inf'))

            probs = F.softmax(scores, dim=-1)                  # softmax
            out = probs @ v_                                   # 加权求和
            out = out.permute(0, 2, 1, 3)

        out = out.reshape(B, T, H * Dh)                        # 合并 heads
        out = self.out_proj(out)                               # 输出线性层
        out = self.out_dropout(out)                            # dropout

        return out, (k, v) if use_cache else None              # 返回结果

class CasualLayer(nn.Module):                                  # Causal Transformer Block
    def __init__(self, dim, n_heads, dropout=0.1,
                 ffn_hidden=None, use_flash=True,
                 use_rope=True, max_seq_len=2048):
        super().__init__()
        ffn_hidden = ffn_hidden or dim * 4                     # FFN 隐藏层大小
        self.ln1 = nn.LayerNorm(dim)                           # LayerNorm1
        self.attn = AttentionLayer(dim, n_heads,
                                   dropout, use_flash,
                                   use_rope, max_seq_len)      # 自注意力
        self.ln2 = nn.LayerNorm(dim)                           # LayerNorm2
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_hidden),                        # FFN 前半部分
            nn.GELU(),
            nn.Linear(ffn_hidden, dim),                        # FFN 输出
            nn.Dropout(dropout)
        )

    def forward(self, x, past_kv=None, attention_mask=None, use_cache=False):
        norm_x = self.ln1(x)                                   # LN
        attn_out, new_kv = self.attn(norm_x,
                                     past_kv=past_kv,
                                     attention_mask=attention_mask,
                                     use_cache=use_cache)      # 注意力
        x = x + attn_out                                       # 残差连接
        ffn_out = self.ffn(self.ln2(x))                        # FFN
        x = x + ffn_out                                        # 残差连接
        return x, new_kv

class CasualTRM(nn.Module):                                    # 整个 Transformer 模型
    def __init__(self, dim=768, d_ff=3072, n_heads=12, n_layers=12,
                 max_seq_len=2048, dropout=0.1,
                 use_flash=True, use_rope=True):
        super(CasualTRM, self).__init__()
        if dim % n_heads != 0:
            raise ValueError(f"Embedding dim {dim} must be divisible by n_heads {n_heads}")
        self.use_rope = use_rope
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, dim)) if not use_rope else None  # 可选位置编码
        self.layers = nn.ModuleList([
            CasualLayer(dim, n_heads, dropout,
                         d_ff, use_flash,
                         use_rope, max_seq_len)               # N 层 Transformer
            for _ in range(n_layers)
        ])
        self.apply(self._init_weights)

    def _init_weights(self, module):                           # 参数初始化
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)           # 线性层初始化
            if module.bias is not None:
                nn.init.zeros_(module.bias)                    # 偏置初始化
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)           # embedding 初始化

    def forward(self, inputs_embeds=None,
                attention_mask=None, past_kv_list=None,
                use_cache=False):

        if use_cache and past_kv_list is not None:
            inputs_embeds = inputs_embeds[:, -1:, :].contiguous()  # 仅使用最后一个 token
        x = inputs_embeds

        B, T, _ = x.shape
        if not self.use_rope:                                  # 不使用 RoPE 则加位置编码
            if use_cache and past_kv_list is not None:
                past_len = past_kv_list[0][0].size(1)
                pos_embed = self.pos_embed[:, past_len:past_len + T, :]
            else:
                pos_embed = self.pos_embed[:, :T, :]
            x = x + pos_embed

        new_past = []
        for i, layer in enumerate(self.layers):
            past_kv = past_kv_list[i] if past_kv_list else None
            x, new_kv = layer(x,
                              past_kv=past_kv,
                              attention_mask=attention_mask,
                              use_cache=use_cache)             # 每层执行
            if use_cache:
                new_past.append(new_kv)                       # 保存 KV

        return x, (new_past if use_cache else None)            # 输出结果
