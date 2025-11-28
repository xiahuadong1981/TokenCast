import torch                                                    # 导入 PyTorch


class TriangularCausalMask():                                   # 定义三角因果 Mask（用于自回归注意力）
    def __init__(self, B, L, device="cpu"):                     # B=batch size, L=序列长度
        mask_shape = [B, 1, L, L]                               # Mask 形状为 [B, 1, L, L]（多头前共享）
        with torch.no_grad():                                   # 不需要梯度（只是构造常量 mask）
            self._mask = torch.triu(                            # 生成上三角矩阵（右上三角为 True）
                torch.ones(mask_shape, dtype=torch.bool),       # 全 1 的布尔矩阵
                diagonal=1                                      # diagonal=1 表示主对角线以上为 True
            ).to(device)                                        # 放到 GPU/CPU 上

    @property
    def mask(self):                                             # 提供 mask 属性接口
        return self._mask                                       # 返回构造好的上三角 Mask（不可访问未来信息）


class ProbMask():                                               # 定义概率 Mask（用于 ProbSparse Attention）
    def __init__(self, B, H, L, index, scores, device="cpu"):   # B=batch, H=heads, L=seq_len, index=采样位置
        _mask = torch.ones(L, scores.shape[-1],                 # 先创建二维全 True 的布尔矩阵 (L, attn_len)
                            dtype=torch.bool                    # dtype=bool
                           ).to(device).triu(1)                 # 取上三角（未来位置为 True）

        _mask_ex = _mask[None, None, :].expand(                 # 扩展成 (B, H, L, attn_len)
            B, H, L, scores.shape[-1]                           # 每个 batch 和 head 都用同样的 mask 模板
        )

        indicator = _mask_ex[                                   # 按照 index 选择对应的行
            torch.arange(B)[:, None, None],                     # B 维度索引
            torch.arange(H)[None, :, None],                     # H 维度索引
            index, :                                            # L 的动态索引（ProbSparse 选出的 query 位置）
        ].to(device)

        self._mask = indicator.view(scores.shape).to(device)    # reshape 成与 scores 相同的形状作为最终 Mask

    @property
    def mask(self):                                             # mask 属性接口
        return self._mask                                       # 返回构造好的 ProbSparse Mask
