import math
from turtle import forward
import torch
import torch.nn as nn

class SelfAttentionV1(nn.Module):
    def __init__(self, hidden_dim: int = 768) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        # 三个线性变换：query、key、value
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj   = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x 的形状假设为 (batch_size, seq_len, hidden_dim)
        输出的形状也是 (batch_size, seq_len, hidden_dim)
        """
        # 先做线性映射
        Q = self.query_proj(x)  # (B, L, D)
        K = self.key_proj(x)    # (B, L, D)
        V = self.value_proj(x)  # (B, L, D)

        # 计算注意力得分：Q @ K^T，得到 (B, L, L)
        # 注意：transpose 的正确写法是 transpose(-1, -2)
        scores = torch.matmul(Q, K.transpose(-1, -2))  # (B, L, D) @ (B, D, L) -> (B, L, L)

        # 缩放：除以 sqrt(D)
        scaled_scores = scores / math.sqrt(self.hidden_dim)  # (B, L, L)

        # 计算注意力权重
        attn_weights = scaled_scores.softmax(dim=-1)  # (B, L, L)

        # 最终输出：权重 @ V，得到 (B, L, D)
        output = torch.matmul(attn_weights, V)  # (B, L, L) @ (B, L, D) -> (B, L, D)
        return output


# 测试一下：
if __name__ == "__main__":
    X = torch.rand(3, 2, 4)          # 假设 batch_size=3, seq_len=2, hidden_dim=4
    self_att_net = SelfAttentionV1(hidden_dim=4)
    out = self_att_net(X)
    print("输入形状:", X.shape)      # torch.Size([3, 2, 4])
    print("输出形状:", out.shape)    # torch.Size([3, 2, 4])















