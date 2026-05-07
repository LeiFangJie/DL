import torch
import torch.nn as nn
import torch.nn.functional as F

# ======================
# 版本1：直观版多头注意力（你最容易理解）
# ======================
class MultiHeadAttention_V1(nn.Module):
    def __init__(self, embed_size=8, hidden_size=16, num_heads=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads  # 每个头维度 8

        # ========== 重点：真·分开h个头 ==========
        # 头1 的 Wq, Wk, Wv
        self.wq1 = nn.Linear(embed_size, self.head_dim)
        self.wk1 = nn.Linear(embed_size, self.head_dim)
        self.wv1 = nn.Linear(embed_size, self.head_dim)

        # 头2 的 Wq, Wk, Wv
        self.wq2 = nn.Linear(embed_size, self.head_dim)
        self.wk2 = nn.Linear(embed_size, self.head_dim)
        self.wv2 = nn.Linear(embed_size, self.head_dim)

        # 输出融合层 Wo（标准必须有）
        self.wo = nn.Linear(hidden_size, hidden_size)

    def scaled_dot_product_attention(self, q, k, v):
        # 缩放点积注意力
        dk = q.size(-1)
        attn_score = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(dk, dtype=torch.float32))
        attn_weight = F.softmax(attn_score, dim=-1)
        output = torch.matmul(attn_weight, v)
        return output

    def forward(self, q, k, v):
        # q: [4,6,8]  k: [4,7,8]  v: [4,7,8]

        # ========== 头1 单独计算 ==========
        q1 = self.wq1(q)  # [4,6,8]
        k1 = self.wk1(k)
        v1 = self.wv1(v)
        out1 = self.scaled_dot_product_attention(q1, k1, v1)  # [4,6,8]

        # ========== 头2 单独计算 ==========
        q2 = self.wq2(q)
        k2 = self.wk2(k)
        v2 = self.wv2(v)
        out2 = self.scaled_dot_product_attention(q2, k2, v2)  # [4,6,8]

        # ========== 拼接两个头 ==========
        concat_out = torch.cat([out1, out2], dim=-1)  # [4,6, 16]

        # ========== 过 Wo 融合 ==========
        final_out = self.wo(concat_out)  # [4,6,16]

        return final_out

# ======================
# 版本2：高效版多头注意力（你手上代码完整版）
# ======================
class MultiHeadAttention_V2(nn.Module):
    def __init__(self, embed_size=8, hidden_size=16, num_heads=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # 大矩阵（高效版）
        self.wq = nn.Linear(embed_size, hidden_size)
        self.wk = nn.Linear(embed_size, hidden_size)
        self.wv = nn.Linear(embed_size, hidden_size)

        self.wo = nn.Linear(hidden_size, hidden_size)

    def split_heads(self, x):
        # [4, seq, 16] → [4, heads, seq, 8]
        batch_size = x.shape[0]
        return x.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

    def scaled_dot_product_attention(self, q, k, v):
        dk = q.size(-1)
        attn_score = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(dk, dtype=torch.float32))
        attn_weight = F.softmax(attn_score, dim=-1)
        output = torch.matmul(attn_weight, v)
        return output

    def forward(self, q, k, v):
        # 1. 投影到 hidden 维度
        q = self.wq(q)  # [4,6,16]
        k = self.wk(k)  # [4,7,16]
        v = self.wv(v)  # [4,7,16]

        # 2. 拆分成多头
        q = self.split_heads(q)  # [4,2,6,8]
        k = self.split_heads(k)
        v = self.split_heads(v)

        # 3. 计算注意力
        attn_out = self.scaled_dot_product_attention(q, k, v)  # [4,2,6,8]

        # 4. 拼接多头
        attn_out = attn_out.transpose(1, 2).contiguous()  # [4,6,2,8]
        concat_out = attn_out.view(q.size(0), -1, self.hidden_size)  # [4,6,16]

        # 5. 输出层 Wo
        final_out = self.wo(concat_out)

        return final_out

# ======================
# 测试：用你的数据维度
# ======================
if __name__ == "__main__":
    # 你的数据
    batch = 4
    q_len = 6    # 解码器序列长度
    kv_len = 7   # 编码器序列长度
    embed = 8
    hidden = 16
    heads = 2

    # 构造输入
    Q = torch.randn(batch, q_len, embed)  # [4,6,8]
    K = torch.randn(batch, kv_len, embed) # [4,7,8]
    V = torch.randn(batch, kv_len, embed) # [4,7,8]

    # ========== 测试版本1 ==========
    print("=== 版本1 输出 ===")
    model1 = MultiHeadAttention_V1(embed, hidden, heads)
    out1 = model1(Q, K, V)
    print("输出形状:", out1.shape)  # 应该输出: torch.Size([4,6,16])

    # ========== 测试版本2 ==========
    print("\n=== 版本2 输出 ===")
    model2 = MultiHeadAttention_V2(embed, hidden, heads)
    out2 = model2(Q, K, V)
    print("输出形状:", out2.shape)  # 同样输出: torch.Size([4,6,16])