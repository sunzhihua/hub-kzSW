import numpy as np
import torch.nn as nn
import torch
import math

"""
通过手动矩阵运算实现Transformer结构

流程:
1、 输入向量
2、 Multi-Head Attention
3、 Add & Norm
4、 Feed Forward Neural Network
5、 Add & Norm
6、 输出向量
7、 Transformer 堆叠 Transformer Block

核心类:
1、 MultiHeadSelfAttention
2、 LayerNorm
3、 FeedForwardNetwork、 Linear、GELU
4、 TransformerBlock
5、 Transformer

编码说明：
    1、勿用torch.nn.Transformer或torch.nn.MultiheadAttention等高级API
    2、勿用torch.nn.LayerNorm等高级API
    3、勿用torch.nn.Linear等高级API

输入向量:
    标准输入 x : x.shape = (batch_size, seq_len, d_model)
    batch_size (批次大小): 
        含义: 一次处理多少个独立的样本 (句子、序列) 。   例: 一次同时处理4个句子
        作用: 并行处理效率，梯度计算的批次单位
    seq_len (序列长度): 
        含义: 每个样本中包含多少个时间步/词元 (token) 。 例: 每个句子有10个词
        作用: 序列的时序维度，模型按这个方向进行循环或自注意力计算
    d_model (模型维度/特征维度): 
        含义: 每个位置用多少维的向量来表示 (嵌入维度) 。 例: 每个词用512维的向量表示
        作用: 模型的表达能力，信息压缩/扩展的维度。
"""


class MultiHeadSelfAttention(nn.Module):
    """
    参数说明：
        d_model: 输入向量的维度
        num_attention_heads: 多头注意力机制的头数
    """

    def __init__(self, d_model, num_attention_heads):
        super().__init__()
        self.d_model = d_model
        self.num_attention_heads = num_attention_heads
        self.head_dim = d_model // num_attention_heads

        self.query = Linear(d_model, d_model)
        self.key = Linear(d_model, d_model)
        self.value = Linear(d_model, d_model)

        self.output_dense = nn.Linear(d_model, d_model)

    # 将 Q、K、V 从 (batch_size, seq_len, head_dim) reshape 为 (batch, seq_len, num_heads, head_dim)
    # 目的：将每个头的表示分开存储，num_heads 维度用于后续并行计算多个注意力头
    # self.num_attention_heads = 12, self.head_dim = 64
    def _reshape_for_multi_head(self, x, batch_size, seq_len):
        """
        将输入从 (batch, seq_len, d_model) reshape 为 (batch, seq_len, num_heads, head_dim)
        """
        x = x.reshape(batch_size, seq_len, self.num_attention_heads, self.head_dim)
        # 转置为 (batch, num_heads, seq_len, head_dim) 便于并行计算
        return x.transpose(1, 2)


    def _scaled_dot_product_attention(self, Q, K, V):
        """
        计算缩放点积注意力
        Args:
            Q: (batch, num_heads, seq_len, head_dim)
            K: (batch, num_heads, seq_len, head_dim)
            V: (batch, num_heads, seq_len, head_dim)
        Returns:
            context: (batch, num_heads, seq_len, head_dim)
            attention_probs: (batch, num_heads, seq_len, seq_len)
        """
        # 计算注意力分数: Q @ K^T / sqrt(head_dim)
        # Q: (batch_size, seq_len, num_heads, head_dim), K^T: (batch, num_heads, head_dim, seq_len) → 结果: (batch, seq_len, num_heads, seq_len)
        # 除以 sqrt(head_dim) 是缩放点积注意力（Scaled Dot-Product Attention），防止点积值过大导致 softmax 梯度消失
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Softmax 获取注意力权重
        attention_probs = torch.softmax(attention_scores, dim=-1)

        # 计算加权和
        context = torch.matmul(attention_probs, V)

        return context, attention_probs

    def _merge_multi_head(self, context_layer, batch_size, seq_len):
        """
        合并多头注意力输出
        Args:
            context_layer: (batch, num_heads, seq_len, head_dim)
        Returns:
            (batch, seq_len, d_model)
        """
        # (batch, num_heads, seq_len, head_dim) -> (batch, seq_len, num_heads, head_dim)
        context_layer = context_layer.transpose(1, 2).contiguous()
        # (batch, seq_len, num_heads, head_dim) -> (batch, seq_len, d_model)
        return context_layer.reshape(batch_size, seq_len, self.d_model)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            output: (batch, seq_len, d_model)
            attention_probs: (batch, num_heads, seq_len, seq_len)
        """
        batch_size, seq_len, _ = x.shape

        # 1. 线性变换得到 Q, K, V
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # 2. 重塑为多头格式
        Q = self._reshape_for_multi_head(Q, batch_size, seq_len)
        K = self._reshape_for_multi_head(K, batch_size, seq_len)
        V = self._reshape_for_multi_head(V, batch_size, seq_len)

        # 3. 缩放点积注意力
        context_layer, attention_probs = self._scaled_dot_product_attention(Q, K, V)

        # 4. 合并多头
        context_layer = self._merge_multi_head(context_layer, batch_size, seq_len)
 
         # 5. 输出投影
        output = self.output_dense(context_layer)

        return output, attention_probs


class LayerNorm(nn.Module):
    """
    参数说明：
        d_model: 输入向量的维度
        gama: 缩放参数
        beta: 平移参数
        eps: 用于数值稳定的小常数
    """

    def __init__(self, d_model, eps=1e-12):
        super().__init__()
        self.d_model = d_model
        self.gama = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        '''
        与原始论文一致：原始的 LayerNorm 论文使用的是总体标准差（除以 n) 
        训练稳定性：在小 batch 或短序列时，有偏估计可以避免分母过小
        计算效率：无需进行 n-1 的额外计算
        实际差异很小：当序列长度足够大时（如 512、1024) ,两种计算方式差异可以忽略：
        '''
        std = x.std(dim=-1, keepdim=True, unbiased=False)
        standard_x = (x - mean) / (std + self.eps)
        output = self.gama * standard_x + self.beta
        return output


class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        """
        初始化权重
        原因:
          数值稳定性:	防止梯度爆炸/消失
          加快收敛: 合适的初始值让模型更快学习
          打破对称性: 避免相同更新导致神经元行为一致
        常用方法: Xavier/Kaiming 初始化
        """
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return x @ self.weight.T + self.bias


class GELU(nn.Module):
    """
    手动实现 GELU 激活函数 \n
    GELU(x) = x * Φ(x)，其中 Φ 是标准正态分布的累积分布函数 \n
    近似公式: 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x^3)))
    """

    def forward(self, x):
        return (
            0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)))
        )


class FeedForwardNetwork(nn.Module):
    """
    参数说明：
        d_model: 输入向量的维度
        dim_ffn: 前馈神经网络的隐藏层维度
    """

    def __init__(self, d_model, dim_ffn):
        super().__init__()
        self.d_model = d_model
        self.dim_ffn = dim_ffn
        self.linear1 = Linear(d_model, dim_ffn)
        self.gelu = GELU()
        self.linear2 = Linear(dim_ffn, d_model)

    def forward(self, x):
        """
        前馈神经网络的前向传播:
            FFN(x) = GELU(xW₁ + b₁)W₂ + b₂
        """
        out = self.linear1(x)
        out = self.gelu.forward(out)
        output = self.linear2(out)
        return output


class TransformerBlock(nn.Module):
    """
    参数说明：
        d_model: 输入向量的维度
        num_attention_heads: 多头注意力机制的头数
        dim_ffn: 前馈神经网络的隐藏层维度
    """

    def __init__(self, d_model, num_attention_heads, dim_ffn):
        super().__init__()
        self.d_model = d_model
        self.num_attention_heads = num_attention_heads
        self.dim_ffn = dim_ffn

        self.MHA = MultiHeadSelfAttention(d_model, num_attention_heads)
        self.norm1 = LayerNorm(d_model)
        self.FFN = FeedForwardNetwork(d_model, dim_ffn)
        self.norm2 = LayerNorm(d_model)

    def forward(self, vec):
        """
        Transformer Block 完整公式:
            ① z = LayerNorm( x + MHA(x) )
            ② output = LayerNorm( z + FFN(z) )
        """
        # 第一层多头自注意力
        attn_output, _ = self.MHA(vec)
        norm1_output = self.norm1(vec + attn_output)

        # 前馈神经网络
        ffn_output = self.FFN(norm1_output)
        output = self.norm2(norm1_output + ffn_output)

        return output


class Transformer(nn.Module):
    """
    参数说明：
        d_model: 输入向量的维度
        num_attention_heads: 多头注意力机制的头数
        dim_ffn: 前馈神经网络的隐藏层维度
        num_hidden_layers: Transformer Block 的层数
    """

    def __init__(
        self, d_model=768, num_attention_heads=12, dim_ffn=None, num_hidden_layers=3
    ):
        super().__init__()
        self.d_model = d_model
        self.num_attention_heads = num_attention_heads
        self.dim_ffn = dim_ffn if dim_ffn is not None else 4 * d_model
        self.num_hidden_layers = num_hidden_layers

        # "d_model": 768,  # d_model 神经网络中某一层的输出特征维度 (在 BERT 中: 每个 token 的向量表示维度)
        # "layer_norm_eps": 1e-12,
        # "max_position_embeddings": 512,
        # "num_attention_heads": 12,
        # "num_hidden_layers": 12,
        # "type_vocab_size": 2,
        # "vocab_size": 21128

        self.layers = nn.ModuleList(
            [
                TransformerBlock(d_model, num_attention_heads, self.dim_ffn)
                for _ in range(num_hidden_layers)
            ]
        )

    def forward(self, vec):
        output = vec
        for layer in self.layers:
            output = layer(output)
        return output

if __name__ == "__main__":
    model = Transformer(d_model=512, num_hidden_layers=6, num_attention_heads=8, dim_ffn=1024)
    x = torch.randn(2, 16, 512)        # [B, T, H]
    print(model(x).shape)              # [2, 16, 512]
