import torch
import torch.nn as nn
import torch.nn.functional as F

from tokenizer import tokenizer
from config import config


class Llama3(nn.Module):
    def __init__(self, name_scope=None, dtype="float32"):
        super().__init__(name_scope, dtype)
        # 初始化嵌入层
        self.embedding_layer = nn.Embedding(config.vocab_size, config.dim)
        # 初始化最终的 RMS 归一化层
        self.rms_norm_final = RMS_Norm(config.dim)
        # 创建模型层列表
        self.layers = nn.ModuleList([
            Llama3_layer() for _ in range(config.n_layers)
        ])
        # 创建输出参数
        self.out = nn.Parameter(torch.empty(size=(4096, 128256)), dtype=dtype)

   # 定义前向传播逻辑
    def forward(self, tokens):

        # 通过嵌入层获取未归一化的嵌入
        token_embeddings_unnormalized = self.embedding_layer(torch.tensor(tokens))
        final_embeddings = token_embeddings_unnormalized

        # 逐层通过模型层处理嵌入
        for layer in self.layers:
            final_embeddings = layer(token_embeddings_unnormalized)
        
        # 应用最终的 RMS 归一化
        final_embeddings = self.rms_norm_final(final_embeddings)
        # 计算 logits
        logits = torch.matmul(final_embeddings, self.out)
        # 选择概率最高的下一个词
        next_token = torch.argmax(logits, axis=-1, keepdim=True)
        return next_token, logits
    

# 定义 Llama3 层的类，继承自 nn.Layer
class Llama3_layer(nn.Layer):
    def __init__(self, name_scope=None, dtype='float32'):
        super().__init__(name_scope, dtype)
        # 初始化两个 RMS 归一化层和注意力模块与前馈网络
        self.rms_norm_embedding = RMS_Norm(config.dim)
        self.rms_norm_attention = RMS_Norm(config.dim)
        self.attention = Attention_block()
        self.ffn = SwiGLU_FFN()
        
    # 定义层的前向传播逻辑
    def forward(self, token_embeddings_unnormalized):
        token_embeddings = self.rms_norm_embedding(token_embeddings_unnormalized)
        embedding_after_edit = self.attention(token_embeddings, token_embeddings_unnormalized)
        embedding_after_edit_normalized = self.rms_norm_attention(embedding_after_edit)
        output_after_feedforward = self.ffn(embedding_after_edit_normalized)
        layer_embedding = embedding_after_edit + output_after_feedforward
        return layer_embedding


# 定义 RMS 归一化类
class RMS_Norm(nn.Layer):
    def __init__(self, normalized_shape, name_scope=None, dtype='float32'):
        super().__init__(name_scope, dtype)
        # 初始化归一化权重
        self.weight = nn.Parameter(torch.empty(shape=(normalized_shape), dtype='float32'))

    # 定义归一化逻辑
    def forward(self, hidden_state):
        variance = hidden_state.pow(2).mean(-1, keepdim=True)
        hidden_state *= torch.rsqrt(variance + config.norm_eps)
        return hidden_state * self.weight   
    
class SwiGLU_FFN(nn.Layer):
    def __init__(self, name_scope=None, dtype="float32"):
        super().__init__(name_scope, dtype)
        # 初始化网络参数
        self.w1 = nn.Parameter(torch.empty(shape=(4096, 14336)), dtype=dtype)
        self.w2 = nn.Parameter(torch.empty(shape=(14336, 4096)), dtype=dtype)
        self.w3 = nn.Parameter(torch.empty(shape=(4096, 14336)), dtype=dtype)

    # 定义前向传播逻辑
    def forward(self, embedding_after_edit_normalized):
        # 实现 SwiGLU 激活和前馈过程
        output_after_feedforward = torch.matmul(nn.functional.silu(torch.matmul(embedding_after_edit_normalized, self.w1)) * torch.matmul(embedding_after_edit_normalized, self.w3), self.w2)
        return output_after_feedforward

# 定义注意力模块类
class Attention_block(nn.Layer):
    def __init__(self, name_scope=None, dtype="float32"):
        super().__init__(name_scope, dtype)
        # 注意力机制的头维度和键值对的头维度
        self.head_dim = config.dim // config.n_heads
        self.kv_head_dim = config.dim // 4 // config.n_kv_heads

        # 权重参数初始化
        self.wq = nn.Parameter(torch.empty(shape=(config.n_heads, self.head_dim, config.dim)), dtype=dtype)
        self.wk = nn.Parameter(torch.empty(shape=(config.n_kv_heads, self.kv_head_dim, config.dim)), dtype=dtype)
        self.wv = nn.Parameter(torch.empty(shape=(config.n_kv_heads, self.kv_head_dim, config.dim)), dtype=dtype)
        self.wo = nn.Parameter(torch.empty(shape=(config.dim, config.dim)), dtype=dtype)
        self.zero_to_one_split_into_64_parts = torch.tensor(torch.arange(64)) / 64

        # 预先计算并缓存频率系数，提高效率
        self.freqs = 1.0 / (config.rope_theta ** self.zero_to_one_split_into_64_parts)

    # 定义前向传播逻辑
    def forward(self, token_embeddings, token_embeddings_unnormalized):
        qkv_attention_store = []

        # 为整个批次的token预先计算复数频率系数
        freqs_cis = self._get_freqs_cis(token_embeddings.shape[0])
        
        for head in range(config.n_heads):

            # 对每个注意力头，计算其注意力并存储
            qkv_attention = self._compute_attention_head(
                token_embeddings, freqs_cis, head
            )

            qkv_attention_store.append(qkv_attention)
        
        # 将所有注意力头的结果合并
        stacked_qkv_attention = torch.cat(qkv_attention_store, axis=-1)
        # 计算最终的输出变化量并添加到未归一化的嵌入上
        embedding_delta = torch.matmul(stacked_qkv_attention, self.wo.T)
        embedding_after_edit = token_embeddings_unnormalized + embedding_delta
        return embedding_after_edit
    
    def _get_freqs_cis(self, length):
        # 计算并返回每个token对应的复数频率系数
        freq_for_each_token = torch.outer(torch.arange(length, dtype='float32'), self.freqs)
        return torch.polar(torch.ones_like(freq_for_each_token) * torch.cos(freq_for_each_token), torch.ones_like(freq_for_each_token) * torch.sin(freq_for_each_token))

    def _compute_attention_head(self, token_embeddings, freqs_cis, head):
        # 计算给定注意力头的Q, K, V
        q_per_token, k_per_token, v_per_token = self._compute_qkv(token_embeddings, head)
        # 应用复数旋转和重塑
        q_per_token_rotated = self._rotate_and_reshape(q_per_token, freqs_cis)
        k_per_token_rotated = self._rotate_and_reshape(k_per_token, freqs_cis)
        # 计算Q和K的点积，并应用缩放因子
        qk_per_token = torch.matmul(q_per_token_rotated, k_per_token_rotated.T) / (self.head_dim ** 0.5)
        # 应用掩码并进行softmax归一化
        qk_per_token_after_masking = self._apply_mask(qk_per_token, len(token_embeddings))
        qk_per_token_after_masking_after_softmax = nn.functional.softmax(qk_per_token_after_masking)
        # 计算最终的注意力输出
        qkv_attention = torch.matmul(qk_per_token_after_masking_after_softmax, v_per_token)
        return qkv_attention

    def _compute_qkv(self, token_embeddings, head):
        # 分别计算Q, K, V矩阵
        q = torch.matmul(token_embeddings, self.wq[head].T)
        k = torch.matmul(token_embeddings, self.wk[head // 4].T)
        v = torch.matmul(token_embeddings, self.wv[head // 4].T)
        return q, k, v

    def _rotate_and_reshape(self, token_tensor, freqs_cis):
        # 将token张量拆分、转换为复数、旋转并重新塑形
        token_tensor_split_into_pairs = torch.reshape(token_tensor, [token_tensor.shape[0], -1, 2])
        token_tensor_as_complex_numbers = torch.as_complex(token_tensor_split_into_pairs)
        token_tensor_rotated = torch.as_real(token_tensor_as_complex_numbers * freqs_cis)
        return torch.reshape(token_tensor_rotated, token_tensor.shape)

    def _apply_mask(self, qk_matrix, length):
        # 创建并应用上三角掩码，用于注意力计算中防止信息泄露
        mask = torch.full((length, length), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        return qk_matrix + mask


class SwiGLU_FFN(nn.Layer):
    def __init__(self, name_scope=None, dtype="float32"):
        super().__init__(name_scope, dtype)
        # 初始化网络参数
        self.w1 = nn.Parameter(torch.empty(shape=(4096, 14336)), dtype=dtype)
        self.w2 = nn.Parameter(torch.empty(shape=(14336, 4096)), dtype=dtype)
        self.w3 = nn.Parameter(torch.empty(shape=(4096, 14336)), dtype=dtype)

    # 定义前向传播逻辑
    def forward(self, embedding_after_edit_normalized):
        # 实现 SwiGLU 激活和前馈过程
        output_after_feedforward = torch.matmul(nn.functional.silu(torch.matmul(embedding_after_edit_normalized, self.w1)) * torch.matmul(embedding_after_edit_normalized, self.w3), self.w2)
        return output_after_feedforward


