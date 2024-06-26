import paddle
import paddle.nn as nn
from config import config
from tokenizer import tokenizer

class Llama3(nn.Layer):
    def __init__(self, name_scope=None, dtype='float32'):
        super().__init__(name_scope, dtype)
        

    def forward(self, x):
        pass

class SwiGLU_FFN(nn.Layer):
    def __init__(self, name_scope=None, dtype="float32"):
        super().__init__(name_scope, dtype)
        self.rms_norm = RMS_Norm(config.dim)
        self.attention_block = Attention_block()
        self.w1 = paddle.create_parameter(shape=[4096, 14336], dtype=dtype)
        self.w2 = paddle.create_parameter(shape=[14336, 4096], dtype=dtype)
        self.w3 = paddle.create_parameter(shape=[4096, 14336], dtype=dtype)

    def forward(self, tokens):
            embedding_after_edit, token_embeddings_unnormalized = self.attention_block(tokens)
            embedding_after_edit_normalized = self.rms_norm(embedding_after_edit)
            # print(embedding_after_edit_normalized.shape)
            output_after_feedforward = paddle.matmul(nn.functional.silu(paddle.matmul(embedding_after_edit_normalized, self.w1)) * paddle.matmul(embedding_after_edit_normalized, self.w3), self.w2)
            # print(output_after_feedforward.shape)
            layer_embedding = embedding_after_edit + output_after_feedforward

class Attention_block(nn.Layer):
    def __init__(self, name_scope=None, dtype="float32"):
        super().__init__(name_scope, dtype)

        self.embedding_layer = Embedding_Layer()

        self.head_dim = config.dim // config.n_heads
        self.kv_head_dim = config.dim // 4 // config.n_kv_heads
        self.wq = paddle.create_parameter(shape=[config.n_heads, self.head_dim, config.dim], dtype=dtype)
        self.wk = paddle.create_parameter(shape=[config.n_kv_heads, self.kv_head_dim, config.dim], dtype=dtype)
        self.wv = paddle.create_parameter(shape=[config.n_kv_heads, self.kv_head_dim, config.dim], dtype=dtype)
        self.wo = paddle.create_parameter(shape=[config.dim, config.dim], dtype=dtype)
        self.zero_to_one_split_into_64_parts = paddle.to_tensor(paddle.arange(64)) / 64
        # print(self.zero_to_one_split_into_64_parts)
        self.freqs = 1.0 / (config.rope_theta ** self.zero_to_one_split_into_64_parts)
        # print(self.wk.shape)
        
        # print(self.wv.shape)

    def forward(self, token):
        qkv_attention_store = []
        for head in range(config.n_heads):

            token_embeddings, token_embeddings_unnormalized = self.embedding_layer(token)
            q_per_token = paddle.matmul(token_embeddings, self.wq[head].T)
            # print(q_per_token.shape)
            q_per_token_split_into_pairs = paddle.reshape(q_per_token, [q_per_token.shape[0], -1, 2])
            # print(q_per_token_split_into_pairs.shape)
            q_per_token_as_complex_numbers = paddle.as_complex(q_per_token_split_into_pairs)
            # print(q_per_token_as_complex_numbers.shape)
            freq_for_each_token = paddle.outer(paddle.arange(int(q_per_token_as_complex_numbers.shape[0]), dtype='float32'), self.freqs)
            freqs_cis = paddle.polar(paddle.ones_like(freq_for_each_token) * paddle.cos(freq_for_each_token), paddle.ones_like(freq_for_each_token) * paddle.sin(freq_for_each_token))
            q_per_token_as_complex_numbers_rotated = q_per_token_as_complex_numbers * freqs_cis
            # print(q_per_token_as_complex_numbers_rotated.shape)
            q_per_token_split_into_pairs_rotated = paddle.as_real(q_per_token_as_complex_numbers_rotated)
            q_per_token_rotated = paddle.reshape(q_per_token_split_into_pairs_rotated, q_per_token.shape)

            k_per_token = paddle.matmul(token_embeddings, self.wk[head // 4].T)
            k_per_token_split_into_pairs = paddle.reshape(k_per_token, [k_per_token.shape[0], -1, 2])
            k_per_token_as_complex_numbers = paddle.as_complex(k_per_token_split_into_pairs)
            k_per_token_split_into_pairs_rotated = paddle.as_real(k_per_token_as_complex_numbers * freqs_cis)
            k_per_token_rotated = paddle.reshape(k_per_token_split_into_pairs_rotated, k_per_token.shape)

            # print(q_per_token_rotated.shape)
            # print(k_per_token_rotated.shape)
            qk_per_token = paddle.matmul(q_per_token_rotated, k_per_token_rotated.T) / (self.head_dim) ** 0.5
            # print(qk_per_token.shape)
            mask = paddle.full((len(token), len(token)), float("-inf"))
            mask = paddle.triu(mask, diagonal=1)
            # print(mask)
            qk_per_token_after_masking = qk_per_token + mask
            qk_per_token_after_masking_after_softmax = nn.functional.softmax(qk_per_token_after_masking)
            
            v_per_token = paddle.matmul(token_embeddings, self.wv[head // 4].T)
            qkv_attention = paddle.matmul(qk_per_token_after_masking_after_softmax, v_per_token)
            qkv_attention_store.append(qkv_attention)
        stacked_qkv_attention = paddle.concat(qkv_attention_store, axis=-1)
        # print(stacked_qkv_attention.shape)
        embedding_delta = paddle.matmul(stacked_qkv_attention, self.wo.T)
        embedding_after_edit = token_embeddings_unnormalized + embedding_delta
        return embedding_after_edit, token_embeddings_unnormalized


class Embedding_Layer(nn.Layer):
    def __init__(self, name_scope=None, dtype='float32'):
        super().__init__(name_scope, dtype)
        
        self.embedding_layer = nn.Embedding(config.vocab_size, config.dim)
        self.rms_norm = RMS_Norm(config.dim)

    def forward(self, tokens):
        tokens = paddle.to_tensor(tokens)
        token_embeddings_unnormalized = self.embedding_layer(tokens).to(paddle.float32)
        token_embeddings = self.rms_norm(token_embeddings_unnormalized)
        return token_embeddings, token_embeddings_unnormalized


class RMS_Norm(nn.Layer):
    def __init__(self, normalized_shape, name_scope=None, dtype='float32'):
        super().__init__(name_scope, dtype)

        self.weight = paddle.create_parameter(shape=[normalized_shape], dtype='float32')

    def forward(self, hidden_state):
        variance = hidden_state.to(paddle.float32).pow(2).mean(-1, keepdim=True)
        hidden_state *= paddle.rsqrt(variance + config.norm_eps)

        return hidden_state * self.weight        
    

if __name__ == "__main__":
    model = SwiGLU_FFN()
    prompt = "the answer to the ultimate question of life, the universe, and everything is "
    tokens = [128000] + tokenizer.encode(prompt)
    # print(tokens)
    # print(model(tokens).shape)
    attn = model(tokens)