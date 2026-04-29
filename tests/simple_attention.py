import torch
import torch.nn as nn

# X.shape: [batch_size, seq_len, embed_dim]
# head_dim = embed_dim // num_heads
# W_Q, W_K, W_V, W_O = nn.Linear(embed_dim, embed_dim), nn.Linear(embed_dim, embed_dim), nn.Linear(embed_dim, embed_dim)
# или можно общую матрицу весов W для [W_Q|W_K|W_V] nn.Linear(embed_dim, embed_dim * 3) и после применения слоя делать сплит на Q, K, V через torch.chunk
# потом делаем решейп на число голов, после применения W_Q берем Q_i как embed_dim[head_i-1:head_i]
#   
# attention matmul shape: 
# [batch_size, num_heads, seq_len, head_dim] @ [batch_size, num_heads, head_dim, seq_len] --> [batch_size, num_heads, seq_len, seq_len] (attention weights)
# torch.softmax(axis=-1) - значения вдоль [seq_len, i] будут в сумме давать 1, то что нужно.
# 
# V.shape: [batch_size, num_heads, seq_len, head_dim] - (для каждой посл-ти внутри батча и каждой головы внимания) новый элемент seq_len_i - взвешенное среднее всех value
#
# W_Q, W_K, W_V, W_O
# Q = W_Q @ X
# K = W_K @ X
# V = W_V @ X
# 
# mask.shape (from tokenizer): [batch_size, seq_len]
# mask.unsqueeze? --> [batch_size, 1, 1, seq_len]
#
#  attn_weights = Q @ K^T
# attn_weights.masked_fill_(mask, float("-inf"))
# O = softmax(attn_weights / sqrt(embed_dim)) @ V    shape: [batch_size, num_heads, seq_len, head_dim]
# result = O @ W_O


class SelfAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int=512, 
        num_heads: int=16,
        bias: bool=False
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.W = nn.Linear(embed_dim, embed_dim * 3, bias=bias)
        self.W_O = nn.Linear(embed_dim, embed_dim, bias=bias)
    
    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None=None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        output = self.W(x)  # [batch_size, seq_len, embed_dim] --> [batch_size, seq_len, embed_dim * 3]
        q, k, v = output.chunk(chunks=3, dim=-1) # [batch_size, seq_len, embed_dim * 3] --> [batch_size, seq_len, embed_dim], ... , ... ;
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2) # [batch_size, seq_len, embed_dim] --> [batch_size, hum_heads, seq_len, head_dim]
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2) # [batch_size, seq_len, embed_dim] --> [batch_size, hum_heads, seq_len, head_dim]
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2) # [batch_size, seq_len, embed_dim] --> [batch_size, hum_heads, seq_len, head_dim]

        q_kt = q @ k.transpose(-1, -2) # [batch_size, hum_heads, seq_len, head_dim] @ [batch_size, hum_heads, head_dim, seq_len] --> [batch_size, hum_heads, seq_len, seq_len]
        if attention_mask is not None:
            # attention_mask.shape: [batch_size, seq_len]
            # Например: [[1, ..., 1, 1, 0], [1, ..., 1, 0, 0]] --> [[0, ..., 0, 0, -inf], [0, ..., 0, -inf, -inf]]
            attention_mask = (1 - attention_mask) * float("-inf") 
            
            # При сложении с q_kt будет broadcast значений attention_mask[:, None, None, :]
            # [batch_size, seq_len] --> [batch_size, 1, 1, seq_len] --> [batch_size, hum_heads, head_dim, seq_len]
            q_kt += attention_mask[:, None, None, :]

        attn_weights = (q_kt / self.head_dim ** 0.5).softmax(dim=-1) # [batch_size, hum_heads, seq_len, seq_len]
        o = torch.matmul(attn_weights, v) # [batch_size, hum_heads, seq_len, seq_len] @ [batch_size, hum_heads, seq_len, head_dim] --> [batch_size, hum_heads, seq_len, head_dim]
        o = o.transpose(1, 2).contiguous().view(batch_size, seq_len, -1) # [batch_size, hum_heads, seq_len, head_dim] --> [batch_size, seq_len, embed_size]
        # использую .contiguous(), потому что transpose и view inplace операции и не работают 2 раза подряд. Или можно использовать reshape вместо view
        
        result = self.W_O(o) # [batch_size, seq_len, embed_size] --> [batch_size, seq_len, embed_size]
        
        return result
