import torch
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaConfig

from .llama_attention import TransformerConfig, CausalSelfAttention


torch.manual_seed(42)


if __name__ == "__main__":
    batch_size, seq_len, embed_dim = 2, 4, 8

    custom_config = TransformerConfig(n_heads=4, n_kv_head=2, hidden_dim=8, intermediate_dim=32, dropout=0)
    custom_block = CausalSelfAttention(custom_config)

    module = LlamaAttention()
    tensor_input = torch.randn(batch_size, seq_len, embed_dim)
    print(f"{tensor_input=}")

    config = LlamaConfig(num_attention_heads=4, num_key_value_heads=2, hidden_size=embed_dim)
    config._attn_implementation = "eager"
    block = LlamaAttention(layer_idx=0, config=config)
    print(block)


    sin, cos = torch.randn(1), torch.randn(1)
    tensor_output = block(tensor_input, position_embeddings=(sin, cos), attention_mask=None)
    print(f"{tensor_output=}")