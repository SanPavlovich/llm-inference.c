from pathlib import Path
import argparse
import torch
from transformers.models.llama.modeling_llama import LlamaModel, LlamaAttention, LlamaConfig

from llama_attention import TransformerConfig, CausalSelfAttention

torch.manual_seed(42)


def block_eq(layer, layer_value):
    assert layer.weight.data.shape == layer_value.weight.data.shape
    layer.weight.data = layer_value.weight.data
    if layer.bias and layer_value.bias:
        layer.bias.data = layer_value.bias.data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-bs", "--batch_size", type=int, default=2, help="batch size")
    parser.add_argument("-sl", "--seq-len", type=int, default=12, help="sequence length")
    parser.add_argument("-ed", "--embed-dim", type=int, default=128, help="embedding dim")
    parser.add_argument("-nh", "--num-heads", type=int, default=32, help="embedding dim")
    parser.add_argument("-v", "--verbose", action="store_true", help="save generated test data or not.")
    parser.add_argument("--save", action="store_true", help="save generated test data or not.")
    args = parser.parse_args()
    batch_size, seq_len, embed_dim, num_heads = args.batch_size, args.seq_len, args.embed_dim, args.num_heads
    print(f"{batch_size=}, {seq_len=}, {embed_dim=}, {num_heads=}")

    config = LlamaConfig(attention_dropout=0, num_hidden_layers=2, hidden_size=embed_dim, intermediate_size=embed_dim*2, num_attention_heads=num_heads, num_key_value_heads=num_heads//2)
    config._attn_implementation = "eager"
    model = LlamaModel(config)

    hidden_states = torch.randn(batch_size, seq_len, embed_dim)
    position_ids = torch.arange(hidden_states.shape[1], device=hidden_states.device).unsqueeze(0)
    cos, sin = model.rotary_emb(hidden_states, position_ids)

    causal_mask = torch.tril(torch.ones(seq_len, seq_len))
    causal_mask = causal_mask.view(1, 1, seq_len, seq_len)
    causal_mask = (1.0 - causal_mask) * torch.finfo(torch.float32).min

    block = LlamaAttention(layer_idx=0, config=config)
    block.eval()

    tensor_output, _ = block(hidden_states, position_embeddings=(cos, sin), attention_mask=causal_mask)


    custom_config = TransformerConfig(
        n_layer=0, n_head=num_heads, n_kv_head=num_heads//2, hidden_dim=embed_dim, intermediate_dim=embed_dim*2, dropout=0,
        use_rope=True
    )
    custom_block = CausalSelfAttention(custom_config)

    block_eq(custom_block.q_proj, block.q_proj)
    block_eq(custom_block.k_proj, block.k_proj)
    block_eq(custom_block.v_proj, block.v_proj)
    block_eq(custom_block.out_proj, block.o_proj)

    custom_block.eval()

    tensor_output_test = custom_block(hidden_states, attention_mask=None)

    assert torch.allclose(tensor_output, tensor_output_test)

    if args.save:
        abspath = Path().resolve().parent / "data" / "attention" / f"bs_{batch_size}_sl_{seq_len}_ed_{embed_dim}_nh_{num_heads}"
        abspath.mkdir(parents=True, exist_ok=False)
        custom_block(hidden_states, attention_mask=None, verbose=args.verbose, save_path=abspath)