from pathlib import Path
import argparse
import torch
from transformers.models.llama.modeling_llama import LlamaModel, LlamaDecoderLayer, LlamaConfig
from utils import save_tensor_to_bin

torch.manual_seed(42)


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

    block = LlamaDecoderLayer(layer_idx=1, config=config)
    block.eval()

    mlp_output = block.mlp(hidden_states)
    tensor_output = block(hidden_states, position_embeddings=(cos, sin), attention_mask=causal_mask)

    if args.verbose:
        print(f"tensor_input.shape: {hidden_states.shape}")
        print(f"tensor_input:\n{hidden_states}\n")
        print(f"{mlp_output.shape=}")
        print(f"{mlp_output=}\n")
        print(f"{tensor_output.shape=}")
        print(f"{tensor_output=}\n")
    
    if args.save:
        abspath = Path().resolve().parent / "data" / "decoder" / f"bs_{batch_size}_sl_{seq_len}_ed_{embed_dim}_nh_{num_heads}"
        abspath.mkdir(parents=True, exist_ok=False)
        save_tensor_to_bin(str(abspath / "tensor_input.bin"), hidden_states)
        save_tensor_to_bin(str(abspath / "mlp_output.bin"), mlp_output)
        save_tensor_to_bin(str(abspath / "tensor_output.bin"), tensor_output)

        for name, p in block.named_parameters():
            save_name = name.replace(".weight", "").replace(".", "__")
            save_tensor_to_bin(
                str(abspath / f"{save_name}.bin"), 
                p.data
            )