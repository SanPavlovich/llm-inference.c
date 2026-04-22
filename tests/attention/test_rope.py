from pathlib import Path
import argparse
import torch
from transformers.models.llama.modeling_llama import LlamaConfig, LlamaModel, apply_rotary_pos_emb

from utils import save_tensor_to_bin

torch.manual_seed(42)


def rotary_emb(seq_len, head_dim, rope_theta):
    half = head_dim // 2
    rope_inv_freq = 1.0 / (rope_theta ** (torch.arange(half, dtype=torch.float32) / half))
    t = torch.arange(seq_len, device=hidden_states.device, dtype=torch.float32)
    freqs = torch.outer(t, rope_inv_freq)  # [L, half]
    cos = freqs.cos().view(1, 1, seq_len, -1)
    sin = freqs.sin().view(1, 1, seq_len, -1)
    return cos, sin


def apply_rotary(query, key, cos, sin):
    half = query.shape[-1] // 2
    q1, q2 = query[..., :half], query[..., half:]
    k1, k2 = key[..., :half], key[..., half:]
    q = torch.cat([q1 * cos - q2 * sin, q2 * cos + q1 * sin], dim=-1)
    k = torch.cat([k1 * cos - k2 * sin, k2 * cos_test + k1 * sin], dim=-1)
    return q, k


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

    # reference implementation
    query = torch.randn(batch_size, num_heads, seq_len, embed_dim // num_heads)
    key = torch.randn(batch_size, num_heads, seq_len, embed_dim // num_heads)

    if args.verbose:
        print(f"query before rotary: {query=}")
        print(f"key   before rotary: {key=}")

    config = LlamaConfig(num_hidden_layers=2, hidden_size=embed_dim, intermediate_size=embed_dim*2, num_attention_heads=num_heads)
    model = LlamaModel(config)
    hidden_states = torch.randn(batch_size, seq_len, embed_dim)
    position_ids = torch.arange(hidden_states.shape[1], device=hidden_states.device)
    position_ids = position_ids.unsqueeze(0)

    cos, sin = model.rotary_emb(hidden_states, position_ids)
    q, k = apply_rotary_pos_emb(query, key, cos, sin)

    if args.verbose:
        print(f"query after rotary: {q=}")
        print(f"key   after rotary: {k=}")

    # custom implementation comparison
    rope_theta = config.rope_parameters["rope_theta"]
    head_dim = embed_dim // num_heads
    cos_test, sin_test = rotary_emb(seq_len, head_dim, rope_theta)
    q_test, k_test = apply_rotary(query, key, cos_test, sin_test)

    assert torch.allclose(q, q_test)
    print("test for q passed!")

    assert torch.allclose(k, k_test)
    print("test for k passed!")

    if args.verbose:
        print(f"cos: {cos_test=}")
        print(f"sin: {sin_test=}")

    if args.verbose:
        print(f"query after rotary: {q=}")
        print(f"key   after rotary: {k=}")

    if args.save:
        abspath = Path().resolve().parent.parent / "data" / "rope" / f"bs_{batch_size}_sl_{seq_len}_ed_{embed_dim}_nh_{num_heads}"
        abspath.mkdir(parents=True, exist_ok=False)
        
        # save queries
        save_tensor_to_bin(filename=str(abspath / "query_before_rope.bin"), tensor=query)
        save_tensor_to_bin(filename=str(abspath / "query_after_rope.bin"), tensor=q)
        
        # save keys
        save_tensor_to_bin(filename=str(abspath / "key_before_rope.bin"), tensor=key)
        save_tensor_to_bin(filename=str(abspath / "key_after_rope.bin"), tensor=k)
        
        # save frequences
        save_tensor_to_bin(filename=str(abspath / "cos.bin"), tensor=cos_test)
        save_tensor_to_bin(filename=str(abspath / "sin.bin"), tensor=sin_test)
