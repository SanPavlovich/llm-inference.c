from pathlib import Path
import struct
import argparse
import numpy as np
import torch
from transformers.models.llama.modeling_llama import LlamaForCausalLM, LlamaConfig
from utils import save_tensor_to_bin

torch.manual_seed(42)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-bs", "--batch_size", type=int, default=2, help="batch size")
    parser.add_argument("-sl", "--seq-len", type=int, default=12, help="sequence length")
    parser.add_argument("-ed", "--embed-dim", type=int, default=128, help="embedding dim")
    parser.add_argument("-id", "--intermediate-dim", type=int, default=128, help="intermediate dim")
    parser.add_argument("-nh", "--num-heads", type=int, default=32, help="embedding dim")
    parser.add_argument("-vs", "--vocab-size", type=int, default=32, help="vocab size")
    parser.add_argument("-nl", "--num-layers", type=int, default=32, help="num hidden layers")
    parser.add_argument("-v", "--verbose", action="store_true", help="save generated test data or not.")
    parser.add_argument("--save", action="store_true", help="save generated test data or not.")
    args = parser.parse_args()
    batch_size, seq_len, embed_dim, intermediate_dim, num_heads, vocab_size, num_layers = \
    args.batch_size, args.seq_len, args.embed_dim, args.intermediate_dim, args.num_heads, args.vocab_size, args.num_layers
    print(f"{batch_size=}, {seq_len=}, {embed_dim=}, {intermediate_dim=}, {num_heads=}, {vocab_size=}, {num_layers=}")

    config = LlamaConfig(
        vocab_size=vocab_size,
        attention_dropout=0, 
        num_hidden_layers=num_layers, 
        hidden_size=embed_dim, 
        intermediate_size=intermediate_dim, 
        num_attention_heads=num_heads, 
        num_key_value_heads=num_heads//2
    )
    config._attn_implementation = "eager"
    model = LlamaForCausalLM(config)
    print(f"model parameters count: {sum([p.numel() for p in model.parameters()]):_}")

    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    inputs_embeds = model.model.embed_tokens(input_ids)
    output = model(input_ids)

    if args.verbose:
        print(f"{input_ids.shape=}")
        print(f"{input_ids=}\n")
        print(f"{inputs_embeds.shape=}")
        print(f"{inputs_embeds=}\n")
        print(f"{output.logits.shape=}")
        print(f"{output.logits=}\n")
    
    if args.save:
        abspath = Path().resolve().parent / "data" / "forward" / f"bs_{batch_size}_sl_{seq_len}_ed_{embed_dim}_nh_{num_heads}"
        abspath.mkdir(parents=True, exist_ok=False)
        save_tensor_to_bin(str(abspath / "tensor_input.bin"), input_ids, np_dtype=np.int64)
        save_tensor_to_bin(str(abspath / "inputs_embeds.bin"), inputs_embeds)
        save_tensor_to_bin(str(abspath / "tensor_output.bin"), output.logits.data)

        # save config:
        filename = str(abspath / "config.bin")
        integers = [
            batch_size,
            seq_len,
            embed_dim,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
            config.intermediate_size,
            config.vocab_size,
            config.num_hidden_layers
        ]
        floats = [
            config.rope_parameters["rope_theta"],
            config.rms_norm_eps
        ]

        format_string = f'{len(integers)}N{len(floats)}f' 
        with open(filename, 'wb') as f:
            packed_data = struct.pack(format_string, *integers, *floats)
            f.write(packed_data)
        
        # save model parameters:
        model_params = []
        for name, p in model.named_parameters():
            print(name, tuple(p.shape))
            model_params.append(p.data.flatten())
        save_tensor_to_bin(
            str(abspath / "model.bin"),
            torch.cat(model_params)
        )