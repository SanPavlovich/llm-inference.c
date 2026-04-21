from pathlib import Path
import argparse
import numpy as np
import torch
import torch.nn as nn


torch.manual_seed(42)


def save_tensor_to_bin(filename: str, tensor: torch.Tensor):
    with open(filename, 'wb') as f:
        arr = tensor.detach().cpu().numpy().astype(np.float32).flatten()
        f.write(arr.tobytes())
        print(f"Written {len(arr)} floats to {filename}")

def rmsnorm(x, weight, eps):
    rms_value = (tensor_input.pow(2).mean(axis=-1, keepdim=True) + eps).sqrt()
    return (x / rms_value) * weight

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-bs", "--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("-sl", "--seq-len", type=int, default=512, help="sequence length")
    parser.add_argument("-ed", "--embed-dim", type=int, default=256, help="embedding dim")
    parser.add_argument("-v", "--verbose", action="store_true", help="save generated test data or not.")
    parser.add_argument("--save", action="store_true", help="save generated test data or not.")
    args = parser.parse_args()
    batch_size, seq_len, embed_dim = args.batch_size, args.seq_len, args.embed_dim
    print(f"{batch_size=}, {seq_len=}, {embed_dim=}")

    tensor_input = torch.randn(batch_size, seq_len, embed_dim)
    print(f"{tensor_input.shape=}")

    rms = nn.RMSNorm(embed_dim, eps=1e-5)
    rms.weight.data = torch.randn(embed_dim)
    with torch.no_grad():
        tensor_output = rms(tensor_input)
    print(f"{tensor_output.shape=}")

    if args.verbose:
        print(f"{tensor_input=}\n")
        print(f"{tensor_output=}\n")
        print(f"{rms.weight.data=}\n")
    
    tensor_input_copy = tensor_input.clone()
    tensor_output_copy = rmsnorm(x=tensor_input_copy, weight=rms.weight, eps=rms.eps)
    assert torch.allclose(tensor_output_copy, tensor_output)
    print(f"mean diff: {(tensor_output - tensor_output_copy).mean().item()}")
    print(f"max diff: {(tensor_output - tensor_output_copy).abs().max().item()}")

    if args.save:
        abspath = Path().resolve().parent.parent / "data" / "rmsnorm" / f"bs_{batch_size}_sl_{seq_len}_ed_{embed_dim}"
        abspath.mkdir(parents=True, exist_ok=False)
        # SAVE INPUT TENSOR
        save_tensor_to_bin(
            filename=str(abspath / "tensor_input.bin"),
            tensor=tensor_input
        )

        # SAVE RMS WEIGHT TENSOR
        save_tensor_to_bin(
            filename=str(abspath / "rms_weight_data.bin"),
            tensor=rms.weight.data
        )

        # SAVE OUTPUT TENSOR
        save_tensor_to_bin(
            filename=str(abspath / "tensor_output.bin"),
            tensor=tensor_output
        )

