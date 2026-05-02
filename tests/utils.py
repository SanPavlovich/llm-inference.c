import torch
import numpy as np


def save_tensor_to_bin(filename: str, tensor: torch.Tensor, np_dtype: np.dtype=np.float32):
    with open(filename, 'wb') as f:
        arr = tensor.detach().cpu().numpy().astype(np_dtype).flatten()
        f.write(arr.tobytes())
        print(f"Written {len(arr)} floats to {filename}")