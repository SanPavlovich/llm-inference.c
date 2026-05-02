# Llama Inference in Pure C

A from-scratch implementation of Llama architecture inference in pure C. No dependencies except `libc` and `libm`. This is not a production engine — it's an exercise in understanding every single byte, pointer, and floating-point operation that makes a Transformer speak.

## 🧠 Philosophy

Modern ML frameworks (PyTorch, JAX, TensorFlow) are black boxes of comfort. This project is the antidote. Here, tensors are `float*`, layers are `for` loops.

## 📁 Project Structure
```text
├── csrc/
│   ├── ops.c # RMSNorm, Softmax, RoPE, Linear, SwiGLU, LlamaAttention, LlamaDecoderLayer, complete Llama forward
│   ├── run.c # Entry point, generation loop
│   ├── utils.c # cli parser, binary config/model loader, memory allocation
│   └── tokenizer.c 
├── src/
│   └── export_weights.py # PyTorch -> Raw Binary converter
├── сtests/ # compare c layers output with pytorch reference
├── tests/  # generate pytorch test data and implement torch.nn layer with pure pytorch 
├── data/ # Reference tensors for debugging
└── README.md
```

## 🗺️ Roadmap

- ✅ **RMSNorm** — `rmsnorm.c`, verified byte-for-byte with `torch.nn.RMSNorm`
- ✅ **RoPE (Rotary Position Embedding)** — `rope.c`, cos/sin precomputation verified
- ✅ **Softmax** — numerically stable version with `max` subtraction
- ✅ **LlamaAttention** — integrate RoPE application directly into attention
- ✅ **Linear (MatMul)** — naive triple-loop, verified against `torch.nn.Linear`
- ✅ **SwiGLU** — `silu(gate) * up` followed by `down_proj`, verified in `swiglu.c`
- ✅ **LlamaDecoderLayer** — RMSNorm → Attention → Residual → RMSNorm → FFN → Residual
- ✅ **Embedding** - pytorch nn.Embedding layer implementation (with memcpy)
- ✅ **LlamaForCausalLM** - forward
- 🟡 **LlamaForCausalLM** - generate
- 🟡 **tokenizer**
- 🟡 **KV Cache**
- 🟡 **MultiQueryAttention**