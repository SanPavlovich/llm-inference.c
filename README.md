# Llama Inference in Pure C

A from-scratch implementation of Llama architecture inference in pure C. No dependencies except `libc` and `libm`. This is not a production engine — it's an exercise in understanding every single byte, pointer, and floating-point operation that makes a Transformer speak.

## 🧠 Philosophy

Modern ML frameworks (PyTorch, JAX, TensorFlow) are black boxes of comfort. This project is the antidote. Here, tensors are `float*`, layers are `for` loops.

## 📁 Project Structure
```text
├── csrc/
│   ├── layers.c # RMSNorm, MatMul, Softmax, RoPE
│   ├── run.c # Entry point, generation loop
│   ├── loader.c # Binary weight loader (custom .bin format)
│   └── tokenizer.c 
├── src/
│   └── export_weights.py # PyTorch -> Raw Binary converter
├── сtests/ # compare c layers output with pytorch reference
├── tests/  # generate pytorch test data and implement torch.nn layer with pure pytorch 
├── data/ # Reference tensors for debugging
└── README.md
```