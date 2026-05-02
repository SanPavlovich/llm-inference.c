#ifndef LLM_STRUCT_H
#define LLM_STRUCT_H

#include <stddef.h>

typedef struct {
    size_t batch_size;
    size_t seq_len;
    size_t embed_dim;
    size_t num_heads;
    size_t num_kv_heads;
    size_t head_dim;
    size_t intermediate_size;
    size_t vocab_size;
    size_t num_hidden_layers;
    float rope_theta;
    float rms_eps;
} LlamaConfig;


typedef struct {
    float* up_proj;
    float* gate_proj;
    float* down_proj;
} LlamaMLP;


typedef struct {
    float* w_q;
    float* w_k;
    float* w_v;
    float* w_o;
} LlamaAttention;


typedef struct {
    float* weight;
} RMSNorm;


typedef struct {
    float* weight;
} Embedding;


typedef struct {
    float* weight;
} Linear;


typedef struct {
    LlamaMLP mlp;
    LlamaAttention self_attn;
    RMSNorm rms_attn;
    RMSNorm rms_ffn;
} LlamaDecoderLayer;


typedef struct {
    Embedding embed_tokens;
    LlamaDecoderLayer* layers;
    RMSNorm norm;
    Linear lm_head;
} LlamaModel;


typedef struct {
    float* output;
} RMSNormActivation;


typedef struct {
    float* gate;
    float* up;
    float* gate_silu;
    float* gate_mul_up;
    float* output;
} LlamaMLPActivation;


typedef struct {
    float* query;
    float* key;
    float* query_rope;
    float* key_rope;
    float* q_kt;
    float* attn_weights;
    float* value;
    float* attn_output;
    float* output;
} LlamaAttentionActivation;


typedef struct {
    LlamaAttentionActivation self_attn;
    LlamaMLPActivation mlp;
    RMSNormActivation rms_attn;
    RMSNormActivation rms_ffn;
} LlamaDecoderActivation;


typedef struct {
    float* output;
} EmbeddingActivation;


typedef struct {
    float* output;
} LinearActivation;


typedef struct {
    EmbeddingActivation embed_tokens;
    LlamaDecoderActivation decoder;
    RMSNormActivation norm;
    LinearActivation lm_head;
} LlamaModelActivation;

#endif