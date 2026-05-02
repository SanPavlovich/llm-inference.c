#ifndef OPS_H
#define OPS_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void rmsnorm(
    size_t batch_size, 
    size_t seq_len, 
    size_t embed_dim,
    float eps,
    float* weight,
    float* input,
    float* output
);


void rotary_emb(
    size_t seq_len, size_t head_dim,
    float rope_theta,
    float* cos_output,
    float* sin_output
);


void apply_rotary_pos_emb(
    size_t batch_size, size_t num_heads, size_t seq_len, size_t head_dim,
    float* query_input,
    float* key_input,
    const float* cos,
    const float* sin,
    float* query_output,
    float* key_output
);


float dot_product(size_t size, float* array1, float* array2);


void softmax_1d(
    size_t size,
    float* input,
    float* output
);


void create_causal_mask(
    size_t seq_len,
    float* output
);


void attention(
    size_t batch_size, size_t num_heads, size_t seq_len, size_t head_dim,
    float* query,        // shape: [batch_size, hum_heads, seq_len, head_dim]
    float* key,          // shape: [batch_size, hum_heads, seq_len, head_dim]
    float* value,        // shape: [batch_size, hum_heads, seq_len, head_dim]
    float* q_kt,         // shape: [seq_len, seq_len]
    float* attn_weights, // shape: [seq_len, seq_len]
    float* causal_mask,  // shape: [seq_len, seq_len]
    float* output        // shape: [batch_size, hum_heads, seq_len, head_dim]
);


void llama_attention(
    size_t batch_size, size_t num_heads, size_t num_kv_heads, size_t seq_len, size_t head_dim,
    float* cos, float* sin, float* causal_mask,
    float* w_q, float* w_k, float* w_v, float* w_o,
    float* query, float* key,
    float* query_with_rope, float* key_with_rope,
    float* q_kt, float* attn_weights,
    float* value, 
    float* attn_output, // shape: [batch_size, hum_heads, seq_len, head_dim]
    float* input,       // shape: [batch_size, seq_len, embed_dim]
    float* output       // shape: [batch_size, seq_len, embed_dim]
);

#endif