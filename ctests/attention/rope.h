#ifndef ROPE_H
#define ROPE_H

#include <stddef.h>  // size_t

void rotary_emb(
    size_t seq_len, size_t head_dim,
    float rope_theta,
    float* cos_output, // shape: [seq_len, head_dim / 2]
    float* sin_output  // shape: [seq_len, head_dim / 2]
);

void apply_rotary_pos_emb(
    size_t batch_size, size_t num_heads, size_t seq_len, size_t head_dim,
    float* query_input,  // shape: [batch_size, num_heads, seq_len, head_dim]
    float* key_input,    // shape: [batch_size, num_heads, seq_len, head_dim]
    const float* cos,    // shape: [seq_len, head_dim / 2]
    const float* sin,    // shape: [seq_len, head_dim / 2]
    float* query_output, // shape: [batch_size, num_heads, seq_len, head_dim]
    float* key_output    // shape: [batch_size, num_heads, seq_len, head_dim]
);

#endif