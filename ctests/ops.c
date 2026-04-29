#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "ops.h"


void rmsnorm(
    size_t batch_size, 
    size_t seq_len, 
    size_t embed_dim,
    float eps,
    float* weight,
    float* input,
    float* output
) {
    float *curr_input_embed, *curr_output_embed;
    for(int b=0; b < batch_size; b++) {
        for(int s=0; s < seq_len; s++) {
            curr_input_embed = input   + b * seq_len * embed_dim + s * embed_dim;
            curr_output_embed = output + b * seq_len * embed_dim + s * embed_dim;
            float rms=0;
            for(int e=0; e < embed_dim; e++) {
                rms += pow(curr_input_embed[e], 2);
            }
            rms = sqrt((rms / (float)embed_dim) + eps);

            for(int e=0; e < embed_dim; e++) {
                curr_output_embed[e] = (curr_input_embed[e] / rms) * weight[e];
            }
        }
    }
}


void rotary_emb(
    size_t seq_len, size_t head_dim,
    float rope_theta,
    float* cos_output,
    float* sin_output
) {
    int half = head_dim / 2;
    float freq;
    for(int m=0; m < seq_len; m++) {
        for(int i=0; i < half; i++) {
            freq = m * (1 / (pow(rope_theta, 2.0 * i / head_dim)));
            cos_output[m * half + i] = cos(freq);
            sin_output[m * half + i] = sin(freq);
        }
    }
}


void apply_rotary_pos_emb(
    size_t batch_size, size_t num_heads, size_t seq_len, size_t head_dim,
    float* query_input,
    float* key_input,
    const float* cos,
    const float* sin,
    float* query_output,
    float* key_output
) {
    int half = head_dim / 2;
    float *curr_query_input, *curr_key_input, *curr_query_output, *curr_key_output;

    for(int b=0; b < batch_size; b++) {
        for(int nh=0; nh < num_heads; nh++) {
            int offset_b_nh = (b * num_heads * seq_len * head_dim) + (nh * seq_len * head_dim);
            curr_query_input = query_input + offset_b_nh;
            curr_key_input = key_input + offset_b_nh;
            curr_query_output = query_output + offset_b_nh;
            curr_key_output = key_output + offset_b_nh;

            for(int s=0; s < seq_len; s++) {
                // first half loop
                for(int hd=0; hd < half; hd++) {
                    int offset_s_hd = s * head_dim + hd;
                    int offset_freq = s * half + hd;
                    float q_1 = curr_query_input[offset_s_hd];
                    float q_2 = curr_query_input[offset_s_hd + half];
                    float k_1 = curr_key_input[offset_s_hd];
                    float k_2 = curr_key_input[offset_s_hd + half];
                    
                    float q_rope = q_1 * cos[offset_freq] - q_2 * sin[offset_freq];
                    float k_rope = k_1 * cos[offset_freq] - k_2 * sin[offset_freq];

                    curr_query_output[offset_s_hd] = q_rope;
                    curr_key_output[offset_s_hd] = k_rope;
                }

                // second half loop
                for(int hd=half; hd < head_dim; hd++) {
                    int offset_s_hd = s * head_dim + hd;
                    int offset_freq = s * half + hd - half;
                    float q_1 = curr_query_input[offset_s_hd];
                    float q_2 = curr_query_input[offset_s_hd - half];
                    float k_1 = curr_key_input[offset_s_hd];
                    float k_2 = curr_key_input[offset_s_hd - half];
                    
                    float q_rope = q_1 * cos[offset_freq] + q_2 * sin[offset_freq];
                    float k_rope = k_1 * cos[offset_freq] + k_2 * sin[offset_freq];

                    curr_query_output[offset_s_hd] = q_rope;
                    curr_key_output[offset_s_hd] = k_rope;
                }
            }
        }
    }
}


float dot_product(size_t size, float* array1, float* array2) {
    float sum = 0;
    for(int i=0; i < size; i++)
        sum += array1[i] * array2[i];
    return sum;
}


void softmax_1d(
    size_t size,
    float* input,
    float* output
) {
    float sum = 0;
    float max = -INFINITY;
    for(int i=0; i < size; i++) {
        if (input[i] > max)
            max = input[i];
    }
    for(int i=0; i < size; i++) {
        sum += exp(input[i] - max);
    }
    for(int i=0; i < size; i++) {
        output[i] = exp(input[i] - max) / sum;
    }
}


void create_causal_mask(
    size_t seq_len,
    float* output
) {
    for(int i=0; i < seq_len; i++) {
        for(int j=0; j < seq_len; j++) {
            if (i >= j) {
                output[i * seq_len + j] = 0.0f;
            } else {
                output[i * seq_len + j] = -INFINITY;
            }
        }
    }
}


void attention(
    size_t batch_size, size_t num_heads, size_t seq_len, size_t head_dim,
    float* query,        // shape: [batch_size, hum_heads, seq_len, head_dim]
    float* key,          // shape: [batch_size, hum_heads, seq_len, head_dim]
    float* value,        // shape: [batch_size, hum_heads, seq_len, head_dim]
    float* q_kt,         // shape: [seq_len, seq_len]
    float* attn_weights, // shape: [seq_len, seq_len]
    float* causal_mask,  // shape: [seq_len, seq_len]
    float* output        // shape: [batch_size, hum_heads, seq_len, head_dim]
) {
    float scale = 1.0f / sqrtf((float)head_dim);

    for(int b=0; b < batch_size; b++) {
        for(int nh=0; nh < num_heads; nh++) {
            int offset_b_nh = (b * num_heads * seq_len * head_dim) + (nh * seq_len * head_dim);
            float* curr_q = query + offset_b_nh;
            float* curr_k = key + offset_b_nh;
            float* curr_v = value + offset_b_nh;
            float* curr_output = output + offset_b_nh;
            
            // Q * K^T for current attention head
            for(int i=0; i < seq_len; i++) {
                for(int j=0; j < seq_len; j++) {
                    float* q_vector = curr_q + i * head_dim;
                    float* k_vector = curr_k + j * head_dim;
                    float dot_ij = dot_product(head_dim, q_vector, k_vector);
                    q_kt[i * seq_len + j] = dot_ij * scale;

                    // apply causal mask
                    q_kt[i * seq_len + j] += causal_mask[i * seq_len + j];
                }
            }

            // softmax
            for(int i=0; i < seq_len; i++) {
                float* q_kt_row = q_kt + i * seq_len;
                float* attn_weights_row = attn_weights + i * seq_len;
                softmax_1d(seq_len, q_kt_row, attn_weights_row);
            }

            // output
            for(int i=0; i < seq_len; i++) {
                for(int j=0; j < seq_len; j++) {
                    float weight = attn_weights[i * seq_len + j];
                    float* v_row = curr_v + j * head_dim;
                    float* out_row = curr_output + i * head_dim;
                    for(int d = 0; d < head_dim; d++) {
                        out_row[d] += weight * v_row[d];
                    }
                }
            }
        }
    }
}


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
) {
    // The code below performs matrix multiplication and immediately converts the result to its transposed form!
    // For the key and value it is applied repeat_interleave immediately.
    // query = (x @ w_q.T).view(batch_size, seq_len, num_heads, head_dim).transope(1, 2);
    // key = (x @ w_k.T).view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2).repeat_interleave(q_per_kv, dim=1);
    // value = (x @ w_v.T).view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2).repeat_interleave(q_per_kv, dim=1);

    int embed_dim = num_heads * head_dim;
    int q_per_kv = num_heads / num_kv_heads;

    for(int bs=0; bs < batch_size; bs++) {
        for(int s=0; s < seq_len; s++) {
            float* curr_input_embed = input + bs * seq_len * embed_dim + s * embed_dim;
            for(int nh=0; nh < num_heads; nh++) {
                int nh_kv = nh / q_per_kv;
                for(int hd=0; hd < head_dim; hd++) {
                    float head_sum_q = 0, head_sum_k = 0, head_sum_v=0;
                    for(int ed=0; ed < embed_dim; ed++) {
                        head_sum_q += curr_input_embed[ed] * w_q[(nh * head_dim + hd) * embed_dim + ed];
                        head_sum_k += curr_input_embed[ed] * w_k[(nh_kv * head_dim + hd) * embed_dim + ed];
                        head_sum_v += curr_input_embed[ed] * w_v[(nh_kv * head_dim + hd) * embed_dim + ed];
                    }
                    int head_offset = bs * num_heads * seq_len * head_dim + nh * seq_len * head_dim + s * head_dim;
                    float* curr_query_head = query + head_offset; 
                    float* curr_key_head = key + head_offset;
                    float* curr_value_head = value + head_offset;
                    curr_query_head[hd] = head_sum_q;
                    curr_key_head[hd] = head_sum_k;
                    curr_value_head[hd] = head_sum_v;
                }
            }
        }
    }

    apply_rotary_pos_emb(
        batch_size, num_heads, seq_len, head_dim,
        query, key,
        cos, sin,
        query_with_rope, key_with_rope
    );

    attention(
        batch_size, num_heads, seq_len, head_dim,
        query_with_rope, key_with_rope, value,
        q_kt, attn_weights,
        causal_mask,
        attn_output
    );

    // output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, embed_dim) @ w_o.T
    for(int bs=0; bs < batch_size; bs++) {
        for(int s=0; s < seq_len; s++) {
            for(int ed=0; ed < embed_dim; ed++) {
                float embed_sum = 0;
                for(int nh=0; nh < num_heads; nh++) {
                    int head_offset = bs * num_heads * seq_len * head_dim + nh * seq_len * head_dim + s * head_dim;
                    float* curr_input_head = attn_output + head_offset;
                    for(int hd=0; hd < head_dim; hd++) {
                        embed_sum += curr_input_head[hd] * w_o[ed * embed_dim + (nh * head_dim + hd)];
                    }
                }
                float* curr_output_embed = output + bs * seq_len * embed_dim + s * embed_dim; 
                curr_output_embed[ed] = embed_sum;
            }
        }
    }
}