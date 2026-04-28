#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>


void rotary_emb(
    size_t seq_len, size_t head_dim,
    float rope_theta,
    float* cos_output, // shape: [seq_len, head_dim / 2]
    float* sin_output  // shape: [seq_len, head_dim / 2]
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
    float* query_input,  // shape: [batch_size, num_heads, seq_len, head_dim]
    float* key_input,    // shape: [batch_size, num_heads, seq_len, head_dim]
    const float* cos,    // shape: [seq_len, head_dim / 2]
    const float* sin,    // shape: [seq_len, head_dim / 2]
    float* query_output, // shape: [batch_size, num_heads, seq_len, head_dim]
    float* key_output    // shape: [batch_size, num_heads, seq_len, head_dim]
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