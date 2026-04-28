#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <string.h>
#include "utils.h"
#include "rope.h"


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


int main(int argc, char *argv[]) {
    size_t batch_size = atoi(argv[1]);
    size_t seq_len = atoi(argv[2]);
    size_t embed_dim = atoi(argv[3]);
    size_t num_heads = atoi(argv[4]);
    size_t num_kv_heads = num_heads / 2;
    if (argc < 5) {
        fprintf(stderr, "Usage: %s <batch> <seq> <dim> <head>\n", argv[0]);
        return 1;
    }

    // indexing: total elements: batch_size * seq_len * embed_dim
    char fn_input[512], fn_w_q[512], fn_w_k[512], fn_w_v[512], fn_w_o[512], fn_output[512], fn_attn_output[512];
    snprintf(fn_input, sizeof(fn_input), "../../data/attention/bs_%zu_sl_%zu_ed_%zu_nh_%zu/tensor_input.bin", batch_size, seq_len, embed_dim, num_heads);
    snprintf(fn_w_q, sizeof(fn_w_q), "../../data/attention/bs_%zu_sl_%zu_ed_%zu_nh_%zu/q_proj.bin", batch_size, seq_len, embed_dim, num_heads);
    snprintf(fn_w_k, sizeof(fn_w_k), "../../data/attention/bs_%zu_sl_%zu_ed_%zu_nh_%zu/k_proj.bin", batch_size, seq_len, embed_dim, num_heads);
    snprintf(fn_w_v, sizeof(fn_w_v), "../../data/attention/bs_%zu_sl_%zu_ed_%zu_nh_%zu/v_proj.bin", batch_size, seq_len, embed_dim, num_heads);
    snprintf(fn_w_o, sizeof(fn_w_o), "../../data/attention/bs_%zu_sl_%zu_ed_%zu_nh_%zu/out_proj.bin", batch_size, seq_len, embed_dim, num_heads);
    snprintf(fn_output, sizeof(fn_output), "../../data/attention/bs_%zu_sl_%zu_ed_%zu_nh_%zu/tensor_output.bin", batch_size, seq_len, embed_dim, num_heads);

    FILE *fp_input = fopen(fn_input, "rb");
    FILE *fp_w_q = fopen(fn_w_q, "rb");
    FILE *fp_w_k = fopen(fn_w_k, "rb");
    FILE *fp_w_v = fopen(fn_w_v, "rb");
    FILE *fp_w_o = fopen(fn_w_o, "rb");
    FILE *fp_output = fopen(fn_output, "rb");
    if (!fp_input || !fp_w_q || !fp_w_k || !fp_w_v || !fp_w_o || !fp_output) {
        fprintf(stderr, "ERROR: Cannot open one of the files\n");
        return 1;
    }

    int head_dim = embed_dim / num_heads;
    size_t total_elements = batch_size * num_heads * seq_len * head_dim;
    int freq_elements = seq_len * head_dim / 2;
    printf("batch_size: %ld, num_heads: %ld, seq_len: %ld, head_dim: %d\n", batch_size, num_heads, seq_len, head_dim);

    size_t weights_size_q = sizeof(float) * embed_dim * head_dim * num_heads;
    size_t weights_size_kv = sizeof(float) * embed_dim * head_dim * num_kv_heads;
    size_t feqs_size = sizeof(float) * freq_elements;
    size_t hidden_size = sizeof(float) * total_elements;

    float* w_q = (float*)malloc(weights_size_q);
    float* w_k = (float*)malloc(weights_size_kv);
    float* w_v = (float*)malloc(weights_size_kv);
    float* w_o = (float*)malloc(weights_size_q);

    float rope_theta = 10000.0;
    float* cos = (float*)malloc(feqs_size);
    float* sin = (float*)malloc(feqs_size);
    float* causal_mask = (float*)malloc(sizeof(float) * seq_len * seq_len);
    float* query = (float*)malloc(sizeof(float) * batch_size * seq_len * embed_dim);
    float* key = (float*)malloc(sizeof(float) * batch_size * seq_len * embed_dim);
    float* query_with_rope = (float*)malloc(sizeof(float) * batch_size * seq_len * embed_dim);
    float* key_with_rope = (float*)malloc(sizeof(float) * batch_size * seq_len * embed_dim);
    float* q_kt = (float*)malloc(sizeof(float) * seq_len * seq_len);
    float* attn_weights = (float*)malloc(sizeof(float) * seq_len * seq_len);
    float* value = (float*)malloc(sizeof(float) * batch_size * seq_len * embed_dim);
    float* attn_output = (float*)calloc(total_elements, sizeof(float)); // calloc ==> init array with zeros 
    float* input = (float*)malloc(hidden_size);
    float* output = (float*)malloc(hidden_size);
    float* output_test = (float*)malloc(hidden_size);

    fread(input, sizeof(float), total_elements, fp_input);
    fclose(fp_input);
    fread(w_q, sizeof(float), total_elements, fp_w_q);
    fclose(fp_w_q);
    fread(w_k, sizeof(float), total_elements, fp_w_k);
    fclose(fp_w_k);
    fread(w_v, sizeof(float), total_elements, fp_w_v);
    fclose(fp_w_v);
    fread(w_o, sizeof(float), total_elements, fp_w_o);
    fclose(fp_w_o);
    fread(output, sizeof(float), total_elements, fp_output);
    fclose(fp_output);

    create_causal_mask(seq_len, causal_mask);
    rotary_emb(seq_len, head_dim, rope_theta, cos, sin);
    llama_attention(
        batch_size, num_heads, num_kv_heads, seq_len, head_dim,
        cos, sin, causal_mask,
        w_q, w_k, w_v, w_o,
        query, key,
        query_with_rope, key_with_rope,
        q_kt, attn_weights,
        value, 
        attn_output,
        input,
        output_test
    );

    printf("w_o:\n");
    print_2d(w_o, embed_dim, num_heads * head_dim);
    printf("\n");
    printf("input:\n");
    print(input, total_elements);
    printf("\n");
    printf("output:\n");
    print(output, total_elements);
    printf("\n");
    printf("output test:\n");
    print(output_test, total_elements);
    printf("\n");

    bool is_close_output = allclose(output, output_test, total_elements, 1e-6);
    printf("attn output is_close: %d\n", is_close_output);

    free(w_q);
    free(w_k);
    free(w_v);
    free(w_o);
    free(cos);
    free(sin);
    free(causal_mask);
    free(query);
    free(key);
    free(query_with_rope);
    free(key_with_rope);
    free(value);
    free(attn_output);
    free(input);
    free(output);
    free(output_test);
}