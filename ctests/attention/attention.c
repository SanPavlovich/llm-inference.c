#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <string.h>
#include "utils.h"


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
    float* query,       // shape: [batch_size, hum_heads, seq_len, head_dim]
    float* key,         // shape: [batch_size, hum_heads, seq_len, head_dim]
    float* value,       // shape: [batch_size, hum_heads, seq_len, head_dim]
    float* causal_mask, // shape: [seq_len, seq_len]
    float* output       // shape: [batch_size, hum_heads, seq_len, head_dim]
) {
    float* q_kt = (float*)malloc(sizeof(float) * seq_len * seq_len);
    float* attn_weights = (float*)malloc(sizeof(float) * seq_len * seq_len);
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

    free(q_kt);
    free(attn_weights);
}

void llama_attention() {

}

int main(int argc, char *argv[]) {
    size_t batch_size = atoi(argv[1]);
    size_t seq_len = atoi(argv[2]);
    size_t embed_dim = atoi(argv[3]);
    size_t num_heads = atoi(argv[4]);
    if (argc < 5) {
        fprintf(stderr, "Usage: %s <batch> <seq> <dim> <head>\n", argv[0]);
        return 1;
    }

    // indexing: total elements: batch_size * seq_len * embed_dim
    char fn_query[512], fn_key[512], fn_value[512], fn_output[512];
    snprintf(fn_query, sizeof(fn_query), "../../data/attention/bs_%zu_sl_%zu_ed_%zu_nh_%zu/query.bin", batch_size, seq_len, embed_dim, num_heads);
    snprintf(fn_key, sizeof(fn_key), "../../data/attention/bs_%zu_sl_%zu_ed_%zu_nh_%zu/key.bin", batch_size, seq_len, embed_dim, num_heads);
    snprintf(fn_value, sizeof(fn_value), "../../data/attention/bs_%zu_sl_%zu_ed_%zu_nh_%zu/value.bin", batch_size, seq_len, embed_dim, num_heads);
    snprintf(fn_output, sizeof(fn_output), "../../data/attention/bs_%zu_sl_%zu_ed_%zu_nh_%zu/attn_output_4d.bin", batch_size, seq_len, embed_dim, num_heads);

    FILE *fp_query = fopen(fn_query, "rb");
    FILE *fp_key = fopen(fn_key, "rb");
    FILE *fp_value = fopen(fn_value, "rb");
    FILE *fp_output = fopen(fn_output, "rb");
    if (!fp_query || !fp_key || !fp_value || !fp_output) {
        fprintf(stderr, "ERROR: Cannot open one of the files\n");
        return 1;
    }

    int head_dim = embed_dim / num_heads;
    size_t total_elements = batch_size * num_heads * seq_len * head_dim;
    printf("batch_size: %ld, num_heads: %ld, seq_len: %ld, head_dim: %d\n", batch_size, num_heads, seq_len, head_dim);

    size_t hidden_size = sizeof(float) * total_elements;

    float* query = (float*)malloc(hidden_size);
    float* key = (float*)malloc(hidden_size);
    float* value = (float*)malloc(hidden_size);
    float* output = (float*)malloc(hidden_size);
    float* output_test = (float*)calloc(total_elements, sizeof(float)); // calloc ==> init array with zeros 

    fread(query, sizeof(float), total_elements, fp_query);
    fclose(fp_query);
    fread(key, sizeof(float), total_elements, fp_key);
    fclose(fp_key);
    fread(value, sizeof(float), total_elements, fp_value);
    fclose(fp_value);
    fread(output, sizeof(float), total_elements, fp_output);
    fclose(fp_output);

    float* causal_mask = (float*)malloc(sizeof(float) * seq_len * seq_len);
    create_causal_mask(seq_len, causal_mask);

    print_2d(causal_mask, seq_len, seq_len);

    printf("output 4d:\n");
    print(output, total_elements);
    printf("\n");

    attention(
        batch_size, num_heads, seq_len, head_dim,
        query, key, value,
        causal_mask,
        output_test
    );

    printf("output 4d test:\n");
    print(output_test, total_elements);
    printf("\n");

    bool is_close_output = allclose(output, output_test, total_elements, 1e-6);
    printf("attn output is_close: %d\n", is_close_output);

    free(query);
    free(key);
    free(value);
    free(output);
    free(causal_mask);
    free(output_test);
}