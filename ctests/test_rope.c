#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>


bool allclose(float* A, float* B, size_t n, float tol) {
    for(int i=0; i < n; i++) {
        double diff = fabs((double)(A[i] - B[i]));
        // printf("i: %d, A[i]: %e, B[i]: %e, diff: %lf\n", i, A[i], B[i], diff);
        if(diff > tol) {
            printf("\ndiff: %lf, idx: %d\n", diff, i);
            return false;
        }
    }
    return true;
}

void print(float* array, size_t size) {
    for(int i=0; i < size; i++) {
        printf("%e ", array[i]);
    }
    printf("\n");
}

void print_2d(float* array, size_t nrows, size_t ncols) {
    for(int i=0; i < nrows; i++) {
        for(int j=0; j < ncols; j++)
            printf("%e ", array[i * ncols + j]);
        printf("\n");
    }
    printf("\n");
}

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
    char fn_query[256], fn_key[256], fn_query_rope[256], fn_key_rope[256], fn_cos[256], fn_sin[256];
    snprintf(fn_query, sizeof(fn_query), "../../data/rope/bs_%zu_sl_%zu_ed_%zu_nh_%zu/query_before_rope.bin", batch_size, seq_len, embed_dim, num_heads);
    snprintf(fn_query_rope, sizeof(fn_query_rope), "../../data/rope/bs_%zu_sl_%zu_ed_%zu_nh_%zu/query_after_rope.bin", batch_size, seq_len, embed_dim, num_heads);
    snprintf(fn_key, sizeof(fn_key), "../../data/rope/bs_%zu_sl_%zu_ed_%zu_nh_%zu/key_before_rope.bin", batch_size, seq_len, embed_dim, num_heads);
    snprintf(fn_key_rope, sizeof(fn_key_rope), "../../data/rope/bs_%zu_sl_%zu_ed_%zu_nh_%zu/key_after_rope.bin", batch_size, seq_len, embed_dim, num_heads);
    snprintf(fn_cos, sizeof(fn_cos), "../../data/rope/bs_%zu_sl_%zu_ed_%zu_nh_%zu/cos.bin", batch_size, seq_len, embed_dim, num_heads);
    snprintf(fn_sin, sizeof(fn_sin), "../../data/rope/bs_%zu_sl_%zu_ed_%zu_nh_%zu/sin.bin", batch_size, seq_len, embed_dim, num_heads);

    FILE *fp_query = fopen(fn_query, "rb");
    FILE *fp_query_rope = fopen(fn_query_rope, "rb");
    FILE *fp_key = fopen(fn_key, "rb");
    FILE *fp_key_rope = fopen(fn_key_rope, "rb");
    FILE *fp_cos = fopen(fn_cos, "rb");
    FILE *fp_sin = fopen(fn_sin, "rb");
    if (!fp_query || !fp_query_rope || !fp_key || !fp_key_rope || !fp_cos || !fp_sin) {
        fprintf(stderr, "ERROR: Cannot open one of the files\n");
        return 1;
    }

    int head_dim = embed_dim / num_heads;
    int total_elements = batch_size * seq_len * embed_dim;
    int freq_elements = seq_len * head_dim / 2;
    printf("freq elements: %d\n", freq_elements);
    printf("batch_size: %ld, num_heads: %ld, seq_len: %ld, head_dim: %d\n", batch_size, num_heads, seq_len, head_dim);

    size_t hidden_size = sizeof(float) * total_elements;
    size_t feqs_size = sizeof(float) * freq_elements;

    float* query = (float*)malloc(hidden_size);
    float* query_with_rope = (float*)malloc(hidden_size);
    float* key = (float*)malloc(hidden_size);
    float* key_with_rope = (float*)malloc(hidden_size);
    float* cos = (float*)malloc(feqs_size);
    float* sin = (float*)malloc(feqs_size);

    fread(query, sizeof(float), total_elements, fp_query);
    fclose(fp_query);
    fread(query_with_rope, sizeof(float), total_elements, fp_query_rope);
    fclose(fp_query_rope);

    fread(key, sizeof(float), total_elements, fp_key);
    fclose(fp_key);
    fread(key_with_rope, sizeof(float), total_elements, fp_key_rope);
    fclose(fp_key_rope);

    fread(cos, sizeof(float), freq_elements, fp_cos);
    fclose(fp_cos);
    fread(sin, sizeof(float), freq_elements, fp_sin);
    fclose(fp_sin);

    float* cos_test = (float*)malloc(feqs_size);
    float* sin_test = (float*)malloc(feqs_size);
    float* query_test = (float*)malloc(hidden_size);
    float* key_test = (float*)malloc(hidden_size);

    float rope_theta = 10000.0;
    rotary_emb(seq_len, head_dim, rope_theta, cos_test, sin_test);

    bool is_close_cos = allclose(cos, cos_test, freq_elements, 1e-6);
    bool is_close_sin = allclose(sin, sin_test, freq_elements, 1e-6);
    printf("\ncos is_close: %d, sin is_close: %d\n", is_close_cos, is_close_sin);

    apply_rotary_pos_emb(
        batch_size, num_heads, seq_len, head_dim,
        query, key, cos, sin, 
        query_test, key_test
    );

    // printf("query after:\n");
    // print(query_with_rope, total_elements);
    // printf("\n");

    // printf("query after test:\n");
    // print(query_test, total_elements);
    // printf("\n");

    bool is_close_query = allclose(query_with_rope, query_test, freq_elements, 1e-6);
    bool is_close_key   = allclose(key_with_rope, key_test, freq_elements, 1e-6);
    printf("\nquery is_close: %d, key is_close: %d\n", is_close_query, is_close_key);
}