#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <string.h>
#include "utils.h"
#include "ops.h"


void silu(
    size_t size,
    float* input,
    float* output
) {
    for(int i=0; i < size; i++) {
        float x = input[i];
        output[i] = x / (1.0f + expf(-x));
    }
}


void linear(
    size_t in_features,
    size_t out_features,
    float* weight,
    float* input,
    float* output
) {
    // output = x @ w.T
    for(int i=0; i < out_features; i++) {
        float dot_sum = 0.0f;
        for(int j=0; j < in_features; j++) {
            dot_sum += input[j] * weight[i * in_features + j];
        }
        output[i] = dot_sum;
    }
}

void swiglu(
    size_t batch_size,
    size_t seq_len,
    size_t embed_dim,
    size_t intermediate_size,
    float* weight_gate_proj,
    float* weight_up_proj,
    float* weight_down_proj,
    float* gate,
    float* up,
    float* gate_silu,
    float* gate_mul_up,
    float* input,
    float* output
) {
    // pytorch: output = self.down_proj(F.silu(self.gate_proj(input)) * self.up_proj(input))
    // numpy:   output = (silu(x @ weight_gate_proj.T) * (x @ weight_up_proj.T)) @ weight_down_proj.T
    for(int b=0; b < batch_size; b++) {
        for(int s=0; s < seq_len; s++) {
            int offset = b * seq_len * embed_dim + s * embed_dim;
            float* curr_input_embed = input + offset;
            float* curr_output_embed = output + offset;

            linear(embed_dim, intermediate_size, weight_gate_proj, curr_input_embed, gate);
            linear(embed_dim, intermediate_size, weight_up_proj, curr_input_embed, up);
            silu(intermediate_size, gate, gate_silu);
            for(int i=0; i < intermediate_size; i++) {
                gate_mul_up[i] = gate_silu[i] * up[i];
            }
            linear(intermediate_size, embed_dim, weight_down_proj, gate_mul_up, curr_output_embed);
        }
    }
}


void residual(
    size_t size,
    float* input,
    float* output
) {
    for(int i=0; i < size; i++) {
        output[i] += input[i];
    }
}


void llama_decoder() {

}


int main(int argc, char *argv[]) {
    size_t batch_size = atoi(argv[1]);
    size_t seq_len = atoi(argv[2]);
    size_t embed_dim = atoi(argv[3]);
    size_t num_heads = atoi(argv[4]);
    size_t num_kv_heads = num_heads / 2;
    size_t head_dim = embed_dim / num_heads;
    size_t intermediate_size = embed_dim * 2;

    char fn_input[512], fn_gate_proj[512], fn_up_proj[512], fn_down_proj[512], fn_output[512];
    snprintf(fn_input, sizeof(fn_input), "../data/decoder/bs_%zu_sl_%zu_ed_%zu_nh_%zu/tensor_input.bin", batch_size, seq_len, embed_dim, num_heads);
    snprintf(fn_gate_proj, sizeof(fn_gate_proj), "../data/decoder/bs_%zu_sl_%zu_ed_%zu_nh_%zu/mlp__gate_proj.bin", batch_size, seq_len, embed_dim, num_heads);
    snprintf(fn_up_proj, sizeof(fn_up_proj), "../data/decoder/bs_%zu_sl_%zu_ed_%zu_nh_%zu/mlp__up_proj.bin", batch_size, seq_len, embed_dim, num_heads);
    snprintf(fn_down_proj, sizeof(fn_down_proj), "../data/decoder/bs_%zu_sl_%zu_ed_%zu_nh_%zu/mlp__down_proj.bin", batch_size, seq_len, embed_dim, num_heads);
    snprintf(fn_output, sizeof(fn_output), "../data/decoder/bs_%zu_sl_%zu_ed_%zu_nh_%zu/mlp_output.bin", batch_size, seq_len, embed_dim, num_heads);

    FILE *fp_input = fopen(fn_input, "rb");
    FILE *fp_gate_proj = fopen(fn_gate_proj, "rb");
    FILE *fp_up_proj = fopen(fn_up_proj, "rb");
    FILE *fp_down_proj = fopen(fn_down_proj, "rb");
    FILE *fp_output = fopen(fn_output, "rb");
    if (!fp_input || !fp_gate_proj || !fp_up_proj || !fp_down_proj || !fp_output) {
        fprintf(stderr, "ERROR: Cannot open one of the files\n");
        return 1;
    }

    size_t total_elements = batch_size * seq_len * embed_dim;
    size_t hidden_size = sizeof(float) * total_elements;

    float* weight_gate_proj = (float*)malloc(sizeof(float) * intermediate_size * embed_dim);
    float* weight_up_proj = (float*)malloc(sizeof(float) * intermediate_size * embed_dim);
    float* weight_down_proj = (float*)malloc(sizeof(float) * embed_dim * intermediate_size);
    float* input = (float*)malloc(hidden_size);
    float* output = (float*)malloc(hidden_size);

    fread(input, sizeof(float), total_elements, fp_input);
    fclose(fp_input);
    fread(weight_gate_proj, sizeof(float), total_elements, fp_gate_proj);
    fclose(fp_gate_proj);
    fread(weight_up_proj, sizeof(float), total_elements, fp_up_proj);
    fclose(fp_up_proj);
    fread(weight_down_proj, sizeof(float), total_elements, fp_down_proj);
    fclose(fp_down_proj);
    fread(output, sizeof(float), total_elements, fp_output);
    fclose(fp_output);

    float* gate = (float*)malloc(sizeof(float) * intermediate_size);
    float* up = (float*)malloc(sizeof(float) * intermediate_size);
    float* gate_silu = (float*)malloc(sizeof(float) * intermediate_size);
    float* gate_mul_up = (float*)malloc(sizeof(float) * intermediate_size);
    float* output_test = (float*)malloc(hidden_size);

    printf("input:\n");
    print(input, total_elements);
    printf("\n");
    printf("output:\n");
    print(output, total_elements);
    printf("\n");

    swiglu(
        batch_size, seq_len, embed_dim, intermediate_size,
        weight_gate_proj, weight_up_proj, weight_down_proj,
        gate, up, gate_silu, gate_mul_up,
        input,
        output_test
    );

    printf("output test:\n");
    print(output_test, total_elements);
    printf("\n");

    bool is_close_output = allclose(output, output_test, total_elements, 1e-6);
    if (is_close_output) {
        printf("swiglu output OK! %d\n", is_close_output);
    } else {
        printf("swiglu output NOT OK! %d\n", is_close_output);
    }

}