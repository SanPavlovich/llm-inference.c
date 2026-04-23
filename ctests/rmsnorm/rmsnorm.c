#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>


bool allclose(float* A, float* B, size_t n, float tol) {
    for(int i=0; i < n; i++) {
        double diff = fabs((double)(A[i] - B[i]));
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

void rmsnorm(
    size_t batch_size, 
    size_t seq_len, 
    size_t embed_dim,
    float eps,
    const float* weight,
    const float* input,
    float* output
) {
    // int * const ptr = &x; // Адрес зафиксирован
    // const int *ptr = &x; // Данные зафиксированы
    float *curr_input_embed, *curr_output_embed;
    for(int b=0; b < batch_size; b++) {
        for(int s=0; s < seq_len; s++) {
            // *(ptr + 1)); // (сдвиг на 1 * sizeof(float))
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

int main(int argc, char *argv[]) {
    size_t batch_size = atoi(argv[1]);
    size_t seq_len = atoi(argv[2]);
    size_t embed_dim = atoi(argv[3]);
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <batch> <seq> <dim>\n", argv[0]);
        return 1;
    }
    float eps=1e-5;

    // indexing: total elements: batch_size * seq_len * embed_dim
    char fname_input[256], fname_output[256], fname_weight[256];
    snprintf(fname_input, sizeof(fname_input), "../../data/rmsnorm/bs_%zu_sl_%zu_ed_%zu/tensor_input.bin", batch_size, seq_len, embed_dim);
    snprintf(fname_output, sizeof(fname_output), "../../data/rmsnorm/bs_%zu_sl_%zu_ed_%zu/tensor_output.bin", batch_size, seq_len, embed_dim);
    snprintf(fname_weight, sizeof(fname_weight), "../../data/rmsnorm/bs_%zu_sl_%zu_ed_%zu/rms_weight_data.bin", batch_size, seq_len, embed_dim);
    printf("Input file:  %s\n", fname_input);

    FILE *fp_input = fopen(fname_input, "rb");
    FILE *fp_output = fopen(fname_output, "rb");
    FILE *fp_weight = fopen(fname_weight, "rb");
    if (!fp_input || !fp_output || !fp_weight) {
        fprintf(stderr, "ERROR: Cannot open one of the files\n");
        return 1;
    }

    int total_elements = batch_size * seq_len * embed_dim;
    size_t array_size = sizeof(float) * total_elements;
    size_t weight_size = sizeof(float) * embed_dim;
    float* input = (float*)malloc(array_size);
    float* output = (float*)malloc(array_size);
    float* output_test = (float*)malloc(array_size);
    float* weight = (float*)malloc(weight_size);
    fread(input, sizeof(float), total_elements, fp_input);
    fclose(fp_input);

    fread(output, sizeof(float), total_elements, fp_output);
    fclose(fp_output);

    fread(weight, sizeof(float), embed_dim, fp_weight);
    fclose(fp_weight);

    // printf("input:\n");
    // print(input, total_elements);
    // printf("\n");
    // printf("output:\n");
    // print(output, total_elements);
    // printf("\n");
    // printf("weight:\n");
    // print(weight, embed_dim);
    // printf("\n");

    rmsnorm(
        batch_size, seq_len, embed_dim, 
        eps, weight, 
        input, output_test
    );

    // printf("output test:\n");
    // print(output_test, total_elements);

    bool is_close = allclose(output, output_test, total_elements, 1e-8);
    printf("\noutput close to reference: %d\n", is_close);
}