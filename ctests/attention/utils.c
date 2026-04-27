#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#include "utils.h"

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

void matmul(float* A, float* B, float* C, size_t m, size_t n, size_t k) {
    for(int m_idx = 0; m_idx < m; m_idx++) {
        for(int k_idx = 0; k_idx < k; k_idx++) {
            float sum = 0;
            for(int n_idx = 0; n_idx < n; n_idx++)
                sum += A[m_idx * n + n_idx] * B[n_idx * k + k_idx];
            C[m_idx * k + k_idx] = sum;
        }    
    }
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