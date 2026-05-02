#ifndef UTILS_H
#define UTILS_H

#include <stddef.h>
#include <stdint.h>

bool allclose(float* A, float* B, size_t n, float tol);

void print(float* array, size_t size);

void print_int64(int64_t* array, size_t size);

void print_2d(float* array, size_t nrows, size_t ncols);

#endif