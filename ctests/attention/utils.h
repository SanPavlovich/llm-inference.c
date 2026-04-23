#include <stddef.h>
#ifndef UTILS_H
#define UTILS_H

bool allclose(float* A, float* B, size_t n, float tol);

void print(float* array, size_t size);

void print_2d(float* array, size_t nrows, size_t ncols);

#endif