**При объявлении статических массивов, указатель на массив передаем как &array[0]:**

```c
void softmax_1d(
    size_t size,
    float* input,
    float* output
)

float array[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 11.0};
float *result = (float*)malloc(sizeof(float) * 6);
softmax_1d(6, &array[0], result);
```

**pointers:**

```c
int * const ptr = &x; // Адрес зафиксирован
const int *ptr = &x; // Данные зафиксированы
*(ptr + 1)); // (сдвиг на 1 * sizeof(float))
```

**math.h, INFINITY:**

```c
#include <math.h>

printf("-inf: %e\n", inf_negative);             // - inf
printf("-inf + 10: %e\n", inf_negative + 10.0); // -inf
printf("-inf - 10: %e\n", inf_negative + 10.0); // -inf
printf("exp(0): %e\n", exp(0));                 // 1
printf("exp(0): %e\n", exp(1));                 // 2.718...
printf("exp(-inf): %e\n", exp(-INFINITY));      // 0
```

**компиляция с дебагом malloc:**

```bash
# И запускай — ASAN сразу покажет строку выхода за границы.
gcc -fsanitize=address -g your_file.c -lm
```