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

**Struct**

memory allocation for struct:

```c
typedef struct {
    float* w_q;  // это указатель (8 байт на x64)
    float* w_k;  // это указатель (8 байт)
    float* w_v;  // это указатель (8 байт)
    float* w_o;  // это указатель (8 байт)
} AttentionWeights;

params.attn_weights = malloc(sizeof(AttentionWeights)); // memory for pointers float* - 32 bytes
params.attn_weights[0].w_q = malloc(embed_dim * num_heads * head_dim * sizeof(float)); // memory for data
```

```text
Куча:
┌─────────────────────────┐
│ AttentionWeights (32B)  │  ← params.attn_weights указывает сюда
│  w_q = 0x7f...1000  ────┼──→ [float, float, float, ...] (массив весов Q)
│  w_k = 0x7f...2000  ────┼──→ [float, float, float, ...] (массив весов K)
│  w_v = 0x7f...3000  ────┼──→ [float, float, float, ...] (массив весов V)
│  w_o = 0x7f...4000  ────┼──→ [float, float, float, ...] (массив весов O)
└─────────────────────────┘
```

```c
int num_layers = 24;
AttentionWeights* attn_weights = malloc(num_layers * sizeof(AttentionWeights));

// Для каждого слоя выделяем данные
for (int l = 0; l < num_layers; l++) {
    attn_weights[l].w_q = malloc(embed_dim * num_heads * head_dim * sizeof(float));
    attn_weights[l].w_k = malloc(embed_dim * num_kv_heads * head_dim * sizeof(float));
    // ...
}

// Освобождение
for (int l = 0; l < num_layers; l++) {
    free(attn_weights[l].w_q);
    free(attn_weights[l].w_k);
}
free(attn_weights);  // освобождаем сам массив структур
```

передача структуры в функцию:
```c
typedef struct {
    size_t in_features;
    size_t out_features;
    float* weights;
} Config;

void forward(
    Config* config
) {
    // *(config.weights) <==> config->weights
    // config->in_features; потому что config передается по ссылке!
    // config->weights - это float* указатель на веса;
    for(int i=0; i < config->in_features; i++) {
        for(int i=0; i < config->out_features; i++) {
            config->weights[i * config->out_features + j] += 0.01f;
        }
    }
}

Config config;
config.in_features = 2;
config.out_features = 12;
config.weights = malloc(sizeof(float) * config.in_features * config.out_features);
forward(&config);
free(config.weights);
```

шпаргалка:
```c
void func_value(LinearLayer layer) {    // структура по значению (копия)
    layer.in_features = 10;             // точка
    layer.weights[0] = 1.0f;           // точка
}

void func_pointer(LinearLayer* layer) { // указатель на структуру
    layer->in_features = 10;            // стрелка
    layer->weights[0] = 1.0f;           // стрелка
    
    // Альтернатива (разыменование):
    (*layer).in_features = 10;          // работает, но никто так не пишет
}
```