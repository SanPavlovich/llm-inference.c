#ifndef UTILS_H
#define UTILS_H

#include <stddef.h>
#include <stdint.h>
#include <llm_struct.h>

int parse_args(int argc, char* argv[], char** model_path);

void load_config(LlamaConfig* config, const char* config_file_path);

void* safe_malloc(size_t size);

void load_model(
    LlamaModel* model,
    LlamaConfig* config,
    const char* model_file_path
);

void free_model(LlamaModel* model, LlamaConfig* config);

void malloc_activation(
    LlamaModelActivation* activation,
    LlamaConfig* config
);

void free_activation(
    LlamaModelActivation* activation,
    LlamaConfig* config
);

void print_config(LlamaConfig *config);

#endif