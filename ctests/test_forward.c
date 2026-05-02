#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <string.h>
#include <stdint.h>
#include "utils.h"
#include "ops.h"
#include "llm_struct.h"


void embedding_forward(
    size_t batch_size,
    size_t seq_len,
    size_t embed_dim,
    size_t vocab_size,
    float* weight,
    int64_t* input,
    float* output
) {
    for(int b=0; b < batch_size; b++) {
        for(int s=0; s < seq_len; s++) {
            int64_t token_id = input[b * seq_len + s];
            if (token_id < 0 || token_id >= vocab_size) {
                printf("Invalid token_id: %ld\n", token_id);
                exit(1);
            }
            float* src = weight + token_id * embed_dim;
            float* dst = output + b * seq_len * embed_dim + s * embed_dim;
            memcpy(dst, src, embed_dim * sizeof(float));
        }
    }
}


void llama_forward(
    LlamaConfig* config,
    LlamaModel* model,
    LlamaModelActivation* activation,
    int64_t* input
) {
    size_t freqs_size = config->seq_len * config->head_dim / 2;
    size_t hidden_size = config->batch_size * config->seq_len * config->embed_dim;
    
    float* cos = (float*)malloc(sizeof(float) * freqs_size);
    float* sin = (float*)malloc(sizeof(float) * freqs_size);
    float* causal_mask = (float*)malloc(sizeof(float) * config->seq_len * config->seq_len);

    embedding_forward(
        config->batch_size, config->seq_len, config->embed_dim, config->vocab_size,
        model->embed_tokens.weight,
        input,
        activation->embed_tokens.output
    );

    printf("embedding_forward ok\n");

    create_causal_mask(config->seq_len, causal_mask);
    rotary_emb(config->seq_len, config->head_dim, config->rope_theta, cos, sin);

    // decoder layers
    for(int l=0; l < config->num_hidden_layers; l++) {
        llama_decoder_forward(
            config, &model->layers[l], &activation->decoder, 
            cos, sin, causal_mask, 
            activation->embed_tokens.output // inplace forward
        );
        memset(activation->decoder.self_attn.attn_output, 0, hidden_size * sizeof(float));
        printf("llama_decoder_forward [layer %d] ok\n", l);
    }

    // final RMSNorm + lm_head
    rmsnorm(
        config->batch_size, config->seq_len, config->embed_dim, config->rms_eps,
        model->norm.weight,
        activation->embed_tokens.output,
        activation->norm.output
    );
    printf("rmsnorm_forward ok\n");

    for(int b=0; b < config->batch_size; b++) {
        for(int s=0; s < config->seq_len; s++) {
            int embed_offset = b * config->seq_len * config->embed_dim + s * config->embed_dim;
            int logit_offset = b * config->seq_len * config->vocab_size + s * config->vocab_size;
            float* curr_input_embed = activation->norm.output + embed_offset;
            float* curr_output_logit = activation->lm_head.output + logit_offset;

            linear_forward(
                config->embed_dim, config->vocab_size,
                model->lm_head.weight,
                curr_input_embed,    // input
                curr_output_logit    // output
            );
        }
    }

    printf("lm_head ok\n");

    free(cos);
    free(sin);
    free(causal_mask);
}


int parse_args(int argc, char* argv[], char** model_path) {
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model-path") == 0 || strcmp(argv[i], "-m") == 0) {
            if (i + 1 < argc) {
                *model_path = argv[++i];
            } else {
                fprintf(stderr, "Error: --model-path requires a path\n");
                return 1;
            }
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printf("Usage: %s --temperature <t> --model <path>\n", argv[0]);
            printf("  -m, --model-path     Path to model weights\n");
            return 0;
        } else {
            fprintf(stderr, "Unknown argument: %s\n", argv[i]);
            fprintf(stderr, "Use --help for usage\n");
            return 1;
        }
    }
}


void load_config(LlamaConfig* config, const char* config_file_path) {
    FILE *fp_config = fopen(config_file_path, "rb");
    if (!fp_config) {
        fprintf(stderr, "ERROR: Cannot open %s\n", config_file_path);
        return;
    }
    fread(&config->batch_size, sizeof(size_t), 1, fp_config);
    fread(&config->seq_len, sizeof(size_t), 1, fp_config);
    fread(&config->embed_dim, sizeof(size_t), 1, fp_config);
    fread(&config->num_heads, sizeof(size_t), 1, fp_config);
    fread(&config->num_kv_heads, sizeof(size_t), 1, fp_config);
    fread(&config->head_dim, sizeof(size_t), 1, fp_config);
    fread(&config->intermediate_size, sizeof(size_t), 1, fp_config);
    fread(&config->vocab_size, sizeof(size_t), 1, fp_config);
    fread(&config->num_hidden_layers, sizeof(size_t), 1, fp_config);
    fread(&config->rope_theta, sizeof(float), 1, fp_config);
    fread(&config->rms_eps, sizeof(float), 1, fp_config);
    fclose(fp_config);
    printf("config loaded successfully from: %s\n", config_file_path);
}


void* safe_malloc(size_t size) {
    void* ptr = malloc(size);
    if (!ptr) {
        fprintf(stderr, "ERROR: malloc(%zu) failed\n", size);
        exit(1);
    }
    return ptr;
}


void load_model(
    LlamaModel* model,
    LlamaConfig* config,
    const char* model_file_path
) {
    FILE* fp = fopen(model_file_path, "rb");
    if (!fp) {
        fprintf(stderr, "ERROR: Cannot open model file: %s\n", model_file_path);
        exit(1);
    }
    
    size_t embed_dim = config->embed_dim;
    size_t num_heads = config->num_heads;
    size_t num_kv_heads = config->num_kv_heads;
    size_t head_dim = config->head_dim;
    size_t intermediate_size = config->intermediate_size;
    size_t vocab_size = config->vocab_size;
    size_t num_layers = config->num_hidden_layers;
    
    model->embed_tokens.weight = (float*)safe_malloc(vocab_size * embed_dim * sizeof(float));
    fread(model->embed_tokens.weight, sizeof(float), vocab_size * embed_dim, fp);
    
    model->layers = (LlamaDecoderLayer*)safe_malloc(num_layers * sizeof(LlamaDecoderLayer));
    
    for (int l = 0; l < num_layers; l++) {
        LlamaDecoderLayer* layer = &model->layers[l];
        
        layer->self_attn.w_q = (float*)safe_malloc(num_heads * head_dim * embed_dim * sizeof(float));
        layer->self_attn.w_k = (float*)safe_malloc(num_kv_heads * head_dim * embed_dim * sizeof(float));
        layer->self_attn.w_v = (float*)safe_malloc(num_kv_heads * head_dim * embed_dim * sizeof(float));
        layer->self_attn.w_o = (float*)safe_malloc(embed_dim * num_heads * head_dim * sizeof(float));
        layer->mlp.gate_proj = (float*)safe_malloc(intermediate_size * embed_dim * sizeof(float));
        layer->mlp.up_proj = (float*)safe_malloc(intermediate_size * embed_dim * sizeof(float));
        layer->mlp.down_proj = (float*)safe_malloc(embed_dim * intermediate_size * sizeof(float));
        layer->rms_attn.weight = (float*)safe_malloc(embed_dim * sizeof(float));
        layer->rms_ffn.weight = (float*)safe_malloc(embed_dim * sizeof(float));
        
        fread(layer->self_attn.w_q, sizeof(float), num_heads * head_dim * embed_dim, fp);
        fread(layer->self_attn.w_k, sizeof(float), num_kv_heads * head_dim * embed_dim, fp);
        fread(layer->self_attn.w_v, sizeof(float), num_kv_heads * head_dim * embed_dim, fp);
        fread(layer->self_attn.w_o, sizeof(float), embed_dim * num_heads * head_dim, fp);
        fread(layer->mlp.gate_proj, sizeof(float), intermediate_size * embed_dim, fp);
        fread(layer->mlp.up_proj, sizeof(float), intermediate_size * embed_dim, fp);
        fread(layer->mlp.down_proj, sizeof(float), embed_dim * intermediate_size, fp);
        fread(layer->rms_attn.weight, sizeof(float), embed_dim, fp);
        fread(layer->rms_ffn.weight, sizeof(float), embed_dim, fp);
    }
    
    model->norm.weight = (float*)safe_malloc(embed_dim * sizeof(float));
    model->lm_head.weight = (float*)safe_malloc(vocab_size * embed_dim * sizeof(float));

    fread(model->norm.weight, sizeof(float), embed_dim, fp);
    fread(model->lm_head.weight, sizeof(float), vocab_size * embed_dim, fp);
    
    fclose(fp);
    printf("model loaded successfully from: %s\n", model_file_path);
}


void free_model(LlamaModel* model, LlamaConfig* config) {
    free(model->embed_tokens.weight);
    
    for (int l = 0; l < config->num_hidden_layers; l++) {
        LlamaDecoderLayer* layer = &model->layers[l];
        free(layer->rms_attn.weight);
        free(layer->self_attn.w_q);
        free(layer->self_attn.w_k);
        free(layer->self_attn.w_v);
        free(layer->self_attn.w_o);
        free(layer->rms_ffn.weight);
        free(layer->mlp.gate_proj);
        free(layer->mlp.up_proj);
        free(layer->mlp.down_proj);
    }
    
    free(model->layers);
    free(model->norm.weight);
    free(model->lm_head.weight);
}


void malloc_activation(
    LlamaModelActivation* activation,
    LlamaConfig* config
) {
    size_t batch_size = config->batch_size; 
    size_t seq_len = config->seq_len;
    size_t embed_dim = config->embed_dim;
    size_t intermediate_size = config->intermediate_size;
    size_t vocab_size = config->vocab_size;
    size_t hidden_size = batch_size * seq_len * embed_dim;

    // embedding
    activation->embed_tokens.output = (float*)safe_malloc(sizeof(float) * hidden_size);

    // decoder 
    activation->decoder.self_attn.query = (float*)safe_malloc(sizeof(float) * batch_size * seq_len * embed_dim);
    activation->decoder.self_attn.key = (float*)safe_malloc(sizeof(float) * batch_size * seq_len * embed_dim);
    activation->decoder.self_attn.query_rope = (float*)safe_malloc(sizeof(float) * batch_size * seq_len * embed_dim);
    activation->decoder.self_attn.key_rope = (float*)safe_malloc(sizeof(float) * batch_size * seq_len * embed_dim);
    activation->decoder.self_attn.q_kt = (float*)safe_malloc(sizeof(float) * seq_len * seq_len);
    activation->decoder.self_attn.attn_weights = (float*)safe_malloc(sizeof(float) * seq_len * seq_len);
    activation->decoder.self_attn.value = (float*)safe_malloc(sizeof(float) * batch_size * seq_len * embed_dim);
    activation->decoder.self_attn.attn_output = (float*)calloc(hidden_size, sizeof(float));
    activation->decoder.self_attn.output = (float*)safe_malloc(sizeof(float) * hidden_size);

    activation->decoder.mlp.gate = (float*)safe_malloc(sizeof(float) * intermediate_size);
    activation->decoder.mlp.up = (float*)safe_malloc(sizeof(float) * intermediate_size);
    activation->decoder.mlp.gate_silu = (float*)safe_malloc(sizeof(float) * intermediate_size);
    activation->decoder.mlp.gate_mul_up = (float*)safe_malloc(sizeof(float) * intermediate_size);
    activation->decoder.mlp.output = (float*)safe_malloc(sizeof(float) * hidden_size);

    activation->decoder.rms_attn.output = (float*)safe_malloc(sizeof(float) * hidden_size);
    activation->decoder.rms_ffn.output = (float*)safe_malloc(sizeof(float) * hidden_size);

    // final
    activation->norm.output = (float*)safe_malloc(sizeof(float) * hidden_size);
    activation->lm_head.output = (float*)safe_malloc(sizeof(float) * batch_size * seq_len * vocab_size);
}


void free_activation(
    LlamaModelActivation* activation,
    LlamaConfig* config
) {
    // embedding
    free(activation->embed_tokens.output);

    // decoder 
    free(activation->decoder.self_attn.query);
    free(activation->decoder.self_attn.key);
    free(activation->decoder.self_attn.query_rope);
    free(activation->decoder.self_attn.key_rope);
    free(activation->decoder.self_attn.q_kt);
    free(activation->decoder.self_attn.attn_weights);
    free(activation->decoder.self_attn.value);
    free(activation->decoder.self_attn.attn_output);
    free(activation->decoder.self_attn.output);

    free(activation->decoder.mlp.gate);
    free(activation->decoder.mlp.up);
    free(activation->decoder.mlp.gate_silu);
    free(activation->decoder.mlp.gate_mul_up);

    free(activation->decoder.rms_attn.output);
    free(activation->decoder.rms_ffn.output);

    // final
    free(activation->norm.output);
    free(activation->lm_head.output);
}


void print_config(LlamaConfig *config) {
    printf(
        "\nmodel config:\nbatch_size: %ld\nseq_len: %ld\nembed_dim: %ld\nnum_heads: %ld\nnum_kv_heads: %ld"
        "\nhead_dim: %ld\nintermediate_size: %ld\nvocab_size: %ld\nnum_hidden_layers: %ld\nrope_theta: %e\nrms_eps: %e\n",
        config->batch_size, config->seq_len, config->embed_dim, config->num_heads, config->num_kv_heads, config->head_dim,
        config->intermediate_size, config->vocab_size, config->num_hidden_layers, config->rope_theta, config->rms_eps
    );
}


int main(int argc, char *argv[]) {
    char *model_path = NULL;
    char model_file_path[256], config_file_path[256];
    char fn_inputs_embeds[256], fn_input[256], fn_output[256];
    parse_args(argc, argv, &model_path);    
    printf("model path:  %s\n", model_path);
    if (!model_path) {
        fprintf(stderr, "Error: --model-path are required\n");
        return 1;
    } else {
        snprintf(model_file_path, sizeof(model_file_path), "%s/%s", model_path, "model.bin");
        snprintf(config_file_path, sizeof(config_file_path), "%s/%s", model_path, "config.bin");
        snprintf(fn_inputs_embeds, sizeof(fn_inputs_embeds), "%s/%s", model_path, "inputs_embeds.bin");
        snprintf(fn_input, sizeof(fn_input), "%s/%s", model_path, "tensor_input.bin");
        snprintf(fn_output, sizeof(fn_output), "%s/%s", model_path, "tensor_output.bin");
    }
    printf("model path: %s\n", model_file_path);
    printf("config path: %s\n", config_file_path);

    FILE *fp_input = fopen(fn_input, "rb");
    FILE *fp_output = fopen(fn_output, "rb");

    if (!fp_input || !fp_output) {
        fprintf(stderr, "ERROR: Cannot open one of the files\n");
        return 1;
    }

    // load config
    LlamaConfig config;
 	load_config(&config, config_file_path);
    print_config(&config);
    size_t hidden_size = config.batch_size * config.seq_len * config.embed_dim;
    size_t logits_size = config.batch_size * config.seq_len * config.vocab_size;

    // load model
    LlamaModel model;
    load_model(&model, &config, model_file_path);

    // allocate memory for model activaions
    LlamaModelActivation activation;
    malloc_activation(&activation, &config);
    
    // read input and output
    int64_t* input_ids = (int64_t*)malloc(sizeof(int64_t) * config.batch_size * config.seq_len);
    fread(input_ids, sizeof(int64_t), config.batch_size * config.seq_len, fp_input);
    fclose(fp_input);

    float* output = (float*)malloc(sizeof(float) * logits_size);
    fread(output, sizeof(float), logits_size, fp_output);
    fclose(fp_output);

    // printf("input ids:\n");
    // print_int64(input_ids, config.batch_size * config.seq_len);
    // printf("\n");

    printf("\nbefore llama forward:\n");

    llama_forward(
        &config,
        &model,
        &activation,
        input_ids
    );

    // printf("output:\n");
    // print(output, logits_size);
    // printf("\n");

    // printf("output test:\n");
    // print(activation.lm_head.output, logits_size);
    // printf("\n");

    bool is_close_output = allclose(output, activation.lm_head.output, logits_size, 1e-5);
    if (is_close_output) {
        printf("output OK! %d\n", is_close_output);
    } else {
        printf("output NOT OK! %d\n", is_close_output);
    }

    free_activation(&activation, &config);
    free_model(&model, &config);

    free(input_ids);
    free(output);
}