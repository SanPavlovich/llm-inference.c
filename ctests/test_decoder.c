#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <string.h>
#include "utils.h"
#include "ops.h"
#include "llm_struct.h"


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


void linear_forward(
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

            linear_forward(embed_dim, intermediate_size, weight_gate_proj, curr_input_embed, gate);
            linear_forward(embed_dim, intermediate_size, weight_up_proj, curr_input_embed, up);
            silu(intermediate_size, gate, gate_silu);
            for(int i=0; i < intermediate_size; i++) {
                gate_mul_up[i] = gate_silu[i] * up[i];
            }
            linear_forward(intermediate_size, embed_dim, weight_down_proj, gate_mul_up, curr_output_embed);
        }
    }
}


void residual(
    size_t size,
    float* input,
    float* residual
) {
    for(int i=0; i < size; i++) {
        input[i] += residual[i];
    }
}


void llama_decoder_forward(
    LlamaConfig* config,
    LlamaDecoderLayer* params,
    LlamaDecoderActivation* activation,
    float* cos,
    float* sin,
    float* causal_mask,
    float* input
) {
    size_t total_elements = config->batch_size * config->seq_len * config->embed_dim;
    rmsnorm(
        config->batch_size, config->seq_len, config->embed_dim, config->rms_eps,
        params->rms_attn.weight,
        input,
        activation->rms_attn.output
    );
    llama_attention(
        config->batch_size, config->num_heads, config->num_kv_heads, config->seq_len, config->head_dim,
        cos, sin, causal_mask, 
        params->self_attn.w_q, params->self_attn.w_k, params->self_attn.w_v, params->self_attn.w_o,
        activation->self_attn.query, activation->self_attn.key, 
        activation->self_attn.query_rope, activation->self_attn.key_rope,
        activation->self_attn.q_kt, activation->self_attn.attn_weights,
        activation->self_attn.value,
        activation->self_attn.attn_output,
        activation->rms_attn.output,    // input
        activation->self_attn.output    // output
    );
    residual(total_elements, input, activation->self_attn.output);
    rmsnorm(
        config->batch_size, config->seq_len, config->embed_dim, config->rms_eps,
        params->rms_ffn.weight,
        input,
        activation->rms_ffn.output
    );
    swiglu(
        config->batch_size, config->seq_len, config->embed_dim, config->intermediate_size,
        params->mlp.gate_proj, params->mlp.up_proj, params->mlp.down_proj,
        activation->mlp.gate, activation->mlp.up, activation->mlp.gate_silu, activation->mlp.gate_mul_up,
        activation->rms_ffn.output,     // input
        activation->mlp.output          // output
    );
    residual(total_elements, input, activation->mlp.output);
}


int main(int argc, char *argv[]) {
    size_t batch_size = atoi(argv[1]);
    size_t seq_len = atoi(argv[2]);
    size_t embed_dim = atoi(argv[3]);
    size_t num_heads = atoi(argv[4]);

    LlamaConfig config;
    config.batch_size = batch_size;
    config.seq_len = seq_len;
    config.embed_dim = embed_dim;
    config.num_heads = num_heads;
    config.num_kv_heads = num_heads / 2;
    config.head_dim = embed_dim / num_heads;
    config.intermediate_size = embed_dim * 2;
    config.rope_theta = 10000.0f;
    config.rms_eps = 1e-6;

    size_t total_elements = batch_size * seq_len * embed_dim;
    size_t hidden_size = sizeof(float) * total_elements;
    size_t freq_elements = seq_len * config.head_dim / 2;
    size_t feqs_size = sizeof(float) * freq_elements;

    // memory allocation for input/output
    float* input = (float*)malloc(hidden_size);
    float* output = (float*)malloc(hidden_size);
    float* output_test = (float*)malloc(hidden_size);

    // memory allocation for model weights
    LlamaDecoderLayer layer;
    layer.self_attn.w_q = malloc(embed_dim * num_heads * config.head_dim * sizeof(float));
    layer.self_attn.w_k = malloc(embed_dim * num_heads * config.head_dim * sizeof(float));
    layer.self_attn.w_v = malloc(embed_dim * num_heads * config.head_dim * sizeof(float));
    layer.self_attn.w_o = malloc(embed_dim * num_heads * config.head_dim * sizeof(float));

    layer.mlp.gate_proj = malloc(embed_dim * config.intermediate_size * sizeof(float));
    layer.mlp.up_proj   = malloc(embed_dim * config.intermediate_size * sizeof(float));
    layer.mlp.down_proj = malloc(config.intermediate_size * embed_dim * sizeof(float));

    layer.rms_attn.weight = malloc(embed_dim * sizeof(float));
    layer.rms_ffn.weight  = malloc(embed_dim * sizeof(float));

    // memory allocation for model activations
    LlamaDecoderActivation activation;
    activation.mlp.gate = (float*)malloc(sizeof(float) * config.intermediate_size);
    activation.mlp.up = (float*)malloc(sizeof(float) * config.intermediate_size);
    activation.mlp.gate_silu = (float*)malloc(sizeof(float) * config.intermediate_size);
    activation.mlp.gate_mul_up = (float*)malloc(sizeof(float) * config.intermediate_size);
    activation.mlp.output = (float*)malloc(sizeof(float) * total_elements);

    activation.self_attn.query = (float*)malloc(sizeof(float) * batch_size * seq_len * embed_dim);
    activation.self_attn.key = (float*)malloc(sizeof(float) * batch_size * seq_len * embed_dim);
    activation.self_attn.query_rope = (float*)malloc(sizeof(float) * batch_size * seq_len * embed_dim);
    activation.self_attn.key_rope = (float*)malloc(sizeof(float) * batch_size * seq_len * embed_dim);
    activation.self_attn.q_kt = (float*)malloc(sizeof(float) * seq_len * seq_len);
    activation.self_attn.attn_weights = (float*)malloc(sizeof(float) * seq_len * seq_len);
    activation.self_attn.value = (float*)malloc(sizeof(float) * batch_size * seq_len * embed_dim);
    activation.self_attn.attn_output = (float*)calloc(total_elements, sizeof(float));
    activation.self_attn.output = (float*)malloc(hidden_size);

    activation.rms_attn.output = (float*)malloc(hidden_size);
    activation.rms_ffn.output = (float*)malloc(hidden_size);

    char fn_input[512], fn_output[512], \
    fn_gate_proj[512], fn_up_proj[512], fn_down_proj[512], \
    fn_w_q[512], fn_w_k[512], fn_w_v[512], fn_w_o[512], \
    fn_rms_attn[512], fn_rms_ffn[512];
    snprintf(fn_input, sizeof(fn_input), "../data/decoder/bs_%zu_sl_%zu_ed_%zu_nh_%zu/tensor_input.bin", batch_size, seq_len, embed_dim, num_heads);
    snprintf(fn_gate_proj, sizeof(fn_gate_proj), "../data/decoder/bs_%zu_sl_%zu_ed_%zu_nh_%zu/mlp__gate_proj.bin", batch_size, seq_len, embed_dim, num_heads);
    snprintf(fn_up_proj, sizeof(fn_up_proj), "../data/decoder/bs_%zu_sl_%zu_ed_%zu_nh_%zu/mlp__up_proj.bin", batch_size, seq_len, embed_dim, num_heads);
    snprintf(fn_down_proj, sizeof(fn_down_proj), "../data/decoder/bs_%zu_sl_%zu_ed_%zu_nh_%zu/mlp__down_proj.bin", batch_size, seq_len, embed_dim, num_heads);
    snprintf(fn_w_q, sizeof(fn_w_q), "../data/decoder/bs_%zu_sl_%zu_ed_%zu_nh_%zu/self_attn__q_proj.bin", batch_size, seq_len, embed_dim, num_heads);
    snprintf(fn_w_k, sizeof(fn_w_k), "../data/decoder/bs_%zu_sl_%zu_ed_%zu_nh_%zu/self_attn__k_proj.bin", batch_size, seq_len, embed_dim, num_heads);
    snprintf(fn_w_v, sizeof(fn_w_v), "../data/decoder/bs_%zu_sl_%zu_ed_%zu_nh_%zu/self_attn__v_proj.bin", batch_size, seq_len, embed_dim, num_heads);
    snprintf(fn_w_o, sizeof(fn_w_o), "../data/decoder/bs_%zu_sl_%zu_ed_%zu_nh_%zu/self_attn__o_proj.bin", batch_size, seq_len, embed_dim, num_heads);
    snprintf(fn_rms_attn, sizeof(fn_rms_attn), "../data/decoder/bs_%zu_sl_%zu_ed_%zu_nh_%zu/input_layernorm.bin", batch_size, seq_len, embed_dim, num_heads);
    snprintf(fn_rms_ffn, sizeof(fn_rms_ffn), "../data/decoder/bs_%zu_sl_%zu_ed_%zu_nh_%zu/post_attention_layernorm.bin", batch_size, seq_len, embed_dim, num_heads);
    snprintf(fn_output, sizeof(fn_output), "../data/decoder/bs_%zu_sl_%zu_ed_%zu_nh_%zu/tensor_output.bin", batch_size, seq_len, embed_dim, num_heads);

    FILE *fp_input = fopen(fn_input, "rb");
    FILE *fp_gate_proj = fopen(fn_gate_proj, "rb");
    FILE *fp_up_proj = fopen(fn_up_proj, "rb");
    FILE *fp_down_proj = fopen(fn_down_proj, "rb");
    FILE *fp_w_q = fopen(fn_w_q, "rb");
    FILE *fp_w_k = fopen(fn_w_k, "rb");
    FILE *fp_w_v = fopen(fn_w_v, "rb");
    FILE *fp_w_o = fopen(fn_w_o, "rb");
    FILE *fp_rms_ffn = fopen(fn_rms_ffn, "rb");
    FILE *fp_rms_attn = fopen(fn_rms_attn, "rb");
    FILE *fp_output = fopen(fn_output, "rb");
    if (!fp_input || !fp_output || \
        !fp_gate_proj || !fp_up_proj || !fp_down_proj || \
        !fp_w_q || !fp_w_k || !fp_w_v || !fp_w_o || \
        !fp_rms_attn || !fp_rms_ffn
    ) {
        fprintf(stderr, "ERROR: Cannot open one of the files\n");
        return 1;
    }

    // read decoder weights
    fread(layer.mlp.gate_proj, sizeof(float), total_elements, fp_gate_proj);
    fclose(fp_gate_proj);
    fread(layer.mlp.up_proj, sizeof(float), total_elements, fp_up_proj);
    fclose(fp_up_proj);
    fread(layer.mlp.down_proj, sizeof(float), total_elements, fp_down_proj);
    fclose(fp_down_proj);
    fread(layer.self_attn.w_q, sizeof(float), total_elements, fp_w_q);
    fclose(fp_w_q);
    fread(layer.self_attn.w_k, sizeof(float), total_elements, fp_w_k);
    fclose(fp_w_k);
    fread(layer.self_attn.w_v, sizeof(float), total_elements, fp_w_v);
    fclose(fp_w_v);
    fread(layer.self_attn.w_o, sizeof(float), total_elements, fp_w_o);
    fclose(fp_w_o);
    fread(layer.rms_attn.weight, sizeof(float), total_elements, fp_rms_attn);
    fclose(fp_rms_attn);
    fread(layer.rms_ffn.weight, sizeof(float), total_elements, fp_rms_ffn);
    fclose(fp_rms_ffn);

    // read reference input and output
    fread(input, sizeof(float), total_elements, fp_input);
    fclose(fp_input);
    fread(output, sizeof(float), total_elements, fp_output);
    fclose(fp_output);

    printf("input:\n");
    print(input, total_elements);
    printf("\n");
    printf("output:\n");
    print(output, total_elements);
    printf("\n");

    float* cos = (float*)malloc(feqs_size);
    float* sin = (float*)malloc(feqs_size);
    float* causal_mask = (float*)malloc(sizeof(float) * seq_len * seq_len);
    create_causal_mask(seq_len, causal_mask);
    rotary_emb(seq_len, config.head_dim, config.rope_theta, cos, sin);

    llama_decoder_forward(
        &config, 
        &layer, 
        &activation, 
        cos, sin, 
        causal_mask,
        input
    );
    // memset(activation.self_attn.attn_output, 0, total_elements);

    printf("output test:\n");
    print(input, total_elements);
    printf("\n");

    bool is_close_output = allclose(output, input, total_elements, 1e-6);
    if (is_close_output) {
        printf("swiglu output OK! %d\n", is_close_output);
    } else {
        printf("swiglu output NOT OK! %d\n", is_close_output);
    }

}