// gpu_profiler.cpp - Profile ALL GPU ops to measure compute and memory bandwidth
// Similar to concurrent_profiler but runs on CUDA instead of CPU

#include "ggml-backend.h"
#include "ggml.h"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

// ============================================================================
// Utilities
// ============================================================================

struct Timer {
    using clk = std::chrono::high_resolution_clock;
    clk::time_point t0;
    void start() { t0 = clk::now(); }
    double stop() { return std::chrono::duration<double>(clk::now() - t0).count(); }
};

enum class Quant { F32, F16, Q8_0, Q4_0, Q4_1, Q5_0, Q2_K };

static ggml_type parse_quant(Quant q) {
    switch (q) {
        case Quant::F32: return GGML_TYPE_F32;
        case Quant::F16: return GGML_TYPE_F16;
        case Quant::Q8_0: return GGML_TYPE_Q8_0;
        case Quant::Q4_0: return GGML_TYPE_Q4_0;
        case Quant::Q4_1: return GGML_TYPE_Q4_1;
        case Quant::Q5_0: return GGML_TYPE_Q5_0;
        case Quant::Q2_K: return GGML_TYPE_Q2_K;
    }
    return GGML_TYPE_F16;
}

static const char* quant_name(Quant q) {
    switch (q) {
        case Quant::F32: return "f32";
        case Quant::F16: return "f16";
        case Quant::Q8_0: return "q8_0";
        case Quant::Q4_0: return "q4_0";
        case Quant::Q4_1: return "q4_1";
        case Quant::Q5_0: return "q5_0";
        case Quant::Q2_K: return "q2_K";
    }
    return "unknown";
}

// ============================================================================
// Benchmark Result Structure
// ============================================================================

struct BenchResult {
    std::string op_name;
    std::string quant_type;
    
    // Dimensions (op-specific)
    int N, K, B;
    int n_tokens, ctx_len, n_heads, head_dim;
    int64_t n_elements;
    
    // Metrics
    double ops;
    double bytes;
    double time_s;
    
    // Derived
    float arithmetic_intensity;
    float effective_gflops;
    float effective_bw_gb_s;
    
    void calculate_derived() {
        arithmetic_intensity = (float)(ops / bytes);
        effective_gflops = (float)(ops / time_s / 1e9);
        effective_bw_gb_s = (float)(bytes / time_s / 1e9);
    }
    
    void print() const {
        printf("%-20s quant=%-6s AI=%.3f FLOP/byte BW=%.2f GB/s Perf=%.2f GFLOP/s",
               op_name.c_str(), quant_type.c_str(),
               arithmetic_intensity, effective_bw_gb_s, effective_gflops);
        
        if (op_name.find("MUL_MAT_ID") != std::string::npos) {
            printf(" [N=%d K=%d experts=%d/%d B=%d]", N, K, n_tokens, ctx_len, B);
        } else if (op_name.find("MUL_MAT") != std::string::npos) {
            printf(" [N=%d K=%d B=%d]", N, K, B);
        } else if (op_name.find("FLASH_ATTN") != std::string::npos) {
            printf(" [tokens=%d ctx=%d heads=%d dim=%d]", n_tokens, ctx_len, n_heads, head_dim);
        } else {
            printf(" [n=%lld]", (long long)n_elements);
        }
        printf("\n");
    }
};

// ============================================================================
// Benchmark Functions
// ============================================================================

double benchmark_mul_mat_raw(ggml_backend_t be, int N, int K, int batch_size, 
                              Quant quant, double* out_time_s, double* out_ops, double* out_bytes) {
    // Use no_alloc for proper GPU backend integration
    ggml_init_params params = {4096ull * 1024 * 1024, NULL, true};  // no_alloc = true
    ggml_context * ctx = ggml_init(params);
    
    ggml_tensor * A = ggml_new_tensor_2d(ctx, parse_quant(quant), K, N);
    ggml_tensor * B_tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, batch_size);
    
    // Build graph first
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_tensor * C = ggml_mul_mat(ctx, A, B_tensor);
    ggml_build_forward_expand(gf, C);
    
    // Allocate all tensors using backend
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, be);
    if (!buffer) {
        fprintf(stderr, "Failed to allocate backend buffer\n");
        ggml_free(ctx);
        return 0.0;
    }
    
    // Now set input data AFTER allocation
    std::vector<float> A_float(K * N, 0.5f);
    if (quant == Quant::F32) {
        ggml_backend_tensor_set(A, A_float.data(), 0, ggml_nbytes(A));
    } else {
        // For quantized types, quantize the float data first
        std::vector<uint8_t> A_quantized(ggml_nbytes(A));
        ggml_quantize_chunk(parse_quant(quant), A_float.data(), A_quantized.data(), 0, K * N / ggml_blck_size(parse_quant(quant)), 1, nullptr);
        ggml_backend_tensor_set(A, A_quantized.data(), 0, ggml_nbytes(A));
    }
    
    std::vector<float> B_data(K * batch_size, 1.0f);
    ggml_backend_tensor_set(B_tensor, B_data.data(), 0, ggml_nbytes(B_tensor));
    
    // Warmup
    for (int i = 0; i < 5; ++i) {
        ggml_backend_graph_compute(be, gf);
    }
    ggml_backend_synchronize(be);
    
    // Timed run
    const int iterations = 10;
    Timer t;
    t.start();
    
    for (int i = 0; i < iterations; ++i) {
        ggml_backend_graph_compute_async(be, gf);
    }
    ggml_backend_synchronize(be);
    double total_time = t.stop();
    
    double time_per_iter = total_time / iterations;
    double ops_total = 2.0 * N * K * batch_size;
    double bytes_total = (double)(ggml_nbytes(A) + ggml_nbytes(B_tensor) + ggml_nbytes(C));
    double gflops = ops_total / time_per_iter / 1e9;
    
    if (out_time_s) *out_time_s = time_per_iter;
    if (out_ops) *out_ops = ops_total;
    if (out_bytes) *out_bytes = bytes_total;
    
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    
    return gflops;
}

BenchResult benchmark_mul_mat(ggml_backend_t be, int N, int K, int batch_size, Quant quant) {
    BenchResult result;
    result.op_name = "MUL_MAT";
    result.quant_type = quant_name(quant);
    result.N = N;
    result.K = K;
    result.B = batch_size;
    
    benchmark_mul_mat_raw(be, N, K, batch_size, quant, &result.time_s, &result.ops, &result.bytes);
    result.calculate_derived();
    
    return result;
}

double benchmark_mul_mat_id_raw(ggml_backend_t be, int N, int K, int n_experts,
                                 int n_experts_used, int batch_size, Quant quant,
                                 double * out_time_s, double * out_ops, double * out_bytes) {
    ggml_init_params params = {8192ull * 1024 * 1024, NULL, true};  // no_alloc = true
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        fprintf(stderr, "Failed to initialize context\n");
        return 0.0;
    }

    // A: expert weights [K, N, n_experts]
    // B: input [K, 1, batch_size] - must be 3D so b->ne[2] == ids->ne[1]
    // ids: expert indices [n_experts_used, batch_size]
    ggml_tensor * A   = ggml_new_tensor_3d(ctx, parse_quant(quant), K, N, n_experts);
    ggml_tensor * B   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, 1, batch_size);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_experts_used, batch_size);

    // check memory requirements
    size_t total_memory_needed = ggml_nbytes(A) + ggml_nbytes(B) + ggml_nbytes(ids);
    size_t gpu_free = 0, gpu_total = 0;
    ggml_backend_dev_memory(ggml_backend_get_device(be), &gpu_free, &gpu_total);

    if (total_memory_needed > gpu_free) {
        fprintf(stderr, "Insufficient GPU memory: need %.2f GB, have %.2f GB free\n",
                total_memory_needed / 1e9, gpu_free / 1e9);
        ggml_free(ctx);
        return 0.0;
    }

    // build graph first
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_tensor * C = ggml_mul_mat_id(ctx, A, B, ids);
    ggml_build_forward_expand(gf, C);

    // allocate all tensors using backend
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, be);
    if (!buffer) {
        fprintf(stderr, "Failed to allocate backend buffer (OOM?)\n");
        ggml_free(ctx);
        return 0.0;
    }

    // set input data AFTER allocation
    // for large tensors, just zero-initialize to avoid "vector too long" errors
    std::vector<uint8_t> A_data(ggml_nbytes(A), 0);
    ggml_backend_tensor_set(A, A_data.data(), 0, ggml_nbytes(A));

    std::vector<float> B_data(K * batch_size, 1.0f);
    ggml_backend_tensor_set(B, B_data.data(), 0, ggml_nbytes(B));
    
    std::vector<int32_t> ids_data(n_experts_used * batch_size);
    for (int i = 0; i < n_experts_used * batch_size; i++) {
        ids_data[i] = i % n_experts;
    }
    ggml_backend_tensor_set(ids, ids_data.data(), 0, ggml_nbytes(ids));
    
    // Warmup
    for (int i = 0; i < 5; ++i) {
        ggml_backend_graph_compute(be, gf);
    }
    ggml_backend_synchronize(be);
    
    // Timed run
    const int iterations = 10;
    Timer t;
    t.start();
    
    for (int i = 0; i < iterations; ++i) {
        ggml_backend_graph_compute_async(be, gf);
    }
    ggml_backend_synchronize(be);
    double total_time = t.stop();
    
    double time_per_iter = total_time / iterations;
    double ops_total = 2.0 * N * K * batch_size * n_experts_used;
    double bytes_total = (ggml_nbytes(A) * n_experts_used / n_experts) + ggml_nbytes(B) + ggml_nbytes(C);
    double gflops = ops_total / time_per_iter / 1e9;
    
    if (out_time_s) *out_time_s = time_per_iter;
    if (out_ops) *out_ops = ops_total;
    if (out_bytes) *out_bytes = bytes_total;
    
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    
    return gflops;
}

BenchResult benchmark_mul_mat_id(ggml_backend_t be, int N, int K, int n_experts, 
                                  int n_experts_used, int batch_size, Quant quant) {
    BenchResult result;
    result.op_name = "MUL_MAT_ID";
    result.quant_type = quant_name(quant);
    result.N = N;
    result.K = K;
    result.B = batch_size;
    result.n_tokens = n_experts_used;
    result.ctx_len = n_experts;
    
    double gflops = benchmark_mul_mat_id_raw(be, N, K, n_experts, n_experts_used, batch_size, 
                                              quant, &result.time_s, &result.ops, &result.bytes);
    
    if (gflops == 0.0) {
        throw std::runtime_error("Benchmark failed (likely OOM)");
    }
    
    result.calculate_derived();
    
    return result;
}

double benchmark_flash_attn_raw(ggml_backend_t be, int n_tokens, int ctx_len, 
                                int n_q_heads, int n_kv_heads, int head_dim, 
                                Quant kv_quant, double* out_time_s, double* out_ops, double* out_bytes) {
    // Use no_alloc for proper GPU backend integration
    ggml_init_params params = {8192ull * 1024 * 1024, NULL, true};  // no_alloc = true
    ggml_context * ctx = ggml_init(params);
    
    const int batch = 1;
    ggml_tensor * Q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_dim, n_tokens, n_q_heads, batch);
    ggml_tensor * K = ggml_new_tensor_4d(ctx, parse_quant(kv_quant), head_dim, ctx_len, n_kv_heads, batch);
    ggml_tensor * V = ggml_new_tensor_4d(ctx, parse_quant(kv_quant), head_dim, ctx_len, n_kv_heads, batch);
    ggml_tensor * mask = nullptr;
    
    // Build graph first
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_tensor * out = ggml_flash_attn_ext(ctx, Q, K, V, mask, 1.0f / sqrtf((float)head_dim), 0.0f, 0.0f);
    ggml_build_forward_expand(gf, out);
    
    // Allocate all tensors using backend
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, be);
    if (!buffer) {
        fprintf(stderr, "Failed to allocate backend buffer\n");
        ggml_free(ctx);
        return 0.0;
    }
    
    // Now set input data AFTER allocation
    std::vector<float> Q_data(ggml_nelements(Q), 1.0f);
    ggml_backend_tensor_set(Q, Q_data.data(), 0, ggml_nbytes(Q));
    
    // Initialize K and V with quantized data
    int64_t kv_elements = ggml_nelements(K);
    std::vector<float> KV_float(kv_elements, 0.5f);
    
    if (kv_quant == Quant::F32) {
        ggml_backend_tensor_set(K, KV_float.data(), 0, ggml_nbytes(K));
        ggml_backend_tensor_set(V, KV_float.data(), 0, ggml_nbytes(V));
    } else {
        std::vector<uint8_t> K_quantized(ggml_nbytes(K));
        std::vector<uint8_t> V_quantized(ggml_nbytes(V));
        ggml_quantize_chunk(parse_quant(kv_quant), KV_float.data(), K_quantized.data(), 0, kv_elements / ggml_blck_size(parse_quant(kv_quant)), 1, nullptr);
        ggml_quantize_chunk(parse_quant(kv_quant), KV_float.data(), V_quantized.data(), 0, kv_elements / ggml_blck_size(parse_quant(kv_quant)), 1, nullptr);
        ggml_backend_tensor_set(K, K_quantized.data(), 0, ggml_nbytes(K));
        ggml_backend_tensor_set(V, V_quantized.data(), 0, ggml_nbytes(V));
    }
    
    // Warmup
    for (int i = 0; i < 5; ++i) {
        ggml_backend_graph_compute(be, gf);
    }
    ggml_backend_synchronize(be);
    
    // Timed
    const int iterations = 10;
    Timer t;
    t.start();
    
    for (int i = 0; i < iterations; ++i) {
        ggml_backend_graph_compute_async(be, gf);
    }
    ggml_backend_synchronize(be);
    double total_time = t.stop();
    
    double time_per_iter = total_time / iterations;
    double ops_total = 2.0 * n_tokens * head_dim * ctx_len * n_q_heads * 2;
    double bytes_total = (double)(ggml_nbytes(Q) + ggml_nbytes(K) + ggml_nbytes(V) + ggml_nbytes(out));
    double gflops = ops_total / time_per_iter / 1e9;
    
    if (out_time_s) *out_time_s = time_per_iter;
    if (out_ops) *out_ops = ops_total;
    if (out_bytes) *out_bytes = bytes_total;
    
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    
    return gflops;
}

BenchResult benchmark_flash_attn(ggml_backend_t be, int n_tokens, int ctx_len, 
                                  int n_q_heads, int n_kv_heads, int head_dim, 
                                  Quant kv_quant, const char* arch_name) {
    BenchResult result;
    result.op_name = std::string("FLASH_ATTN_") + arch_name;
    result.quant_type = quant_name(kv_quant);
    result.n_tokens = n_tokens;
    result.ctx_len = ctx_len;
    result.n_heads = n_kv_heads;
    result.head_dim = head_dim;
    
    benchmark_flash_attn_raw(be, n_tokens, ctx_len, n_q_heads, n_kv_heads, 
                             head_dim, kv_quant, &result.time_s, &result.ops, &result.bytes);
    result.calculate_derived();
    
    return result;
}

// Element-wise operations
double benchmark_elem_raw(ggml_backend_t be, const char* op_name, int64_t n_elements, 
                           double* out_time_s, double* out_ops, double* out_bytes) {
    // Use no_alloc for proper GPU backend integration
    ggml_init_params params = {256ull * 1024 * 1024, NULL, true};  // no_alloc = true
    ggml_context * ctx = ggml_init(params);
    
    ggml_tensor * X = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_elements);
    ggml_tensor * Y = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_elements);
    
    // Build graph first
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_tensor * out = nullptr;
    
    if (strcmp(op_name, "ADD") == 0) out = ggml_add(ctx, X, Y);
    else if (strcmp(op_name, "MUL") == 0) out = ggml_mul(ctx, X, Y);
    else if (strcmp(op_name, "SUB") == 0) out = ggml_sub(ctx, X, Y);
    else if (strcmp(op_name, "DIV") == 0) out = ggml_div(ctx, X, Y);
    
    ggml_build_forward_expand(gf, out);
    
    // Allocate all tensors using backend
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, be);
    if (!buffer) {
        fprintf(stderr, "Failed to allocate backend buffer\n");
        ggml_free(ctx);
        return 0.0;
    }
    
    // Now set input data AFTER allocation
    std::vector<float> X_data(n_elements, 1.5f);
    std::vector<float> Y_data(n_elements, 2.5f);
    ggml_backend_tensor_set(X, X_data.data(), 0, ggml_nbytes(X));
    ggml_backend_tensor_set(Y, Y_data.data(), 0, ggml_nbytes(Y));
    
    // Warmup
    for (int i = 0; i < 5; ++i) {
        ggml_backend_graph_compute(be, gf);
    }
    ggml_backend_synchronize(be);
    
    // Timed
    const int iterations = 10;
    Timer t;
    t.start();
    
    for (int i = 0; i < iterations; ++i) {
        ggml_backend_graph_compute_async(be, gf);
    }
    ggml_backend_synchronize(be);
    double total_time = t.stop();
    
    double time_per_iter = total_time / iterations;
    double ops_total = (double)n_elements;
    double bytes_total = (double)(ggml_nbytes(X) + ggml_nbytes(Y) + ggml_nbytes(out));
    double gflops = ops_total / time_per_iter / 1e9;
    
    if (out_time_s) *out_time_s = time_per_iter;
    if (out_ops) *out_ops = ops_total;
    if (out_bytes) *out_bytes = bytes_total;
    
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    
    return gflops;
}

BenchResult benchmark_elem(ggml_backend_t be, const char* op_name, int64_t n_elements) {
    BenchResult result;
    result.op_name = op_name;
    result.quant_type = "f32";
    result.n_elements = n_elements;
    
    benchmark_elem_raw(be, op_name, n_elements, &result.time_s, &result.ops, &result.bytes);
    result.calculate_derived();
    
    return result;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char ** argv) {
    printf("=== GPU Profiler (CUDA Backend) ===\n\n");

    bool    fast_mode = false;
    int32_t n_ubatch  = 512;  // max ubatch size

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--fast")) {
            fast_mode = true;
        } else if ((!strcmp(argv[i], "-ub") || !strcmp(argv[i], "--ubatch")) && i + 1 < argc) {
            n_ubatch = std::stoi(argv[++i]);
        }
    }

    // Profile key batch sizes: decode (1), continuous batching (64), prompt (512)
    std::vector<int32_t> batch_sizes = {1, 64, 512};

    printf("n_ubatch:     %d\n", n_ubatch);
    printf("Tiers (%zu):   ", batch_sizes.size());
    for (size_t i = 0; i < batch_sizes.size(); ++i) {
        printf("%d%s", batch_sizes[i], (i < batch_sizes.size() - 1) ? ", " : "\n");
    }
    printf("Profile mode: %s\n", fast_mode ? "FAST (reduced configs)" : "FULL (all configs)");
    printf("\n");

    #if defined(_WIN32)
        _putenv("GGML_CUDA_PIPELINE_SHARDING=0");
    #else
        setenv("GGML_CUDA_PIPELINE_SHARDING", "0", 1);
    #endif

    // initialize GPU backend
    ggml_backend_t gpu_be = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    if (!gpu_be) {
        fprintf(stderr, "Failed to initialize GPU backend (CUDA not available?)\n");
        return 1;
    }

    printf("GPU: %s\n\n", ggml_backend_name(gpu_be));

    // initialize quantization tables for k-quants
    ggml_quantize_init(GGML_TYPE_Q2_K);
    ggml_quantize_init(GGML_TYPE_Q4_0);
    ggml_quantize_init(GGML_TYPE_Q4_1);
    ggml_quantize_init(GGML_TYPE_Q5_0);
    ggml_quantize_init(GGML_TYPE_Q8_0);

    std::vector<BenchResult> all_results;

    // all matrix sizes (N, K) - square, wide, tall
    struct matmul_size {
        int32_t N;
        int32_t K;
    };

    std::vector<matmul_size> matmul_sizes;
    if (fast_mode) {
        // fast mode: 6 representative sizes (small, medium, large x square/rect)
        matmul_sizes.push_back({  256,   256});
        matmul_sizes.push_back({ 1024,  1024});
        matmul_sizes.push_back({ 4096,  4096});
        matmul_sizes.push_back({ 4096,  1024});  // wide
        matmul_sizes.push_back({ 1024,  4096});  // tall
        matmul_sizes.push_back({ 8192,  4096});  // large wide
    } else {
        // square matrices
        matmul_sizes.push_back({  128,   128});
        matmul_sizes.push_back({  256,   256});
        matmul_sizes.push_back({  512,   512});
        matmul_sizes.push_back({ 1024,  1024});
        matmul_sizes.push_back({ 2048,  2048});
        matmul_sizes.push_back({ 4096,  4096});
        matmul_sizes.push_back({ 8192,  8192});
        matmul_sizes.push_back({16384, 16384});
        // wide matrices (N > K)
        matmul_sizes.push_back({  256,   128});
        matmul_sizes.push_back({  512,   256});
        matmul_sizes.push_back({ 1024,   512});
        matmul_sizes.push_back({ 2048,  1024});
        matmul_sizes.push_back({ 4096,  2048});
        matmul_sizes.push_back({ 8192,  4096});
        matmul_sizes.push_back({16384,  8192});
        matmul_sizes.push_back({32768, 16384});
        // tall matrices (K > N)
        matmul_sizes.push_back({  128,   256});
        matmul_sizes.push_back({  256,   512});
        matmul_sizes.push_back({  512,  1024});
        matmul_sizes.push_back({ 1024,  2048});
        matmul_sizes.push_back({ 2048,  4096});
        matmul_sizes.push_back({ 4096,  8192});
        matmul_sizes.push_back({ 8192, 16384});
        matmul_sizes.push_back({16384, 32768});
    }

    std::vector<Quant> matmul_quants;
    matmul_quants.push_back(Quant::F32);
    matmul_quants.push_back(Quant::F16);
    matmul_quants.push_back(Quant::Q8_0);
    matmul_quants.push_back(Quant::Q4_0);
    matmul_quants.push_back(Quant::Q4_1);
    matmul_quants.push_back(Quant::Q5_0);
    matmul_quants.push_back(Quant::Q2_K);

    // === MUL_MAT ===
    printf("=== MUL_MAT Operations ===\n\n");
    for (size_t qi = 0; qi < matmul_quants.size(); ++qi) {
        Quant quant = matmul_quants[qi];
        printf("--- Quantization: %s ---\n", quant_name(quant));

        for (size_t bi = 0; bi < batch_sizes.size(); ++bi) {
            int32_t batch_size = batch_sizes[bi];
            printf("  [Batch=%d]\n", batch_size);

            for (size_t si = 0; si < matmul_sizes.size(); ++si) {
                int32_t N = matmul_sizes[si].N;
                int32_t K = matmul_sizes[si].K;

                // Q2_K requires K dimension divisible by 256 (QK_K block size)
                if (quant == Quant::Q2_K && (K % 256 != 0)) {
                    continue;
                }

                BenchResult res = benchmark_mul_mat(gpu_be, N, K, batch_size, quant);
                res.print();
                all_results.push_back(res);
            }
        }
        printf("\n");
    }

    // === MUL_MAT_ID ===
    printf("=== MUL_MAT_ID Operations (MoE) ===\n\n");

    struct moe_config {
        int32_t N;
        int32_t K;
        int32_t n_experts;
        int32_t n_experts_used;
    };

    std::vector<moe_config> moe_configs;
    if (fast_mode) {
        // fast mode: 4 representative MoE sizes
        moe_configs.push_back({  256,   256, 128, 8});
        moe_configs.push_back({ 1024,  1024, 128, 8});
        moe_configs.push_back({ 2048,  1024, 128, 8});  // wide
        moe_configs.push_back({ 1024,  2048, 128, 8});  // tall
    } else {
        // square
        moe_configs.push_back({  128,   128, 128, 8});
        moe_configs.push_back({  256,   256, 128, 8});
        moe_configs.push_back({  512,   512, 128, 8});
        moe_configs.push_back({ 1024,  1024, 128, 8});
        moe_configs.push_back({ 2048,  2048, 128, 8});
        moe_configs.push_back({ 4096,  4096, 128, 8});
        // wide
        moe_configs.push_back({  256,   128, 128, 8});
        moe_configs.push_back({  512,   256, 128, 8});
        moe_configs.push_back({ 1024,   512, 128, 8});
        moe_configs.push_back({ 2048,  1024, 128, 8});
        moe_configs.push_back({ 4096,  2048, 128, 8});
        // tall
        moe_configs.push_back({  128,   256, 128, 8});
        moe_configs.push_back({  256,   512, 128, 8});
        moe_configs.push_back({  512,  1024, 128, 8});
        moe_configs.push_back({ 1024,  2048, 128, 8});
        moe_configs.push_back({ 2048,  4096, 128, 8});
    }

    std::vector<Quant> moe_quants;
    moe_quants.push_back(Quant::F32);
    moe_quants.push_back(Quant::F16);
    moe_quants.push_back(Quant::Q8_0);
    moe_quants.push_back(Quant::Q4_0);
    moe_quants.push_back(Quant::Q4_1);
    moe_quants.push_back(Quant::Q5_0);
    moe_quants.push_back(Quant::Q2_K);

    for (size_t qi = 0; qi < moe_quants.size(); ++qi) {
        Quant quant = moe_quants[qi];
        printf("--- MoE Quantization: %s ---\n", quant_name(quant));

        for (size_t bi = 0; bi < batch_sizes.size(); ++bi) {
            int32_t batch_size = batch_sizes[bi];
            printf("  [Batch=%d]\n", batch_size);

            for (size_t ci = 0; ci < moe_configs.size(); ++ci) {
                int32_t N              = moe_configs[ci].N;
                int32_t K              = moe_configs[ci].K;
                int32_t n_experts      = moe_configs[ci].n_experts;
                int32_t n_experts_used = moe_configs[ci].n_experts_used;

                // Q2_K requires K dimension divisible by 256 (QK_K block size)
                if (quant == Quant::Q2_K && (K % 256 != 0)) {
                    continue;
                }

                try {
                    BenchResult res = benchmark_mul_mat_id(gpu_be, N, K, n_experts, n_experts_used, batch_size, quant);
                    res.print();
                    all_results.push_back(res);
                } catch (const std::exception & e) {
                    printf("SKIPPED: MUL_MAT_ID N=%d K=%d experts=%d/%d quant=%s (error: %s)\n",
                           N, K, n_experts_used, n_experts, quant_name(quant), e.what());
                }
            }
        }
        printf("\n");
    }

    // === FLASH_ATTN ===
    printf("=== FLASH_ATTN Operations ===\n\n");

    std::vector<int32_t> attn_ctx_lens;
    if (fast_mode) {
        // fast mode: 3 representative context lengths
        attn_ctx_lens.push_back( 1024);
        attn_ctx_lens.push_back( 4096);
        attn_ctx_lens.push_back(16384);
    } else {
        attn_ctx_lens.push_back( 1024);
        attn_ctx_lens.push_back( 2048);
        attn_ctx_lens.push_back( 4096);
        attn_ctx_lens.push_back( 8192);
        attn_ctx_lens.push_back(16384);
        attn_ctx_lens.push_back(32768);
        attn_ctx_lens.push_back(65536);
    }

    const int32_t n_q_heads = 32;
    const int32_t head_dim  = 128;

    struct attn_config {
        const char * name;
        int32_t      n_kv_heads;
    };

    std::vector<attn_config> attn_configs;
    if (fast_mode) {
        // fast mode: 3 representative attention types
        attn_configs.push_back({"MHA",   32});
        attn_configs.push_back({"GQA-8",  8});
        attn_configs.push_back({"MQA",    1});
    } else {
        attn_configs.push_back({"MHA",    32});
        attn_configs.push_back({"GQA-16", 16});
        attn_configs.push_back({"GQA-8",   8});
        attn_configs.push_back({"GQA-4",   4});
        attn_configs.push_back({"GQA-2",   2});
        attn_configs.push_back({"MQA",     1});
    }

    for (size_t ai = 0; ai < attn_configs.size(); ++ai) {
        const attn_config & config = attn_configs[ai];
        printf("--- %s (n_kv=%d) ---\n", config.name, config.n_kv_heads);

        // use batch_sizes as n_tokens for tier-based profiling
        for (size_t bi = 0; bi < batch_sizes.size(); ++bi) {
            int32_t n_tokens = batch_sizes[bi];
            printf("  [n_tokens=%d]\n", n_tokens);

            for (size_t li = 0; li < attn_ctx_lens.size(); ++li) {
                int32_t ctx_len = attn_ctx_lens[li];

                BenchResult res = benchmark_flash_attn(
                    gpu_be, n_tokens, ctx_len, n_q_heads, config.n_kv_heads,
                    head_dim, Quant::F16, config.name);
                res.print();
                all_results.push_back(res);
            }
        }
        printf("\n");
    }

    // === Element-wise ===
    // skip in fast mode - element-wise ops are trivial and not worth profiling
    if (!fast_mode) {
        printf("=== Element-wise Operations ===\n\n");

        std::vector<int64_t> elem_sizes;
        elem_sizes.push_back(  512);
        elem_sizes.push_back( 1024);
        elem_sizes.push_back( 2048);
        elem_sizes.push_back( 4096);
        elem_sizes.push_back( 8192);
        elem_sizes.push_back(16384);
        elem_sizes.push_back(32768);
        elem_sizes.push_back(65536);

        const char * elem_ops[] = {"ADD", "MUL", "SUB", "DIV"};
        const int    n_elem_ops = 4;

        for (int oi = 0; oi < n_elem_ops; ++oi) {
            const char * op = elem_ops[oi];
            printf("--- %s ---\n", op);

            for (size_t ni = 0; ni < elem_sizes.size(); ++ni) {
                int64_t n = elem_sizes[ni];

                BenchResult res = benchmark_elem(gpu_be, op, n);
                res.print();
                all_results.push_back(res);
            }
            printf("\n");
        }
    }

    // === Calculate Ridge Points ===
    printf("\n=== Calculating Ridge Points ===\n");

    // group by (op, quant) and find peak compute
    std::map<std::string, std::vector<BenchResult *>> groups;
    for (size_t ri = 0; ri < all_results.size(); ++ri) {
        std::string key = all_results[ri].op_name + "_" + all_results[ri].quant_type;
        groups[key].push_back(&all_results[ri]);
    }

    // for GPU, we need to estimate peak memory BW
    // typically measured from memory-bound operations
    double peak_gpu_bw = 0.0;
    for (size_t ri = 0; ri < all_results.size(); ++ri) {
        const BenchResult & r = all_results[ri];
        if (r.arithmetic_intensity < 2.0) {  // memory-bound operations
            peak_gpu_bw = std::max(peak_gpu_bw, (double)r.effective_bw_gb_s);
        }
    }

    // calculate peak GPU compute from compute-bound operations
    double peak_gpu_compute = 0.0;
    for (size_t ri = 0; ri < all_results.size(); ++ri) {
        const BenchResult & r = all_results[ri];
        // look for compute-bound operations (high AI or large matmuls)
        if (r.arithmetic_intensity > 10.0 ||
            (r.op_name == "MUL_MAT" && r.N >= 4096 && r.K >= 4096)) {
            peak_gpu_compute = std::max(peak_gpu_compute, (double)r.effective_gflops);
        }
    }

    printf("Estimated GPU Memory BW: %.1f GB/s (from memory-bound ops)\n", peak_gpu_bw);
    printf("Estimated GPU Compute:   %.1f GFLOP/s (from compute-bound ops)\n\n", peak_gpu_compute);

    std::map<std::string, double> ridge_points;
    for (std::map<std::string, std::vector<BenchResult *>>::iterator it = groups.begin(); it != groups.end(); ++it) {
        const std::string & key = it->first;
        const std::vector<BenchResult *> & group = it->second;

        double peak_gflops = 0.0;
        for (size_t gi = 0; gi < group.size(); ++gi) {
            peak_gflops = std::max(peak_gflops, (double)group[gi]->effective_gflops);
        }

        // use GPU memory BW for ridge
        double ridge = peak_gpu_bw > 0 ? (peak_gflops / peak_gpu_bw) : 0.0;
        ridge_points[key] = ridge;

        printf("%s: Peak=%.1f GFLOP/s, Ridge=%.3f FLOP/byte\n",
               key.c_str(), peak_gflops, ridge);
    }

    // === Overall Summary ===
    printf("\n=== Overall Summary ===\n");
    printf("Batch sizes tested: %zu\n", batch_sizes.size());
    printf("Total benchmarks:   %zu\n", all_results.size());

    // === Save Results ===
    printf("\n=== Saving Results ===\n");
    FILE * f = fopen("gpu_results.txt", "w");
    if (f) {
        fprintf(f, "# GPU Profiling (backend=%s, batch_sizes=[", ggml_backend_name(gpu_be));
        for (size_t i = 0; i < batch_sizes.size(); ++i) {
            fprintf(f, "%d%s", batch_sizes[i], (i < batch_sizes.size() - 1) ? "," : "");
        }
        fprintf(f, "], GPU_Memory_BW=%.1f GB/s, GPU_Peak_Compute=%.1f GFLOP/s)\n",
                peak_gpu_bw, peak_gpu_compute);
        fprintf(f, "# op_name quant AI(FLOP/byte) BW(GB/s) GFLOP/s Ridge(FLOP/byte) N K B n_tokens ctx_len n_heads head_dim n_elements\n");

        for (size_t ri = 0; ri < all_results.size(); ++ri) {
            const BenchResult & r = all_results[ri];
            std::string key = r.op_name + "_" + r.quant_type;
            double ridge = ridge_points.count(key) ? ridge_points[key] : 0.0;

            fprintf(f, "%s %s %.4f %.2f %.2f %.4f %d %d %d %d %d %d %d %lld\n",
                    r.op_name.c_str(), r.quant_type.c_str(),
                    r.arithmetic_intensity, r.effective_bw_gb_s, r.effective_gflops, ridge,
                    r.N, r.K, r.B, r.n_tokens, r.ctx_len, r.n_heads, r.head_dim,
                    (long long)r.n_elements);
        }
        fclose(f);
        printf("Results saved to gpu_results.txt\n");
        printf("Total benchmarks completed: %zu\n", all_results.size());
    } else {
        fprintf(stderr, "Failed to open gpu_results.txt for writing\n");
    }

    ggml_backend_free(gpu_be);

    // cleanup quantization tables
    ggml_quantize_free();

    return 0;
}
