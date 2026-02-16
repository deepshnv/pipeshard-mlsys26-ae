// concurrent_profiler.cpp - Profile ALL CPU ops with and without concurrent PCIe transfers
// Tests all operations from results8.txt to measure CPU/PCIe overlap efficiency

#include "common.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#if defined(_MSC_VER)
#include <malloc.h>
#include <intrin.h>
#endif

// ============================================================================
// Cache Flushing for Cold Measurements
// ============================================================================

static std::vector<char> g_flush_buffer;

void flush_caches() {
    if (g_flush_buffer.empty()) {
        return;
    }
    
    volatile char sum = 0;
    for (size_t i = 0; i < g_flush_buffer.size(); i += 64) {
        sum += g_flush_buffer[i];
        g_flush_buffer[i] = (char)(i & 0xFF);
    }
    g_flush_buffer[0] = sum;
    
    #if defined(_MSC_VER)
        _mm_mfence();
    #elif defined(__GNUC__) || defined(__clang__)
        __sync_synchronize();
    #endif
}

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

// Test CPU-to-DRAM bandwidth (cache-cold sequential reads)
static double benchmark_cpu_dram_bandwidth(int threads) {
    const size_t pool_bytes = 1024 * 1024 * 1024;  // 1 GB
    const size_t chunk_per_thread = pool_bytes / threads;
    const int iterations = 10;
    
    std::vector<uint8_t> pool(pool_bytes);
    
    // Touch once to commit pages
    for (size_t i = 0; i < pool.size(); i += 4096) {
        pool[i] = (uint8_t)(i & 0xFF);
    }
    
    std::vector<std::thread> workers;
    std::vector<uint64_t> sinks(threads, 0);
    std::vector<double> thread_bytes(threads, 0.0);
    
    Timer t;
    t.start();
    
    for (int tid = 0; tid < threads; ++tid) {
        workers.emplace_back([&pool, &sinks, &thread_bytes, tid, chunk_per_thread, iterations, pool_bytes]() {
            const size_t start = tid * chunk_per_thread;
            const size_t end = (tid == (int)(pool_bytes / chunk_per_thread) - 1) ? pool_bytes : (start + chunk_per_thread);
            
            uint64_t local_sink = 0;
            double local_bytes = 0.0;
            
            for (int iter = 0; iter < iterations; ++iter) {
                const size_t limit = end - sizeof(uint64_t);
                size_t offset = start;
                
                while (offset + 64 <= limit) {
                    local_sink += *(const uint64_t *)(pool.data() + offset + 0);
                    local_sink += *(const uint64_t *)(pool.data() + offset + 8);
                    local_sink += *(const uint64_t *)(pool.data() + offset + 16);
                    local_sink += *(const uint64_t *)(pool.data() + offset + 24);
                    local_sink += *(const uint64_t *)(pool.data() + offset + 32);
                    local_sink += *(const uint64_t *)(pool.data() + offset + 40);
                    local_sink += *(const uint64_t *)(pool.data() + offset + 48);
                    local_sink += *(const uint64_t *)(pool.data() + offset + 56);
                    offset += 64;
                    local_bytes += 64.0;
                }
            }
            
            sinks[tid] = local_sink;
            thread_bytes[tid] = local_bytes;
        });
    }
    
    for (auto& w : workers) {
        w.join();
    }
    
    double elapsed = t.stop();
    
    uint64_t total_sink = 0;
    double total_bytes = 0.0;
    for (int i = 0; i < threads; ++i) {
        total_sink += sinks[i];
        total_bytes += thread_bytes[i];
    }
    
    if (total_sink == 0xFFFFFFFFFFFFFFFFull) {
        printf("");
    }
    
    return total_bytes / elapsed / 1e9;
}

// ============================================================================
// PCIe Stress Infrastructure
// ============================================================================

struct PCIeStress {
    std::atomic<bool> active{false};
    std::atomic<bool> stop{false};
    
    ggml_backend_t gpu_backend = nullptr;
    ggml_backend_buffer_t host_buf = nullptr;
    ggml_backend_buffer_t dev_buf = nullptr;
    ggml_tensor *h_tensor = nullptr, *d_tensor = nullptr;
    ggml_context *ctx = nullptr;
    size_t transfer_size = 256 * 1024 * 1024;
    
    // Measured standalone PCIe bandwidth (for reference)
    double calibrated_bw_gb_s = 0.0;
};

void pcie_stress_loop(PCIeStress* pcie) {
    pcie->active.store(true, std::memory_order_release);
    
    while (!pcie->stop.load(std::memory_order_acquire)) {
        // H2D
        ggml_backend_tensor_set_async(pcie->gpu_backend, pcie->d_tensor, 
                                      pcie->h_tensor->data, 0, pcie->transfer_size);
        ggml_backend_synchronize(pcie->gpu_backend);
        
        // D2H
        ggml_backend_tensor_get_async(pcie->gpu_backend, pcie->d_tensor, 
                                      pcie->h_tensor->data, 0, pcie->transfer_size);
        ggml_backend_synchronize(pcie->gpu_backend);
    }
    
    pcie->active.store(false, std::memory_order_release);
}

void calibrate_pcie(PCIeStress* pcie) {
    printf("Calibrating standalone PCIe bandwidth...\n");
    
    Timer t;
    t.start();
    const int cal_iterations = 20;
    
    for (int i = 0; i < cal_iterations; ++i) {
        ggml_backend_tensor_set_async(pcie->gpu_backend, pcie->d_tensor, 
                                      pcie->h_tensor->data, 0, pcie->transfer_size);
        ggml_backend_synchronize(pcie->gpu_backend);
        
        ggml_backend_tensor_get_async(pcie->gpu_backend, pcie->d_tensor, 
                                      pcie->h_tensor->data, 0, pcie->transfer_size);
        ggml_backend_synchronize(pcie->gpu_backend);
    }
    
    double elapsed = t.stop();
    double bytes = (double)cal_iterations * pcie->transfer_size * 2.0;  // H2D + D2H
    pcie->calibrated_bw_gb_s = bytes / elapsed / 1e9;
    
    printf("  Standalone PCIe BW: %.1f GB/s\n\n", pcie->calibrated_bw_gb_s);
}

// ============================================================================
// Benchmark Result Structure (matches roofline_profiler + concurrent data)
// ============================================================================

struct BenchResult {
    std::string op_name;
    std::string quant_type;
    int threads;
    
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
    
    // Standalone and concurrent execution metrics
    float standalone_gflops;
    float concurrent_gflops;
    float concurrent_efficiency_pct;  // % of standalone peak
    float pcie_standalone_bw_gb_s;  // Max PCIe BW for this thread count
    
    void calculate_derived() {
        arithmetic_intensity = (float)(ops / bytes);
        effective_gflops = (float)(ops / time_s / 1e9);
        effective_bw_gb_s = (float)(bytes / time_s / 1e9);
    }
    
    void print(double pcie_bw_ref = 0.0) const {
        // Print like roofline_profiler
        printf("%-20s quant=%-6s threads=%d AI=%.3f FLOP/byte BW=%.2f GB/s Perf=%.2f GFLOP/s",
               op_name.c_str(), quant_type.c_str(), threads,
               arithmetic_intensity, effective_bw_gb_s, effective_gflops);
        
        // NEW: Add concurrent performance if available
        if (concurrent_gflops > 0) {
            printf(" | Concur=%.2f (%.1f%%)", concurrent_gflops, concurrent_efficiency_pct);
            
            // Show PCIe BW reference
            if (pcie_bw_ref > 0) {
                // Estimate concurrent PCIe BW based on CPU efficiency
                double est_pcie_bw = pcie_bw_ref * (concurrent_efficiency_pct / 100.0) * 0.9;
                double pcie_eff = 100.0 * (est_pcie_bw / pcie_bw_ref);
                printf(" PCIe~%.1f GB/s (%.1f%%)", est_pcie_bw, pcie_eff);
            }
        }
        
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
// Helper: Properly quantize data for k-quant formats
// ============================================================================

std::vector<uint8_t> create_quantized_data(Quant quant, int64_t n_elements) {
    ggml_type type       = parse_quant(quant);
    size_t    quant_size = ggml_row_size(type, n_elements);
    
    std::vector<uint8_t> quant_data(quant_size, 0);  // Just zero-fill
    return quant_data;
}

// ============================================================================
// Benchmark Functions (copied from roofline_profiler.cpp)
// ============================================================================

double benchmark_mul_mat_raw(ggml_backend_t be, int N, int K, int batch_size, 
                              Quant quant, int threads, bool flush_cache, double* out_time_s, double* out_ops, double* out_bytes) {
    ggml_init_params params = {4096ull * 1024 * 1024, NULL, false};
    ggml_context * ctx = ggml_init(params);
    
    ggml_backend_cpu_set_n_threads(be, threads);
    
    ggml_tensor * A = ggml_new_tensor_2d(ctx, parse_quant(quant), K, N);
    ggml_tensor * B_tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, batch_size);
    
    ggml_backend_buffer_t bufA = ggml_backend_alloc_buffer(be, ggml_nbytes(A));
    ggml_backend_buffer_t bufB = ggml_backend_alloc_buffer(be, ggml_nbytes(B_tensor));
    ggml_backend_tensor_alloc(bufA, A, ggml_backend_buffer_get_base(bufA));
    ggml_backend_tensor_alloc(bufB, B_tensor, ggml_backend_buffer_get_base(bufB));
    
    // Only Q2_K requires proper quantization, others work fine with zero-fill
    std::vector<uint8_t> A_data;
    if (quant == Quant::Q2_K) {
        // Q2_K has strict block structure validation
        A_data = create_quantized_data(quant, K * N);
    } else {
        // All others (F32, F16, Q4_0, Q5_0, Q8_0) work fine with zero-fill
        A_data.resize(ggml_nbytes(A), 0);
    }
    std::vector<float> B_data(K * batch_size, 1.0f);
    ggml_backend_tensor_set(A, A_data.data(), 0, ggml_nbytes(A));
    ggml_backend_tensor_set(B_tensor, B_data.data(), 0, ggml_nbytes(B_tensor));
    
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_tensor * C = ggml_mul_mat(ctx, A, B_tensor);
    ggml_build_forward_expand(gf, C);
    
    // Warmup
    for (int i = 0; i < 2; ++i) {
        if (flush_cache) flush_caches();
        ggml_backend_graph_compute_async(be, gf);
    }
    
    // Timed run
    const int iterations = 5;
    double total_time = 0.0;
    Timer t;
    
    for (int i = 0; i < iterations; ++i) {
        if (flush_cache) flush_caches();  // NOT timed
        
        t.start();
        ggml_backend_graph_compute_async(be, gf);
        total_time += t.stop();
    }
    
    double time_per_iter = total_time / iterations;
    double ops_total = 2.0 * N * K * batch_size;
    double bytes_total = (double)(ggml_nbytes(A) + ggml_nbytes(B_tensor) + ggml_nbytes(C));
    double gflops = ops_total / time_per_iter / 1e9;
    
    if (out_time_s) *out_time_s = time_per_iter;
    if (out_ops) *out_ops = ops_total;
    if (out_bytes) *out_bytes = bytes_total;
    
    ggml_backend_buffer_free(bufA);
    ggml_backend_buffer_free(bufB);
    ggml_free(ctx);
    
    return gflops;
}

BenchResult benchmark_mul_mat_concurrent(ggml_backend_t be, int N, int K, int batch_size, 
                                          Quant quant, int threads, bool flush_cache, PCIeStress* pcie) {
    BenchResult result;
    result.op_name = "MUL_MAT";
    result.quant_type = quant_name(quant);
    result.threads = threads;
    result.N = N;
    result.K = K;
    result.B = batch_size;
    result.pcie_standalone_bw_gb_s = pcie ? pcie->calibrated_bw_gb_s : 0.0;  // Track max PCIe BW
    
    // 1. Standalone
    double standalone_gflops = benchmark_mul_mat_raw(be, N, K, batch_size, quant, threads, flush_cache, 
                                                      &result.time_s, &result.ops, &result.bytes);
    result.calculate_derived();  // Calculate AI, BW, GFLOP/s from time_s
    result.standalone_gflops = result.effective_gflops;
    
    // 2. With PCIe stress
    if (pcie && pcie->gpu_backend) {
        // Start PCIe stress thread
        pcie->stop.store(false, std::memory_order_release);
        std::thread pcie_thread(pcie_stress_loop, pcie);
        
        // Wait for it to start
        while (!pcie->active.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        
        // Run benchmark
        result.concurrent_gflops = benchmark_mul_mat_raw(be, N, K, batch_size, quant, threads, flush_cache, nullptr, nullptr, nullptr);
        
        // Stop PCIe
        pcie->stop.store(true, std::memory_order_release);
        pcie_thread.join();
    } else {
        result.concurrent_gflops = result.standalone_gflops;
    }
    
    // Cap efficiency at 100% (measurement noise can cause > 100%)
    result.concurrent_efficiency_pct = (float)(std::min)(100.0, 100.0 * (result.concurrent_gflops / result.standalone_gflops));

    return result;
}

double benchmark_mul_mat_id_raw(ggml_backend_t be, int N, int K, int n_experts,
                                 int n_experts_used, int batch_size, Quant quant,
                                 int threads, bool flush_cache, double * out_time_s, double * out_ops, double * out_bytes) {
    ggml_init_params params = {16384ull * 1024 * 1024, NULL, false};
    ggml_context * ctx = ggml_init(params);

    ggml_backend_cpu_set_n_threads(be, threads);

    // A: expert weights [K, N, n_experts]
    // B: input [K, 1, batch_size] - must be 3D so b->ne[2] == ids->ne[1]
    // ids: expert indices [n_experts_used, batch_size]
    ggml_tensor * A   = ggml_new_tensor_3d(ctx, parse_quant(quant), K, N, n_experts);
    ggml_tensor * B   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, 1, batch_size);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_experts_used, batch_size);

    ggml_backend_buffer_t bufA   = ggml_backend_alloc_buffer(be, ggml_nbytes(A));
    ggml_backend_buffer_t bufB   = ggml_backend_alloc_buffer(be, ggml_nbytes(B));
    ggml_backend_buffer_t bufIds = ggml_backend_alloc_buffer(be, ggml_nbytes(ids));

    ggml_backend_tensor_alloc(bufA,   A,   ggml_backend_buffer_get_base(bufA));
    ggml_backend_tensor_alloc(bufB,   B,   ggml_backend_buffer_get_base(bufB));
    ggml_backend_tensor_alloc(bufIds, ids, ggml_backend_buffer_get_base(bufIds));

    // CRITICAL: use int64_t to avoid overflow (4096*4096*128 = 2^31)
    int64_t n_elements_total = (int64_t)K * N * n_experts;

    std::vector<uint8_t> A_data;
    if (quant == Quant::Q2_K) {
        // Q2_K has strict block structure validation
        A_data = create_quantized_data(quant, n_elements_total);
    } else {
        // all others (F32, F16, Q4_0, Q5_0, Q8_0) work fine with zero-fill
        A_data.resize(ggml_nbytes(A), 0);
    }
    ggml_backend_tensor_set(A, A_data.data(), 0, ggml_nbytes(A));

    std::vector<float> B_data(K * batch_size, 1.0f);
    ggml_backend_tensor_set(B, B_data.data(), 0, ggml_nbytes(B));
    
    std::vector<int32_t> ids_data(n_experts_used * batch_size);
    for (int i = 0; i < n_experts_used * batch_size; i++) {
        ids_data[i] = i % n_experts;
    }
    ggml_backend_tensor_set(ids, ids_data.data(), 0, ggml_nbytes(ids));
    
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_tensor * C = ggml_mul_mat_id(ctx, A, B, ids);
    ggml_build_forward_expand(gf, C);
    
    // Warmup
    for (int i = 0; i < 2; ++i) {
        if (flush_cache) flush_caches();
        ggml_backend_graph_compute_async(be, gf);
    }
    
    // Timed run
    const int iterations = 5;
    double total_time = 0.0;
    Timer t;
    
    for (int i = 0; i < iterations; ++i) {
        if (flush_cache) flush_caches();  // NOT timed
        
        t.start();
        ggml_backend_graph_compute_async(be, gf);
        total_time += t.stop();
    }
    
    double time_per_iter = total_time / iterations;
    double ops_total = 2.0 * N * K * batch_size * n_experts_used;
    double bytes_total = (ggml_nbytes(A) * n_experts_used / n_experts) + ggml_nbytes(B) + ggml_nbytes(C);
    double gflops = ops_total / time_per_iter / 1e9;
    
    if (out_time_s) *out_time_s = time_per_iter;
    if (out_ops) *out_ops = ops_total;
    if (out_bytes) *out_bytes = bytes_total;
    
    ggml_backend_buffer_free(bufA);
    ggml_backend_buffer_free(bufB);
    ggml_backend_buffer_free(bufIds);
    ggml_free(ctx);
    
    return gflops;
}

BenchResult benchmark_mul_mat_id_concurrent(ggml_backend_t be, int N, int K, int n_experts, 
                                              int n_experts_used, int batch_size, Quant quant, 
                                              int threads, bool flush_cache, PCIeStress* pcie) {
    BenchResult result;
    result.op_name = "MUL_MAT_ID";
    result.quant_type = quant_name(quant);
    result.threads = threads;
    result.N = N;
    result.K = K;
    result.B = batch_size;
    result.n_tokens = n_experts_used;
    result.ctx_len = n_experts;
    result.pcie_standalone_bw_gb_s = pcie ? pcie->calibrated_bw_gb_s : 0.0;
    
    // 1. Standalone
    double standalone_gflops = benchmark_mul_mat_id_raw(be, N, K, n_experts, n_experts_used, batch_size, 
                                                         quant, threads, flush_cache, 
                                                         &result.time_s, &result.ops, &result.bytes);
    result.calculate_derived();
    result.standalone_gflops = result.effective_gflops;
    
    // 2. With PCIe stress
    if (pcie && pcie->gpu_backend) {
        pcie->stop.store(false, std::memory_order_release);
        std::thread pcie_thread(pcie_stress_loop, pcie);
        
        while (!pcie->active.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        
        result.concurrent_gflops = benchmark_mul_mat_id_raw(be, N, K, n_experts, n_experts_used, batch_size, 
                                                             quant, threads, flush_cache, nullptr, nullptr, nullptr);
        
        pcie->stop.store(true, std::memory_order_release);
        pcie_thread.join();
    } else {
        result.concurrent_gflops = result.standalone_gflops;
    }
    
    // Cap efficiency at 100%
    result.concurrent_efficiency_pct = (float)(std::min)(100.0, 100.0 * (result.concurrent_gflops / result.standalone_gflops));

    return result;
}

double benchmark_flash_attn_raw(ggml_backend_t be, int n_tokens, int ctx_len, 
                                int n_q_heads, int n_kv_heads, int head_dim, 
                                Quant kv_quant, int threads, bool flush_cache, double* out_time_s, double* out_ops, double* out_bytes) {
    ggml_init_params params = {8192ull * 1024 * 1024, NULL, false};
    ggml_context * ctx = ggml_init(params);
    
    ggml_backend_cpu_set_n_threads(be, threads);
    
    const int batch = 1;
    ggml_tensor * Q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_dim, n_tokens, n_q_heads, batch);
    ggml_tensor * K = ggml_new_tensor_4d(ctx, parse_quant(kv_quant), head_dim, ctx_len, n_kv_heads, batch);
    ggml_tensor * V = ggml_new_tensor_4d(ctx, parse_quant(kv_quant), head_dim, ctx_len, n_kv_heads, batch);
    ggml_tensor * mask = nullptr;
    
    ggml_backend_buffer_t bufQ = ggml_backend_alloc_buffer(be, ggml_nbytes(Q));
    ggml_backend_buffer_t bufK = ggml_backend_alloc_buffer(be, ggml_nbytes(K));
    ggml_backend_buffer_t bufV = ggml_backend_alloc_buffer(be, ggml_nbytes(V));
    
    ggml_backend_tensor_alloc(bufQ, Q, ggml_backend_buffer_get_base(bufQ));
    ggml_backend_tensor_alloc(bufK, K, ggml_backend_buffer_get_base(bufK));
    ggml_backend_tensor_alloc(bufV, V, ggml_backend_buffer_get_base(bufV));
    
    std::vector<float> Q_data(ggml_nelements(Q), 1.0f);
    // Only Q2_K requires proper quantization for K/V
    std::vector<uint8_t> K_data, V_data;
    if (kv_quant == Quant::Q2_K) {
        K_data = create_quantized_data(kv_quant, ggml_nelements(K));
        V_data = create_quantized_data(kv_quant, ggml_nelements(V));
    } else {
        // FP16 and others work fine with zero-fill
        K_data.resize(ggml_nbytes(K), 0);
        V_data.resize(ggml_nbytes(V), 0);
    }
    ggml_backend_tensor_set(Q, Q_data.data(), 0, ggml_nbytes(Q));
    ggml_backend_tensor_set(K, K_data.data(), 0, ggml_nbytes(K));
    ggml_backend_tensor_set(V, V_data.data(), 0, ggml_nbytes(V));
    
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_tensor * out = ggml_flash_attn_ext(ctx, Q, K, V, mask, 1.0f / sqrtf((float)head_dim), 0.0f, 0.0f);
    ggml_build_forward_expand(gf, out);
    
    // Warmup
    for (int i = 0; i < 2; ++i) {
        if (flush_cache) flush_caches();
        ggml_backend_graph_compute_async(be, gf);
    }
    
    // Timed
    const int iterations = 5;
    double total_time = 0.0;
    Timer t;
    
    for (int i = 0; i < iterations; ++i) {
        if (flush_cache) flush_caches();  // NOT timed
        
        t.start();
        ggml_backend_graph_compute_async(be, gf);
        total_time += t.stop();
    }
    
    double time_per_iter = total_time / iterations;
    double ops_total = 2.0 * n_tokens * head_dim * ctx_len * n_q_heads * 2;
    double bytes_total = (double)(ggml_nbytes(Q) + ggml_nbytes(K) + ggml_nbytes(V) + ggml_nbytes(out));
    double gflops = ops_total / time_per_iter / 1e9;
    
    if (out_time_s) *out_time_s = time_per_iter;
    if (out_ops) *out_ops = ops_total;
    if (out_bytes) *out_bytes = bytes_total;
    
    ggml_backend_buffer_free(bufQ);
    ggml_backend_buffer_free(bufK);
    ggml_backend_buffer_free(bufV);
    ggml_free(ctx);
    
    return gflops;
}

BenchResult benchmark_flash_attn_concurrent(ggml_backend_t be, int n_tokens, int ctx_len, 
                                              int n_q_heads, int n_kv_heads, int head_dim, 
                                              Quant kv_quant, int threads, bool flush_cache, const char* arch_name, 
                                              PCIeStress* pcie) {
    BenchResult result;
    result.op_name = std::string("FLASH_ATTN_") + arch_name;
    result.quant_type = quant_name(kv_quant);
    result.threads = threads;
    result.n_tokens = n_tokens;
    result.ctx_len = ctx_len;
    result.n_heads = n_kv_heads;
    result.head_dim = head_dim;
    result.pcie_standalone_bw_gb_s = pcie ? pcie->calibrated_bw_gb_s : 0.0;  // Track max PCIe BW
    
    // 1. Standalone
    double standalone_gflops = benchmark_flash_attn_raw(be, n_tokens, ctx_len, n_q_heads, n_kv_heads, 
                                                         head_dim, kv_quant, threads, flush_cache, 
                                                         &result.time_s, &result.ops, &result.bytes);
    result.calculate_derived();
    result.standalone_gflops = result.effective_gflops;
    
    // 2. With PCIe stress
    if (pcie && pcie->gpu_backend) {
        pcie->stop.store(false, std::memory_order_release);
        std::thread pcie_thread(pcie_stress_loop, pcie);
        
        while (!pcie->active.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        
        result.concurrent_gflops = benchmark_flash_attn_raw(be, n_tokens, ctx_len, n_q_heads, n_kv_heads, 
                                                            head_dim, kv_quant, threads, flush_cache, nullptr, nullptr, nullptr);
        
        pcie->stop.store(true, std::memory_order_release);
        pcie_thread.join();
    } else {
        result.concurrent_gflops = result.standalone_gflops;
    }
    
    // Cap efficiency at 100% (measurement noise can cause > 100%)
    result.concurrent_efficiency_pct = (float)(std::min)(100.0, 100.0 * (result.concurrent_gflops / result.standalone_gflops));

    return result;
}

// Element-wise operations
double benchmark_elem_raw(ggml_backend_t be, const char* op_name, int64_t n_elements, 
                           int threads, bool flush_cache, double* out_time_s, double* out_ops, double* out_bytes) {
    ggml_init_params params = {256ull * 1024 * 1024, NULL, false};
    ggml_context * ctx = ggml_init(params);
    
    ggml_backend_cpu_set_n_threads(be, threads);
    
    ggml_tensor * X = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_elements);
    ggml_tensor * Y = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_elements);
    
    ggml_backend_buffer_t bufX = ggml_backend_alloc_buffer(be, ggml_nbytes(X));
    ggml_backend_buffer_t bufY = ggml_backend_alloc_buffer(be, ggml_nbytes(Y));
    ggml_backend_tensor_alloc(bufX, X, ggml_backend_buffer_get_base(bufX));
    ggml_backend_tensor_alloc(bufY, Y, ggml_backend_buffer_get_base(bufY));
    
    std::vector<float> X_data(n_elements, 1.5f);
    std::vector<float> Y_data(n_elements, 2.5f);
    ggml_backend_tensor_set(X, X_data.data(), 0, ggml_nbytes(X));
    ggml_backend_tensor_set(Y, Y_data.data(), 0, ggml_nbytes(Y));
    
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_tensor * out = nullptr;
    
    if (strcmp(op_name, "ADD") == 0) out = ggml_add(ctx, X, Y);
    else if (strcmp(op_name, "MUL") == 0) out = ggml_mul(ctx, X, Y);
    else if (strcmp(op_name, "SUB") == 0) out = ggml_sub(ctx, X, Y);
    else if (strcmp(op_name, "DIV") == 0) out = ggml_div(ctx, X, Y);
    
    ggml_build_forward_expand(gf, out);
    
    // Warmup
    for (int i = 0; i < 2; ++i) {
        if (flush_cache) flush_caches();
        ggml_backend_graph_compute_async(be, gf);
    }
    
    // Timed
    const int iterations = 5;
    double total_time = 0.0;
    Timer t;
    
    for (int i = 0; i < iterations; ++i) {
        if (flush_cache) flush_caches();  // NOT timed
        
        t.start();
        ggml_backend_graph_compute_async(be, gf);
        total_time += t.stop();
    }
    
    double time_per_iter = total_time / iterations;
    double ops_total = (double)n_elements;
    double bytes_total = (double)(ggml_nbytes(X) + ggml_nbytes(Y) + ggml_nbytes(out));
    double gflops = ops_total / time_per_iter / 1e9;
    
    if (out_time_s) *out_time_s = time_per_iter;
    if (out_ops) *out_ops = ops_total;
    if (out_bytes) *out_bytes = bytes_total;
    
    ggml_backend_buffer_free(bufX);
    ggml_backend_buffer_free(bufY);
    ggml_free(ctx);
    
    return gflops;
}

BenchResult benchmark_elem_concurrent(ggml_backend_t be, const char* op_name, int64_t n_elements, 
                                       int threads, bool flush_cache, PCIeStress* pcie) {
    BenchResult result;
    result.op_name = op_name;
    result.quant_type = "f32";
    result.threads = threads;
    result.n_elements = n_elements;
    result.pcie_standalone_bw_gb_s = pcie ? pcie->calibrated_bw_gb_s : 0.0;  // Track max PCIe BW
    
    // 1. Standalone
    double standalone_gflops = benchmark_elem_raw(be, op_name, n_elements, threads, flush_cache, 
                                                   &result.time_s, &result.ops, &result.bytes);
    result.calculate_derived();
    result.standalone_gflops = result.effective_gflops;
    
    // 2. With PCIe
    if (pcie && pcie->gpu_backend) {
        pcie->stop.store(false, std::memory_order_release);
        std::thread pcie_thread(pcie_stress_loop, pcie);
        
        while (!pcie->active.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        
        result.concurrent_gflops = benchmark_elem_raw(be, op_name, n_elements, threads, flush_cache, nullptr, nullptr, nullptr);
        
        pcie->stop.store(true, std::memory_order_release);
        pcie_thread.join();
    } else {
        result.concurrent_gflops = result.standalone_gflops;
    }
    
    // Cap efficiency at 100% (measurement noise can cause > 100%)
    result.concurrent_efficiency_pct = (float)(std::min)(100.0, 100.0 * (result.concurrent_gflops / result.standalone_gflops));

    return result;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char ** argv) {
    // Use cpu_get_num_math() to match llama-cli's default thread count.
    // This excludes E-cores on hybrid Intel CPUs and skips hyperthreads,
    // matching the exact thread count llama-cli uses when -t is not specified.
    int32_t default_threads = cpu_get_num_math();

    int32_t fixed_threads = -1;  // -1 means use default (cpu_get_num_math)
    bool    flush_cache   = false;
    bool    fast_mode     = false;
    int32_t n_ubatch      = 512;  // max ubatch size

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--threads") && i + 1 < argc) {
            fixed_threads = std::stoi(argv[++i]);
        } else if (!strcmp(argv[i], "--cold")) {
            flush_cache = true;
        } else if (!strcmp(argv[i], "--fast")) {
            fast_mode = true;
        } else if ((!strcmp(argv[i], "-ub") || !strcmp(argv[i], "--ubatch")) && i + 1 < argc) {
            n_ubatch = std::stoi(argv[++i]);
        }
    }

    // Profile key batch sizes: decode (1), continuous batching (64), prompt (512)
    std::vector<int32_t> batch_sizes = {1, 64, 512};

    // Use a single thread count — either user-specified or the llama-cli default
    std::vector<int> thread_counts;
    thread_counts.push_back(fixed_threads > 0 ? fixed_threads : default_threads);

    printf("=== System Configuration ===\n");
    printf("Math cores detected:      %d\n", default_threads);
    printf("Testing thread count:     %d%s\n", thread_counts[0],
           fixed_threads > 0 ? " (user-specified)" : " (auto, matches llama-cli default)");
    printf("n_ubatch:                 %d\n", n_ubatch);
    printf("Tiers (%zu):               ", batch_sizes.size());
    for (size_t i = 0; i < batch_sizes.size(); ++i) {
        printf("%d%s", batch_sizes[i], (i < batch_sizes.size() - 1) ? ", " : "\n");
    }
    printf("Cache mode:               %s\n", flush_cache ? "COLD (flushing)" : "WARM (cached)");
    printf("Profile mode:             %s\n", fast_mode ? "FAST (reduced configs)" : "FULL (all configs)");
    printf("\n");
    
    #if defined(_WIN32)
        _putenv("GGML_CUDA_PIPELINE_SHARDING=0");
    #else
        setenv("GGML_CUDA_PIPELINE_SHARDING", "0", 1);
    #endif
    
    // Pre-allocate flush buffer if needed
    if (flush_cache) {
        printf("Pre-allocating cache flush buffer (256 MB)...\n");
        const size_t flush_size = 256 * 1024 * 1024;
        g_flush_buffer.resize(flush_size);
        for (size_t i = 0; i < flush_size; i += 4096) {
            g_flush_buffer[i] = (char)(i & 0xFF);
        }
        printf("Flush buffer ready.\n\n");
    }
    
    // Initialize CPU backend
    ggml_backend_t cpu_be = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!cpu_be) {
        fprintf(stderr, "Failed to initialize CPU backend\n");
        return 1;
    }
    
    // Initialize quantization tables for k-quants
    ggml_quantize_init(GGML_TYPE_Q2_K);
    ggml_quantize_init(GGML_TYPE_Q4_0);
    ggml_quantize_init(GGML_TYPE_Q4_1);
    ggml_quantize_init(GGML_TYPE_Q5_0);
    ggml_quantize_init(GGML_TYPE_Q8_0);
    
    // Setup PCIe stress context
    PCIeStress pcie;
    pcie.gpu_backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    bool has_gpu = (pcie.gpu_backend != nullptr);
    
    if (has_gpu) {
        ggml_backend_buffer_type_t host_buft = ggml_backend_dev_host_buffer_type(
            ggml_backend_get_device(pcie.gpu_backend));
        
        if (host_buft) {
            pcie.host_buf = ggml_backend_buft_alloc_buffer(host_buft, pcie.transfer_size);
            pcie.dev_buf = ggml_backend_alloc_buffer(pcie.gpu_backend, pcie.transfer_size);
            
            ggml_init_params params = {pcie.transfer_size + 8 * 1024 * 1024, NULL, true};
            pcie.ctx = ggml_init(params);
            
            pcie.h_tensor = ggml_new_tensor_1d(pcie.ctx, GGML_TYPE_F32, pcie.transfer_size / 4);
            pcie.d_tensor = ggml_new_tensor_1d(pcie.ctx, GGML_TYPE_F32, pcie.transfer_size / 4);
            
            ggml_backend_tensor_alloc(pcie.host_buf, pcie.h_tensor, 
                                     ggml_backend_buffer_get_base(pcie.host_buf));
            ggml_backend_tensor_alloc(pcie.dev_buf, pcie.d_tensor, 
                                     ggml_backend_buffer_get_base(pcie.dev_buf));
            
            std::vector<float> init(pcie.transfer_size / 4, 1.0f);
            ggml_backend_tensor_set(pcie.h_tensor, init.data(), 0, pcie.transfer_size);
            
            printf("=== Concurrent Profiler: CPU Ops + PCIe Stress ===\n");
            printf("GPU: %s\n", ggml_backend_name(pcie.gpu_backend));
            printf("Will calibrate PCIe BW for each thread count\n\n");
        } else {
            has_gpu = false;
        }
    }
    
    if (!has_gpu) {
        printf("=== Standalone Mode (No GPU Available) ===\n\n");
    }
    
    std::vector<BenchResult> all_results;
    
    // Track PCIe BW and DRAM BW per thread count
    struct PerThreadBandwidths {
        int threads;
        double dram_bw;           // Measured DRAM BW for this thread count
        double pcie_standalone_bw;
        double pcie_concurrent_bw;
        double cpu_efficiency_pct;
    };
    std::vector<PerThreadBandwidths> per_thread_bw;
    
    // === Main Thread Loop ===
    for (int threads : thread_counts) {
        printf("\n");
        printf("========================================\n");
        printf("=== Testing with %d thread(s) ===\n", threads);
        printf("========================================\n\n");
        
        // Measure DRAM bandwidth for this specific thread count
        printf("Measuring DRAM bandwidth with %d thread(s)...\n", threads);
        double thread_dram_bw = benchmark_cpu_dram_bandwidth(threads);
        printf("  DRAM BW (%d threads): %.1f GB/s\n", threads, thread_dram_bw);
        
        // Recalibrate PCIe for this thread count (if GPU available)
        if (has_gpu) {
            calibrate_pcie(&pcie);
        }
        
        PerThreadBandwidths bw_stat;
        bw_stat.threads = threads;
        bw_stat.dram_bw = thread_dram_bw;
        bw_stat.pcie_standalone_bw = pcie.calibrated_bw_gb_s;
        
        std::vector<BenchResult> thread_results;

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

                    BenchResult res = benchmark_mul_mat_concurrent(
                        cpu_be, N, K, batch_size, quant, threads, flush_cache,
                        has_gpu ? &pcie : nullptr);
                    res.print(pcie.calibrated_bw_gb_s);
                    thread_results.push_back(res);
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
                    int32_t N             = moe_configs[ci].N;
                    int32_t K             = moe_configs[ci].K;
                    int32_t n_experts      = moe_configs[ci].n_experts;
                    int32_t n_experts_used = moe_configs[ci].n_experts_used;

                    // Q2_K requires K dimension divisible by 256 (QK_K block size)
                    if (quant == Quant::Q2_K && (K % 256 != 0)) {
                        continue;
                    }

                    BenchResult res = benchmark_mul_mat_id_concurrent(
                        cpu_be, N, K, n_experts, n_experts_used, batch_size,
                        quant, threads, flush_cache, has_gpu ? &pcie : nullptr);
                    res.print(pcie.calibrated_bw_gb_s);
                    thread_results.push_back(res);
                    all_results.push_back(res);
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

                    BenchResult res = benchmark_flash_attn_concurrent(
                        cpu_be, n_tokens, ctx_len, n_q_heads, config.n_kv_heads,
                        head_dim, Quant::F16, threads, flush_cache, config.name,
                        has_gpu ? &pcie : nullptr);
                    res.print(pcie.calibrated_bw_gb_s);
                    thread_results.push_back(res);
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

                BenchResult res = benchmark_elem_concurrent(
                    cpu_be, op, n, threads, flush_cache, has_gpu ? &pcie : nullptr);
                res.print(pcie.calibrated_bw_gb_s);
                thread_results.push_back(res);
                all_results.push_back(res);
            }
            printf("\n");
        }
        }  // end if (!fast_mode) for element-wise ops
        
        // calculate average concurrent efficiency for this thread count
        if (has_gpu) {
            double sum_standalone = 0.0;
            double sum_concurrent = 0.0;
            for (size_t ri = 0; ri < thread_results.size(); ++ri) {
                sum_standalone += thread_results[ri].standalone_gflops;
                sum_concurrent += thread_results[ri].concurrent_gflops;
            }
            double avg_efficiency = 100.0 * (sum_concurrent / sum_standalone);
            bw_stat.pcie_concurrent_bw = bw_stat.pcie_standalone_bw * (avg_efficiency / 100.0) * 0.9;
            bw_stat.cpu_efficiency_pct = avg_efficiency;

            printf("Thread %d Summary: CPU efficiency = %.1f%%, DRAM BW = %.1f GB/s, PCIe BW = %.1f GB/s\n",
                   threads, avg_efficiency, thread_dram_bw, bw_stat.pcie_concurrent_bw);
        } else {
            bw_stat.pcie_standalone_bw = 0.0;
            bw_stat.pcie_concurrent_bw = 0.0;
            bw_stat.cpu_efficiency_pct = 100.0;

            printf("Thread %d Summary: DRAM BW = %.1f GB/s\n", threads, thread_dram_bw);
        }
        per_thread_bw.push_back(bw_stat);
    }  // end thread loop

    // === Bandwidth Scaling Summary ===
    if (!per_thread_bw.empty()) {
        printf("\n=== Bandwidth Scaling vs Thread Count ===\n");
        printf("%-10s | %-15s | %-15s | %-15s\n", "Threads", "DRAM BW", "PCIe Standalone", "PCIe Concurrent");
        printf("-----------|-----------------|-----------------|----------------\n");
        for (size_t pi = 0; pi < per_thread_bw.size(); ++pi) {
            const PerThreadBandwidths & p = per_thread_bw[pi];
            if (has_gpu) {
                printf("%-10d | %13.1f GB/s | %13.1f GB/s | %13.1f GB/s\n",
                       p.threads, p.dram_bw, p.pcie_standalone_bw, p.pcie_concurrent_bw);
            } else {
                printf("%-10d | %13.1f GB/s | %15s | %15s\n",
                       p.threads, p.dram_bw, "N/A", "N/A");
            }
        }
    }

    // === Calculate Ridge Points (PER THREAD COUNT) ===
    printf("\n=== Calculating Ridge Points ===\n");

    // create lookup map: thread_count -> measured DRAM BW
    std::map<int32_t, double> thread_to_dram_bw;
    for (size_t pi = 0; pi < per_thread_bw.size(); ++pi) {
        thread_to_dram_bw[per_thread_bw[pi].threads] = per_thread_bw[pi].dram_bw;
    }

    // group by (op, quant, threads) - thread-specific ridge points
    std::map<std::string, double> ridge_points;  // key = "op_quant_threads"
    std::map<std::string, std::vector<BenchResult *>> groups;

    for (size_t ri = 0; ri < all_results.size(); ++ri) {
        std::string key = all_results[ri].op_name + "_" + all_results[ri].quant_type + "_" + std::to_string(all_results[ri].threads);
        groups[key].push_back(&all_results[ri]);
    }

    for (std::map<std::string, std::vector<BenchResult *>>::iterator it = groups.begin(); it != groups.end(); ++it) {
        const std::string & key = it->first;
        const std::vector<BenchResult *> & group = it->second;

        double peak_gflops = 0.0;
        for (size_t gi = 0; gi < group.size(); ++gi) {
            peak_gflops = (std::max)(peak_gflops, (double)group[gi]->standalone_gflops);
        }

        // use MEASURED DRAM BW for this thread count
        int32_t thread_count = group[0]->threads;
        double measured_dram_bw = thread_to_dram_bw[thread_count];

        double ridge = peak_gflops / measured_dram_bw;
        ridge_points[key] = ridge;

        std::string base_key = group[0]->op_name + "_" + group[0]->quant_type;
        printf("%s (threads=%d): Peak=%.1f GFLOP/s, DRAM=%.1f GB/s, Ridge=%.3f FLOP/byte\n",
               base_key.c_str(), thread_count, peak_gflops, measured_dram_bw, ridge);
    }
    
    // === Overall Summary ===
    printf("\n=== Overall Summary ===\n");
    printf("Thread counts tested: %zu\n", thread_counts.size());
    printf("Batch sizes tested:   %zu\n", batch_sizes.size());
    printf("Total benchmarks:     %zu\n", all_results.size());

    if (has_gpu) {
        double sum_standalone = 0.0;
        double sum_concurrent = 0.0;

        for (size_t ri = 0; ri < all_results.size(); ++ri) {
            sum_standalone += all_results[ri].standalone_gflops;
            sum_concurrent += all_results[ri].concurrent_gflops;
        }

        double avg_standalone_gflops = sum_standalone / all_results.size();
        double avg_concurrent_gflops = sum_concurrent / all_results.size();
        double avg_efficiency_pct = 100.0 * (avg_concurrent_gflops / avg_standalone_gflops);

        printf("\n--- CPU Performance (averaged across all tests) ---\n");
        printf("Average standalone:   %.1f GFLOP/s\n", avg_standalone_gflops);
        printf("Average concurrent:   %.1f GFLOP/s (%.1f%% efficiency)\n", avg_concurrent_gflops, avg_efficiency_pct);

        printf("\n--- Conclusion ---\n");
        if (avg_efficiency_pct >= 95.0) {
            printf("Excellent overlap (%.1f%% efficiency) - PCIe and CPU can pipeline efficiently!\n", avg_efficiency_pct);
        } else if (avg_efficiency_pct >= 85.0) {
            printf("Good overlap (%.1f%% efficiency) - acceptable for pipelining\n", avg_efficiency_pct);
        } else if (avg_efficiency_pct >= 70.0) {
            printf("Moderate overlap (%.1f%% efficiency) - some contention\n", avg_efficiency_pct);
        } else {
            printf("Poor overlap (%.1f%% efficiency) - significant memory controller contention\n", avg_efficiency_pct);
        }
    }

    // === Save Results ===
    printf("\n=== Saving Results ===\n");
    FILE * f = fopen("concurrent_results.txt", "w");
    if (f) {
        fprintf(f, "# Concurrent Profiling (thread_counts=[");
        for (size_t i = 0; i < thread_counts.size(); ++i) {
            fprintf(f, "%d%s", thread_counts[i], (i < thread_counts.size() - 1) ? "," : "");
        }
        fprintf(f, "], batch_sizes=[");
        for (size_t i = 0; i < batch_sizes.size(); ++i) {
            fprintf(f, "%d%s", batch_sizes[i], (i < batch_sizes.size() - 1) ? "," : "");
        }
        fprintf(f, "])\n");

        // measured bandwidths per thread count
        fprintf(f, "# Measured Bandwidths Per Thread Count:\n");
        for (size_t pi = 0; pi < per_thread_bw.size(); ++pi) {
            const PerThreadBandwidths & p = per_thread_bw[pi];
            if (has_gpu) {
                fprintf(f, "#   Threads=%d: DRAM_BW=%.1f GB/s, PCIe_Standalone=%.1f GB/s, PCIe_Concurrent=%.1f GB/s (CPU_Eff=%.1f%%)\n",
                       p.threads, p.dram_bw, p.pcie_standalone_bw, p.pcie_concurrent_bw, p.cpu_efficiency_pct);
            } else {
                fprintf(f, "#   Threads=%d: DRAM_BW=%.1f GB/s\n", p.threads, p.dram_bw);
            }
        }
        fprintf(f, "# op_name quant threads AI(FLOP/byte) BW(GB/s) GFLOP/s Ridge(FLOP/byte) Concurrent_GFLOP/s PCIe_Concurrent_BW N K B n_tokens ctx_len n_heads head_dim n_elements\n");

        for (size_t ri = 0; ri < all_results.size(); ++ri) {
            const BenchResult & r = all_results[ri];
            std::string key = r.op_name + "_" + r.quant_type + "_" + std::to_string(r.threads);
            double ridge = ridge_points.count(key) ? ridge_points[key] : 0.0;

            // estimate concurrent PCIe BW based on CPU performance ratio
            double est_pcie_bw = pcie.calibrated_bw_gb_s * (r.concurrent_gflops / r.standalone_gflops) * 0.9;

            // output: op_name quant threads AI BW GFLOP/s Ridge Concurrent_GFLOP/s PCIe_BW N K B n_tokens ctx_len n_heads head_dim n_elements
            fprintf(f, "%s %s %d %.4f %.2f %.2f %.4f %.2f %.2f %d %d %d %d %d %d %d %lld\n",
                    r.op_name.c_str(), r.quant_type.c_str(), r.threads,
                    r.arithmetic_intensity, r.effective_bw_gb_s, r.effective_gflops, ridge,
                    r.concurrent_gflops,
                    est_pcie_bw,
                    r.N, r.K, r.B, r.n_tokens, r.ctx_len, r.n_heads, r.head_dim,
                    (long long)r.n_elements);
        }
        fclose(f);
        printf("Results saved to concurrent_results.txt\n");
        printf("Total benchmarks completed: %zu\n", all_results.size());
    } else {
        fprintf(stderr, "Failed to open concurrent_results.txt for writing\n");
    }

    // cleanup
    if (has_gpu) {
        if (pcie.ctx) {
            ggml_free(pcie.ctx);
        }
        if (pcie.host_buf) {
            ggml_backend_buffer_free(pcie.host_buf);
        }
        if (pcie.dev_buf) {
            ggml_backend_buffer_free(pcie.dev_buf);
        }
        ggml_backend_free(pcie.gpu_backend);
    }
    ggml_backend_free(cpu_be);

    // cleanup quantization tables
    ggml_quantize_free();

    return 0;
}
