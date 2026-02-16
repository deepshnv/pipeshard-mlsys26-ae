#pragma once

#include "ggml-backend.h"

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

struct GlobalBenchmarkStats {
    double peak_system_bw;
    double peak_pcie_bw;
    double eff_system_bw;
    double eff_pcie_bw;
    double peak_gpu_memory_bw;
    double peak_gpu_compute;  // Peak GPU compute in GFLOP/s
};

struct OpPrediction {
    std::string  op_name;
    std::string  node_name;
    std::string  quant_type;
    double       predicted_time_ms;
    double       flops;
    double       bytes;
    double       ai;
    double       ridge;
    bool         exact_match;
    const char * bottleneck;
};

// Add after OpPrediction struct:
struct BenchmarkEntry {
    std::string op_name;
    std::string quant;
    int threads;
    double ai;
    double bw_gb_s;
    double peak_gflop_s;
    double ridge;
    double eff_gflop_s;
    double eff_pcie_bw;
    int64_t N, K, B;
    int64_t n_tokens, ctx_len, n_heads, head_dim, n_elements;
};


// Interpolated benchmark metrics (used when no exact match found)
struct InterpolatedMetrics {
    double eff_gflop_s;
    double peak_gflop_s;
    double bw_gb_s;
    double ridge;
    double eff_pcie_bw;
    bool   valid;        // false if no benchmarks found
    bool   exact_match;  // true if exact hit, false if interpolated/nearest
};

struct OpTimingDetail {
    std::string op_name;
    std::string node_name;
    double time_ms;
    double flops;
    double bytes;
    bool exact_match;
};

struct SplitTimingResult {
    double time_ms;
    double eff_pcie_bw;
    int op_count;
    int n_cache_hit;   // ops matched from timing cache
    int n_exact;       // ops matched exact benchmark
    int n_interp;      // ops used interpolated benchmark
    int n_fallback;    // ops fell back to memory bandwidth
    std::vector<OpTimingDetail> op_details;
};

// Extracted metrics from a single ggml operation
struct OpMetrics {
    double      ops;
    double      bytes;
    const char* quant_type;
    int64_t     N, K, M, ctx_len, n_elements, n_kv_heads, n_experts_used;  // M = batch, n_experts_used for MoE
    int64_t     head_dim, n_q_heads, n_tokens;  // For FLASH_ATTN
};

// Extract ops/bytes metrics from a tensor (shared by predict and debug functions)
OpMetrics calculate_op_metrics(const ggml_tensor * node);

// Encapsulates benchmark data and provides timing predictions
class BenchmarkPredictor {
    const std::vector<BenchmarkEntry> & m_cpu_benchmarks;
    const std::vector<BenchmarkEntry> & m_gpu_benchmarks;
    const GlobalBenchmarkStats       & m_stats;

    // Hash maps for O(1) lookup (key = "OP|quant|N|K" or similar)
    std::unordered_map<std::string, const BenchmarkEntry *> m_cpu_map;
    std::unordered_map<std::string, const BenchmarkEntry *> m_gpu_map;

    void build_lookup_maps();

    // Key generation for hash maps (includes batch for MUL_MAT)
    static std::string make_key(const std::string & op, const std::string & quant, int64_t N, int64_t K, int64_t batch);
    static std::string make_attn_key(int64_t ctx_len, int64_t n_heads, int64_t n_tokens);
    static std::string make_elem_key(const std::string & op, int64_t n_elements);

    // Debug print implementation
    void debug_print_split_impl(ggml_backend_sched_t sched, int split_id, bool is_gpu, int32_t batch_size) const;

    // Interpolate between nearest benchmarks (by dimension)
    InterpolatedMetrics interpolate_benchmark(
        const std::vector<BenchmarkEntry> & benchmarks,
        const char * op_name, const char * quant,
        int64_t target_size) const;

    // Interpolate between batch sizes (1, 64, 512)
    InterpolatedMetrics interpolate_by_batch(
        const std::vector<BenchmarkEntry> & benchmarks,
        const char * op_name, const char * quant,
        int64_t N, int64_t K, int64_t target_batch) const;

    // Interpolate by quant type using bits-per-weight similarity
    // When exact quant not found, interpolate between nearest quant types (e.g., q5_0 <-> q8_0 for q6_K)
    InterpolatedMetrics interpolate_by_quant(
        const std::vector<BenchmarkEntry> & benchmarks,
        const char * op_name, const char * target_quant,
        int64_t N, int64_t K, int64_t target_batch) const;

    // Timing cache: key = "GPU|MUL_MAT|Q4_0|4096|1" -> (time_ms, pcie_bw)
    mutable std::unordered_map<std::string, std::pair<double, double>> m_timing_cache;

    static std::string make_timing_key(bool is_gpu, bool async_copy, const char * op_name, const OpMetrics & m, int32_t batch_size);

    // Nearest benchmark search with scoring for dimensions, batch, and quant
    static const BenchmarkEntry * find_nearest_benchmark(
        const std::vector<BenchmarkEntry> & benchmarks,
        const char * op_name, const char * quant,
        int64_t N, int64_t K, int64_t ctx_len, int64_t n_elements,
        int64_t n_kv_heads = -1, int64_t target_batch = 1);

    // Flag to switch between old and new prediction methods
    mutable bool m_use_old_prediction = false;

public:
    // Static loaders
    static std::vector<BenchmarkEntry> load_cpu_benchmarks(const char * filepath, int n_threads, GlobalBenchmarkStats * out_stats);
    static std::vector<BenchmarkEntry> load_gpu_benchmarks(const char * filepath, GlobalBenchmarkStats * out_stats);

    BenchmarkPredictor(const std::vector<BenchmarkEntry> & cpu_benchmarks,
                       const std::vector<BenchmarkEntry> & gpu_benchmarks,
                       const GlobalBenchmarkStats       & stats)
        : m_cpu_benchmarks(cpu_benchmarks)
        , m_gpu_benchmarks(gpu_benchmarks)
        , m_stats(stats) {
        build_lookup_maps();
    }

    SplitTimingResult predict(ggml_backend_sched_t sched, size_t split_id, bool is_gpu, int32_t batch_size, bool async_copy = true) const;

    // Debug: print split operations with interpolation details
    void debug_print_split(ggml_backend_sched_t sched, size_t split_id, bool is_gpu, int32_t batch_size) const;

    // Clear timing cache (call when graph structure changes)
    void clear_cache() const { m_timing_cache.clear(); }

    const GlobalBenchmarkStats & stats() const { return m_stats; }

    // Switch between old (find_nearest_benchmark) and new (batch interpolation) prediction
    void set_use_old_prediction(bool use_old) const { m_use_old_prediction = use_old; }
    bool use_old_prediction() const { return m_use_old_prediction; }
};

