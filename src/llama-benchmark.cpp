#include "llama-benchmark.h"
#include "llama.h"
#include "llama-impl.h"
#include <cstring>
#include <cstdio>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>

// Load CPU benchmark database with concurrent metrics
std::vector<BenchmarkEntry> BenchmarkPredictor::load_cpu_benchmarks(const char* filepath, int n_threads, GlobalBenchmarkStats* out_stats) {
    std::vector<BenchmarkEntry> db;
    FILE* f = fopen(filepath, "r");
    if (!f) {
        LLAMA_LOG_INFO("Warning: Could not open benchmark file: %s\n", filepath);
        return db;
    }
    
    // Initialize stats
    if (out_stats) {
        out_stats->peak_system_bw = 0.0;
        out_stats->peak_pcie_bw = 0.0;
        out_stats->eff_system_bw = 0.0;
        out_stats->eff_pcie_bw = 0.0;
    }
    
    double max_system_dram_bw = 0.0;  // Track absolute max DRAM BW across all thread counts
    
    char line[512];
    while (fgets(line, sizeof(line), f)) {
        // Parse header for global stats
        if (line[0] == '#' && out_stats) {
            // New format: "#   Threads=N: DRAM_BW=X.X GB/s, PCIe_Standalone=Y.Y GB/s, PCIe_Concurrent=Z.Z GB/s (CPU_Eff=W.W%)"
            int thread_count = 0;
            double dram_bw = 0.0, pcie_standalone = 0.0, pcie_concurrent = 0.0, cpu_eff = 0.0;
            
            if (sscanf(line, "#   Threads=%d: DRAM_BW=%lf GB/s, PCIe_Standalone=%lf GB/s, PCIe_Concurrent=%lf GB/s (CPU_Eff=%lf%%)",
                      &thread_count, &dram_bw, &pcie_standalone, &pcie_concurrent, &cpu_eff) == 5) {
                
                // Track maximum DRAM BW across all thread counts
                if (dram_bw > max_system_dram_bw) {
                    max_system_dram_bw = dram_bw;
                }
                
                if (thread_count == n_threads) {
                    // Found the matching thread count
                    out_stats->peak_system_bw = dram_bw;
                    out_stats->peak_pcie_bw = pcie_standalone;
                    out_stats->eff_system_bw = dram_bw * (cpu_eff / 100.0);  // Effective DRAM under PCIe load
                    out_stats->eff_pcie_bw = std::min(pcie_concurrent, dram_bw);
                }
            }
            
            // Fallback: Old format compatibility
            // Look for "Sys_DRAM_BW=XX.X GB/s"
            const char* sys_bw_str = strstr(line, "Sys_DRAM_BW=");
            if (sys_bw_str) {
                double sys_bw = 0.0;
                sscanf(sys_bw_str, "Sys_DRAM_BW=%lf", &sys_bw);
                if (sys_bw > max_system_dram_bw) {
                    max_system_dram_bw = sys_bw;
                }
                if (out_stats->peak_system_bw == 0.0) {
                    out_stats->peak_system_bw = sys_bw;
                }
            }
            
            // Look for "PCIe_Max_BW=XX.X GB/s"
            const char* pcie_str = strstr(line, "PCIe_Max_BW=");
            if (pcie_str && out_stats->peak_pcie_bw == 0.0) {
                sscanf(pcie_str, "PCIe_Max_BW=%lf", &out_stats->peak_pcie_bw);
            }
            continue;
        }
        
        if (line[0] == '#' || line[0] == '\n') continue;
        
        BenchmarkEntry entry;
        char op_name[64], quant[16];
        
        // Updated parse for new format: op quant threads AI BW peak_GFLOP/s Ridge eff_GFLOP/s eff_PCIe_BW N K B ...
        int parsed = sscanf(line, "%s %s %d %lf %lf %lf %lf %lf %lf %lld %lld %lld %lld %lld %lld %lld %lld",
                           op_name, quant, &entry.threads,
                           &entry.ai, &entry.bw_gb_s, &entry.peak_gflop_s, &entry.ridge,
                           &entry.eff_gflop_s, &entry.eff_pcie_bw,
                           &entry.N, &entry.K, &entry.B, &entry.n_tokens, &entry.ctx_len,
                           &entry.n_heads, &entry.head_dim, &entry.n_elements);
        
        if (parsed >= 9 && entry.threads == n_threads) {  // Need at least 9 fields (up to eff_pcie_bw)
            entry.op_name = op_name;
            entry.quant = quant;
            db.push_back(entry);
        }
    }
    fclose(f);

    // Cap PCIe BW by the absolute maximum system DRAM BW (independent of thread count)
    if (max_system_dram_bw > 0.0) {
        out_stats->peak_pcie_bw = std::min(out_stats->peak_pcie_bw, max_system_dram_bw);
    } else {
        // Fallback to thread-specific value if we didn't find max
        out_stats->peak_pcie_bw = std::min(out_stats->peak_pcie_bw, out_stats->peak_system_bw);
    }
    
    // Fallback: if effective values not found, assume no contention
    if (out_stats->eff_system_bw == 0.0) {
        out_stats->eff_system_bw = out_stats->peak_system_bw;
    }
    if (out_stats->eff_pcie_bw == 0.0) {
        out_stats->eff_pcie_bw = out_stats->peak_pcie_bw;
    }
    
    if (out_stats) {
        LLAMA_LOG_INFO("Loaded %zu benchmark entries for %d threads (Peak: DRAM=%.1f GB/s, PCIe=%.1f GB/s | Concurrent: DRAM=%.1f GB/s, PCIe=%.1f GB/s)\n", 
               db.size(), n_threads, out_stats->peak_system_bw, out_stats->peak_pcie_bw,
               out_stats->eff_system_bw, out_stats->eff_pcie_bw);
    } else {
        LLAMA_LOG_INFO("Loaded %zu benchmark entries for %d threads\n", db.size(), n_threads);
    }
    
    return db;
}

std::vector<BenchmarkEntry> BenchmarkPredictor::load_gpu_benchmarks(const char* filepath, GlobalBenchmarkStats* out_stats) {
    std::vector<BenchmarkEntry> db;
    FILE* f = fopen(filepath, "r");
    if (!f) {
        LLAMA_LOG_INFO("Warning: Could not open GPU benchmark file: %s\n", filepath);
        return db;
    }
    
    // Initialize stats
    if (out_stats) {
        out_stats->peak_gpu_memory_bw = 0.0;
        out_stats->peak_gpu_compute = 0.0;
    }
    
    char line[512];
    while (fgets(line, sizeof(line), f)) {
        // Parse header for GPU memory BW and compute
        if (line[0] == '#') {
            double gpu_memory_bw = 0.0;
            double gpu_compute = 0.0;
            
            // Try new format first (with both memory BW and compute)
            if (sscanf(line, "# GPU Profiling (backend=%*[^,], GPU_Memory_BW=%lf GB/s, GPU_Peak_Compute=%lf GFLOP/s)", 
                       &gpu_memory_bw, &gpu_compute) == 2) {
                if (out_stats) {
                    out_stats->peak_gpu_memory_bw = gpu_memory_bw;
                    out_stats->peak_gpu_compute = gpu_compute;
                }
                LLAMA_LOG_INFO("Parsed GPU Stats: GPU_Memory_BW=%.1f GB/s, GPU_Peak_Compute=%.1f GFLOP/s\n", 
                               gpu_memory_bw, gpu_compute);
            }
            // Fallback to old format (only memory BW)
            else if (sscanf(line, "# GPU Profiling (backend=%*[^,], GPU_Memory_BW=%lf GB/s)", &gpu_memory_bw) == 1) {
                if (out_stats) {
                    out_stats->peak_gpu_memory_bw = gpu_memory_bw;
                    // Estimate compute from memory BW if not available
                    out_stats->peak_gpu_compute = gpu_memory_bw * 150.0;  // Rough heuristic
                }
                LLAMA_LOG_INFO("Parsed GPU Stats: GPU_Memory_BW=%.1f GB/s (estimating compute=%.1f GFLOP/s)\n", 
                               gpu_memory_bw, out_stats->peak_gpu_compute);
            }
            continue;
        }
        if (line[0] == '\n') continue; // Skip empty lines
        
        BenchmarkEntry entry;
        char op_name[64], quant[16];
        
        // GPU results don't have threads or concurrent metrics
        // Format: op_name quant AI BW GFLOP/s Ridge N K B n_tokens ctx_len n_heads head_dim n_elements
        int parsed = sscanf(line, "%s %s %lf %lf %lf %lf %lld %lld %lld %lld %lld %lld %lld %lld",
                           op_name, quant, &entry.ai, &entry.bw_gb_s, &entry.peak_gflop_s, &entry.ridge,
                           &entry.N, &entry.K, &entry.B, &entry.n_tokens, &entry.ctx_len,
                           &entry.n_heads, &entry.head_dim, &entry.n_elements);
        
        if (parsed == 14) {  // All fields parsed
            entry.op_name = op_name;
            entry.quant = quant;
            entry.threads = -1;  // GPU doesn't use CPU threads
            entry.eff_gflop_s = entry.peak_gflop_s;  // GPU has no concurrent mode
            entry.eff_pcie_bw = 0.0;  // Not applicable for GPU-only benchmarks
            db.push_back(entry);
        }
    }
    fclose(f);
    LLAMA_LOG_INFO("Loaded %zu GPU benchmark entries\n", db.size());
    return db;
}

// --- BenchmarkPredictor key helpers ---

std::string BenchmarkPredictor::make_key(const std::string & op, const std::string & quant, int64_t N, int64_t K, int64_t batch) {
    return op + "|" + quant + "|" + std::to_string(N) + "|" + std::to_string(K) + "|" + std::to_string(batch);
}

std::string BenchmarkPredictor::make_attn_key(int64_t ctx_len, int64_t n_heads, int64_t n_tokens) {
    return "FLASH_ATTN|" + std::to_string(ctx_len) + "|" + std::to_string(n_heads) + "|" + std::to_string(n_tokens);
}

std::string BenchmarkPredictor::make_elem_key(const std::string & op, int64_t n_elements) {
    return op + "|" + std::to_string(n_elements);
}

// Linear interpolation helper
static double lerp(double a, double b, double t) {
    return a + t * (b - a);
}

// Get the primary size dimension for a benchmark entry based on op type
static int64_t get_bench_size(const BenchmarkEntry & b, const char * op_name) {
    if (strstr(op_name, "MUL_MAT")) {
        return b.K;  // K varies with batch size
    } else if (strstr(op_name, "FLASH_ATTN")) {
        return b.ctx_len;
        } else {
        return b.n_elements;
    }
}

InterpolatedMetrics BenchmarkPredictor::interpolate_benchmark(
        const std::vector<BenchmarkEntry> & benchmarks,
        const char * op_name, const char * quant,
        int64_t target_size) const {

    InterpolatedMetrics result = {0, 0, 0, 0, 0, false, false};

    const BenchmarkEntry * lower = nullptr;
    const BenchmarkEntry * upper = nullptr;
    int64_t lower_size = 0;
    int64_t upper_size = INT64_MAX;

    for (const auto & b : benchmarks) {
        // Filter by op name - exact match to avoid "MUL" matching "MUL_MAT"
        if (b.op_name != op_name) continue;
        // Filter by quant type if specified
        if (quant && !b.quant.empty() && b.quant != quant) continue;

        int64_t size = get_bench_size(b, op_name);

        // Track lower bound (largest size <= target)
        if (size <= target_size && size > lower_size) {
            lower = &b;
            lower_size = size;
        }
        // Track upper bound (smallest size >= target)
        if (size >= target_size && size < upper_size) {
            upper = &b;
            upper_size = size;
        }
    }

    // Handle cases
    if (lower && upper) {
        if (lower == upper) {
            // Exact match
            result.eff_gflop_s  = lower->eff_gflop_s;
            result.peak_gflop_s = lower->peak_gflop_s;
            result.bw_gb_s      = lower->bw_gb_s;
            result.ridge        = lower->ridge;
            result.eff_pcie_bw  = lower->eff_pcie_bw;
            result.valid        = true;
            result.exact_match  = true;
        } else {
            // Interpolate between lower and upper
            double t = (double)(target_size - lower_size) / (double)(upper_size - lower_size);
            result.eff_gflop_s  = lerp(lower->eff_gflop_s,  upper->eff_gflop_s,  t);
            result.peak_gflop_s = lerp(lower->peak_gflop_s, upper->peak_gflop_s, t);
            result.bw_gb_s      = lerp(lower->bw_gb_s,      upper->bw_gb_s,      t);
            result.ridge        = lerp(lower->ridge,        upper->ridge,        t);
            result.eff_pcie_bw  = lerp(lower->eff_pcie_bw,  upper->eff_pcie_bw,  t);
            result.valid        = true;
            result.exact_match  = false;
                    }
    } else if (lower) {
        // Extrapolate from lower (target > all benchmarks)
        result.eff_gflop_s  = lower->eff_gflop_s;
        result.peak_gflop_s = lower->peak_gflop_s;
        result.bw_gb_s      = lower->bw_gb_s;
        result.ridge        = lower->ridge;
        result.eff_pcie_bw  = lower->eff_pcie_bw;
        result.valid        = true;
        result.exact_match  = false;
    } else if (upper) {
        // Extrapolate from upper (target < all benchmarks)
        result.eff_gflop_s  = upper->eff_gflop_s;
        result.peak_gflop_s = upper->peak_gflop_s;
        result.bw_gb_s      = upper->bw_gb_s;
        result.ridge        = upper->ridge;
        result.eff_pcie_bw  = upper->eff_pcie_bw;
        result.valid        = true;
        result.exact_match  = false;
    }
    // else: no matching benchmarks found, result.valid = false

    return result;
}

// Interpolate by batch size for MUL_MAT operations
// Uses dimension scoring to find closest (N,K) with SAME quant,
// then interpolates between nearest batch benchmarks (1, 64, 512)
InterpolatedMetrics BenchmarkPredictor::interpolate_by_batch(
        const std::vector<BenchmarkEntry> & benchmarks,
        const char * op_name, const char * quant,
        int64_t N, int64_t K, int64_t target_batch) const {

    InterpolatedMetrics result = {0, 0, 0, 0, 0, false, false};

    // Step 1: Find best (N,K) match using dimension scoring (same quant only)
    double best_dim_score = 1e20;
    int64_t best_N = 0, best_K = 0;

    for (const auto & b : benchmarks) {
        if (b.op_name != op_name) continue;
        if (quant && !b.quant.empty() && b.quant != quant) continue;

        double n_diff = std::abs((double)b.N - N) / std::max(N, (int64_t)1);
        double k_diff = std::abs((double)b.K - K) / std::max(K, (int64_t)1);
        double score = n_diff + k_diff;

        if (score < best_dim_score) {
            best_dim_score = score;
            best_N = b.N;
            best_K = b.K;
        }
    }

    if (best_dim_score >= 1e20) {
        return result;  // No matching quant found at all
    }

    // Step 2: Among entries with (best_N, best_K, same quant), interpolate by batch
    const BenchmarkEntry * lower = nullptr;
    const BenchmarkEntry * upper = nullptr;
    int64_t lower_batch = 0;
    int64_t upper_batch = INT64_MAX;
                
    for (const auto & b : benchmarks) {
        if (b.op_name != op_name) continue;
        if (quant && !b.quant.empty() && b.quant != quant) continue;
        if (b.N != best_N || b.K != best_K) continue;

        int64_t batch = b.B;

        // Track lower bound (largest batch <= target)
        if (batch <= target_batch && batch > lower_batch) {
            lower = &b;
            lower_batch = batch;
                }
        // Track upper bound (smallest batch >= target)
        if (batch >= target_batch && batch < upper_batch) {
            upper = &b;
            upper_batch = batch;
        }
    }

    // Step 3: Interpolate
    if (lower && upper) {
        if (lower == upper) {
            // Exact batch match
            result.eff_gflop_s  = lower->eff_gflop_s;
            result.peak_gflop_s = lower->peak_gflop_s;
            result.bw_gb_s      = lower->bw_gb_s;
            result.ridge        = lower->ridge;
            result.eff_pcie_bw  = lower->eff_pcie_bw;
            result.valid        = true;
            result.exact_match  = (best_dim_score < 0.001);  // Near-exact dims
        } else {
            // Interpolate between lower and upper batch sizes
            double t = (double)(target_batch - lower_batch) / (double)(upper_batch - lower_batch);
            result.eff_gflop_s  = lerp(lower->eff_gflop_s,  upper->eff_gflop_s,  t);
            result.peak_gflop_s = lerp(lower->peak_gflop_s, upper->peak_gflop_s, t);
            result.bw_gb_s      = lerp(lower->bw_gb_s,      upper->bw_gb_s,      t);
            result.ridge        = lerp(lower->ridge,        upper->ridge,        t);
            result.eff_pcie_bw  = lerp(lower->eff_pcie_bw,  upper->eff_pcie_bw,  t);
            result.valid        = true;
            result.exact_match  = false;
        }
    } else if (lower) {
        // Extrapolate from lower (target > all benchmarks)
        result.eff_gflop_s  = lower->eff_gflop_s;
        result.peak_gflop_s = lower->peak_gflop_s;
        result.bw_gb_s      = lower->bw_gb_s;
        result.ridge        = lower->ridge;
        result.eff_pcie_bw  = lower->eff_pcie_bw;
        result.valid        = true;
        result.exact_match  = false;
    } else if (upper) {
        // Extrapolate from upper (target < all benchmarks)
        result.eff_gflop_s  = upper->eff_gflop_s;
        result.peak_gflop_s = upper->peak_gflop_s;
        result.bw_gb_s      = upper->bw_gb_s;
        result.ridge        = upper->ridge;
        result.eff_pcie_bw  = upper->eff_pcie_bw;
        result.valid        = true;
        result.exact_match  = false;
    }
    // else: no matching benchmarks found, result.valid = false

    return result;
}

// Bits-per-weight mapping for quant interpolation
static double get_bits_per_weight(const std::string & quant) {
    static const std::unordered_map<std::string, double> bpw = {
        {"q2_K", 2.6}, {"q3_K", 3.4}, {"q4_0", 4.5}, {"q4_1", 5.0}, {"q4_K", 4.5},
        {"q5_0", 5.5}, {"q5_1", 6.0}, {"q5_K", 5.5}, {"q6_K", 6.6},
        {"q8_0", 8.5}, {"q8_1", 9.0}, {"f16", 16.0}, {"f32", 32.0}
    };
    auto it = bpw.find(quant);
    return it != bpw.end() ? it->second : -1.0;  // -1 = unknown quant
        }
        
// Interpolate by quant type using bits-per-weight similarity
// Uses dimension scoring to find closest (N,K) across ALL quants,
// then interpolates by BPW
InterpolatedMetrics BenchmarkPredictor::interpolate_by_quant(
        const std::vector<BenchmarkEntry> & benchmarks,
        const char * op_name, const char * target_quant,
        int64_t N, int64_t K, int64_t target_batch) const {

    InterpolatedMetrics result = {0, 0, 0, 0, 0, false, false};

    // Get target BPW - return invalid if unknown quant
    if (!target_quant) {
        return result;  // No quant specified, can't interpolate
    }
    double target_bpw = get_bits_per_weight(target_quant);
    if (target_bpw < 0) {
        return result;  // Unknown quant type, can't interpolate
    }

    // Step 1: Find best (N,K) match using dimension scoring (ANY quant)
    double best_dim_score = 1e20;
    int64_t best_N = 0, best_K = 0;

    for (const auto & b : benchmarks) {
        if (b.op_name != op_name) continue;

        double n_diff = std::abs((double)b.N - N) / std::max(N, (int64_t)1);
        double k_diff = std::abs((double)b.K - K) / std::max(K, (int64_t)1);
        double score = n_diff + k_diff;
            
        if (score < best_dim_score) {
            best_dim_score = score;
            best_N = b.N;
            best_K = b.K;
            }
    }

    if (best_dim_score >= 1e20) {
        return result;  // No benchmarks at all
    }

    // Step 2: Among entries with (best_N, best_K), find batch match then interpolate by quant
    const BenchmarkEntry * lower = nullptr;
    const BenchmarkEntry * upper = nullptr;
    double lower_bpw = 0.0, upper_bpw = 100.0;

    for (const auto & b : benchmarks) {
        if (b.op_name != op_name) continue;
        if (b.N != best_N || b.K != best_K) continue;

        // Match batch approximately (within 2x)
        if (target_batch > 0 && b.B > 0) {
            double batch_ratio = (double)b.B / target_batch;
            if (batch_ratio < 0.5 || batch_ratio > 2.0) continue;
                }

        double q_bpw = get_bits_per_weight(b.quant);
        if (q_bpw < 0) continue;  // Unknown quant type

        // Track lower bound (largest bpw <= target)
        if (q_bpw <= target_bpw && q_bpw > lower_bpw) {
            lower = &b;
            lower_bpw = q_bpw;
        }
        // Track upper bound (smallest bpw >= target)
        if (q_bpw >= target_bpw && q_bpw < upper_bpw) {
            upper = &b;
            upper_bpw = q_bpw;
        }
    }

    // Step 3: Interpolate between quant types
    if (lower && upper) {
        if (lower == upper) {
            // Same quant type found
            result.eff_gflop_s  = lower->eff_gflop_s;
            result.peak_gflop_s = lower->peak_gflop_s;
            result.bw_gb_s      = lower->bw_gb_s;
            result.ridge        = lower->ridge;
            result.eff_pcie_bw  = lower->eff_pcie_bw;
            result.valid        = true;
            result.exact_match  = false;
    } else {
            double t = (target_bpw - lower_bpw) / (upper_bpw - lower_bpw);
            result.eff_gflop_s  = lerp(lower->eff_gflop_s,  upper->eff_gflop_s,  t);
            result.peak_gflop_s = lerp(lower->peak_gflop_s, upper->peak_gflop_s, t);
            result.bw_gb_s      = lerp(lower->bw_gb_s,      upper->bw_gb_s,      t);
            result.ridge        = lerp(lower->ridge,        upper->ridge,        t);
            result.eff_pcie_bw  = lerp(lower->eff_pcie_bw,  upper->eff_pcie_bw,  t);
            result.valid        = true;
            result.exact_match  = false;
        }
    } else if (lower) {
        // Extrapolate from lower (target bpw > all benchmarks)
        result.eff_gflop_s  = lower->eff_gflop_s;
        result.peak_gflop_s = lower->peak_gflop_s;
        result.bw_gb_s      = lower->bw_gb_s;
        result.ridge        = lower->ridge;
        result.eff_pcie_bw  = lower->eff_pcie_bw;
        result.valid        = true;
        result.exact_match  = false;
    } else if (upper) {
        // Extrapolate from upper (target bpw < all benchmarks)
        result.eff_gflop_s  = upper->eff_gflop_s;
        result.peak_gflop_s = upper->peak_gflop_s;
        result.bw_gb_s      = upper->bw_gb_s;
        result.ridge        = upper->ridge;
        result.eff_pcie_bw  = upper->eff_pcie_bw;
        result.valid        = true;
        result.exact_match  = false;
    }
    
    return result;
}

// Create cache key for timing: "GPU|MUL_MAT|Q4_0|4096|1|32" or "CPU_ASYNC|FLASH_ATTN|1024|32"
// batch_size param is used for MUL_MAT/FLASH_ATTN to match benchmark lookup (not tensor m.M)
std::string BenchmarkPredictor::make_timing_key(bool is_gpu, bool async_copy, const char * op_name, const OpMetrics & m, int32_t batch_size) {
    // CPU timing varies based on async_copy (uses eff vs peak bandwidth)
    std::string key = is_gpu ? "GPU|" : (async_copy ? "CPU_ASYNC|" : "CPU_SYNC|");
    key += op_name;
    key += "|";

    if (m.quant_type) {
        key += m.quant_type;
        key += "|";
    }

    // Include shape dimensions based on op type
    // Use BOTH batch_size (for benchmark lookup context) AND m.M (actual tensor size that affects timing)
    if (strstr(op_name, "MUL_MAT_ID")) {
        // MoE: include N, K, batch_size, M, and n_experts_used
        key += std::to_string(m.N) + "|" + std::to_string(m.K) + "|" + std::to_string(batch_size) + "|" + std::to_string(m.M) + "|" + std::to_string(m.n_experts_used);
    } else if (strstr(op_name, "MUL_MAT")) {
        // Dense: include N, K, batch_size, and M (tensor batch)
        key += std::to_string(m.N) + "|" + std::to_string(m.K) + "|" + std::to_string(batch_size) + "|" + std::to_string(m.M);
    } else if (strstr(op_name, "FLASH_ATTN")) {
        // Include dimensions + batch_size + n_tokens (actual tensor batch)
        key += std::to_string(m.head_dim) + "|" + std::to_string(m.n_q_heads) + "|" + 
               std::to_string(batch_size) + "|" + std::to_string(m.n_tokens) + "|" + std::to_string(m.ctx_len) + "|" + std::to_string(m.n_kv_heads);
    } else {
        key += std::to_string(m.n_elements);
    }

    return key;
}

// Extract ops/bytes metrics from a tensor (shared by predict and debug functions)
OpMetrics calculate_op_metrics(const ggml_tensor * node) {
    OpMetrics m = {0.0, 0.0, nullptr, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        
        switch (node->op) {
            case GGML_OP_MUL_MAT: {
            m.N = node->ne[0];
            m.M = node->ne[1];  // batch dimension - needed for cache key
            m.K = node->src[0]->ne[0];
            m.ops = 2.0 * m.N * m.K * m.M;
            m.bytes = ggml_nbytes(node->src[0]) + ggml_nbytes(node->src[1]) + ggml_nbytes(node);
            m.quant_type = ggml_type_name(node->src[0]->type);
                break;
            }
            case GGML_OP_MUL_MAT_ID: {
            m.N = node->ne[0];
            m.n_experts_used = node->ne[1];  // needed for cache key
                int64_t total_experts = node->src[0]->ne[2];
            m.M = node->ne[2];  // batch dimension for MoE
            m.K = node->src[0]->ne[0];
            m.ops = 2.0 * m.N * m.K * m.M * m.n_experts_used;
            m.bytes = (ggml_nbytes(node->src[0]) * m.n_experts_used / total_experts) +
                        ggml_nbytes(node->src[1]) + ggml_nbytes(node);
            m.quant_type = ggml_type_name(node->src[0]->type);
                break;
            }
            case GGML_OP_FLASH_ATTN_EXT: {
            m.head_dim = node->ne[0];
            m.n_q_heads = node->ne[1];
            m.n_tokens = node->ne[2];
            m.ctx_len = node->src[1]->ne[1];
            m.n_kv_heads = node->src[1]->ne[2];
            m.ops = 2.0 * m.n_tokens * m.head_dim * m.ctx_len * m.n_q_heads * 2.0;
            m.bytes = ggml_nbytes(node->src[0]) + ggml_nbytes(node->src[1]) +
                        ggml_nbytes(node->src[2]) + ggml_nbytes(node);
                        break;
                    }
        case GGML_OP_ROPE:
        case GGML_OP_ROPE_BACK: {
            m.n_elements = ggml_nelements(node);
            m.ops = 6.0 * m.n_elements;
            m.bytes = ggml_nbytes(node->src[0]) + ggml_nbytes(node);
            m.quant_type = ggml_type_name(node->type);
            break;
        }
        case GGML_OP_RMS_NORM: {
            m.n_elements = ggml_nelements(node);
            m.ops = 3.0 * m.n_elements;
            m.bytes = ggml_nbytes(node->src[0]) + ggml_nbytes(node);
            m.quant_type = ggml_type_name(node->type);
            break;
        }
        case GGML_OP_GLU: {
            m.n_elements = ggml_nelements(node);
            m.ops = 2.0 * m.n_elements;
            m.bytes = ggml_nbytes(node->src[0]);
            if (node->src[1]) m.bytes += ggml_nbytes(node->src[1]);
            m.bytes += ggml_nbytes(node);
            m.quant_type = ggml_type_name(node->type);
                break;
            }
        case GGML_OP_ADD:
        case GGML_OP_MUL:
        case GGML_OP_SUB:
        case GGML_OP_DIV: {
            m.n_elements = ggml_nelements(node);
            m.ops = m.n_elements;
            m.bytes = ggml_nbytes(node->src[0]);
            if (node->src[1]) m.bytes += ggml_nbytes(node->src[1]);
            m.bytes += ggml_nbytes(node);
            m.quant_type = ggml_type_name(node->type);  // For cache key
                break;
            }
            case GGML_OP_GET_ROWS: {
            m.n_elements = ggml_nelements(node);  // For unique cache key per size
            m.bytes = ggml_nbytes(node);
            m.quant_type = ggml_type_name(node->src[0]->type);  // Source type affects timing
                break;
            }
            case GGML_OP_SET_ROWS:
            case GGML_OP_CPY: {
            m.n_elements = ggml_nelements(node);  // For unique cache key per size
            m.bytes = ggml_nbytes(node->src[0]) + ggml_nbytes(node);
            m.quant_type = ggml_type_name(node->src[0]->type);  // Source type affects timing
                break;
            }
        // Zero-cost metadata/view ops
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_NONE:
            break;
        default: {
            // Unknown ops: estimate as memory-bound
            m.n_elements = ggml_nelements(node);  // For unique cache key per size
            m.bytes = ggml_nbytes(node);
            if (node->src[0]) m.bytes += ggml_nbytes(node->src[0]);
            m.quant_type = ggml_type_name(node->type);
            break;
        }
    }
    return m;
}

void BenchmarkPredictor::build_lookup_maps() {
    // Build CPU benchmark map (keys include batch size for MUL_MAT ops)
    for (const auto & bench : m_cpu_benchmarks) {
        std::string key;
        if (bench.op_name == "MUL_MAT" || bench.op_name == "MUL_MAT_ID") {
            key = make_key(bench.op_name, bench.quant, bench.N, bench.K, bench.B);
        } else if (bench.op_name.find("FLASH_ATTN") == 0) {
            key = make_attn_key(bench.ctx_len, bench.n_heads, bench.n_tokens);
            } else {
            key = make_elem_key(bench.op_name, bench.n_elements);
        }
        m_cpu_map[key] = &bench;
    }

    // Build GPU benchmark map
    for (const auto & bench : m_gpu_benchmarks) {
        std::string key;
        if (bench.op_name == "MUL_MAT" || bench.op_name == "MUL_MAT_ID") {
            key = make_key(bench.op_name, bench.quant, bench.N, bench.K, bench.B);
        } else if (bench.op_name.find("FLASH_ATTN") == 0) {
            key = make_attn_key(bench.ctx_len, bench.n_heads, bench.n_tokens);
                } else {
            key = make_elem_key(bench.op_name, bench.n_elements);
        }
        m_gpu_map[key] = &bench;
    }
}

// Nearest benchmark search with scoring for dimensions, batch, and quant
// Uses weighted scoring to find closest match without requiring exact values
const BenchmarkEntry * BenchmarkPredictor::find_nearest_benchmark(
        const std::vector<BenchmarkEntry> & benchmarks,
        const char * op_name, const char * quant,
        int64_t N, int64_t K, int64_t ctx_len, int64_t n_elements,
        int64_t n_kv_heads, int64_t target_batch) {

    const BenchmarkEntry * best_match = nullptr;
    double best_score = 1e20;

    // Get target BPW for quant scoring
    double target_bpw = quant ? get_bits_per_weight(quant) : -1.0;

    for (const auto & bench : benchmarks) {
        // Exact match to avoid "MUL" matching "MUL_MAT"
        if (bench.op_name != op_name) continue;

        double dim_score = 0.0;
        double batch_score = 0.0;
        double quant_score = 0.0;

        // Dimension scoring based on op type
        if (strcmp(op_name, "MUL_MAT") == 0 || strcmp(op_name, "MUL_MAT_ID") == 0) {
            double n_diff = std::abs((double)bench.N - N) / std::max(N, (int64_t)1);
            double k_diff = std::abs((double)bench.K - K) / std::max(K, (int64_t)1);
            dim_score = n_diff + k_diff;
        } else if (strcmp(op_name, "FLASH_ATTN") == 0) {
            double ctx_diff = std::abs((double)bench.ctx_len - ctx_len) / std::max(ctx_len, (int64_t)1);
            double kv_diff = 0.0;
            if (n_kv_heads > 0 && bench.n_heads > 0) {
                kv_diff = std::abs((double)bench.n_heads - n_kv_heads) / std::max(n_kv_heads, (int64_t)1);
            }
            dim_score = ctx_diff + kv_diff * 0.5;
        } else {
            double elem_diff = std::abs((double)bench.n_elements - n_elements) / std::max(n_elements, (int64_t)1);
            dim_score = elem_diff;
        }

        // Batch scoring (normalized difference)
        if (target_batch > 0 && bench.B > 0) {
            batch_score = std::abs((double)bench.B - target_batch) / std::max(target_batch, (int64_t)1);
        }

        // Quant scoring (BPW difference) - only if not requiring exact quant match
        if (target_bpw > 0) {
            double bench_bpw = get_bits_per_weight(bench.quant);
            if (bench_bpw > 0) {
                quant_score = std::abs(bench_bpw - target_bpw) / target_bpw;
            } else {
                quant_score = 1.0;  // Penalty for unknown quant
            }
        } else if (quant && bench.quant != quant) {
            continue;  // Require exact quant match if BPW unknown
        }

        // Combined score: batch most important (memory vs compute bound), then dims, then quant
        double score = batch_score * 1.0 + dim_score * 0.5 + quant_score * 0.3;

        if (score < best_score) {
            best_score = score;
            best_match = &bench;
        }
    }

    return best_match;
}

void BenchmarkPredictor::debug_print_split_impl(ggml_backend_sched_t sched, int split_id, bool is_gpu, int32_t batch_size) const {
    const auto & benchmarks = is_gpu ? m_gpu_benchmarks : m_cpu_benchmarks;
    const auto & map = is_gpu ? m_gpu_map : m_cpu_map;

    struct ggml_cgraph split_graph = ggml_backend_get_split_graph(sched, split_id);
    
    // Collect all ops with their predicted times
    struct OpInfo {
        int index;
        const char * op_name;
        double time_ms;
        double ops;
        double bytes;
        int64_t N, K;
        const char * quant;
        bool exact_match;
    };
    std::vector<OpInfo> op_infos;
    double total_time = 0.0;
    
    for (int i = 0; i < split_graph.n_nodes; ++i) {
        ggml_tensor * node = split_graph.nodes[i];
        OpMetrics m = calculate_op_metrics(node);
        
        if (m.ops == 0.0 && m.bytes == 0.0) continue;  // Skip unknown ops
        
        OpInfo info;
        info.index = i;
        info.op_name = ggml_op_name(node->op);
        info.ops = m.ops;
        info.bytes = m.bytes;
        info.N = m.N;
        info.K = m.K;
        info.quant = m.quant_type ? m.quant_type : "N/A";
        info.time_ms = 0.0;
        info.exact_match = false;
        
        // Try exact match first
        std::string key;
        if (node->op == GGML_OP_MUL_MAT) {
            key = make_key("MUL_MAT", m.quant_type ? m.quant_type : "", m.N, m.K, batch_size);
        } else if (node->op == GGML_OP_MUL_MAT_ID) {
            key = make_key("MUL_MAT_ID", m.quant_type ? m.quant_type : "", m.N, m.K, batch_size);
        }
        
        auto it = map.find(key);
        if (it != map.end()) {
            const BenchmarkEntry * b = it->second;
            double gflop_s = is_gpu ? b->peak_gflop_s : b->eff_gflop_s;
            info.time_ms = (m.ops / 1e9) / gflop_s * 1000.0;
            info.exact_match = true;
        } else {
            // Nearest match
            const BenchmarkEntry * match = nullptr;
        switch (node->op) {
                case GGML_OP_MUL_MAT:
                    match = find_nearest_benchmark(benchmarks, "MUL_MAT", m.quant_type, m.N, m.K, 0, 0, -1, batch_size);
                break;
                case GGML_OP_MUL_MAT_ID:
                    match = find_nearest_benchmark(benchmarks, "MUL_MAT_ID", m.quant_type, m.N, m.K, 0, 0, -1, batch_size);
                break;
                case GGML_OP_FLASH_ATTN_EXT:
                    match = find_nearest_benchmark(benchmarks, "FLASH_ATTN", nullptr, 0, 0, m.ctx_len, 0, m.n_kv_heads, batch_size);
                break;
                default:
                    match = find_nearest_benchmark(benchmarks, info.op_name, nullptr, 0, 0, 0, m.n_elements, -1, batch_size);
                break;
            }
            
            if (match) {
                double gflop_s = is_gpu ? match->peak_gflop_s : match->eff_gflop_s;
                double ai = m.bytes > 0 ? (m.ops / m.bytes) : 0.0;
                if (ai < match->ridge) {
                    info.time_ms = (m.bytes / 1e9) / match->bw_gb_s * 1000.0;
                } else {
                    info.time_ms = (m.ops / 1e9) / gflop_s * 1000.0;
                }
            } else {
                // Memory BW fallback
                double bw = is_gpu ? m_stats.peak_gpu_memory_bw : m_stats.eff_system_bw;
                if (bw > 0 && m.bytes > 0) {
                    info.time_ms = (m.bytes / 1e9) / bw * 1000.0;
                }
            }
        }
        
        total_time += info.time_ms;
        op_infos.push_back(info);
    }
    
    // Sort by time (descending)
    std::sort(op_infos.begin(), op_infos.end(), [](const OpInfo & a, const OpInfo & b) {
        return a.time_ms > b.time_ms;
    });
    
    // Print header and top 10
    LLAMA_LOG_INFO("\n=== Split %d: TOP 10 Time Contributors (%s, batch=%d, total=%.3f ms) ===\n", 
                   split_id, is_gpu ? "GPU" : "CPU", batch_size, total_time);
    LLAMA_LOG_INFO("%-4s %-12s %8s %8s %10s %10s %6s %s\n",
                   "Rank", "Op", "Time(ms)", "Pct", "FLOPs", "Bytes", "Quant", "Match");
    LLAMA_LOG_INFO("---- ------------ -------- -------- ---------- ---------- ------ -----\n");
    
    int count = std::min((int)op_infos.size(), 20);
    for (int i = 0; i < count; ++i) {
        const OpInfo & info = op_infos[i];
        double pct = total_time > 0 ? (info.time_ms / total_time * 100.0) : 0.0;
        LLAMA_LOG_INFO("%4d %-12s %8.3f %7.1f%% %10.2e %10.2e %6s %s\n",
                       i + 1, info.op_name, info.time_ms, pct, info.ops, info.bytes,
                       info.quant, info.exact_match ? "EXACT" : "NEAR");
    }
    
    LLAMA_LOG_INFO("=== End Split %d ===\n\n", split_id);
}

void BenchmarkPredictor::debug_print_split(ggml_backend_sched_t sched, size_t split_id, bool is_gpu, int32_t batch_size) const {
    debug_print_split_impl(sched, (int)split_id, is_gpu, batch_size);
}

// --- BenchmarkPredictor methods ---

SplitTimingResult BenchmarkPredictor::predict(ggml_backend_sched_t sched, size_t split_id, bool is_gpu, int32_t batch_size, bool async_copy) const {
    SplitTimingResult result = {0.0, 0.0, 0, 0, 0, 0, 0, {}};
    struct ggml_cgraph split_graph = ggml_backend_get_split_graph(sched, (int)split_id);

    // Select appropriate map and benchmarks based on is_gpu
    const auto & map = is_gpu ? m_gpu_map : m_cpu_map;
    const auto & benchmarks = is_gpu ? m_gpu_benchmarks : m_cpu_benchmarks;

    for (int i = 0; i < split_graph.n_nodes; ++i) {
        ggml_tensor * node = split_graph.nodes[i];

        // Use shared helper for ops/bytes extraction
        OpMetrics m = calculate_op_metrics(node);
        if (m.ops == 0.0 && m.bytes == 0.0) continue;  // Skip unknown ops
        double op_time_ms = 0.0;

        const BenchmarkEntry * match = nullptr;
        bool exact_match = false;

        // Hash map lookup based on op type (keys include batch_size for MUL_MAT)
        std::string key;
        switch (node->op) {
            case GGML_OP_MUL_MAT:
                key = make_key("MUL_MAT", m.quant_type ? m.quant_type : "", m.N, m.K, batch_size);
                break;
            case GGML_OP_MUL_MAT_ID:
                key = make_key("MUL_MAT_ID", m.quant_type ? m.quant_type : "", m.N, m.K, batch_size);
                break;
            case GGML_OP_FLASH_ATTN_EXT:
                key = make_attn_key(m.ctx_len, m.n_kv_heads, batch_size);
                break;
            case GGML_OP_ROPE:
            case GGML_OP_RMS_NORM:
            case GGML_OP_GLU:
            case GGML_OP_ADD:
            case GGML_OP_MUL:
            case GGML_OP_SUB:
            case GGML_OP_DIV:
                key = make_elem_key(ggml_op_name(node->op), m.n_elements);
                break;
            default:
                break;  // Memory ops - no benchmark lookup
        }

        // Check timing cache first (same shape + async mode + batch_size = same timing across layers)
        std::string timing_key = make_timing_key(is_gpu, async_copy, ggml_op_name(node->op), m, batch_size);
        auto cache_it = m_timing_cache.find(timing_key);
        if (cache_it != m_timing_cache.end()) {
            // Cache hit - use cached timing
            op_time_ms = cache_it->second.first;
            double pcie_contrib = cache_it->second.second;
            result.time_ms += op_time_ms;
            if (!is_gpu) {
                result.eff_pcie_bw += pcie_contrib;
            }
            result.n_exact++;  // Count cache hits as "exact"
        } else {
            // Cache miss - compute timing
            InterpolatedMetrics interp = {0, 0, 0, 0, 0, false, false};
            double pcie_contrib = 0.0;

            if (!key.empty()) {
                auto it = map.find(key);
                if (it != map.end()) {
                    // Exact hash match
                    const BenchmarkEntry * b = it->second;
                    interp.eff_gflop_s  = b->eff_gflop_s;
                    interp.peak_gflop_s = b->peak_gflop_s;
                    interp.bw_gb_s      = b->bw_gb_s;
                    interp.ridge        = b->ridge;
                    interp.eff_pcie_bw  = b->eff_pcie_bw;
                    interp.valid        = true;
                    interp.exact_match  = true;
                    exact_match         = true;
                    result.n_exact++;
                } else {
                    // Fallback: use old or new prediction method based on flag
                    if (m_use_old_prediction) {
                        // OLD METHOD: find_nearest_benchmark (scoring-based with batch/quant)
                        const BenchmarkEntry * match = nullptr;
                        switch (node->op) {
                            case GGML_OP_MUL_MAT:
                                match = find_nearest_benchmark(benchmarks, "MUL_MAT", m.quant_type, m.N, m.K, 0, 0, -1, batch_size);
                                break;
                            case GGML_OP_MUL_MAT_ID:
                                match = find_nearest_benchmark(benchmarks, "MUL_MAT_ID", m.quant_type, m.N, m.K, 0, 0, -1, batch_size);
                                break;
                            case GGML_OP_FLASH_ATTN_EXT:
                                match = find_nearest_benchmark(benchmarks, "FLASH_ATTN", nullptr, 0, 0, m.ctx_len, 0, m.n_kv_heads, batch_size);
                                break;
                            default:
                                match = find_nearest_benchmark(benchmarks, ggml_op_name(node->op), nullptr, 0, 0, 0, m.n_elements, -1, batch_size);
                break;
            }
                        if (match) {
                            interp.eff_gflop_s  = match->eff_gflop_s;
                            interp.peak_gflop_s = match->peak_gflop_s;
                            interp.bw_gb_s      = match->bw_gb_s;
                            interp.ridge        = match->ridge;
                            interp.eff_pcie_bw  = match->eff_pcie_bw;
                            interp.valid        = true;
                            interp.exact_match  = false;  // Nearest match, not exact
                            result.n_interp++;
                        }
                    } else {
                        // NEW METHOD: batch interpolation first, then dimension interpolation
                        switch (node->op) {
                            case GGML_OP_MUL_MAT:
                                interp = interpolate_by_batch(benchmarks, "MUL_MAT", m.quant_type, m.N, m.K, batch_size);
                                if (!interp.valid) {
                                    // Batch interp failed (no matching N,K or quant), try quant interpolation
                                    interp = interpolate_by_quant(benchmarks, "MUL_MAT", m.quant_type, m.N, m.K, batch_size);
                                }
                break;
                            case GGML_OP_MUL_MAT_ID:
                                interp = interpolate_by_batch(benchmarks, "MUL_MAT_ID", m.quant_type, m.N, m.K, batch_size);
                                if (!interp.valid) {
                                    interp = interpolate_by_quant(benchmarks, "MUL_MAT_ID", m.quant_type, m.N, m.K, batch_size);
            }
                                break;
                            case GGML_OP_FLASH_ATTN_EXT:
                                interp = interpolate_benchmark(benchmarks, "FLASH_ATTN", m.quant_type, m.ctx_len);
                                break;
            default:
                                interp = interpolate_benchmark(benchmarks, ggml_op_name(node->op), m.quant_type, m.n_elements);
                                break;
                        }
                        if (interp.valid) {
                            result.n_interp++;
                        }
        }
                }
            }

            // Calculate timing using interpolated metrics
            if (interp.valid) {
                // GPU always uses peak; CPU uses eff if async_copy (PCIe contention), peak otherwise
                double gflop_s = is_gpu ? interp.peak_gflop_s : (async_copy ? interp.eff_gflop_s : interp.peak_gflop_s);

                if (interp.exact_match) {
                    op_time_ms = (m.ops / 1e9) / gflop_s * 1000.0;
                } else {
                    double ai = m.bytes > 0 ? (m.ops / m.bytes) : 0.0;
                    if (ai < interp.ridge) {
                        op_time_ms = (m.bytes / 1e9) / interp.bw_gb_s * 1000.0;
                    } else {
                        op_time_ms = (m.ops / 1e9) / gflop_s * 1000.0;
                    }
                }

                result.time_ms += op_time_ms;
                pcie_contrib = op_time_ms * interp.eff_pcie_bw;

                if (!is_gpu) {
                    result.eff_pcie_bw += pcie_contrib;
                }
            } else {
                // Fallback to memory bandwidth
                result.n_fallback++;
                if (m.bytes > 0.0) {
                    // GPU uses peak; CPU uses eff if async_copy (PCIe contention), peak otherwise
                    double bw = is_gpu ? m_stats.peak_gpu_memory_bw : (async_copy ? m_stats.eff_system_bw : m_stats.peak_system_bw);
                    if (bw > 0.0) {
                        op_time_ms = (m.bytes / 1e9) / bw * 1000.0;
                        result.time_ms += op_time_ms;
                    }

                    if (!is_gpu && m_stats.peak_pcie_bw > 0.0) {
                        pcie_contrib = op_time_ms * m_stats.eff_pcie_bw;
                        result.eff_pcie_bw += pcie_contrib;
        }
    }
            }

            // Store in cache for future lookups
            m_timing_cache[timing_key] = {op_time_ms, pcie_contrib};
        }

        OpTimingDetail detail;
        detail.op_name = ggml_op_name(node->op);
        detail.node_name = ggml_get_name(node) ? ggml_get_name(node) : "unnamed";
        detail.time_ms = op_time_ms;
        detail.flops = m.ops;
        detail.bytes = m.bytes;
        detail.exact_match = exact_match;
        result.op_details.push_back(detail);
    }

    result.op_count = (int)result.op_details.size();

    if (!is_gpu) {
        if (result.time_ms > 0.0) {
            result.eff_pcie_bw /= result.time_ms;
        } else {
            result.eff_pcie_bw = m_stats.eff_pcie_bw;
        }
    } else {
        result.eff_pcie_bw = 0.0;
    }

    return result;
}
