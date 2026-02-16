#pragma once

#include <vector>
#include <set>
#include <string>
#include <unordered_map>
#include <map>
#include "llama-benchmark.h"

// Dynamic plan building toggle
// false = DISABLED - use pre-built tier plans only (original behavior, ceiling tier lookup)
// true  = ENABLED  - build plans for exact n_tokens using plan_build_for_batch
static constexpr bool DYNAMIC_PLAN_ENABLED = false;

// All the enums and structs related to the pipeline plan
// Defines the backend where a layer or a split is executed.
enum LAYER_BACKENDS {
    CPU = 0,
    GPU_PINNED = 1,  // The layer is executed on the GPU and its weights are pinned in VRAM.
    GPU_SHARDED = 2, // The layer is executed on the GPU, but its weights are sharded and moved to VRAM on demand.
    GPU = 3,
};

// GEN_PHASE_MODE removed - now using exec_strategy enum

// Execution strategy - determines overall plan building approach
// PRIO strategies use progressive logic: first fit attn, then try FFN (for MoE: attn→output→FFN)
enum exec_strategy {
    EXEC_STRATEGY_FULL_GPU           = 0,  // Everything on GPU (including output)
    EXEC_STRATEGY_FULL_GPU_NO_OUTPUT = 1,  // Everything on GPU except output
    EXEC_STRATEGY_STATIC_ATTN_PRIO   = 2,  // Progressive: attn first → then FFN (static, no sharding) - DEFAULT
    EXEC_STRATEGY_SHARD_ATTN_PRIO    = 3,  // Progressive: attn first → then FFN (with sharding fallback)
    EXEC_STRATEGY_FULL_SHARD         = 4,  // Stream all layers (double-buffered sharding)
    EXEC_STRATEGY_COUNT              = 5,
};

// Strategy names - single source of truth, must match exec_strategy enum order
inline const char * exec_strategy_name(exec_strategy s) {
    static const char * names[] = {
        "FULL_GPU",
        "FULL_GPU_NO_OUTPUT",
        "STATIC_ATTN_PRIO",
        "SHARD_ATTN_PRIO",
        "FULL_SHARD",
    };
    return (s >= 0 && s < EXEC_STRATEGY_COUNT) ? names[s] : "UNKNOWN";
}

// Layer placement strategy - determines where attention and FFN weights are placed
// These values are used in ExecutionPlan::strategies[] per layer
enum layer_strategy {
    LAYER_STRATEGY_ALL_GPU_SHARDED         = 1,  // attn: GPU[layer+1], FFN: GPU[layer+1]
    LAYER_STRATEGY_ATTN_GPU_PINNED         = 2,  // attn: GPU[0],       FFN: CPU
    LAYER_STRATEGY_FFN_GPU_PINNED          = 3,  // attn: CPU,          FFN: GPU[0]
    LAYER_STRATEGY_ALL_CPU                 = 4,  // attn: CPU,          FFN: CPU
    LAYER_STRATEGY_ALL_GPU_PINNED          = 5,  // attn: GPU[0],       FFN: GPU[0]
    LAYER_STRATEGY_ATTN_GPU_SHARDED        = 6,  // attn: GPU[1],       FFN: CPU
    LAYER_STRATEGY_FFN_GPU_SHARDED         = 7,  // attn: CPU,          FFN: GPU[1]
    LAYER_STRATEGY_ATTN_PINNED_FFN_SHARDED = 8,  // attn: GPU[0],       FFN: GPU[layer+1]
    LAYER_STRATEGY_COUNT                   = 9,
};

// Configuration for each exec_strategy - defines layer placement behavior
struct strategy_config {
    layer_strategy base;           // default for unpinned layers
    layer_strategy pin;            // for pinned layers
    layer_strategy shard;          // for sharded fallback (if can_shard)
    bool           can_shard;      // whether partial sharding is allowed
    int            token_emb;      // -1 = CPU, 0 = GPU[0]
    int            output;         // -1 = CPU, 0 = GPU[0]
    int            n_shard_slots;  // number of VRAM slots for sharding (1 = alternating, 2 = double-buffer)
};

// Strategy configuration table - one entry per exec_strategy
// Note: -1 represents GGML_CUDA_MAX_DEVICES (CPU backend)
static const strategy_config STRATEGY_CONFIGS[EXEC_STRATEGY_COUNT] = {
    //                                             base                               pin                                shard                            can_shard  emb  out n_shard
    /* FULL_GPU           */ { LAYER_STRATEGY_ALL_GPU_PINNED,   LAYER_STRATEGY_ALL_GPU_PINNED,  LAYER_STRATEGY_ALL_GPU_PINNED,   false,      0,   0,  0 },
    /* FULL_GPU_NO_OUTPUT */ { LAYER_STRATEGY_ALL_GPU_PINNED,   LAYER_STRATEGY_ALL_GPU_PINNED,  LAYER_STRATEGY_ALL_GPU_PINNED,   false,     -1,  -1,  0 },
    /* STATIC_ATTN_PRIO   */ { LAYER_STRATEGY_ALL_CPU,          LAYER_STRATEGY_ATTN_GPU_PINNED, LAYER_STRATEGY_ATTN_GPU_SHARDED, false,     -1,  -1,  0 },
    /* SHARD_ATTN_PRIO    */ { LAYER_STRATEGY_ALL_CPU,          LAYER_STRATEGY_ATTN_GPU_PINNED, LAYER_STRATEGY_ATTN_GPU_SHARDED, true,      -1,  -1,  1 },
    /* FULL_SHARD         */ { LAYER_STRATEGY_ALL_GPU_SHARDED,  LAYER_STRATEGY_ALL_GPU_PINNED,  LAYER_STRATEGY_ALL_GPU_SHARDED,  true,      -1,  -1,  2 },
};

// PipelinePhase removed - unified architecture uses tier-based plan switching

// Represents a single unit of work in the pipeline.
// A split is a group of layers that are executed together on the same backend.
struct PipelineSplit {
    LAYER_BACKENDS                    backend_type;  // The backend where this split is executed.
    std::vector<struct ggml_tensor *> leafs;
    size_t                            size;
    int                               backend_id;
    std::set<int>                     kv_layer_ids;
    size_t                            pinned_size;
};

// Batch tier configuration for unified plan registry
struct batch_tier_config {
    std::vector<uint32_t> sizes;

    //void init(uint32_t n_ubatch, uint32_t n_parallel = 1, uint32_t n_draft = 0) {
    //    sizes.clear();
    //    
    //    // Single parallel: typically 2 tiers (decode=1, prefill=n_ubatch)
    //    // With speculative decoding: 3 tiers (decode=1, verify=n_draft+1, prefill=n_ubatch)
    //    if (n_parallel <= 1) {
    //        sizes.push_back(1);
    //        if (n_draft > 0) {
    //            uint32_t verify_tier = n_draft + 1;
    //            if (verify_tier > 1 && verify_tier < n_ubatch) {
    //                sizes.push_back(verify_tier);
    //            }
    //        }
    //        if (n_ubatch > 1 && (sizes.empty() || sizes.back() != n_ubatch)) {
    //            sizes.push_back(n_ubatch);
    //        }
    //        return;
    //    }
    //    
    //    // Multi-parallel: two-phase exponential growth
    //    // - Decode phase (t <= 64): ×4 growth -> 1, 4, 16, 64
    //    // - Prefill phase (t > 64): ×8 growth -> 512, 4096...
    //    const uint32_t threshold = 64;
    //    const uint32_t slow_mult = 4;
    //    const uint32_t fast_mult = 8;
    //    
    //    for (uint32_t t = 1; t < n_ubatch; ) {
    //        sizes.push_back(t);
    //        t *= (t < threshold) ? slow_mult : fast_mult;
    //    }

    //    // Always add n_ubatch as the final tier
    //    if (sizes.empty() || sizes.back() != n_ubatch) {
    //        sizes.push_back(n_ubatch);
    //    }
    //}


    void init(uint32_t n_ubatch, uint32_t n_parallel = 1, uint32_t n_draft = 0) {
        sizes.clear();
    
        // Decode tiers
        if (n_parallel <= 1) {
            sizes.push_back(1);
            if (n_draft > 0) {
                uint32_t verify_tier = n_draft + 1;
                if (verify_tier > 1) {
                    sizes.push_back(verify_tier);
                }
            }
        } else {
            // Decode phase: ×4 growth -> 1, 4, 16, 64
            for (uint32_t t = 1; t <= 64 && t < 512; t *= 4) {
                sizes.push_back(t);
            }
        }

        // Prefill tiers: 512, 1024, 2048, ... up to n_ubatch (×2 growth)
        for (uint32_t t = 512; t < n_ubatch; t *= 2) {
            if (sizes.empty() || sizes.back() < t) {
                sizes.push_back(t);
            }
        }

        // Always add n_ubatch as final tier
        if (sizes.empty() || sizes.back() != n_ubatch) {
            sizes.push_back(n_ubatch);
        }
    }

    // Returns the ceiling tier (smallest tier where n_tokens <= tier_size)
    size_t tier_index(uint32_t n_tokens) const {
        for (size_t i = 0; i < sizes.size(); i++) {
            if (n_tokens <= sizes[i]) {
                return i;
            }
        }
        return sizes.size() - 1;
    }

    // Returns the nearest tier by absolute distance
    // Used when rebuilding plans on-the-fly for exact n_tokens
    size_t tier_index_nearest(uint32_t n_tokens) const {
        if (sizes.empty()) return 0;
        
        size_t best_tier = 0;
        int64_t best_dist = std::abs((int64_t)n_tokens - (int64_t)sizes[0]);
        
        for (size_t i = 1; i < sizes.size(); i++) {
            int64_t dist = std::abs((int64_t)n_tokens - (int64_t)sizes[i]);
            if (dist < best_dist) {
                best_dist = dist;
                best_tier = i;
            }
        }
        return best_tier;
    }

    uint32_t tier_size(size_t idx) const { return sizes[idx]; }
    size_t   n_tiers()             const { return sizes.size(); }
};

// Per-plan allocator state for restore after plan switching
// Stores the complete allocator state to avoid fragmentation from fresh allocation
struct PlanAllocState {
    std::vector<uint8_t> node_allocs;           // Serialized node_alloc array from galloc
    std::vector<uint8_t> leaf_allocs;           // Serialized leaf_alloc array from galloc
    std::vector<int>     node_backend_ids;      // Backend IDs to avoid backend_ids_changed check
    std::vector<int>     leaf_backend_ids;
    int n_nodes = 0;
    int n_leafs = 0;
    bool valid = false;
};

// Defines the execution plan for a single phase (prompt or generation).
// This includes the splits, VRAM requirements, and other metadata.
struct ExecutionPlan {
    std::vector<PipelineSplit> pipeline_splits;
    size_t                     sharded_weights_vram_req  = 0;
    size_t                     pinned_weights_vram_req = 0;
    size_t                     scratch_vram_req = 0;
    size_t                     total_vram_req = 0;

    std::vector<int>                strategies;
    int                             token_emb_strategy;
    int                             output_strategy;
    std::map<struct ggml_tensor *, int32_t> leaf_to_buf_id_map;
    std::vector<ggml_tensor *> sorted_pinned_weights;
    size_t kv_vram_req = 0;
    bool is_viable = false;
    bool predict_tps = false;
    float tps = 0;
    
    // Per-plan allocator state - saved after planning, restored before execution
    PlanAllocState alloc_state;
};

// The single, unified structure holding all pipeline-sharding plans.
// This includes the plans for both the prompt and generation phases, as well as
// other global parameters that control the pipeline's behavior.
struct PipelinePlan {
    batch_tier_config                        tier_config;
    std::vector<std::vector<ExecutionPlan>>  plans;

    void init_plans(uint32_t n_ubatch, uint32_t n_parallel, uint32_t n_draft = 0) {
        tier_config.init(n_ubatch, n_parallel, n_draft);
        plans.resize(tier_config.n_tiers());
        for (size_t i = 0; i < plans.size(); i++) {
            plans[i].resize(EXEC_STRATEGY_COUNT);
        }
    }

    ExecutionPlan & plan_get(uint32_t n_tokens, exec_strategy strategy) {
        size_t tier = tier_config.tier_index(n_tokens);
        return plans[tier][strategy];
    }

    // Unified plan mode - when FULL_GPU fits at max tier
    bool           single_plan_mode = false;  // true = no plan switching needed
    ExecutionPlan  unified_plan;              // The single plan used when single_plan_mode=true
    
    // Runtime state
    size_t         current_tier     = 0;
    exec_strategy  current_strategy = EXEC_STRATEGY_FULL_GPU;

    size_t         total_kv_size     = 0; // The total size of the KV cache.
    size_t         kv_size_per_layer = 0; // The size of the KV cache per layer.
    size_t         cuda_host_size    = 0; // The size of the host memory required by CUDA.
    size_t         max_vram_alloc    = 0; // The maximum VRAM that can be allocated.

    bool           vram_allocated = false;   // Whether VRAM has been allocated.
    int            skip_weight_copy_id = 0;
    bool           is_moe = false;

    // Cache for dynamically built plans (keyed by n_tokens)
    // These plans reuse the strategy from the nearest tier but are sized for exact n_tokens
    std::unordered_map<uint32_t, ExecutionPlan> dynamic_plans;

    std::vector<BenchmarkEntry> benchmarks;
    std::vector<BenchmarkEntry> gpu_benchmarks;
    GlobalBenchmarkStats global_stats;
    std::vector<uint32_t> last_copied_kv_pos;
};
