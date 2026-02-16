#include "llama-pipeline-executor.h"
#include "llama-context.h"
#include "llama-model.h"
#include "llama-model-loader.h"
#include "llama-graph.h"
#include "ggml-alloc.h"
#include <thread>
#include <chrono>
#include <algorithm>
#include <regex>

// Scratch margin factor applied to all planning budget checks.
static constexpr float SCRATCH_MARGIN = 1.5f;

//
// plan_apply_layer_strategies
//
void PipelineExecutor::plan_apply_layer_strategies(ExecutionPlan & plan) {

    const int32_t cpu_backend_id = GGML_CUDA_MAX_DEVICES;
    const std::vector<int> & strategies = plan.strategies;

    for (int32_t il = 0; il < m_ctx.model.hparams.n_layer; ++il) {
        auto it = m_ctx.model.layer_to_tensor_map.find(il);
        if (it == m_ctx.model.layer_to_tensor_map.end()) {
            continue;
        }

        const std::vector<ggml_tensor *> & tensors = it->second;
        const layer_strategy strat = (layer_strategy) strategies[il];

        // Determine buffer IDs for attention and FFN based on strategy
        int32_t attn_buf_id = cpu_backend_id;
        int32_t ffn_buf_id  = cpu_backend_id;

        switch (strat) {
            case LAYER_STRATEGY_ALL_GPU_SHARDED:
                attn_buf_id = il + 1;
                ffn_buf_id  = il + 1;
                break;
            case LAYER_STRATEGY_ATTN_GPU_PINNED:
                attn_buf_id = 0;
                ffn_buf_id  = cpu_backend_id;
                break;
            case LAYER_STRATEGY_FFN_GPU_PINNED:
                attn_buf_id = cpu_backend_id;
                ffn_buf_id  = 0;
                break;
            case LAYER_STRATEGY_ALL_CPU:
                attn_buf_id = cpu_backend_id;
                ffn_buf_id  = cpu_backend_id;
                break;
            case LAYER_STRATEGY_ALL_GPU_PINNED:
                attn_buf_id = 0;
                ffn_buf_id  = 0;
                break;
            case LAYER_STRATEGY_ATTN_GPU_SHARDED:
                attn_buf_id = 1;
                ffn_buf_id  = cpu_backend_id;
                break;
            case LAYER_STRATEGY_FFN_GPU_SHARDED:
                attn_buf_id = cpu_backend_id;
                ffn_buf_id  = 1;
                break;
            case LAYER_STRATEGY_ATTN_PINNED_FFN_SHARDED:
                attn_buf_id = 0;
                ffn_buf_id  = il + 1;
                break;
            default:
                continue;
        }

        // Apply buffer IDs to tensors
        for (ggml_tensor * tensor : tensors) {
            const char * name = ggml_get_name(tensor);
            if (name == nullptr) {
                continue;
            }

            int32_t buf_id = cpu_backend_id;

            // Match tensor name to component
            if (strstr(name, "attn") || strstr(name, "ffn_norm") || strstr(name, "ffn_gate_inp")) {
                buf_id = attn_buf_id;
            } else if (strstr(name, "ffn_up") || strstr(name, "ffn_down") || strstr(name, "ffn_gate")) {
                buf_id = ffn_buf_id;
            }

            tensor->buffer_id = buf_id;
            tensor->buft = ggml_backend_sched_get_buft(m_ctx.sched.get(), buf_id);
            ggml_backend_sched_set_tensor_backend(m_ctx.sched.get(), tensor, m_ctx.backends[buf_id].get());
        }
    }

    // Handle token_emb and output strategies
    auto it = m_ctx.model.layer_to_tensor_map.find(cpu_backend_id);
    if (it != m_ctx.model.layer_to_tensor_map.end()) {
        for (ggml_tensor * tensor : it->second) {
            const char * name = ggml_get_name(tensor);
            if (name == nullptr) {
                continue;
            }

            int32_t buf_id = cpu_backend_id;
            if (strstr(name, "output")) {
                buf_id = (plan.output_strategy < 0) ? cpu_backend_id : plan.output_strategy;
            } else if (strstr(name, "token_embd")) {
                buf_id = cpu_backend_id;
            }

            tensor->buffer_id = buf_id;
            tensor->buft = ggml_backend_sched_get_buft(m_ctx.sched.get(), buf_id);
            ggml_backend_sched_set_tensor_backend(m_ctx.sched.get(), tensor, m_ctx.backends[buf_id].get());
        }
    }
}

int32_t PipelineExecutor::plan_find_max_pinnable(
        ExecutionPlan      & plan,
        layer_strategy       base,
        layer_strategy       pin,
        uint32_t             batch_size,
        int                  n_shard_slots,
        int32_t              n_outputs_override) {

    const int32_t n_layer = m_ctx.model.hparams.n_layer;
    const size_t available_vram = m_plan.max_vram_alloc;

    // Special case: when base == pin, all layers are always the same strategy
    // No progressive search needed - just check if full config fits
    if (base == pin) {
        plan.strategies.assign(n_layer, pin);
        
        size_t scratch_req = plan_compute_requirements(plan, batch_size, true, false, n_outputs_override);
        
        size_t pinned_size = 0;
        for (const auto & split : plan.pipeline_splits) {
            if (split.backend_id == 0) {
                pinned_size += split.size;
            }
        }
        
        size_t kv_size = std::min((size_t) n_layer * m_plan.kv_size_per_layer, m_plan.total_kv_size);
        size_t total_req = scratch_req + pinned_size + kv_size;
        
        // Either all fit or none fit
        return (total_req <= available_vram) ? n_layer : 0;
    }
        
    // Binary search for max pinnable layers - O(log n) instead of O(n)
    // VRAM requirement is monotonically increasing with number of pinned layers
    int32_t lo = 0, hi = n_layer;
    int32_t result = 0;

    while (lo <= hi) {
        int32_t mid = (lo + hi) / 2;

        // Configure plan: first mid layers pinned, rest use base
        plan.strategies.assign(n_layer, base);
        for (int32_t i = 0; i < mid; i++) {
            plan.strategies[i] = pin;
        }

        // Measure VRAM requirement
        size_t scratch_req = plan_compute_requirements(plan, batch_size, true, false, n_outputs_override);

        // Calculate pinned and sharded weights
        size_t pinned_size = 0;
        size_t max_sharded_size = 0;
        for (const auto & split : plan.pipeline_splits) {
            if (split.backend_id == 0) {
                pinned_size += split.size;
            } else if (split.backend_id != GGML_CUDA_MAX_DEVICES) {
                max_sharded_size = std::max(max_sharded_size, split.size);
            }
        }
        
        // Sharded weights VRAM = n_shard_slots × max_sharded_split_size
        size_t sharded_size = n_shard_slots * max_sharded_size;

        // KV = pinned layers + shard slots
        size_t kv_size = std::min(
            (size_t)(mid + n_shard_slots) * m_plan.kv_size_per_layer,
            m_plan.total_kv_size
        );
        
        size_t total_req = scratch_req + pinned_size + sharded_size + kv_size;

        if (total_req <= available_vram) {
            result = mid;  // mid fits, try more
            lo = mid + 1;
        } else {
            hi = mid - 1;  // mid doesn't fit, try fewer
        }
    }

    // Restore final valid configuration
    plan.strategies.assign(n_layer, base);
    for (int32_t i = 0; i < result; i++) {
        plan.strategies[i] = pin;
    }

    return result;
}

//
// plan_measure_total - Helper to measure total VRAM requirement for a plan
//
size_t PipelineExecutor::plan_measure_total(
        ExecutionPlan & plan,
        uint32_t        batch_size,
        int             n_shard_slots,
        int32_t         n_outputs_override) {

    size_t scratch_req = plan_compute_requirements(plan, batch_size, true, false, n_outputs_override);

    size_t pinned_size = 0;
    size_t max_sharded_size = 0;
    int32_t n_pinned_layers = 0;

    for (const auto & split : plan.pipeline_splits) {
        if (split.backend_id == 0) {
            pinned_size += split.size;
        } else if (split.backend_id != GGML_CUDA_MAX_DEVICES) {
            max_sharded_size = std::max(max_sharded_size, split.size);
        }
    }

    // Count pinned layers from strategies
    for (int32_t s : plan.strategies) {
        if (s == LAYER_STRATEGY_ALL_GPU_PINNED ||
            s == LAYER_STRATEGY_ATTN_GPU_PINNED ||
            s == LAYER_STRATEGY_ATTN_PINNED_FFN_SHARDED) {
            n_pinned_layers++;
        }
    }

    size_t sharded_size = n_shard_slots * max_sharded_size;
    size_t kv_size = std::min(
        (size_t)(n_pinned_layers + n_shard_slots) * m_plan.kv_size_per_layer,
        m_plan.total_kv_size
    );

    return scratch_req + pinned_size + sharded_size + kv_size;
}

//
// plan_shard_remaining_with_recalc - Handle case when not all attention fits
// Recalculates available VRAM after accounting for shard slot overhead
//
int32_t PipelineExecutor::plan_shard_remaining_with_recalc(
        ExecutionPlan  & plan,
        layer_strategy   pin_strategy,
        layer_strategy   shard_strategy,
        uint32_t         batch_size,
        int32_t          n_outputs_override) {

    const int32_t n_layer = m_ctx.model.hparams.n_layer;

    // Set all layers to shard to measure max shard size
    plan.strategies.assign(n_layer, shard_strategy);
    plan_compute_requirements(plan, batch_size, true, false, n_outputs_override);

    size_t max_shard_size = 0;
    for (const auto & split : plan.pipeline_splits) {
        if (split.backend_id != GGML_CUDA_MAX_DEVICES && split.backend_id != 0) {
            max_shard_size = std::max(max_shard_size, split.size);
        }
    }
    plan.sharded_weights_vram_req = max_shard_size;

    // Recalculate available VRAM minus shard slot and 1 KV layer
    size_t adjusted_vram = m_plan.max_vram_alloc - plan.sharded_weights_vram_req - m_plan.kv_size_per_layer;

    // Binary search for how many attention layers can be pinned within adjusted budget
    // VRAM requirement is monotonically increasing with n, so we find the largest n where total_req <= adjusted_vram
    auto test_pinned_layers = [&, n_outputs_override](int32_t n) -> bool {
        for (int32_t i = 0; i < n; i++) {
            plan.strategies[i] = pin_strategy;
        }
        for (int32_t i = n; i < n_layer; i++) {
            plan.strategies[i] = shard_strategy;
        }

        size_t scratch_req = plan_compute_requirements(plan, batch_size, true, false, n_outputs_override);

        size_t pinned_size = 0;
        for (const auto & split : plan.pipeline_splits) {
            pinned_size += (split.backend_id == 0) ? split.size : 0;
        }
        size_t total_req = scratch_req + pinned_size + n * m_plan.kv_size_per_layer;

        return total_req <= adjusted_vram;
    };

    int32_t n_pinned = 0;
    int32_t lo = 0, hi = n_layer;
    while (lo <= hi) {
        int32_t mid = lo + (hi - lo) / 2;
        if (mid == 0 || test_pinned_layers(mid)) {
            n_pinned = mid;
            lo = mid + 1;
        } else {
            hi = mid - 1;
        }
    }

    // Restore best configuration
    for (int32_t i = 0; i < n_pinned; i++) {
        plan.strategies[i] = pin_strategy;
    }
    for (int32_t i = n_pinned; i < n_layer; i++) {
        plan.strategies[i] = shard_strategy;
    }

    // Set KV for pinned + 1 shard slot
    plan.kv_vram_req = std::min((size_t)(n_pinned + 1) * m_plan.kv_size_per_layer, m_plan.total_kv_size);

    return n_pinned;
}

//
// plan_apply_phase2_progressive - Progressive attn→FFN logic for ATTN_PRIO strategies
// Called after Phase 1 finds max pinnable layers
//
void PipelineExecutor::plan_apply_phase2_progressive(
        ExecutionPlan & plan,
        uint32_t        batch_size,
        exec_strategy   strategy,
        int32_t       & n_pinned,
        int32_t         n_outputs_override) {

    const int32_t n_layer = m_ctx.model.hparams.n_layer;
    const strategy_config & cfg = STRATEGY_CONFIGS[strategy];

    // Only applies to ATTN_PRIO strategies
    bool is_static_attn_prio = (strategy == EXEC_STRATEGY_STATIC_ATTN_PRIO);
    bool is_shard_attn_prio  = (strategy == EXEC_STRATEGY_SHARD_ATTN_PRIO);

    if (!is_static_attn_prio && !is_shard_attn_prio) {
        return;  // Not an ATTN_PRIO strategy, skip Phase 2
    }

    // Case A: Not all attention fits - recalculate with shard overhead
    if (is_shard_attn_prio && n_pinned < n_layer) {
        n_pinned = plan_shard_remaining_with_recalc(plan, cfg.pin, cfg.shard, batch_size, n_outputs_override);
        return;
    }

    // Case B: All attention fits - try to also fit FFN
    if (n_pinned == n_layer) {
        // Step 1: For MoE, try output first
        if (m_plan.is_moe) {
            plan.token_emb_strategy = 0;
            plan.output_strategy    = 0;
            size_t req = plan_measure_total(plan, batch_size, cfg.n_shard_slots, n_outputs_override);
            if (req > m_plan.max_vram_alloc) {
                plan.token_emb_strategy = GGML_CUDA_MAX_DEVICES;
                plan.output_strategy    = GGML_CUDA_MAX_DEVICES;
            }
        }

        // Step 2: Determine FFN strategy based on exec_strategy
        layer_strategy remaining_strat = is_shard_attn_prio
            ? LAYER_STRATEGY_ATTN_PINNED_FFN_SHARDED   // Shard FFN for remaining
            : LAYER_STRATEGY_ATTN_GPU_PINNED;          // Keep FFN on CPU for static

        // For SHARD_ATTN_PRIO, check if FFN sharding is viable
        if (is_shard_attn_prio) {
            plan.strategies.assign(n_layer, LAYER_STRATEGY_ATTN_PINNED_FFN_SHARDED);
            size_t req = plan_measure_total(plan, batch_size, cfg.n_shard_slots, n_outputs_override);
            if (req > m_plan.max_vram_alloc) {
                plan.is_viable = false;
                return;
            }
        }

        // Step 3: Binary search to pin full layers (attn + FFN)
        // VRAM requirement is monotonically increasing with n, so we find the largest n where req <= max_vram_alloc
        auto test_full_pinned = [&, n_outputs_override](int32_t n) -> bool {
            for (int32_t i = 0; i < n; i++) {
                plan.strategies[i] = LAYER_STRATEGY_ALL_GPU_PINNED;
            }
            for (int32_t i = n; i < n_layer; i++) {
                plan.strategies[i] = remaining_strat;
            }
            size_t req = plan_measure_total(plan, batch_size, cfg.n_shard_slots, n_outputs_override);
            return req <= m_plan.max_vram_alloc;
        };

        int32_t n_full_pinned = 0;
        int32_t lo = 0, hi = n_layer;
        while (lo <= hi) {
            int32_t mid = lo + (hi - lo) / 2;
            if (mid == 0 || test_full_pinned(mid)) {
                n_full_pinned = mid;
                lo = mid + 1;
            } else {
                hi = mid - 1;
            }
        }

        // Restore best configuration
        for (int32_t i = 0; i < n_full_pinned; i++) {
            plan.strategies[i] = LAYER_STRATEGY_ALL_GPU_PINNED;
        }
        for (int32_t i = n_full_pinned; i < n_layer; i++) {
            plan.strategies[i] = remaining_strat;
        }
    }
}

//
// plan_finalize - finalize plan with all VRAM calculations
//
void PipelineExecutor::plan_finalize(
        ExecutionPlan & plan,
        uint32_t        batch_size,
        int             n_shard_slots,
        int32_t         n_pinned,
        bool            calc_tps,
        int32_t         n_outputs_override) {

    const int32_t n_layer = m_ctx.model.hparams.n_layer;

    // plan_compute_requirements already applies SCRATCH_MARGIN internally
    plan.scratch_vram_req = plan_compute_requirements(plan, batch_size, true, calc_tps, n_outputs_override);

    plan.pinned_weights_vram_req  = 0;
    plan.sharded_weights_vram_req = 0;
    plan.sorted_pinned_weights.clear();
    size_t max_sharded_split_size = 0;

    for (auto & split : plan.pipeline_splits) {
        if (split.backend_id == 0) {
            split.backend_type = GPU_PINNED;
            plan.pinned_weights_vram_req += split.size;
            for (ggml_tensor * t : split.leafs) {
                plan.sorted_pinned_weights.push_back(t);
            }
        } else if (split.backend_id != GGML_CUDA_MAX_DEVICES) {
            split.backend_type = GPU_SHARDED;
            max_sharded_split_size = std::max(max_sharded_split_size, split.size);
        } else {
            split.backend_type = CPU;
        }
    }

    sort_weights_canonical(plan.sorted_pinned_weights);

    plan.sharded_weights_vram_req = n_shard_slots * max_sharded_split_size;

    // Count actual pinned layers from strategies (not from n_pinned parameter)
    int32_t n_pinned_layers = 0;
    for (int32_t s : plan.strategies) {
        if (s == LAYER_STRATEGY_ALL_GPU_PINNED || 
            s == LAYER_STRATEGY_ATTN_GPU_PINNED ||
            s == LAYER_STRATEGY_ATTN_PINNED_FFN_SHARDED) {
            n_pinned_layers++;
        }
    }

    bool has_sharding = (plan.sharded_weights_vram_req > 0);
    size_t n_gpu_kv_layers = n_pinned_layers;
    if (has_sharding) {
        n_gpu_kv_layers += n_shard_slots;
    }
    
    plan.kv_vram_req = std::min(n_gpu_kv_layers * m_plan.kv_size_per_layer, m_plan.total_kv_size);
    plan.total_vram_req = plan.pinned_weights_vram_req + plan.sharded_weights_vram_req + plan.scratch_vram_req + plan.kv_vram_req;

    if (plan.total_vram_req <= m_plan.max_vram_alloc) {
        plan.scratch_vram_req = m_plan.max_vram_alloc - plan.pinned_weights_vram_req - plan.sharded_weights_vram_req - plan.kv_vram_req;
    }

    //LLAMA_LOG_INFO("  -> finalized: pinned=%.1fMB sharded=%.1fMB scratch=%.1fMB kv=%.1fMB total=%.1fMB splits=%zu\n",
    //    plan.pinned_weights_vram_req / (1024.0 * 1024.0),
    //    plan.sharded_weights_vram_req / (1024.0 * 1024.0),
    //    plan.scratch_vram_req / (1024.0 * 1024.0),
    //    plan.kv_vram_req / (1024.0 * 1024.0),
    //    plan.total_vram_req / (1024.0 * 1024.0),
    //    plan.pipeline_splits.size());

    //// Save per-plan allocator state (for restoring when switching back to this plan)
    //// This captures the exact allocation layout computed during planning
    {
        // Use full available VRAM between weights and KV for scratch allocation.
        // KV is placed at the END of the VRAM buffer, so the actual available scratch
        // is the entire gap between weights_end and kv_start, not just plan.scratch_vram_req.
        // This reduces unnecessary tensor aliasing within scratch.
        size_t weights_size = plan.pinned_weights_vram_req + plan.sharded_weights_vram_req;
        ggml_backend_alloc_set_scratch_offsets(m_ctx.sched.get(), plan.scratch_vram_req, weights_size);
        plan_compute_requirements(plan, batch_size, false, false, n_outputs_override);
        
        auto * sched = m_ctx.sched.get();
        
        // Get sizes and save allocator state
        size_t node_size = 0, leaf_size = 0;
        ggml_gallocr_get_state_sizes(ggml_backend_sched_get_galloc(sched), &node_size, &leaf_size);
        
        plan.alloc_state.node_allocs.resize(node_size);
        plan.alloc_state.leaf_allocs.resize(leaf_size);
        
        ggml_gallocr_save_state(ggml_backend_sched_get_galloc(sched),
                                 plan.alloc_state.node_allocs.data(),
                                 plan.alloc_state.leaf_allocs.data(),
                                 &plan.alloc_state.n_nodes,
                                 &plan.alloc_state.n_leafs);
        
        // Save backend_ids (needed to make backend_ids_changed = false)
        int sched_n_nodes = 0, sched_n_leafs = 0;
        ggml_backend_sched_get_backend_ids_sizes(sched, &sched_n_nodes, &sched_n_leafs);
        
        plan.alloc_state.node_backend_ids.resize(sched_n_nodes);
        plan.alloc_state.leaf_backend_ids.resize(sched_n_leafs);
        
        ggml_backend_sched_save_backend_ids(sched,
                                             plan.alloc_state.node_backend_ids.data(),
                                             plan.alloc_state.leaf_backend_ids.data(),
                                             nullptr, nullptr);
        
        plan.alloc_state.valid = (plan.alloc_state.n_nodes > 0);
    }
}

//
// plan_build_for_batch - Build execution plan for a given batch size and strategy
// Always uses init_full() for worst-case KV sizing (n_kv_max).
//
void PipelineExecutor::plan_build_for_batch(
        ExecutionPlan      & plan,
        uint32_t             batch_size,
        exec_strategy        strategy,
        bool                 calc_tps,
        int32_t              n_outputs_override) {

    const auto n_outputs_save       = m_ctx.n_outputs;
    size_t     saved_scratch_offset = ggml_backend_get_scratch_offset(m_ctx.sched.get());
    size_t     saved_scratch_size   = ggml_backend_get_scratch_size(m_ctx.sched.get());
    ggml_backend_alloc_set_scratch_offsets(m_ctx.sched.get(), SIZE_MAX, 0);

    const strategy_config & cfg = STRATEGY_CONFIGS[strategy];
    const int32_t n_layer = m_ctx.model.hparams.n_layer;
    const int n_shard_slots = cfg.n_shard_slots;

    // Set token_emb and output strategies
    plan.token_emb_strategy = (cfg.token_emb < 0) ? GGML_CUDA_MAX_DEVICES : cfg.token_emb;
    plan.output_strategy    = (cfg.output < 0)    ? GGML_CUDA_MAX_DEVICES : cfg.output;

    // Phase 1: Find max pinnable layers
    int32_t n_pinned = plan_find_max_pinnable(plan, cfg.base, cfg.pin, batch_size, n_shard_slots, n_outputs_override);

    // Phase 2: Progressive attn→FFN logic (only for ATTN_PRIO strategies)
    bool is_attn_prio = (strategy == EXEC_STRATEGY_STATIC_ATTN_PRIO || 
                         strategy == EXEC_STRATEGY_SHARD_ATTN_PRIO);

    plan_apply_phase2_progressive(plan, batch_size, strategy, n_pinned, n_outputs_override);

    // Generic sharding fallback (for non-ATTN_PRIO strategies when not all layers pinned)
    // ATTN_PRIO strategies handle their own sharding in plan_apply_phase2_progressive
    if (!is_attn_prio && cfg.can_shard && n_pinned < n_layer) {
        // Set remaining layers to shard strategy
        for (int32_t i = n_pinned; i < n_layer; i++) {
            plan.strategies[i] = cfg.shard;
        }
    }

    // Final measurement and assignment
    plan_finalize(plan, batch_size, n_shard_slots, n_pinned, calc_tps, n_outputs_override);
    m_ctx.n_outputs = n_outputs_save;
    ggml_backend_alloc_reset_scratch_offsets(m_ctx.sched.get(), saved_scratch_offset);
}

// plan_measure_scratch_only - lightweight scratch measurement using existing plan's strategies
// Returns the scratch requirement for the given batch_size without building a full plan
//
size_t PipelineExecutor::plan_measure_scratch_only(const ExecutionPlan & plan, uint32_t batch_size, int32_t n_outputs_override) {
    // Save n_outputs to avoid corrupting decode state during measurement
    const auto n_outputs_save = m_ctx.n_outputs;
    size_t saved_scratch_offset = ggml_backend_get_scratch_offset(m_ctx.sched.get());

    // Save and NULL weight buffer pointers (mimics planning phase where buffers are unallocated)
    std::vector<std::pair<ggml_tensor*, ggml_backend_buffer_t>> saved_buffers;
    for (const auto & pair : m_ctx.model.layer_to_tensor_map) {
        for (ggml_tensor * tensor : pair.second) {
            if (tensor->buffer != nullptr) {
                saved_buffers.push_back({tensor, tensor->buffer});
                tensor->buffer = nullptr;
            }
        }
    }

    ggml_backend_alloc_set_scratch_offsets(m_ctx.sched.get(), SIZE_MAX, 0);

    // Create a mutable copy with just the strategies we need
    ExecutionPlan temp_plan;
    temp_plan.strategies = plan.strategies;
    temp_plan.token_emb_strategy = plan.token_emb_strategy;
    temp_plan.output_strategy = plan.output_strategy;
    
    //LLAMA_LOG_INFO("%s: [RUNTIME MEASURE] batch_size=%u, strategies[0]=%d, output_strategy=%d\n",
    //               __func__, batch_size,
    //               temp_plan.strategies.empty() ? -1 : temp_plan.strategies[0],
    //               temp_plan.output_strategy);

    // NEW: Build measurement graph with decode-mode output count
    // Still uses init_full() for KV (can't use init_batch without real batch data)
    // but fixes n_outputs to match actual decode behavior
    uint32_t n_seqs   = m_ctx.cparams.kv_unified ? 1 : m_ctx.cparams.n_seq_max;
    uint32_t n_tokens = batch_size;

    if (n_tokens % n_seqs != 0) {
        n_tokens = ((n_tokens + (n_seqs - 1)) / n_seqs) * n_seqs;
    }

    // Use override n_outputs if provided (runtime path), otherwise use n_tokens (planning default)
    if (n_outputs_override >= 0) {
        m_ctx.n_outputs = n_outputs_override;
    } else {
        m_ctx.n_outputs = n_tokens;
    }

    ggml_backend_sched_reset(m_ctx.sched.get());

    auto balloc = std::make_unique<llama_batch_allocr>(m_ctx.model.hparams.n_pos_per_embd());
    llama_ubatch ubatch = balloc->ubatch_reserve(n_tokens / n_seqs, n_seqs);

    llm_graph_result res(m_ctx.graph_max_nodes());
    
    // Use init_full for KV - can't use init_batch without actual batch data
    auto mctx = m_ctx.get_memory()->init_full();
    if (!mctx) {
        LLAMA_LOG_ERROR("%s: failed to initialize memory context\n", __func__);
        // Restore buffer pointers before returning
        for (auto & [tensor, buffer] : saved_buffers) {
            tensor->buffer = buffer;
        }
        return SIZE_MAX;
    }
    
    const auto gparams = m_ctx.graph_params(&res, ubatch, mctx.get(), LLM_GRAPH_TYPE_DEFAULT);
    ggml_cgraph * gf = m_ctx.model.build_graph(gparams);

    // Mark weights as leaves for scheduler measurement
    for (const auto & pair : m_ctx.model.layer_to_tensor_map) {
        for (ggml_tensor * tensor : pair.second) {
            tensor->flags |= GGML_TENSOR_FLAG_LEAF;
        }
    }

    // Apply layer strategies to weight tensors
    plan_apply_layer_strategies(temp_plan);

    // Apply node backend overrides
    plan_apply_node_overrides(gf, temp_plan);

    // Measure scratch
    size_t measured_scratch = plan_measure_and_extract(temp_plan, gf, false);
    measured_scratch = GGML_PAD(measured_scratch, 128);

    // Cleanup: restore tensor flags
    for (const auto & pair : m_ctx.model.layer_to_tensor_map) {
        for (ggml_tensor * tensor : pair.second) {
            tensor->flags &= ~GGML_TENSOR_FLAG_LEAF;
        }
    }

    // Restore buffer pointers and n_outputs
    for (auto & [tensor, buffer] : saved_buffers) {
        tensor->buffer = buffer;
    }
    m_ctx.n_outputs = n_outputs_save;
    ggml_backend_alloc_reset_scratch_offsets(m_ctx.sched.get(), saved_scratch_offset);
    return measured_scratch;
}

//
// plan_compute_requirements - builds graph and measures VRAM requirements
//
// Always uses init_full() for planning to ensure worst-case KV sizing (n_kv_max).
//
size_t PipelineExecutor::plan_compute_requirements(
        ExecutionPlan & plan,
        uint32_t        batch_size,
        bool            save_splits,
        bool            calc_tps,
        int32_t         n_outputs_override) {

    // Helper to mark weight tensors as leaves for scheduler measurement
    auto fake_alloc_weights = [&]() {
        for (const auto & pair : m_ctx.model.layer_to_tensor_map) {
            const std::vector<ggml_tensor *> & tensors = pair.second;
            for (ggml_tensor * tensor : tensors) {
                tensor->flags |= GGML_TENSOR_FLAG_LEAF;
            }
        }
    };

    // Helper to restore weight tensor flags after measurement
    auto reset_fake_alloc = [&]() {
        for (const auto & pair : m_ctx.model.layer_to_tensor_map) {
            const std::vector<ggml_tensor *> & tensors = pair.second;
            for (ggml_tensor * tensor : tensors) {
                tensor->flags &= ~GGML_TENSOR_FLAG_LEAF;
            }
        }
    };

    // Step 1: Apply layer strategies to weight tensors
    plan_apply_layer_strategies(plan);

    // Step 2: Build measurement graph
    // Match baseline: use n_seq_max for non-unified KV, but pipe-shard needs worst-case
    // n_seqs=1 because a single long sequence can require more scratch than many short ones.
    uint32_t n_seqs   = m_ctx.cparams.kv_unified ? 1 : m_ctx.cparams.n_seq_max;
    uint32_t n_tokens = batch_size;

    if (n_tokens % n_seqs != 0) {
        n_tokens = ((n_tokens + (n_seqs - 1)) / n_seqs) * n_seqs;
    }

    // Save n_outputs to avoid corrupting decode state during plan measurement
    const auto n_outputs_save = m_ctx.n_outputs;
    
    // Use override n_outputs if provided, otherwise use n_tokens (planning default)
    if (n_outputs_override >= 0) {
        m_ctx.n_outputs = n_outputs_override;
    } else {
        m_ctx.n_outputs = n_tokens;
    }

    ggml_backend_sched_reset(m_ctx.sched.get());

    llm_graph_result res(m_ctx.graph_max_nodes());
    
    // Planning path: create synthetic ubatch with init_full() for worst-case KV sizing
    auto balloc = std::make_unique<llama_batch_allocr>(m_ctx.model.hparams.n_pos_per_embd());
    llama_ubatch ubatch = balloc->ubatch_reserve(n_tokens / n_seqs, n_seqs);
    auto mctx = m_ctx.get_memory()->init_full();
    
    const auto gparams = m_ctx.graph_params(&res, ubatch, mctx.get(), LLM_GRAPH_TYPE_DEFAULT);

    ggml_cgraph * gf = m_ctx.model.build_graph(gparams);

    // Mark weights as leaves for scheduler measurement
    fake_alloc_weights();

    // Step 3: Apply node backend overrides
    plan_apply_node_overrides(gf, plan);

    // Step 4: Measure and extract splits
    size_t measured_scratch = plan_measure_and_extract(plan, gf, save_splits);
    
    // Apply scratch margin to handle runtime allocation pattern differences.
    // When runtime n_stream differs from planning n_stream (e.g., 3 users vs planned 16),
    // self_kq_mask can exceed its cached size_max due to GGML_PAD non-monotonicity.
    // This triggers needs_realloc -> reserve_n, which recomputes all offsets within scratch.
    // The margin ensures reserve_n's best-fit allocation has enough room despite different
    // fragmentation patterns between the planning and runtime graphs.
    measured_scratch = GGML_PAD((size_t)((float)measured_scratch * SCRATCH_MARGIN), 128);
    
    // Step 5: Predict TPS while scheduler still has split info
    if (calc_tps && !plan.pipeline_splits.empty()) {
        //char label[64];
        //snprintf(label, sizeof(label), "batch=%u splits=%zu", batch_size, plan.pipeline_splits.size());
        plan.tps = (float) predict_tps(plan, batch_size, nullptr);
    }

    // Cleanup: restore tensor flags and n_outputs
    reset_fake_alloc();
    m_ctx.n_outputs = n_outputs_save;

    return measured_scratch;
}

// Get attention buffer ID for a layer strategy
static int32_t get_attn_buffer_id(layer_strategy strat, int32_t layer_idx, int32_t cpu_id) {
    switch (strat) {
        case LAYER_STRATEGY_ALL_GPU_SHARDED:         return layer_idx + 1;
        case LAYER_STRATEGY_ATTN_GPU_PINNED:         return 0;
        case LAYER_STRATEGY_FFN_GPU_PINNED:          return cpu_id;
        case LAYER_STRATEGY_ALL_CPU:                 return cpu_id;
        case LAYER_STRATEGY_ALL_GPU_PINNED:          return 0;
        case LAYER_STRATEGY_ATTN_GPU_SHARDED:        return 1;
        case LAYER_STRATEGY_FFN_GPU_SHARDED:         return cpu_id;
        case LAYER_STRATEGY_ATTN_PINNED_FFN_SHARDED: return 0;
        default:                                     return cpu_id;
    }
}

// Get FFN buffer ID for a layer strategy
static int32_t get_ffn_buffer_id(layer_strategy strat, int32_t layer_idx, int32_t cpu_id) {
    switch (strat) {
        case LAYER_STRATEGY_ALL_GPU_SHARDED:         return layer_idx + 1;
        case LAYER_STRATEGY_ATTN_GPU_PINNED:         return cpu_id;
        case LAYER_STRATEGY_FFN_GPU_PINNED:          return 0;
        case LAYER_STRATEGY_ALL_CPU:                 return cpu_id;
        case LAYER_STRATEGY_ALL_GPU_PINNED:          return 0;
        case LAYER_STRATEGY_ATTN_GPU_SHARDED:        return cpu_id;
        case LAYER_STRATEGY_FFN_GPU_SHARDED:         return 1;
        case LAYER_STRATEGY_ATTN_PINNED_FFN_SHARDED: return layer_idx + 1;
        default:                                     return cpu_id;
    }
}

//
// Apply backend overrides to all graph nodes and leaves
// Handles: norm nodes, result nodes, KV cache, FFN ops, ROPE ops, inp_embd
//
void PipelineExecutor::plan_apply_node_overrides(
        ggml_cgraph         * gf,
        const ExecutionPlan & plan) {

    const std::vector<int> & strategies = plan.strategies;
    const int32_t cpu_id    = GGML_CUDA_MAX_DEVICES;
    const int32_t output_id = (plan.output_strategy < 0) ? cpu_id : plan.output_strategy;
    const int32_t emb_id    = (plan.token_emb_strategy < 0) ? cpu_id : plan.token_emb_strategy;

    // Helper to set tensor backend
    auto set_backend = [&](ggml_tensor * t, int32_t buf_id) {
        t->buffer_id = buf_id;
        t->buft = ggml_backend_sched_get_buft(m_ctx.sched.get(), buf_id);
        ggml_backend_sched_set_tensor_backend(m_ctx.sched.get(), t, m_ctx.backends[buf_id].get());
    };

    // Helper to extract layer index from name (supports "-X" and "_lX" patterns)
    auto get_layer_idx = [](const char * name) -> int32_t {
        if (name == nullptr) {
            return -1;
        }
        // Try "-X" pattern first (e.g., "blk.attn_norm-5")
        const char * pos = strrchr(name, '-');
        if (pos != nullptr) {
            return atoi(pos + 1);
        }
        // Try "_lX" pattern (e.g., "cache_k_l5")
        pos = strstr(name, "_l");
        if (pos != nullptr) {
            return atoi(pos + 2);
        }
        return -1;
    };

    // Helper to check if tensor has output parent
    auto has_output_parent = [](ggml_tensor * node) -> bool {
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            ggml_tensor * parent = node->src[j];
            if (parent != nullptr) {
                const char * pname = ggml_get_name(parent);
                if (pname != nullptr && strstr(pname, "output") != nullptr) {
                    return true;
                }
            }
        }
        return false;
    };

    //
    // Process graph nodes
    //
    for (int i = 0; i < gf->n_nodes; ++i) {
        ggml_tensor * node = gf->nodes[i];
        const char  * name = ggml_get_name(node);

        if (name == nullptr) {
            continue;
        }

        int32_t layer_idx = get_layer_idx(name);

        if (strstr(name, "norm") != nullptr && strstr(name, "ffn_moe_weights_norm") == nullptr) {
            // Final output norm → output backend
            if (strstr(name, "result") != nullptr) {
                set_backend(node, output_id);
                continue;
            }

            // Norm connected to output → output backend
            if (has_output_parent(node)) {
                set_backend(node, output_id);
                continue;
            }

            // Layer norm → attention backend for that layer
            if (layer_idx >= 0 && layer_idx < (int32_t)strategies.size()) {
                layer_strategy strat = (layer_strategy) strategies[layer_idx];
                set_backend(node, get_attn_buffer_id(strat, layer_idx, cpu_id));
            }
            continue;
        }

        // 2. Result output node
        if (strstr(name, "result_output") != nullptr) {
            // Also handle token_embd.weight parent
            for (int j = 0; j < GGML_MAX_SRC; j++) {
                ggml_tensor * parent = node->src[j];
                if (parent != nullptr) {
                    const char * pname = ggml_get_name(parent);
                    if (pname != nullptr && strstr(pname, "token_embd.weight") != nullptr) {
                        set_backend(parent, emb_id);
                        break;
                    }
                }
            }
            continue;
        }

        // 3. inp_embd → always CPU
        if (strstr(name, "inp_embd") != nullptr) {
            set_backend(node, cpu_id);
            continue;
        }

        // 4. FFN ops (MUL_MAT, MUL_MAT_ID, GLU)
        if (node->op == GGML_OP_MUL_MAT || node->op == GGML_OP_MUL_MAT_ID || node->op == GGML_OP_GLU) {
            // Skip MoE router ops - they should follow attention placement
            if (strstr(name, "ffn_gate_inp") != nullptr) {
                continue;
            }
            
            // Only process FFN-related ops
            if (strstr(name, "ffn_up") != nullptr || strstr(name, "ffn_down") != nullptr ||
                strstr(name, "ffn_gate") != nullptr || strstr(name, "ffn_out") != nullptr ||
                strstr(name, "ffn_moe_up") != nullptr || strstr(name, "ffn_moe_down") != nullptr ||
                strstr(name, "ffn_moe_gate") != nullptr || strstr(name, "ffn_moe_weighted") != nullptr) {

                if (layer_idx >= 0 && layer_idx < (int32_t)strategies.size()) {
                    layer_strategy strat = (layer_strategy) strategies[layer_idx];
                    set_backend(node, get_ffn_buffer_id(strat, layer_idx, cpu_id));
                }
            }
            continue;
        }
        
        // 5. ROPE ops → attention backend, also handle rope_freqs parent
        if (node->op == GGML_OP_ROPE || node->op == GGML_OP_ROPE_BACK) {
            if (layer_idx >= 0 && layer_idx < (int32_t)strategies.size()) {
                layer_strategy strat = (layer_strategy) strategies[layer_idx];
                int32_t buf_id = get_attn_buffer_id(strat, layer_idx, cpu_id);

                for (int j = 0; j < GGML_MAX_SRC; j++) {
                    ggml_tensor * parent = node->src[j];
                    if (parent != nullptr) {
                        const char * pname = ggml_get_name(parent);
                        if (pname != nullptr && strstr(pname, "rope_freqs.weight") != nullptr) {
                            set_backend(parent, buf_id);
                            break;
                        }
                    }
                }
            }
            continue;
        }
    }

    //
    // Process graph leaves (KV cache tensors)
    //
    for (int i = 0; i < gf->n_leafs; ++i) {
        ggml_tensor * leaf = gf->leafs[i];

        if (!(leaf->flags & GGML_TENSOR_FLAG_KV)) {
            continue;
        }

        int32_t layer_idx = get_layer_idx(ggml_get_name(leaf));
        if (layer_idx >= 0 && layer_idx < (int32_t)strategies.size()) {
            layer_strategy strat = (layer_strategy) strategies[layer_idx];
            set_backend(leaf, get_attn_buffer_id(strat, layer_idx, cpu_id));
        }
    }
}

//
// plan_measure_and_extract
//
size_t PipelineExecutor::plan_measure_and_extract(
        ExecutionPlan      & plan,
        ggml_cgraph        * gf,
        bool                 save_splits) {

    // Reserve scheduler with measurement graph
    ggml_backend_sched_reserve(m_ctx.sched.get(), gf);

    size_t scratch_offset = ggml_backend_get_scratch_offset(m_ctx.sched.get());

    // Calculate max VRAM across splits
    size_t scratch_vram_req = 0;
    for (size_t i = 0; i < m_ctx.backends.size() - 1; ++i) {
        size_t vram_req = ggml_backend_get_per_split_vram_req(m_ctx.sched.get(), i);
        size_t scratch_size = (vram_req > scratch_offset) ? (vram_req - scratch_offset) : vram_req;
        scratch_vram_req = std::max(scratch_vram_req, scratch_size);
    }

    size_t cuda_host_size = 4 * ggml_backend_get_per_split_vram_req(m_ctx.sched.get(), m_ctx.backends.size() - 1);
    m_plan.cuda_host_size = std::max(m_plan.cuda_host_size, cuda_host_size);

    // Extract pipeline splits from scheduler
    if (save_splits) {
        const int num_splits = ggml_backend_sched_get_num_splits(m_ctx.sched.get());
        plan.pipeline_splits.clear();
        plan.pipeline_splits.reserve(num_splits);

        for (int split_id = 0; split_id < num_splits; ++split_id) {
            PipelineSplit split;

            // Get backend info for this split
            ggml_backend_t     split_backend = ggml_backend_sched_get_split_backend(m_ctx.sched.get(), split_id);
            split.backend_id                 = ggml_backend_sched_backend_id(m_ctx.sched.get(), split_backend);
            ggml_backend_dev_t dev           = ggml_backend_get_device(split_backend);

            // Determine backend type from device
            if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
                split.backend_type = CPU;
            } else {
                split.backend_type = GPU;  // Will be refined to GPU_PINNED or GPU_SHARDED later
            }

            // Get leaf indices for this split
            std::vector<int> leaf_ids(gf->n_leafs, -1);
            ggml_backend_graph_leafs_in_split(m_ctx.sched.get(), gf, split_id, leaf_ids.data());

            // Process each leaf in the split
            size_t count = 0;
            while (count < leaf_ids.size() && leaf_ids[count] != -1) {
                int leaf_idx = leaf_ids[count];
                ggml_tensor * leaf = gf->leafs[leaf_idx];

                if (leaf->buft == nullptr) {
                    count++;
                    continue;
                }

                // Track KV layer IDs
                if (leaf->flags & GGML_TENSOR_FLAG_KV) {
                    const char * name = ggml_get_name(leaf);
                    if (name != nullptr) {
                        // Extract layer index from name like "cache_k_l0"
                        const char * pos = strstr(name, "_l");
                        if (pos != nullptr) {
                            int32_t layer_idx = atoi(pos + 2);
                            split.kv_layer_ids.insert(layer_idx);
                        }
                    }
                }

                // Track weight tensors
                if (leaf->flags & GGML_TENSOR_FLAG_LEAF) {
                    split.leafs.push_back(leaf);
                    plan.leaf_to_buf_id_map[leaf] = leaf->buffer_id;
                }

                count++;
            }

            // Calculate split sizes
            size_t split_size  = 0;
            size_t pinned_size = 0;
            for (ggml_tensor * tensor : split.leafs) {
                const size_t tensor_size = ggml_backend_buft_get_alloc_size(tensor->buft, tensor);
                size_t padded = GGML_PAD(tensor_size, 128);
                split_size += padded;

                // Check if tensor is marked for pinning
                auto it = m_ctx.model.tensor_pin_decisions.find(tensor);
                if (it != m_ctx.model.tensor_pin_decisions.end() && it->second) {
                    pinned_size += padded;
                }
            }

            split.size        = split_size;
            split.pinned_size = pinned_size;
            plan.pipeline_splits.push_back(split);
        }
    }

    ggml_backend_sched_reset(m_ctx.sched.get());
    return scratch_vram_req;
}
