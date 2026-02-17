#include "llama-pipeline-executor.h"

#include "llama-context.h"
#include "llama-model.h"
#include "llama-model-loader.h"
#include "llama-graph.h"
#include "llama-benchmark.h"
#include "ggml-cuda.h"  // DEBUG STREAM CHECK
#include "ggml-alloc.h" // For per-plan allocator state restore
#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <regex>
#include <unordered_set>
#include <chrono>

// ===================================================================
// Public Interface
// ===================================================================

PipelineExecutor::PipelineExecutor(llama_context & ctx) : m_ctx(ctx) {
    m_plan.last_copied_kv_pos.resize(m_ctx.model.hparams.n_layer, 0);
}

PipelineExecutor::~PipelineExecutor() {}

// Builds the execution plan for both the prompt and generation phases.
// This function calculates VRAM requirements, determines the number of pinned and sharded splits,
// and allocates the necessary VRAM for the pipeline.
void PipelineExecutor::build_plan() {

    const auto n_outputs_save = m_ctx.n_outputs;

    const char* cpu_profile = (m_ctx.cparams.cpu_profile_path && m_ctx.cparams.cpu_profile_path[0]) ? m_ctx.cparams.cpu_profile_path : "concurrent_results.txt";
    const char* gpu_profile = (m_ctx.cparams.gpu_profile_path && m_ctx.cparams.gpu_profile_path[0]) ? m_ctx.cparams.gpu_profile_path : "gpu_results.txt";

    m_plan.benchmarks = BenchmarkPredictor::load_cpu_benchmarks(cpu_profile, m_ctx.cparams.n_threads, &m_plan.global_stats);
    m_plan.gpu_benchmarks = BenchmarkPredictor::load_gpu_benchmarks(gpu_profile, &m_plan.global_stats);

    // Initialize the benchmark predictor with loaded data
    m_predictor = std::make_unique<BenchmarkPredictor>(m_plan.benchmarks, m_plan.gpu_benchmarks, m_plan.global_stats);
    m_predictor->set_use_old_prediction(true);  // Use old scoring-based method for now

    m_plan.init_plans(m_ctx.cparams.n_ubatch, m_ctx.cparams.n_seq_max, m_ctx.cparams.n_draft);

    // Populate basic plan parameters from the now-initialized context.
    auto * mem = m_ctx.get_memory();
    m_plan.total_kv_size = 0;
    m_plan.kv_size_per_layer = 0;

    if (auto * unified = dynamic_cast<llama_kv_cache_unified*>(mem)) {
        m_plan.total_kv_size = unified->size_k_bytes() + unified->size_v_bytes();
        m_plan.kv_size_per_layer = unified->max_layer_size_k_bytes() + unified->max_layer_size_v_bytes();
    } 
    else if (auto * iswa = dynamic_cast<llama_kv_cache_unified_iswa*>(mem)) {
        m_plan.total_kv_size = iswa->get_base()->size_k_bytes() + iswa->get_base()->size_v_bytes() + iswa->get_swa()->size_k_bytes() + iswa->get_swa()->size_v_bytes();
        size_t base_layer = iswa->get_base()->max_layer_size_k_bytes() + iswa->get_base()->max_layer_size_v_bytes();
        size_t swa_layer  = iswa->get_swa()->max_layer_size_k_bytes() + iswa->get_swa()->max_layer_size_v_bytes();
        m_plan.kv_size_per_layer = std::max(base_layer, swa_layer);
    }
    else {
        LLAMA_LOG_ERROR("%s: Unknown memory type, KV size planning might be incorrect\n", __func__);
    }

    // Build the complete execution plan. This function will call the scheduler internally.
    m_plan.max_vram_alloc = m_ctx.cparams.max_vram_alloc * 1024 * 1024;
    is_moe();
    populate_plan_registry();
    
    // Initialize runtime tracking
    m_use_single_plan = m_plan.single_plan_mode;
    if (!m_use_single_plan) {
        size_t prompt_tier = m_plan.tier_config.n_tiers() - 1;
        m_current_tier     = prompt_tier;
        m_current_strategy = select_best_strategy(prompt_tier);
    }

    ExecutionPlan & initial_plan = m_use_single_plan ? m_plan.unified_plan : m_plan.plans[m_current_tier][m_current_strategy];
    
    // Set initial active plan
    m_active_plan = &initial_plan;

    // Allocate VRAM buffer
    set_scratch_offsets();
    ggml_backend_alloc_vram_buffer(m_ctx.sched.get(), m_plan.max_vram_alloc, m_plan.cuda_host_size);
    plan_compute_requirements(initial_plan, m_plan.tier_config.tier_size(m_plan.tier_config.n_tiers() - 1), false, false);
    
    // Allocate weights using new unified function
    statically_allocate_weights(initial_plan, m_use_single_plan ? EXEC_STRATEGY_FULL_GPU : m_current_strategy);
    prefetch_pinned_weights();
    allocate_kv_for_plan(initial_plan, m_use_single_plan ? EXEC_STRATEGY_FULL_GPU : m_current_strategy);

    // Zero all GPU KV tensors after initial VRAM allocation to prevent NaN
    // from uninitialized VRAM.
    {
        const int32_t n_layer = m_ctx.model.hparams.n_layer;
        for (int32_t il = 0; il < n_layer; il++) {
            pipeline_zero_kv_gpu(il);
        }
    }

    log_pipeline_summary();

    m_ctx.n_outputs = n_outputs_save;
    m_plan.vram_allocated = true;
}

void PipelineExecutor::populate_plan_registry() {
    LLAMA_LOG_INFO("=== populate_plan_registry: START ===\n");
    size_t   n_tiers   = m_plan.tier_config.n_tiers();
    size_t   max_tier  = n_tiers - 1;
    uint32_t max_batch = m_plan.tier_config.tier_size(max_tier);
    int      forced    = m_ctx.cparams.force_exec_strategy;

    // Step 1: Check if FULL_GPU works at max tier
    plan_build_for_batch(m_plan.unified_plan, max_batch, EXEC_STRATEGY_FULL_GPU);

    bool unified_viable = (m_plan.unified_plan.total_vram_req <= m_plan.max_vram_alloc);

    if (unified_viable) {
        m_plan.single_plan_mode       = true;
        m_plan.unified_plan.is_viable = true;
        return;
    }

    m_plan.single_plan_mode       = false;
    m_plan.unified_plan.is_viable = false;
    m_plan.plans[max_tier][EXEC_STRATEGY_FULL_GPU] = m_plan.unified_plan;

    // === FORCED STRATEGY MODE ===
    if (forced >= 0 && forced < EXEC_STRATEGY_COUNT) {
        // Build forced strategy for all tiers
        for (size_t tier = 0; tier < n_tiers; ++tier) {
            exec_strategy   forced_strat = static_cast<exec_strategy>(forced);
            uint32_t batch_size = m_plan.tier_config.tier_size(tier);
            ExecutionPlan & plan = m_plan.plans[tier][forced];
            plan_build_for_batch(plan, batch_size, forced_strat);
            plan.is_viable = (plan.total_vram_req <= m_plan.max_vram_alloc);
        }
        return;
    }

    // === AUTO MODE ===
    // Step 2: Check FULL_GPU_NO_OUTPUT at max tier
    plan_build_for_batch(m_plan.plans[max_tier][EXEC_STRATEGY_FULL_GPU_NO_OUTPUT], max_batch, EXEC_STRATEGY_FULL_GPU_NO_OUTPUT);
    m_plan.plans[max_tier][EXEC_STRATEGY_FULL_GPU_NO_OUTPUT].is_viable = (m_plan.plans[max_tier][EXEC_STRATEGY_FULL_GPU_NO_OUTPUT].total_vram_req <= m_plan.max_vram_alloc);
    bool max_full_gpu_no_output_viable = m_plan.plans[max_tier][EXEC_STRATEGY_FULL_GPU_NO_OUTPUT].is_viable;

    // Build remaining strategies for max tier (only if FULL_GPU_NO_OUTPUT not viable)
    if (!max_full_gpu_no_output_viable) {
        for (int32_t s = 0; s < EXEC_STRATEGY_COUNT; ++s) {
            if (s == EXEC_STRATEGY_FULL_GPU || s == EXEC_STRATEGY_FULL_GPU_NO_OUTPUT) {
                continue;  // Already handled
            }
            exec_strategy   strategy = static_cast<exec_strategy>(s);
            ExecutionPlan & plan     = m_plan.plans[max_tier][s];
            plan_build_for_batch(plan, max_batch, strategy);
            plan.is_viable = (plan.total_vram_req <= m_plan.max_vram_alloc);
        }
    }

    // Step 3: Apply monotonicity rules for lower tiers
    for (size_t tier = max_tier; tier-- > 0;) {
        uint32_t batch_size = m_plan.tier_config.tier_size(tier);

        if (max_full_gpu_no_output_viable) {
            // Only check if FULL_GPU becomes viable at this smaller batch
            plan_build_for_batch(m_plan.plans[tier][EXEC_STRATEGY_FULL_GPU], batch_size, EXEC_STRATEGY_FULL_GPU);
            m_plan.plans[tier][EXEC_STRATEGY_FULL_GPU].is_viable = 
                (m_plan.plans[tier][EXEC_STRATEGY_FULL_GPU].total_vram_req <= m_plan.max_vram_alloc);

            if (!m_plan.plans[tier][EXEC_STRATEGY_FULL_GPU].is_viable) {
                // FULL_GPU not viable → copy FULL_GPU_NO_OUTPUT from max tier (monotonic)
                m_plan.plans[tier][EXEC_STRATEGY_FULL_GPU_NO_OUTPUT] = m_plan.plans[max_tier][EXEC_STRATEGY_FULL_GPU_NO_OUTPUT];
                m_plan.plans[tier][EXEC_STRATEGY_FULL_GPU_NO_OUTPUT].is_viable = true;
            }
            // Rest are non-viable (already false by default)
            continue;
        }

        // max_full_gpu_no_output_viable == false: check strategies in priority order
        for (int32_t s = 0; s < EXEC_STRATEGY_COUNT; ++s) {
            exec_strategy   strategy = static_cast<exec_strategy>(s);
            ExecutionPlan & plan     = m_plan.plans[tier][s];

            plan_build_for_batch(plan, batch_size, strategy);
            plan.is_viable = (plan.total_vram_req <= m_plan.max_vram_alloc);

            // If FULL_GPU or FULL_GPU_NO_OUTPUT viable, skip building rest
            if (plan.is_viable && 
                (strategy == EXEC_STRATEGY_FULL_GPU || strategy == EXEC_STRATEGY_FULL_GPU_NO_OUTPUT)) {
                break;
        }
    }
    }
    // TPS prediction happens inside plan_compute_requirements before scheduler reset
}

// Allocates VRAM for the pipeline when LoRA is enabled.
// It re-calculates the VRAM requirements and re-allocates VRAM.
void PipelineExecutor::alloc_vram() {
    if (m_plan.vram_allocated) {
        return;
    }

    m_plan.vram_allocated = true;
}

ggml_status PipelineExecutor::compute(ggml_cgraph * gf) {
    GGML_ASSERT(m_active_plan != nullptr);
    const ExecutionPlan & plan = *m_active_plan;

    const int n_splits = ggml_backend_sched_get_num_splits(m_ctx.sched.get());
    const int n_plan_splits = (int)plan.pipeline_splits.size();

    // Detect split offset between plan and runtime graph.
    // The plan's pipeline_splits are built from a token-input graph during initialization.
    // When running with embedding input (e.g., image tokens from CLIP), the token embedding
    // CPU split is absent, causing the runtime to have fewer splits. We detect this and
    // offset into the plan's splits so that plan metadata (KV layers, weights, backend type)
    // correctly maps to each runtime split.
    int plan_offset = 0;
    if (!m_plan.single_plan_mode && n_splits < n_plan_splits) {
        ggml_backend_t first_runtime_backend = ggml_backend_sched_get_split_backend(m_ctx.sched.get(), 0);
        if (!ggml_backend_is_cpu(first_runtime_backend) &&
            !plan.pipeline_splits.empty() && plan.pipeline_splits[0].backend_type == CPU) {
            plan_offset = n_plan_splits - n_splits;
        }
    }

    const bool has_sharding = (plan.sharded_weights_vram_req > 0);

    // Reuse pre-allocated vectors instead of allocating new ones
    m_compute_seq_ids = m_mctx->get_seq_ids();
    m_compute_write_cells = m_mctx->get_write_cells();
    m_compute_per_seq_write_cells = m_mctx->get_write_cells_per_seq();

    int last_copied_idx = has_sharding ? 0 : -1;
    ggml_status err = GGML_STATUS_SUCCESS;

    for (int i = 0; i < n_splits; ++i) {
        const int pi = i + plan_offset;
        if (pi >= n_plan_splits) break;

        const auto & split = plan.pipeline_splits[pi];
        const int next_i = i + 1;
        const int next_pi = next_i + plan_offset;
        const bool next_valid = (next_i < n_splits) && (next_pi < n_plan_splits);
        const bool next_is_sharded = next_valid && (plan.pipeline_splits[next_pi].backend_type == GPU_SHARDED);

        // Step 1: CopyInputs
        err = ggml_backend_sched_compute_split_copy_inputs(m_ctx.sched.get(), i);
        if (err != GGML_STATUS_SUCCESS) return err;

        // Step 2: Queue next split's async ops (overlaps with Run)
        if (next_is_sharded) {
            const auto & next_split = plan.pipeline_splits[next_pi];
            for (int layer_id : next_split.kv_layer_ids) {
                pipeline_upload_kv_cells_for_seqs(layer_id, m_compute_seq_ids);
            }
            prefetch_split_weights(plan, next_i, next_pi, &last_copied_idx);
        }

        // Step 3: Run computation
        err = ggml_backend_sched_compute_split_run(m_ctx.sched.get(), i);
        if (err != GGML_STATUS_SUCCESS) return err;

        // Step 4: Sync GPU + Download KV for non-single-plan splits
        if (split.backend_type != CPU) {
            bool need_sync = (m_ctx.n_queued_tokens <= 1) || !m_plan.single_plan_mode;
            if (need_sync) {ggml_backend_synchronize(m_ctx.backends.at(0).get());}
            if (!m_plan.single_plan_mode) {
                for (int layer_id : split.kv_layer_ids) {
                    pipeline_download_kv_cells_for_seqs(layer_id, m_compute_seq_ids, m_compute_write_cells, m_compute_per_seq_write_cells);
                }
            }
        }
    }

    ggml_backend_sched_compute_split_finalize(m_ctx.sched.get());

    return GGML_STATUS_SUCCESS;
}

// Simple prefetch helper
// sched_split_id: index into the scheduler's runtime splits (for backend query)
// plan_split_id:  index into plan.pipeline_splits (for leaf tensor access)
void PipelineExecutor::prefetch_split_weights(const ExecutionPlan & plan, int sched_split_id, int plan_split_id, int * last_idx) {
    if (plan_split_id <= *last_idx) return;
    
    ggml_backend_t backend = ggml_backend_sched_get_split_backend(m_ctx.sched.get(), sched_split_id);
    if (ggml_backend_is_cpu(backend)) {
        *last_idx = plan_split_id;
        return;
    }
    
    if (plan_split_id < (int)plan.pipeline_splits.size()) {
        for (ggml_tensor * leaf : plan.pipeline_splits[plan_split_id].leafs) {
            pipeline_set_tensor_data(leaf, backend, true);  // async
        }
    }
    *last_idx = plan_split_id;
}

void PipelineExecutor::set_scratch_offsets(){
    ExecutionPlan * plan = m_active_plan;
    
    // Fallback if m_active_plan not set yet (during init)
    if (!plan) {
        if (m_plan.single_plan_mode) {
            plan = &m_plan.unified_plan;
        } else {
            plan = &m_plan.plans[m_current_tier][m_current_strategy];
        }
    }
    size_t scratch_offset = plan->pinned_weights_vram_req + plan->sharded_weights_vram_req;
    ggml_backend_alloc_set_scratch_offsets(m_ctx.sched.get(), plan->scratch_vram_req, scratch_offset);
}

void PipelineExecutor::restore_plan_alloc_state(const ExecutionPlan & plan) {
    // Skip if no saved state (plan wasn't finalized properly)
    if (!plan.alloc_state.valid) {
        LLAMA_LOG_ERROR("%s: skipping - alloc_state not valid\n", __func__);
        return;
    }
        
    auto * sched = m_ctx.sched.get();
    auto * galloc = ggml_backend_sched_get_galloc(sched);
    
    // Restore allocator state (node_allocs, leaf_allocs)
    ggml_gallocr_restore_state(galloc,
                                plan.alloc_state.node_allocs.data(),
                                plan.alloc_state.node_allocs.size(),
                                plan.alloc_state.leaf_allocs.data(),
                                plan.alloc_state.leaf_allocs.size(),
                                plan.alloc_state.n_nodes,
                                plan.alloc_state.n_leafs);
    
    // Restore prev_backend_ids so backend_ids_changed check passes
    ggml_backend_sched_restore_backend_ids(sched,
                                            plan.alloc_state.node_backend_ids.data(),
                                            (int)plan.alloc_state.node_backend_ids.size(),
                                            plan.alloc_state.leaf_backend_ids.data(),
                                            (int)plan.alloc_state.leaf_backend_ids.size());
}

// Dynamic plan building:
// Build or retrieve a plan for exact n_tokens
//
// Flow:
//   1. If DYNAMIC_PLAN_ENABLED=false -> use ceiling tier lookup (simple path)
//   2. If n_tokens matches a tier exactly -> use pre-built tier plan
//   3. If dynamic plan already cached for n_tokens -> return cached
//   4. Measure scratch for n_tokens using ceiling tier's strategies
//   5. If measured scratch fits -> clone ceiling plan into dynamic cache
//   6. Otherwise -> build fresh dynamic plan (SLOW PATH)
//
ExecutionPlan * PipelineExecutor::get_or_build_plan_for_tokens(uint32_t n_tokens) {
    //LLAMA_LOG_INFO("%s: n_tokens=%u, DYNAMIC_PLAN_ENABLED=%d\n", __func__, n_tokens, DYNAMIC_PLAN_ENABLED);
    
    // DISABLED: use ceiling tier lookup (original behavior)
    // This path avoids dynamic plan building entirely - just find the smallest tier >= n_tokens
    if (!DYNAMIC_PLAN_ENABLED) {
        size_t tier = m_plan.tier_config.tier_index(n_tokens);  // ceiling tier
        exec_strategy strategy = select_best_strategy(tier);
        //LLAMA_LOG_INFO("%s: [STATIC PATH] n_tokens=%u -> ceiling tier=%zu (%s), scratch=%.2fMB\n",
        //               __func__, n_tokens, tier, exec_strategy_name(strategy),
        //               m_plan.plans[tier][strategy].scratch_vram_req / (1024.0 * 1024.0));
        return &m_plan.plans[tier][strategy];
    }
    
    // Step 1: Check if n_tokens matches any tier exactly - use pre-built plan
    // This is the fast path for tier-aligned batch sizes (1, 4, 16, 64, 512, etc.)
    for (size_t tier = 0; tier < m_plan.tier_config.n_tiers(); tier++) {
        if (m_plan.tier_config.tier_size(tier) == n_tokens) {
            exec_strategy strategy = select_best_strategy(tier);
            //LLAMA_LOG_INFO("%s: [EXACT TIER MATCH] n_tokens=%u matches tier=%zu (%s), scratch=%.2fMB\n",
            //               __func__, n_tokens, tier, exec_strategy_name(strategy),
            //               m_plan.plans[tier][strategy].scratch_vram_req / (1024.0 * 1024.0));
            return &m_plan.plans[tier][strategy];
        }
    }
    
    // Step 2: Check if we already have a cached dynamic plan for this n_tokens
    // Dynamic plans are built on-demand and cached for reuse
    auto it = m_plan.dynamic_plans.find(n_tokens);
    if (it != m_plan.dynamic_plans.end()) {
        //LLAMA_LOG_INFO("%s: [CACHED DYNAMIC] n_tokens=%u found in cache, scratch=%.2fMB\n",
        //               __func__, n_tokens, it->second.scratch_vram_req / (1024.0 * 1024.0));
        return &it->second;
    }
    
    // Step 3: Get ceiling tier plan and check if it's sufficient
    // Ceiling tier = smallest pre-built tier that can handle n_tokens
    size_t ceiling_tier = m_plan.tier_config.tier_index(n_tokens);
    exec_strategy strategy = select_best_strategy(ceiling_tier);
    ExecutionPlan & ceiling_plan = m_plan.plans[ceiling_tier][strategy];

    //LLAMA_LOG_INFO("%s: [DYNAMIC BUILD] n_tokens=%u, ceiling_tier=%zu (batch=%u), strategy=%s\n",
    //               __func__, n_tokens, ceiling_tier, m_plan.tier_config.tier_size(ceiling_tier),
    //               exec_strategy_name(strategy));
    //LLAMA_LOG_INFO("%s:   ceiling_plan: scratch=%.2fMB, pinned=%.2fMB, sharded=%.2fMB\n",
    //               __func__, ceiling_plan.scratch_vram_req / (1024.0 * 1024.0),
    //               ceiling_plan.pinned_weights_vram_req / (1024.0 * 1024.0),
    //               ceiling_plan.sharded_weights_vram_req / (1024.0 * 1024.0));

    allocate_kv_for_plan(ceiling_plan, strategy);

    // Step 4: Quick scratch check - measure scratch for n_tokens with ceiling plan's strategies
    // This avoids a full plan rebuild if the ceiling plan's scratch is already sufficient
    size_t measured_scratch = plan_measure_scratch_only(ceiling_plan, n_tokens, m_ctx.n_outputs);
    
    //LLAMA_LOG_INFO("%s:   measured_scratch=%.2fMB for n_tokens=%u (ceiling has %.2fMB)\n",
    //               __func__, measured_scratch / (1024.0 * 1024.0), n_tokens,
    //               ceiling_plan.scratch_vram_req / (1024.0 * 1024.0));
    
    m_ctx.gf_res_prev->reset();

    // Step 5: If it fits, use ceiling plan (FAST PATH)
    // Clone ceiling plan into dynamic cache so we don't re-measure next time
    if (measured_scratch <= ceiling_plan.scratch_vram_req) {
        //LLAMA_LOG_INFO("%s:   [FAST PATH] measured %.2fMB <= ceiling %.2fMB, cloning ceiling plan\n",
        //               __func__, measured_scratch / (1024.0 * 1024.0),
        //               ceiling_plan.scratch_vram_req / (1024.0 * 1024.0));
        m_plan.dynamic_plans[n_tokens] = ceiling_plan;
        return &m_plan.dynamic_plans[n_tokens];
    }

    // SLOW PATH: Build dynamic plan using RUNTIME graph parameters
    // This happens when ceiling plan's scratch isn't big enough for n_tokens
    // (rare - usually means fragmentation or non-monotonic scratch scaling)
    //LLAMA_LOG_INFO("%s:   [SLOW PATH] measured %.2fMB > ceiling %.2fMB, building fresh plan with runtime params\n",
    //               __func__, measured_scratch / (1024.0 * 1024.0),
    //               ceiling_plan.scratch_vram_req / (1024.0 * 1024.0));
     
    //LLAMA_LOG_INFO("%s:   runtime n_outputs=%d (vs planning default=%u)\n", __func__, n_outputs_actual, n_tokens);
    
    ExecutionPlan & dynamic_plan = m_plan.dynamic_plans[n_tokens];
    // Build dynamic plan with runtime n_outputs
    plan_build_for_batch(dynamic_plan, n_tokens, strategy, false, m_ctx.n_outputs);

    //LLAMA_LOG_INFO("%s:   built dynamic plan: scratch=%.2fMB, pinned=%.2fMB, sharded=%.2fMB\n",
    //               __func__, dynamic_plan.scratch_vram_req / (1024.0 * 1024.0),
    //               dynamic_plan.pinned_weights_vram_req / (1024.0 * 1024.0),
    //               dynamic_plan.sharded_weights_vram_req / (1024.0 * 1024.0));
        
    return &dynamic_plan;
}

// Prepares the pipeline for a new decoding batch.
// This function sets the decode state, configures the VRAM offsets, and assigns tensors
// for the KV cache.
void PipelineExecutor::prepare_for_decode(int64_t n_tokens_all, llama_memory_context_i * mctx) {
    m_mctx = mctx;
    
    // FAST PATH: single plan mode (FULL_GPU fits at max tier)
    if (m_plan.single_plan_mode) {
        //LLAMA_LOG_INFO("%s: n_tokens_all=%lld (single_plan_mode, using unified_plan)\n", __func__, (long long)n_tokens_all);
        m_active_plan = &m_plan.unified_plan;
        restore_plan_alloc_state(*m_active_plan);
        // No tier/strategy logic needed - use unified_plan for everything
        set_scratch_offsets();
        reload_leaf_buffer_ids();

        // Handle KV-shift: reactivate GPU KV and upload shifted data
        if (get_kv_status()) {
            sync_kv_after_shift(m_plan.unified_plan);
            set_kv_status(false);
        }

        return;
    }

    const auto & ubatch = mctx->get_ubatch();
    uint32_t n_tokens = ubatch.n_tokens;
    
    // Get or build plan for exact n_tokens
    // This may return a pre-built tier plan or a dynamically built plan
    ExecutionPlan * new_plan = get_or_build_plan_for_tokens(n_tokens);

    // Check if plan switch is needed (different plan than currently active)
    if (new_plan != m_active_plan && m_active_plan != nullptr) {
        // Determine tier and strategy for tracking
        size_t ceiling_tier = m_plan.tier_config.tier_index(n_tokens);
        exec_strategy ceiling_strategy = select_best_strategy(ceiling_tier);

        // Skip switch if same strategy and current tier is larger (can handle more tokens)
        if (ceiling_strategy == m_current_strategy && m_current_tier >= ceiling_tier) {
            // Keep using current plan, just update n_tokens tracking
            allocate_kv_for_plan(*m_active_plan, ceiling_strategy);
            restore_plan_alloc_state(*m_active_plan);
            set_scratch_offsets();
            reload_leaf_buffer_ids();

            if (get_kv_status()) {
                sync_kv_after_shift(*m_active_plan);
                set_kv_status(false);
            }
            return;
        }
        switch_plan(*m_active_plan, *new_plan, ceiling_strategy);
        
        m_current_tier = ceiling_tier;
        m_current_strategy = ceiling_strategy;
    }
    
    // Update active plan
    m_active_plan = new_plan;
    restore_plan_alloc_state(*m_active_plan);
    set_scratch_offsets();
    reload_leaf_buffer_ids();
    
    if (get_kv_status()) {
        sync_kv_after_shift(*m_active_plan);
        set_kv_status(false);
    }
    return;
}

// A hook called after the graph is built.
// This function sets the pinned buffers for the scheduler and resets the leaf nodes
// for the current phase.
void PipelineExecutor::post_graph_build(ggml_cgraph * gf) {
    GGML_ASSERT(m_active_plan != nullptr);
    const ExecutionPlan & plan = *m_active_plan;

    // Apply node backend overrides using the unified function
    plan_apply_node_overrides(gf, plan);
}

// ===================================================================
// Private Helper Functions
// ===================================================================

void PipelineExecutor::prefetch_pinned_weights() {
    // Use active plan if available, otherwise fallback to tier/strategy lookup
    ExecutionPlan * plan_ptr = m_active_plan;
    if (!plan_ptr) {
        if (m_plan.single_plan_mode) {
            plan_ptr = &m_plan.unified_plan;
        } else {
            plan_ptr = &m_plan.plans[m_current_tier][m_current_strategy];
        }
    }
    const auto & plan = *plan_ptr;

    // Get the backend for the main VRAM buffer (GPU)
    ggml_backend_t gpu_backend = m_ctx.backends.at(0).get();
    
    for (const auto & split : plan.pipeline_splits) {
        if (split.backend_type == GPU_PINNED && !split.leafs.empty()) {
            for (struct ggml_tensor * leaf : split.leafs) {
                pipeline_set_tensor_data(leaf, gpu_backend, true);
            }
        }
    }
}

static LAYER_BACKENDS get_backend_type_for_dev(ggml_backend_dev_t dev) {
    if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
        return CPU;
    }
    return GPU;
}

//
// select_best_strategy - choose best viable strategy for a tier
//
exec_strategy PipelineExecutor::select_best_strategy(size_t tier) {
    // Check for forced strategy
    int forced = m_ctx.cparams.force_exec_strategy;
    if (forced >= 0 && forced < EXEC_STRATEGY_COUNT) {        
        const ExecutionPlan & forced_plan = m_plan.plans[tier][forced];
        if (forced_plan.is_viable) {
            return static_cast<exec_strategy>(forced);
        }
    }

    // Auto-select best strategy
    exec_strategy best_strategy = EXEC_STRATEGY_STATIC_ATTN_PRIO;
    float         best_tps      = -1.0f;

    for (int32_t s = 0; s < EXEC_STRATEGY_COUNT; ++s) {
        const ExecutionPlan & plan = m_plan.plans[tier][s];

        if (plan.is_viable && plan.tps > best_tps) {
            best_tps      = plan.tps;
            best_strategy = static_cast<exec_strategy>(s);
        }
    }
    return best_strategy;
}

//
// switch_plan - handle transition between execution plans
//
void PipelineExecutor::switch_plan(const ExecutionPlan & old_plan, ExecutionPlan & new_plan, exec_strategy new_strategy) {
    const int32_t n_layer = m_ctx.model.hparams.n_layer;

    // Phase 1: Reallocate weights with common prefix optimization
    const auto & old_sorted = old_plan.sorted_pinned_weights;
    const auto & new_sorted = new_plan.sorted_pinned_weights;

    statically_allocate_weights(new_plan, new_strategy);

    // Find common prefix length
    size_t prefix_len = 0;
    size_t min_len = std::min(old_sorted.size(), new_sorted.size());
    while (prefix_len < min_len && old_sorted[prefix_len] == new_sorted[prefix_len]) {
        prefix_len++;
    }
    
    // Copy only weights AFTER the common prefix
    size_t weights_uploaded = 0;
    size_t weights_bytes = 0;
    ggml_backend_t gpu = m_ctx.backends.at(0).get();
    for (size_t i = prefix_len; i < new_sorted.size(); i++) {
        pipeline_set_tensor_data(new_sorted[i], gpu, true);
        weights_uploaded++;
        weights_bytes += ggml_nbytes(new_sorted[i]);
    }

    allocate_kv_for_plan(new_plan, new_strategy);
    
    // Phase 2: Zero + Upload KV
    for (int32_t il = 0; il < n_layer; il++) {
        bool now_gpu = (new_plan.strategies[il] != LAYER_STRATEGY_ALL_CPU);
        if (!now_gpu) {continue; }

        auto * kv_cache = get_kv_cache_for_layer(il);
        auto * pipe_shard = kv_cache ? kv_cache->get_pipe_shard() : nullptr;
        if (!pipe_shard) continue;

        ggml_tensor * k = pipe_shard->k_gpu(il);
        ggml_tensor * v = pipe_shard->v_gpu(il);
        if (!k || !v) continue;

        bool addr_changed = pipe_shard->state_address_changed(il, k->data, v->data);
        //if (!addr_changed) { continue; }

        // Zero GPU KV — old VRAM at new address may contain weight/scratch garbage (NaN)
        pipeline_zero_kv_gpu(il);

        // Upload pinned layers' KV data from CPU (sharded layers uploaded in compute loop)
        bool now_pinned = (new_plan.strategies[il] == LAYER_STRATEGY_ALL_GPU_PINNED ||
                           new_plan.strategies[il] == LAYER_STRATEGY_ATTN_GPU_PINNED ||
                           new_plan.strategies[il] == LAYER_STRATEGY_ATTN_PINNED_FFN_SHARDED);
        if (now_pinned) {
            std::vector<llama_seq_id> seq_ids;
            for (uint32_t s = 0; s < m_ctx.cparams.n_seq_max; ++s) {
                if (!kv_cache->get_cells_for_seq(s).empty()) {
                    seq_ids.push_back(s);
                }
            }

            if (!seq_ids.empty()) {
                pipeline_upload_kv_cells_for_seqs(il, seq_ids);
            }
        }

        pipe_shard->state_update_addresses(il, k->data, v->data);
    }
}

void PipelineExecutor::log_pipeline_summary() const {
    LLAMA_LOG_INFO("\n");
    LLAMA_LOG_INFO("========== PIPELINE PLAN SUMMARY ==========\n");
    
    // --- Global Parameters ---
    LLAMA_LOG_INFO("\n--- Global Parameters ---\n");
    LLAMA_LOG_INFO("  %-20s %8.2f MB\n", "max_vram_alloc:", m_plan.max_vram_alloc / (1024.0 * 1024.0));
    LLAMA_LOG_INFO("  %-20s %8.2f MB\n", "total_kv_size:", m_plan.total_kv_size / (1024.0 * 1024.0));
    LLAMA_LOG_INFO("  %-20s %8.2f MB\n", "kv_size_per_layer:", m_plan.kv_size_per_layer / (1024.0 * 1024.0));
    LLAMA_LOG_INFO("  %-20s %8.2f MB\n", "cuda_host_size:", m_plan.cuda_host_size / (1024.0 * 1024.0));
    LLAMA_LOG_INFO("  %-20s %8d\n", "n_layers:", m_ctx.model.hparams.n_layer);
    
    // --- Plan Mode ---
    LLAMA_LOG_INFO("\n--- Plan Mode ---\n");
    if (m_plan.single_plan_mode) {
        LLAMA_LOG_INFO("  %-20s %s\n", "mode:", "SINGLE_PLAN (FULL_GPU fits at max tier)");
        
        const auto & plan = m_plan.unified_plan;
        LLAMA_LOG_INFO("\n--- Unified Plan Details ---\n");
        LLAMA_LOG_INFO("  %-20s %8.2f MB\n", "pinned_weights:", plan.pinned_weights_vram_req / (1024.0 * 1024.0));
        LLAMA_LOG_INFO("  %-20s %8.2f MB\n", "sharded_weights:", plan.sharded_weights_vram_req / (1024.0 * 1024.0));
        LLAMA_LOG_INFO("  %-20s %8.2f MB\n", "scratch:", plan.scratch_vram_req / (1024.0 * 1024.0));
        LLAMA_LOG_INFO("  %-20s %8.2f MB\n", "kv_cache:", plan.kv_vram_req / (1024.0 * 1024.0));
        LLAMA_LOG_INFO("  %-20s %8.2f MB\n", "total:", plan.total_vram_req / (1024.0 * 1024.0));
        LLAMA_LOG_INFO("  %-20s %8zu\n", "n_splits:", plan.pipeline_splits.size());
        LLAMA_LOG_INFO("  %-20s %8d\n", "n_pinned_layers:", (int) std::count_if(plan.strategies.begin(), plan.strategies.end(),
            [](int s) { return s == LAYER_STRATEGY_ALL_GPU_PINNED || s == LAYER_STRATEGY_ATTN_GPU_PINNED; }));
        LLAMA_LOG_INFO("  %-20s %8.2f tps\n", "predicted_tps:", plan.tps);
    } else {
        LLAMA_LOG_INFO("  %-20s %s\n", "mode:", "MULTI_PLAN (tier x strategy registry)");
        LLAMA_LOG_INFO("  %-20s %zu (batch_size=%u)\n", "current_tier:", m_current_tier, m_plan.tier_config.tier_size(m_current_tier));
        LLAMA_LOG_INFO("  %-20s %s\n", "current_strategy:", exec_strategy_name(m_current_strategy));
        
        // --- Strategy x Tier Viability Matrix ---
        LLAMA_LOG_INFO("\n--- Plan Registry: Strategy x Tier Viability ---\n\n");
        
        // Header row - each tier column is 19 chars: " |" (2) + marker (1) + cell (16)
        LLAMA_LOG_INFO("  %-18s", "Strategy");
        for (size_t tier = 0; tier < m_plan.tier_config.n_tiers(); tier++) {
            uint32_t batch_size = m_plan.tier_config.tier_size(tier);
            char hdr[32];
            snprintf(hdr, sizeof(hdr), "Tier %zu (b=%u)", (size_t)tier, batch_size);
            LLAMA_LOG_INFO(" | %-16s", hdr);
        }
        LLAMA_LOG_INFO("\n");
        
        // Separator - matches column width: 18 for strategy + 19 per tier
        LLAMA_LOG_INFO("  ------------------");
        for (size_t tier = 0; tier < m_plan.tier_config.n_tiers(); tier++) {
            LLAMA_LOG_INFO("+-------------------");
        }
        LLAMA_LOG_INFO("\n");
        
        // Data rows
            for (int strat = 0; strat < EXEC_STRATEGY_COUNT; strat++) {
            LLAMA_LOG_INFO("  %-18s", exec_strategy_name((exec_strategy)strat));

            for (size_t tier = 0; tier < m_plan.tier_config.n_tiers(); tier++) {
                const ExecutionPlan & plan = m_plan.plans[tier][strat];
                
                bool is_current = (tier == m_current_tier && strat == m_current_strategy);
                const char * marker = is_current ? ">" : " ";

                // Check if plan was built (has splits) vs just not built
                bool was_built = !plan.pipeline_splits.empty();
                
                // Format cell content into fixed-width buffer to guarantee alignment
                // All cases produce exactly 16 chars, printed with %-16s for safety
                char cell[32];
                if (!was_built) {
                    snprintf(cell, sizeof(cell), "      SKIP      ");  // 16 chars
                } else if (plan.is_viable) {
                    snprintf(cell, sizeof(cell), "%6.1fMB %4.0ftps", plan.total_vram_req / (1024.0 * 1024.0), plan.tps);  // 16 chars
                } else {
                    snprintf(cell, sizeof(cell), "%6.1fMB    --  ", plan.total_vram_req / (1024.0 * 1024.0));  // 16 chars
                }
                LLAMA_LOG_INFO(" |%s%-16s", marker, cell);
            }
            LLAMA_LOG_INFO("\n");
        }
        
        // --- All Viable Plans ---
        LLAMA_LOG_INFO("\n--- All Viable Plans ---\n");

        bool found_any = false;
        for (size_t tier = 0; tier < m_plan.tier_config.n_tiers(); tier++) {
            uint32_t batch_size = m_plan.tier_config.tier_size(tier);

            for (int strat = 0; strat < EXEC_STRATEGY_COUNT; strat++) {
                const ExecutionPlan & p = m_plan.plans[tier][strat];

                if (!p.is_viable) {
                    continue;
                }

                found_any = true;
                bool is_current = (tier == m_current_tier && strat == m_current_strategy);
                const char * marker = is_current ? ">>>" : "   ";

                LLAMA_LOG_INFO("\n%s Tier %zu, %s (batch=%u)\n", marker, tier, exec_strategy_name((exec_strategy)strat), batch_size);

                int n_pin = 0;
                for (int s : p.strategies) {
                    if (s == LAYER_STRATEGY_ALL_GPU_PINNED || s == LAYER_STRATEGY_ATTN_GPU_PINNED) {
                        n_pin++;
                    }
                }

                LLAMA_LOG_INFO("    %-20s %8.2f MB\n", "pinned_weights:", p.pinned_weights_vram_req / (1024.0 * 1024.0));
                LLAMA_LOG_INFO("    %-20s %8.2f MB\n", "sharded_weights:", p.sharded_weights_vram_req / (1024.0 * 1024.0));
                LLAMA_LOG_INFO("    %-20s %8.2f MB\n", "scratch:", p.scratch_vram_req / (1024.0 * 1024.0));
                LLAMA_LOG_INFO("    %-20s %8.2f MB\n", "kv_cache:", p.kv_vram_req / (1024.0 * 1024.0));
                LLAMA_LOG_INFO("    %-20s %8.2f / %.2f MB\n", "total:", p.total_vram_req / (1024.0 * 1024.0), m_plan.max_vram_alloc / (1024.0 * 1024.0));
                LLAMA_LOG_INFO("    %-20s %8zu\n", "n_splits:", p.pipeline_splits.size());
                LLAMA_LOG_INFO("    %-20s %5d / %d\n", "n_pinned_layers:", n_pin, (int) p.strategies.size());

                LLAMA_LOG_INFO("    %-20s %8.2f tps\n", "predicted_tps:", p.tps);

                // Strategies on one line
                std::ostringstream ss;
                ss << "[";
                for (size_t i = 0; i < p.strategies.size(); ++i) {
                    ss << p.strategies[i];
                    if (i < p.strategies.size() - 1) ss << ",";
        }
                ss << "]";
                LLAMA_LOG_INFO("    %-20s %s\n", "strategies:", ss.str().c_str());
            }
        }

        if (!found_any) {
            LLAMA_LOG_INFO("  (no viable plans found)\n");
        }

        // --- Non-Viable Plans ---
        LLAMA_LOG_INFO("\n--- Non-Viable Plans (exceeds budget) ---\n");
        bool found_nonviable = false;
        for (size_t tier = 0; tier < m_plan.tier_config.n_tiers(); tier++) {
            uint32_t batch_size = m_plan.tier_config.tier_size(tier);
            for (int strat = 0; strat < EXEC_STRATEGY_COUNT; strat++) {
                const ExecutionPlan & p = m_plan.plans[tier][strat];
                if (!p.is_viable && p.total_vram_req > 0) {
                    LLAMA_LOG_INFO("  Tier %zu %-18s batch=%4u -> needs %8.2f MB\n",
                        tier, exec_strategy_name((exec_strategy)strat), batch_size, p.total_vram_req / (1024.0 * 1024.0));
                    found_nonviable = true;
                }
            }
        }
        if (!found_nonviable) {
            LLAMA_LOG_INFO("  (none)\n");
        }

        // --- Skipped Plans ---
        LLAMA_LOG_INFO("\n--- Skipped Plans ---\n");
        bool found_skipped = false;
        for (size_t tier = 0; tier < m_plan.tier_config.n_tiers(); tier++) {
            uint32_t batch_size = m_plan.tier_config.tier_size(tier);
            for (int strat = 0; strat < EXEC_STRATEGY_COUNT; strat++) {
                const ExecutionPlan & p = m_plan.plans[tier][strat];
                if (p.total_vram_req == 0 && !p.is_viable) {
                    LLAMA_LOG_INFO("  Tier %zu %-18s batch=%4u\n", tier, exec_strategy_name((exec_strategy)strat), batch_size);
                    found_skipped = true;
                }
            }
        }
        if (!found_skipped) {
            LLAMA_LOG_INFO("  (none)\n");
        }
    }

    LLAMA_LOG_INFO("\n===========================================\n");
}

void PipelineExecutor::is_moe() {
    m_plan.is_moe = false; // Default to false
    const std::regex moe_pattern("\\.ffn_(up|down|gate)_exps");

    for (const auto & pair : m_ctx.model.layer_to_tensor_map) {
        const std::vector<struct ggml_tensor *> & tensors_in_layer = pair.second;
        for (struct ggml_tensor * tensor : tensors_in_layer) {
            const char* tensor_name = ggml_get_name(tensor);
            if (tensor_name && std::regex_search(tensor_name, moe_pattern)) {
                m_plan.is_moe = true;
                LLAMA_LOG_INFO("%s: MoE model detected\n", __func__);
                return; // Found one, no need to search further
            }
        }
    }
}

void PipelineExecutor::reload_leaf_buffer_ids() {
    ExecutionPlan * plan = m_active_plan;
    
    // Fallback if m_active_plan not set yet (during init)
    if (!plan) {
        if (m_plan.single_plan_mode) {
            plan = &m_plan.unified_plan;
        } else {
            plan = &m_plan.plans[m_current_tier][m_current_strategy];
        }
    }

    for (const auto & pair : plan->leaf_to_buf_id_map) {
        ggml_tensor * leaf = pair.first;
        leaf->buffer_id    = pair.second;
        leaf->buft         = ggml_backend_sched_get_buft(m_ctx.sched.get(), leaf->buffer_id);
        ggml_backend_sched_set_tensor_backend(m_ctx.sched.get(), leaf, m_ctx.backends[leaf->buffer_id].get());
    }
}

llama_kv_cache_unified * PipelineExecutor::get_kv_cache_for_layer(int layer_id) {
    auto * mem = m_ctx.get_memory();

    // 1. Try ISWA (The path you are asking about)
    if (auto * iswa = dynamic_cast<llama_kv_cache_unified_iswa *>(mem)) {
        // Route to the correct internal unified cache
        if (m_ctx.model.hparams.is_swa(layer_id)) {
            return iswa->get_swa();
        } else {
            return iswa->get_base();
        }
    }

    // 2. Try Standard Unified
    if (auto * unified = dynamic_cast<llama_kv_cache_unified *>(mem)) {
        return unified;
    }

    return nullptr;
}

void PipelineExecutor::print_kv(int layer_num) {
    auto * kv_cache = get_kv_cache_for_layer(layer_num);
    auto src_k = kv_cache->get_k_l_gpu_tensor(layer_num);
    auto src_v = kv_cache->get_v_l_gpu_tensor(layer_num);
    int i = 0;
    {
        size_t    nbytes = ggml_nbytes(src_k);
        uint8_t * data   = (uint8_t *) malloc(nbytes);
        ggml_backend_tensor_get(src_k, data, 0, nbytes);
        size_t diff_src_k = (size_t) src_k->data - (size_t) ggml_backend_buffer_get_base(src_k->buffer);
        LLAMA_LOG_INFO("%s, tensor: %s, offset: %f, data[%d]: %d, size: %f, buffer-id: %d, backend: %s, base-ptr: %p, data-ptr: %p\n",
                       __func__,
                       src_k->name,
                       diff_src_k / (1024.0 * 1024.0),
                       i,
                       data[i],
                       nbytes / (1024.0 * 1024.0),
                       src_k->buffer_id,
                       ggml_backend_buffer_name(src_k->buffer),
                       ggml_backend_buffer_get_base(src_k->buffer),
                       src_k->data);
    }
    {
        size_t    nbytes = ggml_nbytes(src_v);
        uint8_t * data   = (uint8_t *) malloc(nbytes);
        ggml_backend_tensor_get(src_v, data, 0, nbytes);
        size_t diff_src_v = (size_t) src_v->data - (size_t) ggml_backend_buffer_get_base(src_v->buffer);
        LLAMA_LOG_INFO("%s, tensor: %s, offset: %f, data[%d]: %d, size: %f, buffer-id: %d, backend: %s, base-ptr: %p, data-ptr: %p\n",
                       __func__,
                       src_v->name,
                       diff_src_v / (1024.0 * 1024.0),
                       i,
                       data[i],
                       nbytes / (1024.0 * 1024.0),
                       src_v->buffer_id,
                       ggml_backend_buffer_name(src_v->buffer),
                       ggml_backend_buffer_get_base(src_v->buffer),
                       src_v->data);
    }
}

// Zero GPU KV tensors for a layer (eliminates NaN from old weight/scratch data in VRAM)
void PipelineExecutor::pipeline_zero_kv_gpu(int layer_num) {
    auto * kv_cache = get_kv_cache_for_layer(layer_num);
    if (!kv_cache || !kv_cache->get_pipe_shard()) return;
    auto * ps = kv_cache->get_pipe_shard();
    ggml_tensor * k = ps->k_gpu(layer_num);
    ggml_tensor * v = ps->v_gpu(layer_num);
    if (k && k->data) ggml_backend_tensor_memset(k, 0, 0, ggml_nbytes(k));
    if (v && v->data) ggml_backend_tensor_memset(v, 0, 0, ggml_nbytes(v));
}

// Upload KV cells needed for the given sequences (CPU -> GPU)
void PipelineExecutor::pipeline_upload_kv_cells_for_seqs(int layer_num, const std::vector<llama_seq_id> & seq_ids) {
    auto * kv_cache = get_kv_cache_for_layer(layer_num);
    if (!kv_cache || !kv_cache->get_pipe_shard()) {
        return;
    }

    auto * pipe_shard   = kv_cache->get_pipe_shard();
    
    // Use the destination tensor's backend for async upload (overlaps with current split's compute)
    ggml_tensor * dst_k = pipe_shard->k_gpu(layer_num);
    if (!dst_k) {
        return;
    }
    ggml_backend_t gpu = ggml_backend_sched_get_tensor_backend(m_ctx.sched.get(), dst_k);
    if (!gpu) {
        gpu = m_ctx.backends.at(0).get();  // fallback
    }
    
    const uint32_t n_stream = kv_cache->get_n_stream();
    const bool is_unified   = (n_stream == 1);

    if (is_unified) {
        // collect all cells from all sequences
        std::vector<uint32_t> all_cells;
        for (llama_seq_id seq_id : seq_ids) {
            std::vector<uint32_t> cells = kv_cache->get_cells_for_seq(seq_id);
            all_cells.insert(all_cells.end(), cells.begin(), cells.end());
        }

        if (all_cells.empty()) {
            return;
        }

        // remove duplicates
        std::sort(all_cells.begin(), all_cells.end());
        all_cells.erase(std::unique(all_cells.begin(), all_cells.end()), all_cells.end());

        // upload to tensor's stream (async with current compute!)
        pipe_shard->cells_upload(layer_num, all_cells, 0, gpu);
    } else {
        // Batch all streams together for fewer API calls
        std::unordered_map<uint32_t, std::vector<uint32_t>> stream_cells;

        for (llama_seq_id seq_id : seq_ids) {
            uint32_t stream_id = kv_cache->get_stream_for_seq(seq_id);
            std::vector<uint32_t> cells = kv_cache->get_cells_for_seq(seq_id);

            if (!cells.empty()) {
                auto & existing = stream_cells[stream_id];
                existing.insert(existing.end(), cells.begin(), cells.end());
            }
        }

        if (!stream_cells.empty()) {
            pipe_shard->cells_upload_batch(layer_num, stream_cells, gpu);
        }
    }
}

// Download specific KV cells that were written (GPU -> CPU)
// cell_indices are per-stream indices, seq_ids tells us which streams
// per_seq_write_cells maps each seq_id to only the cells it wrote this step
void PipelineExecutor::pipeline_download_kv_cells_for_seqs(int layer_num, const std::vector<llama_seq_id> & seq_ids, const std::vector<uint32_t> & cell_indices, const std::unordered_map<llama_seq_id, std::vector<uint32_t>> & per_seq_write_cells) {
    if (cell_indices.empty() || seq_ids.empty()) {
        return;
    }
    
    auto * kv_cache = get_kv_cache_for_layer(layer_num);
    if (!kv_cache || !kv_cache->get_pipe_shard()) {
        return;
    }

    auto * pipe_shard = kv_cache->get_pipe_shard();

    ggml_tensor * src_k = pipe_shard->k_gpu(layer_num);
    ggml_tensor * src_v = pipe_shard->v_gpu(layer_num);

    if (!src_k || !src_v) {
        return;
    }
    
    ggml_backend_t backend_k = ggml_backend_sched_get_tensor_backend(m_ctx.sched.get(), src_k);
    ggml_backend_t backend_v = ggml_backend_sched_get_tensor_backend(m_ctx.sched.get(), src_v);
    
    const uint32_t n_stream = kv_cache->get_n_stream();
    const bool is_unified   = (n_stream == 1);

    if (is_unified) {
        pipe_shard->cells_download(layer_num, cell_indices, 0, backend_k, backend_v);
    } else {
        // Multi-stream: download only the WRITTEN cells per-sequence (not the entire history).
        // per_seq_write_cells provides the exact cells each seq wrote this step.

        // Collect per-stream write cells and compute total written bytes
        std::unordered_map<uint32_t, std::vector<uint32_t>> stream_cells;
        for (llama_seq_id seq_id : seq_ids) {
            uint32_t stream_id = kv_cache->get_stream_for_seq(seq_id);

            auto it = per_seq_write_cells.find(seq_id);
            std::vector<uint32_t> cells;
            if (it != per_seq_write_cells.end() && !it->second.empty()) {
                cells = it->second;
            } else {
                cells = kv_cache->get_cells_for_seq(seq_id);
            }

            if (!cells.empty()) {
                auto & existing = stream_cells[stream_id];
                existing.insert(existing.end(), cells.begin(), cells.end());
            }
        }

        if (!stream_cells.empty()) {
            pipe_shard->cells_download_batch(layer_num, stream_cells, backend_k, backend_v);
        }
    }
}

// Prints the leaf nodes in a split.
void PipelineExecutor::print_leafs_in_split(ggml_cgraph * gf, int split_id) {
    GGML_ASSERT(m_active_plan != nullptr);
    const auto & plan = *m_active_plan;
    for (auto leaf : plan.pipeline_splits[split_id].leafs) {
        print_tensor_from_graph(gf, leaf->name, 1, false);
    }
}

void PipelineExecutor::print_split_details(ggml_cgraph * gf, int split_id) {
    const int sched_num_splits = ggml_backend_sched_get_num_splits(m_ctx.sched.get());
    if (split_id >= sched_num_splits) return;
    
    ggml_backend_t split_backend = ggml_backend_sched_get_split_backend(m_ctx.sched.get(), split_id);
    
    LLAMA_LOG_INFO("\n## SPLIT #%d: %s # inputs\n", split_id, ggml_backend_name(split_backend));

    GGML_ASSERT(m_active_plan != nullptr);
    const auto & plan = *m_active_plan;
    LLAMA_LOG_INFO("Leafs in split:\n");
    for (auto leaf : plan.pipeline_splits[split_id].leafs) {
        const char * name = ggml_get_name(leaf);
        const char * type_name = ggml_type_name(leaf->type);
        size_t size_kb = ggml_nbytes(leaf) / 1024;
        
        LLAMA_LOG_INFO("  Leaf: %-30s type=%6s size=%6zuK dims=[%lld, %lld, %lld, %lld]\n",
            name ? name : "unnamed",
            type_name,
            size_kb,
            leaf->ne[0], leaf->ne[1], leaf->ne[2], leaf->ne[3]);
    }
    LLAMA_LOG_INFO("\nNodes in split:\n");
    // Note: We don't have direct split node indices, but we can iterate through the graph
    // and check which backend each node belongs to
    for (int i = 0; i < gf->n_nodes; ++i) {
        ggml_tensor * node = gf->nodes[i];
        ggml_backend_t node_backend = ggml_backend_sched_get_tensor_backend(m_ctx.sched.get(), node);
        
        if (node_backend != split_backend) continue;
        
        const char * name = ggml_get_name(node);
        const char * op_name = ggml_op_name(node->op);
        const char * type_name = ggml_type_name(node->type);
        size_t size_kb = ggml_nbytes(node) / 1024;
        
        LLAMA_LOG_INFO("  Node #%4d (%-12s): op=%-12s type=%6s size=%6zuK dims=[%lld, %lld, %lld, %lld]\n",
            i,
            name ? name : "unnamed",
            op_name,
            type_name,
            size_kb,
            node->ne[0], node->ne[1], node->ne[2], node->ne[3]);
        
        // Print source tensors
        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            ggml_tensor * src = node->src[j];
            if (!src) continue;
            
            const char * src_name = ggml_get_name(src);
            const char * src_type = ggml_type_name(src->type);
            const char * src_op = ggml_op_name(src->op);
            size_t src_size_kb = ggml_nbytes(src) / 1024;
            
            LLAMA_LOG_INFO("    src%d (%-30s): op=%-12s type=%6s size=%6zuK dims=[%lld, %lld, %lld, %lld]\n",
                j,
                src_name ? src_name : "unnamed",
                src_op,
                src_type,
                src_size_kb,
                src->ne[0], src->ne[1], src->ne[2], src->ne[3]);
        }
    }
}

// TPS prediction with proper amortization for batch size
double PipelineExecutor::predict_tps(const ExecutionPlan & plan, uint32_t n_tokens, const char * label) {
    double total_time_ms = 0.0;
    // K+V bytes for ONE cell, ONE layer
    const uint32_t n_ctx = m_ctx.cparams.n_ctx;
    const size_t kv_cell_bytes = m_plan.kv_size_per_layer / n_ctx;

    if (label) {
        LLAMA_LOG_INFO("\n=== predict_tps: %s (n_tokens=%u, n_splits=%zu) ===\n", 
                       label, n_tokens, plan.pipeline_splits.size());
    }

    for (size_t i = 0; i < plan.pipeline_splits.size(); ++i) {
        const auto & split = plan.pipeline_splits[i];
        double split_time_ms = 0.0;
        
        // ============================================
        // 1. Calculate prefetch for NEXT split
        //    - Weight transfer: AMORTIZED by n_tokens
        //    - KV upload: NOT amortized (scales with n_tokens)
        // ============================================
        double prefetch_size = 0.0;
        if (i + 1 < plan.pipeline_splits.size()) {
            const auto & next_split = plan.pipeline_splits[i + 1];
            if (next_split.backend_id != GGML_CUDA_MAX_DEVICES && next_split.backend_id != 0) {
                prefetch_size = (double)next_split.size;
                prefetch_size += (double) (next_split.kv_layer_ids.size() * m_plan.kv_size_per_layer);
            }
        }

        // ============================================
        // 2. Compute time + transfer time (backend-specific)
        // ============================================
        SplitTimingResult split_result;
        double pcie_bw = 0.0;
        
        bool is_gpu = (split.backend_id != GGML_CUDA_MAX_DEVICES);
        
        //// Debug: print first split details
        //if (split.backend_id == GGML_CUDA_MAX_DEVICES && split.backend_id != 0) {
        //    m_predictor->debug_print_split(m_ctx.sched.get(), i, is_gpu, n_tokens);
        //}
        
        // async_copy = true when there's concurrent PCIe prefetch (uses eff_system_bw)
        // async_copy = false when no prefetch (uses peak_system_bw)
        bool async_copy = (prefetch_size > 0);
        split_result = m_predictor->predict(m_ctx.sched.get(), i, is_gpu, n_tokens, async_copy);
        pcie_bw = is_gpu ? m_predictor->stats().peak_pcie_bw : split_result.eff_pcie_bw;
        
        double compute_time_ms = split_result.time_ms;

        // ============================================
        // 3. Calculate prefetch time
        //    - Weight: amortized by n_tokens
        //    - KV upload: per-token cost (not amortized)
        // ============================================
        double prefetch_time_ms = 0.0;
        if (prefetch_size > 0 && pcie_bw > 0) {
            prefetch_time_ms = (prefetch_size / 1e9 / pcie_bw) * 1000.0;
        }

        // ============================================
        // 4. KV download (current split, after compute)
        //    - Only for GPU splits (PINNED and SHARDED)
        //    - NOT amortized (1 cell per token per layer)
        // ============================================
        double kv_download_time_ms = 0.0;
        if (split.backend_id != GGML_CUDA_MAX_DEVICES && pcie_bw > 0) {
            double kv_download_bytes = (double)(split.kv_layer_ids.size() * kv_cell_bytes * n_tokens);
            kv_download_time_ms = (kv_download_bytes / 1e9 / pcie_bw) * 1000.0;
        }

        // ============================================
        // 5. Split time = max(compute, prefetch) + kv_download
        // ============================================
        split_time_ms = std::max(compute_time_ms, prefetch_time_ms) + kv_download_time_ms;
        if (label) {
            LLAMA_LOG_INFO("  split[%zu] backend_id=%d: compute=%.3fms prefetch=%.3fms kv_dl=%.3fms -> %.3fms (pcie_bw=%.1f, ops=%d [cache=%d exact=%d interp=%d fall=%d])\n",
                            i, split.backend_id, compute_time_ms, prefetch_time_ms, kv_download_time_ms, 
                            split_time_ms, pcie_bw, split_result.op_count,
                            split_result.n_cache_hit, split_result.n_exact, split_result.n_interp, split_result.n_fallback);
        }
        
        total_time_ms += split_time_ms;
    }
    if (label) {
        LLAMA_LOG_INFO("  total_time=%.3fms -> TPS=%.2f\n", total_time_ms, total_time_ms > 0 ? 1000.0 / total_time_ms : 0.0);
    }
    
    if (total_time_ms <= 0.0) {
        return 0.0;
    }
    
    return (n_tokens * 1000.0) / total_time_ms;  // TPS
}
