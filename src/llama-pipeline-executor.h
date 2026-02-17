#pragma once

#include "llama-pipeline.h"
#include "llama-memory.h"
#include "llama-benchmark.h"
#include "ggml-cpp.h"
#include "llama-kv-cache-unified.h"
#include "llama-kv-cache-unified-iswa.h"
#include <memory>

// Forward declarations
struct llama_context;
struct ggml_cgraph;
struct llama_ubatch;
struct ggml_tensor;
struct ggml_backend;
struct llama_memory_context_i;

// The PipelineExecutor class is responsible for managing the pipelined execution of the model.
// It creates an execution plan, manages the state of the pipeline, and orchestrates the
// computation of the graph in a pipelined fashion.
class PipelineExecutor {
public:
    PipelineExecutor(llama_context & ctx);
    ~PipelineExecutor();

    void prefetch_pinned_weights();
    void sort_weights_canonical(std::vector<ggml_tensor *> & weights);
    void allocate_kv_for_plan(ExecutionPlan & plan, exec_strategy strategy);
    void sync_kv_after_shift(const ExecutionPlan & plan);

    // Called once during context initialization to create the execution plan.
    void build_plan();

    // The main entry point for computing a graph in a pipelined fashion.
    // This is called from llama_context::graph_compute.
    ggml_status compute(ggml_cgraph * gf);

    // Hooks called from llama_context::decode during different stages of processing a batch.
    void set_scratch_offsets();
    void prepare_for_decode(int64_t n_tokens_all, llama_memory_context_i * mctx);
    void post_graph_build(ggml_cgraph * gf);

    // Called to allocate VRAM, especially when LoRA is enabled after initial setup.
    void alloc_vram();

    // Exposes the plan for other components (like llama_model) that need to know about device mapping.
    const PipelinePlan & get_plan() const { return m_plan; }

    double predict_tps(const ExecutionPlan & plan, uint32_t n_tokens = 1, const char * label = nullptr);

    void set_kv_status(bool status) { m_kv_updated = status; }
    bool get_kv_status() { return m_kv_updated; }

  private:
    llama_context & m_ctx; // A reference to the parent context to access shared state (model, kv_cache, scheduler, etc.)
    llama_memory_context_i * m_mctx = nullptr;

    PipelinePlan  m_plan;
    std::unique_ptr<BenchmarkPredictor> m_predictor;
    bool          m_kv_updated = false;  // Tracks if KV-shift happened and GPU KV needs sync

    size_t        m_current_tier     = 0;
    exec_strategy m_current_strategy = EXEC_STRATEGY_FULL_GPU;
    bool          m_use_single_plan  = false;

    // Active plan pointer - points to current execution plan (tier plan or dynamic)
    ExecutionPlan * m_active_plan = nullptr;

    std::set<ggml_tensor *> m_current_pinned_weights;

    std::vector<llama_seq_id> m_compute_seq_ids;
    std::vector<uint32_t>     m_compute_write_cells;
    std::unordered_map<llama_seq_id, std::vector<uint32_t>> m_compute_per_seq_write_cells;
        
    // --- Core Planning Methods ---
    void statically_allocate_weights(ExecutionPlan & plan, exec_strategy strategy);
    exec_strategy select_best_strategy(size_t tier);
    void switch_plan(const ExecutionPlan & old_plan, ExecutionPlan & new_plan, exec_strategy new_strategy);
    void restore_plan_alloc_state(const ExecutionPlan & plan);  // Restore per-plan allocator state
    void log_pipeline_summary() const;
    
    // Build or retrieve a plan for exact n_tokens (uses nearest tier's strategy)
    ExecutionPlan * get_or_build_plan_for_tokens(uint32_t n_tokens);

    // --- Tensor & Weight Management ---
    void pipeline_set_tensor_data(ggml_tensor * leaf, ggml_backend_t backend, bool is_async);
    void prefetch_split_weights(const ExecutionPlan & plan, int sched_split_id, int plan_split_id, int * last_idx);

    // --- KV Cache Management ---
    llama_kv_cache_unified * get_kv_cache_for_layer(int layer_num);
    void print_kv(int layer);
    // Multi-user cell-level KV copy
    void pipeline_zero_kv_gpu(int layer_num);
    void pipeline_upload_kv_cells_for_seqs(int layer_num, const std::vector<llama_seq_id> & seq_ids);
    void pipeline_download_kv_cells_for_seqs(int layer_num, const std::vector<llama_seq_id>& seq_ids, const std::vector<uint32_t>& cell_indices, const std::unordered_map<llama_seq_id, std::vector<uint32_t>>& per_seq_write_cells);

    // --- Phase Transition Management ---
    void   reload_leaf_buffer_ids();
    void   is_moe();

    // Plan building helpers
    void plan_apply_layer_strategies(ExecutionPlan & plan);
    void plan_apply_node_overrides(ggml_cgraph * gf, const ExecutionPlan & plan);
    size_t plan_measure_and_extract(ExecutionPlan & plan, ggml_cgraph * gf, bool save_splits);
    size_t plan_compute_requirements(ExecutionPlan & plan, uint32_t batch_size, bool save_splits, bool calc_tps, int32_t n_outputs_override = -1);
    int32_t plan_find_max_pinnable(ExecutionPlan & plan, layer_strategy base, layer_strategy pin, uint32_t batch_size, int n_shard_slots, int32_t n_outputs_override = -1);
    
    // Phase 2 progressive logic helpers
    size_t plan_measure_total(ExecutionPlan & plan, uint32_t batch_size, int n_shard_slots, int32_t n_outputs_override = -1);
    int32_t plan_shard_remaining_with_recalc(ExecutionPlan & plan, layer_strategy pin, layer_strategy shard, uint32_t batch_size, int32_t n_outputs_override = -1);
    void plan_apply_phase2_progressive(ExecutionPlan & plan, uint32_t batch_size, exec_strategy strategy, int32_t & n_pinned, int32_t n_outputs_override = -1);
    void plan_finalize(ExecutionPlan & plan, uint32_t batch_size, int n_shard_slots, int32_t n_pinned, bool calc_tps = true, int32_t n_outputs_override = -1);
    void plan_build_for_batch(ExecutionPlan & plan, uint32_t batch_size, exec_strategy strategy, bool calc_tps = true, int32_t n_outputs_override = -1);
    size_t plan_measure_scratch_only(const ExecutionPlan & plan, uint32_t batch_size, int32_t n_outputs_override = -1);
    
    void populate_plan_registry();
    
    // --- Debugging ---
    void print_leafs_in_split(ggml_cgraph * gf, int split_id);
    void print_split_details(ggml_cgraph * gf, int split_id);
};
