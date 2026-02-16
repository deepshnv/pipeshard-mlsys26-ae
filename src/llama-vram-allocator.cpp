#include "llama-pipeline-executor.h"
#include "llama-context.h"
#include "llama-model.h"
#include "llama-model-loader.h"
#include "llama-kv-cache-unified.h"
#include <algorithm>
#include <unordered_map>

//
// Weight category for canonical ordering
// Categories ensure maximum common prefix when switching strategies
//
// Category 0: Attention weights (always needed, highest reuse across strategies)
// Category 1: FFN norm + MoE router (small, always grouped with attention)
// Category 2: FFN computation weights (dense or MoE experts, can be offloaded)
// Category 3: Other (token_embd, output, rope_freqs)
//
static int get_weight_category(const char * name) {
    if (name == nullptr) {
        return 3;
    }

    // Category 0: Attention weights (always needed, highest reuse across strategies)
    if (strstr(name, "attn_q") || strstr(name, "attn_k") || strstr(name, "attn_v") ||
        strstr(name, "attn_o") || strstr(name, "attn_norm")) {
        return 0;
    }

    // Category 1: FFN norm + MoE router (small, always grouped with attention)
    if (strstr(name, "ffn_norm") || strstr(name, "ffn_gate_inp")) {
        return 1;
    }

    // Category 2: FFN computation weights (dense or MoE experts, can be offloaded)
    // Matches: ffn_up, ffn_down, ffn_gate (dense)
    //          ffn_up_exps, ffn_down_exps, ffn_gate_exps (MoE experts)
    //          ffn_up_shexp, ffn_down_shexp, ffn_gate_shexp (MoE shared experts)
    if (strstr(name, "ffn_up") || strstr(name, "ffn_down") || strstr(name, "ffn_gate")) {
        return 2;
    }

    // Category 3: Other (token_embd, output, rope_freqs)
    return 3;
}

//
// sort_weights_canonical - sort weights by category, then by file offset
// This ensures maximum common prefix when switching between strategies
//
void PipelineExecutor::sort_weights_canonical(std::vector<ggml_tensor *> & weights) {
    // Pre-compute categories to avoid repeated strstr() calls during sort
    // (Sort comparator is called O(n log n) times, each with up to 11 strstr)
    std::unordered_map<ggml_tensor *, int> category_cache;
    category_cache.reserve(weights.size());
    for (ggml_tensor * t : weights) {
        category_cache[t] = get_weight_category(ggml_get_name(t));
    }

    std::sort(weights.begin(), weights.end(),
    [&](ggml_tensor * a, ggml_tensor * b) {
        // Primary: sort by category (using cached values)
        int cat_a = category_cache[a];
        int cat_b = category_cache[b];
        if (cat_a != cat_b) {
            return cat_a < cat_b;
        }

        // Secondary: sort by file offset (for streaming efficiency)
        const auto & wa = m_ctx.model.ml->require_weight(ggml_get_name(a));
        const auto & wb = m_ctx.model.ml->require_weight(ggml_get_name(b));
        if (wa.idx != wb.idx) {
            return wa.idx < wb.idx;
        }
        return wa.offs < wb.offs;
    });
}

//
// statically_allocate_weights - unified weight allocation for any plan
//
void PipelineExecutor::statically_allocate_weights(ExecutionPlan & plan, exec_strategy strategy) {
    const strategy_config & cfg = STRATEGY_CONFIGS[strategy];
    const int n_shard_slots = cfg.n_shard_slots;

    // Get VRAM buffer properties
    ggml_backend_buffer_t vram_buffer = ggml_backend_sched_get_buffer(m_ctx.sched.get(), m_ctx.backends.at(0).get());
    const size_t align     = ggml_backend_buffer_get_alignment(vram_buffer);
    void *       base_addr = ggml_backend_buffer_get_base(vram_buffer);

    // Phase 1: Allocate pinned weights using sorted_pinned_weights
    size_t pinned_offset = 0;
    for (ggml_tensor * tensor : plan.sorted_pinned_weights) {
        const size_t tensor_size = ggml_backend_buft_get_alloc_size(tensor->buft, tensor);
        const size_t padded_size = GGML_PAD(tensor_size, align);

        size_t final_offset = GGML_PAD(pinned_offset, align);
        void * addr = (char *) base_addr + final_offset;

        if (ggml_backend_tensor_alloc(vram_buffer, tensor, addr) != GGML_STATUS_SUCCESS) {
            LLAMA_LOG_ERROR("%s: failed to allocate pinned tensor %s\n", __func__, tensor->name);
            throw std::runtime_error("failed to allocate weight tensor");
        }

        pinned_offset = final_offset + padded_size;
    }

    // Phase 2: Handle CPU and sharded splits
    const size_t sharded_weights_base_offset = plan.pinned_weights_vram_req;

    for (size_t split_idx = 0; split_idx < plan.pipeline_splits.size(); ++split_idx) {
        auto & split = plan.pipeline_splits[split_idx];

        if (split.leafs.empty()) {
            continue;
        }

        // CPU splits: reallocate tensors to CPU buffer
        if (split.backend_type == CPU) {
            for (ggml_tensor * tensor : split.leafs) {
                if (tensor->buft == nullptr) {
                    continue;
                }
                const auto * weight = m_ctx.model.ml->get_weight(ggml_get_name(tensor));
                if (weight == nullptr) {
                    continue;
                }

                uint8_t * data = nullptr;
                auto hybrid_it = m_ctx.model.hybrid_tensor_data.find(tensor);
                if (hybrid_it != m_ctx.model.hybrid_tensor_data.end()) {
                    data = (uint8_t *) hybrid_it->second;
                } else if (m_ctx.model.ml->use_mmap) {
                    data = (uint8_t *) m_ctx.model.get_mapping_addr(weight->idx) + weight->offs;
                } else {
                    data = (uint8_t *) m_ctx.model.ml->get_tensor_data(ggml_get_name(tensor));
                }
                ggml_backend_tensor_realloc_to_cpu(m_ctx.model.get_cpu_buf(), tensor, data);
            }
            continue;
        }

        // Skip GPU_PINNED - already handled in Phase 1
        if (split.backend_type == GPU_PINNED) {
            continue;
        }

        // GPU_SHARDED: cycle through slots
        const int slot_idx = (n_shard_slots > 0) ? (split_idx % n_shard_slots) : 0;
        const size_t slot_size = (n_shard_slots > 0) ? (plan.sharded_weights_vram_req / n_shard_slots) : 0;
        size_t split_base_offset = sharded_weights_base_offset + (slot_idx * slot_size);

        size_t offset_in_split = 0;
        for (ggml_tensor * tensor : split.leafs) {
            const size_t tensor_size = ggml_backend_buft_get_alloc_size(tensor->buft, tensor);
            const size_t padded_size = GGML_PAD(tensor_size, align);

            size_t final_offset = GGML_PAD(split_base_offset + offset_in_split, align);
            void * addr = (char *) base_addr + final_offset;

            if (ggml_backend_tensor_alloc(vram_buffer, tensor, addr) != GGML_STATUS_SUCCESS) {
                LLAMA_LOG_ERROR("%s: failed to allocate tensor %s\n", __func__, tensor->name);
                throw std::runtime_error("failed to allocate weight tensor");
            }

            offset_in_split += padded_size;
        }
    }
}
//
// allocate_kv_for_plan - allocate KV cache tensors for a plan
//
void PipelineExecutor::allocate_kv_for_plan(ExecutionPlan & plan, exec_strategy strategy) {
    // Get n_shard_slots from strategy config
    const strategy_config & cfg = STRATEGY_CONFIGS[strategy];
    const size_t n_shard_slots = cfg.n_shard_slots;

    ggml_backend_buffer_t vram_buffer = ggml_backend_sched_get_buffer(m_ctx.sched.get(), m_ctx.backends.at(0).get());
    void * kv_base_addr = (char *)ggml_backend_buffer_get_base(vram_buffer) + ggml_backend_buffer_get_size(vram_buffer) - plan.kv_vram_req;

    size_t gpu_layer_offset_idx = 0;
    size_t shard_idx = 0;
    
    for (const auto & split : plan.pipeline_splits) {
        for (int layer_id : split.kv_layer_ids) {
            auto * kv_cache = get_kv_cache_for_layer(layer_id);
            ggml_tensor * k = kv_cache->get_k_l_gpu_tensor(layer_id);
            ggml_tensor * v = kv_cache->get_v_l_gpu_tensor(layer_id);
            const size_t k_size_per_layer = kv_cache->max_layer_size_k_bytes();
            size_t offset;
            
            if (split.backend_id == 0) {
                // Pinned: sequential allocation
                kv_cache->activate_gpu_kv(layer_id);
                offset = gpu_layer_offset_idx * m_plan.kv_size_per_layer;
                ggml_backend_tensor_alloc(vram_buffer, k, (char *)kv_base_addr + offset);
                ggml_backend_tensor_alloc(vram_buffer, v, (char *)kv_base_addr + offset + k_size_per_layer);
                kv_cache->get_pipe_shard()->refresh_gpu_views(layer_id);
                gpu_layer_offset_idx++;
            } else if (split.backend_id != GGML_CUDA_MAX_DEVICES) {
                // Sharded: use modulo for cycling through shard slots
                kv_cache->activate_gpu_kv(layer_id);
                offset = (gpu_layer_offset_idx + shard_idx % n_shard_slots) * m_plan.kv_size_per_layer;
                ggml_backend_tensor_alloc(vram_buffer, k, (char *)kv_base_addr + offset);
                ggml_backend_tensor_alloc(vram_buffer, v, (char *)kv_base_addr + offset + k_size_per_layer);
                kv_cache->get_pipe_shard()->refresh_gpu_views(layer_id);
                shard_idx++;
            } else {
                // CPU backend
                kv_cache->activate_cpu_kv(layer_id);
            }
        }
    }
}

void PipelineExecutor::sync_kv_after_shift(const ExecutionPlan & plan) {
    const int32_t n_layer = m_ctx.model.hparams.n_layer;
    std::vector<llama_seq_id> seq_ids = m_mctx->get_seq_ids();

    for (int32_t il = 0; il < n_layer; il++) {
        // Check if this layer has GPU-pinned KV in current plan
        bool is_pinned = (plan.strategies[il] == LAYER_STRATEGY_ALL_GPU_PINNED ||
                          plan.strategies[il] == LAYER_STRATEGY_ATTN_GPU_PINNED ||
                          plan.strategies[il] == LAYER_STRATEGY_ATTN_PINNED_FFN_SHARDED);

        if (!is_pinned) {
            continue;
        }

        auto * kv_cache = get_kv_cache_for_layer(il);
        if (!kv_cache || !kv_cache->get_pipe_shard()) {
            continue;
        }

        // Reactivate GPU KV (was set to CPU during shift)
        kv_cache->activate_gpu_kv(il);

        // Upload all active cells from CPU to GPU
        pipeline_upload_kv_cells_for_seqs(il, seq_ids);
    }

    LLAMA_LOG_DEBUG("%s: synced GPU KV after shift\n", __func__);
}

//
// pipeline_set_tensor_data - load weight data from CPU to GPU
// Supports both synchronous and asynchronous transfers
//
void PipelineExecutor::pipeline_set_tensor_data(ggml_tensor * leaf, ggml_backend_t backend, bool is_async) {
    if (leaf->flags & GGML_TENSOR_FLAG_LORA) {
        for (const auto & lora : m_ctx.loras) {
            if (lora.first->gpu_to_cpu_lora_tensor_map.count(leaf)) {
                struct ggml_tensor * src_cpu_lora = lora.first->gpu_to_cpu_lora_tensor_map[leaf];
                if (is_async) {
                    ggml_backend_tensor_set_async(backend, leaf, src_cpu_lora->data, 0, ggml_nbytes(leaf));
                } else {
                    ggml_backend_tensor_set(leaf, src_cpu_lora->data, 0, ggml_nbytes(leaf));
                }
            }
        }
        return;
    }

    const auto * weight = m_ctx.model.ml->get_weight(ggml_get_name(leaf));
    if (weight == nullptr) {
        return;
    }

    uint8_t * data = nullptr;
    auto hybrid_it = m_ctx.model.hybrid_tensor_data.find(leaf);
    if (hybrid_it != m_ctx.model.hybrid_tensor_data.end()) {
        // Hybrid mode: use pre-computed pointer (pinned or mmap)
        data = (uint8_t *) hybrid_it->second;
    } else if (m_ctx.model.ml->use_mmap) {
        // Legacy mmap path
        data = (uint8_t *) m_ctx.model.get_mapping_addr(weight->idx) + weight->offs;
    } else {
        // Legacy non-mmap path
        data = (uint8_t *) m_ctx.model.ml->get_tensor_data(ggml_get_name(leaf));
    }

    if (is_async) {
        ggml_backend_tensor_set_async(backend, leaf, data, 0, ggml_nbytes(leaf));
    } else {
        ggml_backend_tensor_set(leaf, data, 0, ggml_nbytes(leaf));
    }
    //size_t diff_src_k = (size_t) leaf->data - (size_t) ggml_backend_buffer_get_base(leaf->buffer);
    //LLAMA_LOG_INFO("%s, %d, tensor->name: %s, diff: %f\n", __func__, __LINE__, leaf->name, diff_src_k/1024.0/1024.0);
}

