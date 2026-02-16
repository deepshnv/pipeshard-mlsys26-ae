#pragma once

#include "ggml-cpp.h"

#include <vector>
#include <unordered_map>

struct llama_model;
struct llama_hparams;
struct llama_kv_cache_unified;
struct ggml_tensor;

//
// llama_kv_cache_pipe_shard
//

struct llama_kv_cache_pipe_shard {
    struct kv_layer {
        uint32_t il;

        ggml_tensor * k_cpu;
        ggml_tensor * v_cpu;
        ggml_tensor * k_gpu;
        ggml_tensor * v_gpu;

        std::vector<ggml_tensor *> k_stream_cpu;
        std::vector<ggml_tensor *> v_stream_cpu;
        std::vector<ggml_tensor *> k_stream_gpu;
        std::vector<ggml_tensor *> v_stream_gpu;
    };

    struct cell_range {
        uint32_t start;
        uint32_t count;
    };

    struct layer_kv_state {
        void * last_k_addr    = nullptr;
        void * last_v_addr    = nullptr;
        bool   needs_cpu_sync = false;
    };

    llama_kv_cache_pipe_shard() = default;
    ~llama_kv_cache_pipe_shard() = default;

    bool init(
              llama_kv_cache_unified * kv,
        const llama_model            & model,
        const llama_hparams          & hparams,
                         ggml_type     type_k,
                         ggml_type     type_v,
                          uint32_t     kv_size,
                          uint32_t     n_stream,
                          uint32_t     n_layer_cache,
                              bool     v_trans);

    void activate_cpu(llama_kv_cache_unified * kv, int32_t il);
    void activate_gpu(llama_kv_cache_unified * kv, int32_t il);

    ggml_tensor * k_cpu(int32_t il) const;
    ggml_tensor * v_cpu(int32_t il) const;
    ggml_tensor * k_gpu(int32_t il) const;
    ggml_tensor * v_gpu(int32_t il) const;

    size_t size_k_bytes_max() const;
    size_t size_v_bytes_max() const;

    void   clear_buffers();
    size_t get_total_size() const;

    bool   initialized() const { return !layers.empty(); }
    size_t n_layers()    const { return layers.size(); }

    // upload cells from CPU to GPU (single stream)
    void cells_upload(
                         int32_t     layer_id,
        const std::vector<uint32_t> & cells,
                        uint32_t     stream_id,
                  ggml_backend_t     gpu_backend);

    // upload cells from CPU to GPU (batched multi-stream)
    // stream_cells: map of stream_id -> cell indices
    void cells_upload_batch(
                         int32_t     layer_id,
        const std::unordered_map<uint32_t, std::vector<uint32_t>> & stream_cells,
                  ggml_backend_t     gpu_backend);

    // upload entire K and V tensors (all streams) in one shot each
    void cells_upload_full(
                         int32_t     layer_id,
                  ggml_backend_t     gpu_backend);

    // download cells from GPU to CPU (single stream)
    void cells_download(
                         int32_t     layer_id,
        const std::vector<uint32_t> & cells,
                        uint32_t     stream_id,
                  ggml_backend_t     backend_k,
                  ggml_backend_t     backend_v);

    // download cells from GPU to CPU (batched multi-stream)
    void cells_download_batch(
                         int32_t     layer_id,
        const std::unordered_map<uint32_t, std::vector<uint32_t>> & stream_cells,
                  ggml_backend_t     backend_k,
                  ggml_backend_t     backend_v);

    // download entire K and V tensors (all streams) in one shot each
    void cells_download_full(
                         int32_t     layer_id,
                  ggml_backend_t     backend_k,
                  ggml_backend_t     backend_v);

    // zero a stream region on GPU
    void stream_zero(int32_t layer_id, uint32_t stream_id);

    // Refresh GPU stream view data pointers after parent tensor's data is relocated
    void refresh_gpu_views(int32_t layer_id);

    void state_mark_dirty(int32_t layer_id);
    void state_mark_synced(int32_t layer_id);
    bool state_needs_download(int32_t layer_id) const;
    bool state_address_changed(int32_t layer_id, void * k_addr, void * v_addr) const;
    void state_update_addresses(int32_t layer_id, void * k_addr, void * v_addr);

private:
    std::vector<kv_layer> layers;

    std::vector<ggml_context_ptr>        ctxs;
    std::vector<ggml_backend_buffer_ptr> bufs;

    std::unordered_map<int32_t, int32_t> map_layer_ids;

    std::vector<layer_kv_state> kv_state;

    static std::vector<cell_range> cells_batch_ranges(const std::vector<uint32_t> & cells);
};