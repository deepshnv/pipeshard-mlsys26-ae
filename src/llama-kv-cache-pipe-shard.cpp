#include "llama-kv-cache-pipe-shard.h"

#include "llama-impl.h"
#include "llama-model.h"
#include "llama-kv-cache-unified.h"

#include <algorithm>
#include <unordered_map>

//
// llama_kv_cache_pipe_shard
//

bool llama_kv_cache_pipe_shard::init(
          llama_kv_cache_unified * kv,
    const llama_model            & model,
    const llama_hparams          & hparams,
                     ggml_type     type_k,
                     ggml_type     type_v,
                      uint32_t     kv_size,
                      uint32_t     n_stream,
                      uint32_t     n_layer_cache,
                          bool     v_trans) {

    layers.clear();
    ctxs.clear();
    bufs.clear();
    map_layer_ids.clear();

    // GPU tensor definitions
    {
        ggml_init_params params = {
            /*.mem_size   =*/ size_t(2u * (1 + n_stream) * n_layer_cache * ggml_tensor_overhead()),
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true,
        };

        ggml_context * ctx_gpu = ggml_init(params);
        if (!ctx_gpu) {
            LLAMA_LOG_ERROR("%s: failed to create GPU context for KV cache\n", __func__);
            return false;
        }
        ctxs.emplace_back(ctx_gpu);

        auto * main_gpu_dev = model.dev_layer(0);
        auto * buft_gpu     = ggml_backend_dev_buffer_type(main_gpu_dev);

        ggml_backend_buffer_t buf_gpu_def = ggml_backend_buft_alloc_buffer(buft_gpu, 0);
        bufs.emplace_back(buf_gpu_def);

        for (uint32_t il = 0; il < n_layer_cache; il++) {
            const uint32_t n_embd_k_gqa = hparams.n_embd_k_gqa(il);
            const uint32_t n_embd_v_gqa = !v_trans ? hparams.n_embd_v_gqa(il) : hparams.n_embd_v_gqa_max();

            ggml_tensor * k = ggml_new_tensor_3d(ctx_gpu, type_k, n_embd_k_gqa, kv_size, n_stream);
            ggml_tensor * v = ggml_new_tensor_3d(ctx_gpu, type_v, n_embd_v_gqa, kv_size, n_stream);

            ggml_format_name(k, "cache_k_l%d", il);
            ggml_format_name(v, "cache_v_l%d", il);

            k->buft   = ggml_backend_dev_buffer_type(model.dev_layer(il));
            v->buft   = ggml_backend_dev_buffer_type(model.dev_layer(il));
            k->buffer = buf_gpu_def;
            v->buffer = buf_gpu_def;
            k->flags |= GGML_TENSOR_FLAG_KV;
            v->flags |= GGML_TENSOR_FLAG_KV;

            std::vector<ggml_tensor *> k_stream_gpu;
            std::vector<ggml_tensor *> v_stream_gpu;

            for (uint32_t s = 0; s < n_stream; ++s) {
                k_stream_gpu.push_back(ggml_view_2d(ctx_gpu, k, n_embd_k_gqa, kv_size, k->nb[1], s * k->nb[2]));
                v_stream_gpu.push_back(ggml_view_2d(ctx_gpu, v, n_embd_v_gqa, kv_size, v->nb[1], s * v->nb[2]));
            }

            map_layer_ids[il] = layers.size();

            kv_layer layer;
            layer.il           = il;
            layer.k_gpu        = k;
            layer.v_gpu        = v;
            layer.k_stream_gpu = std::move(k_stream_gpu);
            layer.v_stream_gpu = std::move(v_stream_gpu);
            layer.k_cpu        = nullptr;
            layer.v_cpu        = nullptr;

            layers.push_back(std::move(layer));
        }
    }

    // populate kv->layers from GPU tensors
    kv->layers.resize(layers.size());
    for (size_t i = 0; i < layers.size(); ++i) {
        kv->layers[i].il       = layers[i].il;
        kv->layers[i].k        = layers[i].k_gpu;
        kv->layers[i].v        = layers[i].v_gpu;
        kv->layers[i].k_stream = layers[i].k_stream_gpu;
        kv->layers[i].v_stream = layers[i].v_stream_gpu;

        kv->map_layer_ids[layers[i].il] = i;
    }

    // CPU-pinned host tensors
    {
        ggml_init_params params = {
            /*.mem_size   =*/ size_t(2u * (1 + n_stream) * n_layer_cache * ggml_tensor_overhead()),
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true,
        };

        ggml_context * ctx_cpu = ggml_init(params);
        if (!ctx_cpu) {
            LLAMA_LOG_ERROR("%s: failed to create CPU context for KV cache\n", __func__);
            return false;
        }
        ctxs.emplace_back(ctx_cpu);

        auto * main_gpu_dev    = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
        auto * buft_cpu_pinned = ggml_backend_dev_host_buffer_type(main_gpu_dev);

        for (size_t i = 0; i < layers.size(); ++i) {
            uint32_t il = layers[i].il;

            const uint32_t n_embd_k_gqa = hparams.n_embd_k_gqa(il);
            const uint32_t n_embd_v_gqa = !v_trans ? hparams.n_embd_v_gqa(il) : hparams.n_embd_v_gqa_max();

            ggml_tensor * k = ggml_new_tensor_3d(ctx_cpu, type_k, n_embd_k_gqa, kv_size, n_stream);
            ggml_tensor * v = ggml_new_tensor_3d(ctx_cpu, type_v, n_embd_v_gqa, kv_size, n_stream);

            ggml_format_name(k, "cache_k_cpu_l%d", (int)i);
            ggml_format_name(v, "cache_v_cpu_l%d", (int)i);

            k->buft = buft_cpu_pinned;
            v->buft = buft_cpu_pinned;

            layers[i].k_cpu = k;
            layers[i].v_cpu = v;

            for (uint32_t s = 0; s < n_stream; ++s) {
                layers[i].k_stream_cpu.push_back(ggml_view_2d(ctx_cpu, k, n_embd_k_gqa, kv_size, k->nb[1], s * k->nb[2]));
                layers[i].v_stream_cpu.push_back(ggml_view_2d(ctx_cpu, v, n_embd_v_gqa, kv_size, v->nb[1], s * v->nb[2]));
            }
        }

        ggml_backend_buffer_t buf_cpu = ggml_backend_alloc_ctx_tensors_from_buft(ctx_cpu, buft_cpu_pinned);
        if (!buf_cpu) {
            LLAMA_LOG_ERROR("%s: failed to allocate CPU-pinned buffer for KV cache\n", __func__);
            return false;
        }

        ggml_backend_buffer_clear(buf_cpu, 0);
        bufs.emplace_back(buf_cpu);
    }

    kv_state.resize(layers.size());

    LLAMA_LOG_INFO("%s: initialized pipe shard KV cache with %zu layers\n", __func__, layers.size());

    return true;
}

void llama_kv_cache_pipe_shard::activate_cpu(llama_kv_cache_unified * kv, int32_t il) {
    auto it = map_layer_ids.find(il);
    if (it == map_layer_ids.end()) {
        return;
    }

    const size_t idx = it->second;

    kv->layers[idx].k        = layers[idx].k_cpu;
    kv->layers[idx].v        = layers[idx].v_cpu;
    kv->layers[idx].k_stream = layers[idx].k_stream_cpu;
    kv->layers[idx].v_stream = layers[idx].v_stream_cpu;
}

void llama_kv_cache_pipe_shard::activate_gpu(llama_kv_cache_unified * kv, int32_t il) {
    auto it = map_layer_ids.find(il);
    if (it == map_layer_ids.end()) {
        return;
    }

    const size_t idx = it->second;

    kv->layers[idx].k        = layers[idx].k_gpu;
    kv->layers[idx].v        = layers[idx].v_gpu;
    kv->layers[idx].k_stream = layers[idx].k_stream_gpu;
    kv->layers[idx].v_stream = layers[idx].v_stream_gpu;
}

ggml_tensor * llama_kv_cache_pipe_shard::k_cpu(int32_t il) const {
    auto it = map_layer_ids.find(il);
    if (it == map_layer_ids.end()) {
        return nullptr;
    }
    return layers[it->second].k_cpu;
}

ggml_tensor * llama_kv_cache_pipe_shard::v_cpu(int32_t il) const {
    auto it = map_layer_ids.find(il);
    if (it == map_layer_ids.end()) {
        return nullptr;
    }
    return layers[it->second].v_cpu;
}

ggml_tensor * llama_kv_cache_pipe_shard::k_gpu(int32_t il) const {
    auto it = map_layer_ids.find(il);
    if (it == map_layer_ids.end()) {
        return nullptr;
    }
    return layers[it->second].k_gpu;
}

ggml_tensor * llama_kv_cache_pipe_shard::v_gpu(int32_t il) const {
    auto it = map_layer_ids.find(il);
    if (it == map_layer_ids.end()) {
        return nullptr;
    }
    return layers[it->second].v_gpu;
}

size_t llama_kv_cache_pipe_shard::size_k_bytes_max() const {
    size_t max_size = 0;
    for (const auto & layer : layers) {
        if (layer.k_gpu) {
            max_size = std::max(max_size, ggml_nbytes(layer.k_gpu));
        }
    }
    return max_size;
}

size_t llama_kv_cache_pipe_shard::size_v_bytes_max() const {
    size_t max_size = 0;
    for (const auto & layer : layers) {
        if (layer.v_gpu) {
            max_size = std::max(max_size, ggml_nbytes(layer.v_gpu));
        }
    }
    return max_size;
}

void llama_kv_cache_pipe_shard::clear_buffers() {
    for (auto & buf : bufs) {
        ggml_backend_buffer_clear(buf.get(), 0);
    }
}

size_t llama_kv_cache_pipe_shard::get_total_size() const {
    size_t size = 0;
    for (const auto & buf : bufs) {
        size += ggml_backend_buffer_get_size(buf.get());
    }
    return size;
}

std::vector<llama_kv_cache_pipe_shard::cell_range> llama_kv_cache_pipe_shard::cells_batch_ranges(
        const std::vector<uint32_t> & cells) {
    std::vector<cell_range> ranges;
    if (cells.empty()) {
        return ranges;
    }

    uint32_t range_start = cells[0];
    uint32_t range_count = 1;

    for (size_t i = 1; i < cells.size(); ++i) {
        if (cells[i] == range_start + range_count) {
            range_count++;
        } else {
            ranges.push_back({range_start, range_count});
            range_start = cells[i];
            range_count = 1;
        }
    }
    ranges.push_back({range_start, range_count});

    return ranges;
}

void llama_kv_cache_pipe_shard::stream_zero(int32_t layer_id, uint32_t stream_id) {
    auto it = map_layer_ids.find(layer_id);
    if (it == map_layer_ids.end()) {
        return;
    }

    const size_t idx = it->second;

    ggml_tensor * k = layers[idx].k_gpu;
    ggml_tensor * v = layers[idx].v_gpu;

    if (!k || !v) {
        return;
    }

    const size_t kv_size       = k->ne[1];
    const size_t k_row_bytes   = k->ne[0] * ggml_element_size(k);
    const size_t v_row_bytes   = v->ne[0] * ggml_element_size(v);
    const size_t k_stream_size = kv_size * k_row_bytes;
    const size_t v_stream_size = kv_size * v_row_bytes;

    size_t k_offset = stream_id * k_stream_size;
    size_t v_offset = stream_id * v_stream_size;

    ggml_backend_tensor_memset(k, 0, k_offset, k_stream_size);
    ggml_backend_tensor_memset(v, 0, v_offset, v_stream_size);
}

void llama_kv_cache_pipe_shard::cells_upload(
                     int32_t     layer_id,
    const std::vector<uint32_t> & cells,
                    uint32_t     stream_id,
              ggml_backend_t     gpu_backend) {
    if (cells.empty()) {
        return;
    }

    auto it = map_layer_ids.find(layer_id);
    if (it == map_layer_ids.end()) {
        return;
    }

    const size_t idx = it->second;

    ggml_tensor * src_k = layers[idx].k_cpu;
    ggml_tensor * dst_k = layers[idx].k_gpu;
    ggml_tensor * src_v = layers[idx].v_cpu;
    ggml_tensor * dst_v = layers[idx].v_gpu;

    if (!src_k || !dst_k || !src_v || !dst_v) {
        return;
    }

    const size_t elem_size     = ggml_element_size(src_k);
    const size_t k_row_bytes   = src_k->ne[0] * elem_size;
    const size_t v_row_bytes   = src_v->ne[0] * ggml_element_size(src_v);
    const size_t kv_size       = src_k->ne[1];
    const size_t k_stream_size = kv_size * k_row_bytes;
    const size_t v_stream_size = kv_size * v_row_bytes;

    size_t k_stream_offset = stream_id * k_stream_size;
    size_t v_stream_offset = stream_id * v_stream_size;

    std::vector<uint32_t> sorted_cells = cells;
    std::sort(sorted_cells.begin(), sorted_cells.end());

    auto ranges = cells_batch_ranges(sorted_cells);

    for (const auto & range : ranges) {
        size_t k_offset = k_stream_offset + range.start * k_row_bytes;
        size_t k_size   = range.count * k_row_bytes;
        ggml_backend_tensor_set_async(gpu_backend, dst_k, (char *)src_k->data + k_offset, k_offset, k_size);

        size_t v_offset = v_stream_offset + range.start * v_row_bytes;
        size_t v_size   = range.count * v_row_bytes;
        ggml_backend_tensor_set_async(gpu_backend, dst_v, (char *)src_v->data + v_offset, v_offset, v_size);
    }
}

void llama_kv_cache_pipe_shard::cells_download(
                     int32_t     layer_id,
    const std::vector<uint32_t> & cells,
                    uint32_t     stream_id,
              ggml_backend_t     backend_k,
              ggml_backend_t     backend_v) {
    if (cells.empty()) {
        return;
    }

    auto it = map_layer_ids.find(layer_id);
    if (it == map_layer_ids.end()) {
        return;
    }

    const size_t idx = it->second;

    ggml_tensor * src_k = layers[idx].k_gpu;
    ggml_tensor * dst_k = layers[idx].k_cpu;
    ggml_tensor * src_v = layers[idx].v_gpu;
    ggml_tensor * dst_v = layers[idx].v_cpu;

    if (!src_k || !dst_k || !src_v || !dst_v) {
        return;
    }

    const size_t elem_size     = ggml_element_size(src_k);
    const size_t k_row_bytes   = src_k->ne[0] * elem_size;
    const size_t v_row_bytes   = src_v->ne[0] * ggml_element_size(src_v);
    const size_t kv_size       = src_k->ne[1];
    const size_t k_stream_size = kv_size * k_row_bytes;
    const size_t v_stream_size = kv_size * v_row_bytes;

    size_t k_stream_offset = stream_id * k_stream_size;
    size_t v_stream_offset = stream_id * v_stream_size;

    std::vector<uint32_t> sorted_cells = cells;
    std::sort(sorted_cells.begin(), sorted_cells.end());

    auto ranges = cells_batch_ranges(sorted_cells);

    for (const auto & range : ranges) {
        size_t k_offset = k_stream_offset + range.start * k_row_bytes;
        size_t k_size   = range.count * k_row_bytes;
        ggml_backend_tensor_get_async(backend_k, src_k, (char *)dst_k->data + k_offset, k_offset, k_size);

        size_t v_offset = v_stream_offset + range.start * v_row_bytes;
        size_t v_size   = range.count * v_row_bytes;
        ggml_backend_tensor_get_async(backend_v, src_v, (char *)dst_v->data + v_offset, v_offset, v_size);
    }

    state_mark_synced(layer_id);
}

// Batched upload: processes all streams in one pass with fewer API calls
void llama_kv_cache_pipe_shard::cells_upload_batch(
                     int32_t     layer_id,
    const std::unordered_map<uint32_t, std::vector<uint32_t>> & stream_cells,
              ggml_backend_t     gpu_backend) {
    if (stream_cells.empty()) {
        return;
    }

    auto it = map_layer_ids.find(layer_id);
    if (it == map_layer_ids.end()) {
        return;
    }

    const size_t idx = it->second;

    ggml_tensor * src_k = layers[idx].k_cpu;
    ggml_tensor * dst_k = layers[idx].k_gpu;
    ggml_tensor * src_v = layers[idx].v_cpu;
    ggml_tensor * dst_v = layers[idx].v_gpu;

    if (!src_k || !dst_k || !src_v || !dst_v) {
        return;
    }

    const size_t elem_size     = ggml_element_size(src_k);
    const size_t k_row_bytes   = src_k->ne[0] * elem_size;
    const size_t v_row_bytes   = src_v->ne[0] * ggml_element_size(src_v);
    const size_t kv_size       = src_k->ne[1];
    const size_t k_stream_size = kv_size * k_row_bytes;
    const size_t v_stream_size = kv_size * v_row_bytes;

    // Collect all transfer operations across all streams
    struct transfer_op {
        size_t offset;
        size_t size;
    };
    std::vector<transfer_op> k_ops;
    std::vector<transfer_op> v_ops;

    for (const auto & [stream_id, cells] : stream_cells) {
        if (cells.empty()) continue;

        size_t k_stream_offset = stream_id * k_stream_size;
        size_t v_stream_offset = stream_id * v_stream_size;

        std::vector<uint32_t> sorted_cells = cells;
        std::sort(sorted_cells.begin(), sorted_cells.end());
        auto ranges = cells_batch_ranges(sorted_cells);

        for (const auto & range : ranges) {
            k_ops.push_back({k_stream_offset + range.start * k_row_bytes, range.count * k_row_bytes});
            v_ops.push_back({v_stream_offset + range.start * v_row_bytes, range.count * v_row_bytes});
        }
    }

    // Phase 3: Execute all transfers (async, will overlap)
    for (const auto & op : k_ops) {
        ggml_backend_tensor_set_async(gpu_backend, dst_k, (char *)src_k->data + op.offset, op.offset, op.size);
    }
    for (const auto & op : v_ops) {
        ggml_backend_tensor_set_async(gpu_backend, dst_v, (char *)src_v->data + op.offset, op.offset, op.size);
    }
}

// Full upload: transfers entire K and V tensors (all streams) in 2 async calls
void llama_kv_cache_pipe_shard::cells_upload_full(
                     int32_t     layer_id,
              ggml_backend_t     gpu_backend) {
    auto it = map_layer_ids.find(layer_id);
    if (it == map_layer_ids.end()) {
        return;
    }

    const size_t idx = it->second;

    ggml_tensor * src_k = layers[idx].k_cpu;
    ggml_tensor * dst_k = layers[idx].k_gpu;
    ggml_tensor * src_v = layers[idx].v_cpu;
    ggml_tensor * dst_v = layers[idx].v_gpu;

    if (!src_k || !dst_k || !src_v || !dst_v) {
        return;
    }

    const size_t k_bytes = ggml_nbytes(src_k);
    const size_t v_bytes = ggml_nbytes(src_v);

    ggml_backend_tensor_set_async(gpu_backend, dst_k, src_k->data, 0, k_bytes);
    ggml_backend_tensor_set_async(gpu_backend, dst_v, src_v->data, 0, v_bytes);
}

// Batched download: processes all streams in one pass
void llama_kv_cache_pipe_shard::cells_download_batch(
                     int32_t     layer_id,
    const std::unordered_map<uint32_t, std::vector<uint32_t>> & stream_cells,
              ggml_backend_t     backend_k,
              ggml_backend_t     backend_v) {
    if (stream_cells.empty()) {
        return;
    }

    auto it = map_layer_ids.find(layer_id);
    if (it == map_layer_ids.end()) {
        return;
    }

    const size_t idx = it->second;

    ggml_tensor * src_k = layers[idx].k_gpu;
    ggml_tensor * dst_k = layers[idx].k_cpu;
    ggml_tensor * src_v = layers[idx].v_gpu;
    ggml_tensor * dst_v = layers[idx].v_cpu;

    if (!src_k || !dst_k || !src_v || !dst_v) {
        return;
    }

    const size_t elem_size     = ggml_element_size(src_k);
    const size_t k_row_bytes   = src_k->ne[0] * elem_size;
    const size_t v_row_bytes   = src_v->ne[0] * ggml_element_size(src_v);
    const size_t kv_size       = src_k->ne[1];
    const size_t k_stream_size = kv_size * k_row_bytes;
    const size_t v_stream_size = kv_size * v_row_bytes;

    // Collect all transfer operations
    struct transfer_op {
        size_t offset;
        size_t size;
    };
    std::vector<transfer_op> k_ops;
    std::vector<transfer_op> v_ops;

    for (const auto & [stream_id, cells] : stream_cells) {
        if (cells.empty()) continue;

        size_t k_stream_offset = stream_id * k_stream_size;
        size_t v_stream_offset = stream_id * v_stream_size;

        std::vector<uint32_t> sorted_cells = cells;
        std::sort(sorted_cells.begin(), sorted_cells.end());
        auto ranges = cells_batch_ranges(sorted_cells);

        for (const auto & range : ranges) {
            k_ops.push_back({k_stream_offset + range.start * k_row_bytes, range.count * k_row_bytes});
            v_ops.push_back({v_stream_offset + range.start * v_row_bytes, range.count * v_row_bytes});
        }
    }

    // Execute all downloads
    for (const auto & op : k_ops) {
        ggml_backend_tensor_get_async(backend_k, src_k, (char *)dst_k->data + op.offset, op.offset, op.size);
    }
    for (const auto & op : v_ops) {
        ggml_backend_tensor_get_async(backend_v, src_v, (char *)dst_v->data + op.offset, op.offset, op.size);
    }

    state_mark_synced(layer_id);
}

// Full download: transfers entire K and V tensors (all streams) in 2 async calls
void llama_kv_cache_pipe_shard::cells_download_full(
                     int32_t     layer_id,
              ggml_backend_t     backend_k,
              ggml_backend_t     backend_v) {
    auto it = map_layer_ids.find(layer_id);
    if (it == map_layer_ids.end()) {
        return;
    }

    const size_t idx = it->second;

    ggml_tensor * src_k = layers[idx].k_gpu;
    ggml_tensor * dst_k = layers[idx].k_cpu;
    ggml_tensor * src_v = layers[idx].v_gpu;
    ggml_tensor * dst_v = layers[idx].v_cpu;

    if (!src_k || !dst_k || !src_v || !dst_v) {
        return;
    }

    const size_t k_bytes = ggml_nbytes(src_k);
    const size_t v_bytes = ggml_nbytes(src_v);

    ggml_backend_tensor_get_async(backend_k, src_k, dst_k->data, 0, k_bytes);
    ggml_backend_tensor_get_async(backend_v, src_v, dst_v->data, 0, v_bytes);

    state_mark_synced(layer_id);
}

void llama_kv_cache_pipe_shard::refresh_gpu_views(int32_t layer_id) {
    auto it = map_layer_ids.find(layer_id);
    if (it == map_layer_ids.end()) {
        return;
    }

    const size_t idx = it->second;
    ggml_tensor * k = layers[idx].k_gpu;
    ggml_tensor * v = layers[idx].v_gpu;

    if (!k || !k->data || !v || !v->data) {
        return;
    }

    for (size_t s = 0; s < layers[idx].k_stream_gpu.size(); s++) {
        layers[idx].k_stream_gpu[s]->data = (char *)k->data + s * k->nb[2];
    }
    for (size_t s = 0; s < layers[idx].v_stream_gpu.size(); s++) {
        layers[idx].v_stream_gpu[s]->data = (char *)v->data + s * v->nb[2];
    }
}

void llama_kv_cache_pipe_shard::state_mark_dirty(int32_t layer_id) {
    auto it = map_layer_ids.find(layer_id);
    if (it != map_layer_ids.end()) {
        kv_state[it->second].needs_cpu_sync = true;
    }
}

void llama_kv_cache_pipe_shard::state_mark_synced(int32_t layer_id) {
    auto it = map_layer_ids.find(layer_id);
    if (it != map_layer_ids.end()) {
        kv_state[it->second].needs_cpu_sync = false;
    }
}

bool llama_kv_cache_pipe_shard::state_needs_download(int32_t layer_id) const {
    auto it = map_layer_ids.find(layer_id);
    if (it == map_layer_ids.end()) {
        return false;
    }
    return kv_state[it->second].needs_cpu_sync;
}

bool llama_kv_cache_pipe_shard::state_address_changed(int32_t layer_id, void * k_addr, void * v_addr) const {
    auto it = map_layer_ids.find(layer_id);
    if (it == map_layer_ids.end()) {
        return true;
    }
    const auto & state = kv_state[it->second];
    return (state.last_k_addr != k_addr) || (state.last_v_addr != v_addr);
}

void llama_kv_cache_pipe_shard::state_update_addresses(int32_t layer_id, void * k_addr, void * v_addr) {
    auto it = map_layer_ids.find(layer_id);
    if (it != map_layer_ids.end()) {
        kv_state[it->second].last_k_addr = k_addr;
        kv_state[it->second].last_v_addr = v_addr;
    }
}
