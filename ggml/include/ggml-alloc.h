#pragma once

#include "ggml.h"

#ifdef  __cplusplus
extern "C" {
#endif

typedef struct ggml_backend_buffer_type * ggml_backend_buffer_type_t;
typedef struct      ggml_backend_buffer * ggml_backend_buffer_t;
typedef struct             ggml_backend * ggml_backend_t;

// Tensor allocator
struct ggml_tallocr {
    ggml_backend_buffer_t buffer;
    void * base;
    size_t alignment;
    size_t offset;
};

GGML_API struct ggml_tallocr ggml_tallocr_new(ggml_backend_buffer_t buffer);
GGML_API enum ggml_status    ggml_tallocr_alloc(struct ggml_tallocr * talloc, struct ggml_tensor * tensor);

// Graph allocator
/*
  Example usage:
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_cpu_buffer_type());

    // optional: create a worst-case graph and reserve the buffers to avoid reallocations
    ggml_gallocr_reserve(galloc, build_graph(max_batch));

    // allocate the graph
    struct ggml_cgraph * graph = build_graph(batch);
    ggml_gallocr_alloc_graph(galloc, graph);

    printf("compute buffer size: %zu bytes\n", ggml_gallocr_get_buffer_size(galloc, 0));

    // evaluate the graph
    ggml_backend_graph_compute(backend, graph);
*/

// special tensor flags for use with the graph allocator:
//   ggml_set_input(): all input tensors are allocated at the beginning of the graph in non-overlapping addresses
//   ggml_set_output(): output tensors are never freed and never overwritten

typedef struct ggml_gallocr * ggml_gallocr_t;

GGML_API ggml_gallocr_t ggml_gallocr_new(ggml_backend_buffer_type_t buft);
GGML_API ggml_gallocr_t ggml_gallocr_new_n(ggml_backend_buffer_type_t * bufts, int n_bufs);
GGML_API void           ggml_gallocr_free(ggml_gallocr_t galloc);

// pre-allocate buffers from a measure graph - does not allocate or modify the graph
// call with a worst-case graph to avoid buffer reallocations
// not strictly required for single buffer usage: ggml_gallocr_alloc_graph will reallocate the buffers automatically if needed
// returns false if the buffer allocation failed
GGML_API bool ggml_gallocr_reserve(ggml_gallocr_t galloc, struct ggml_cgraph * graph);
GGML_API bool ggml_gallocr_reserve_n(
    ggml_gallocr_t galloc,
    struct ggml_cgraph * graph,
    const int * node_buffer_ids,
    const int * leaf_buffer_ids,
    bool alloc_memory);

// Calculates the VRAM requirement for each split of the graph.
GGML_API size_t ggml_gallocr_calculate_per_split_vram_req(ggml_gallocr_t galloc);
// Gets the VRAM requirement for a specific split.
GGML_API size_t ggml_gallocr_get_per_split_vram_req(ggml_gallocr_t galloc, int split_id);
// Allocates all the required VRAM for the pipeline.
GGML_API bool ggml_gallocr_alloc_vram_buffer(ggml_gallocr_t galloc, size_t max_vram_alloc, size_t host_size);
GGML_API bool ggml_gallocr_alloc_set_scratch_offsets(ggml_gallocr_t galloc, size_t scratch_size, size_t scratch_offset);
GGML_API bool ggml_gallocr_alloc_reset_scratch_offsets(ggml_gallocr_t galloc, size_t scratch_offset);
GGML_API size_t ggml_gallocr_get_scratch_offset(ggml_gallocr_t galloc);
GGML_API size_t ggml_gallocr_get_scratch_size(ggml_gallocr_t galloc);
GGML_API void ggml_gallocr_alloc_set_pipe_shard(ggml_gallocr_t galloc, bool pipe_shard);
// Gets the buffer for a specific buffer ID.
GGML_API ggml_backend_buffer_t ggml_gallocr_get_buffer(ggml_gallocr_t galloc, int buffer_id);
// automatic reallocation if the topology changes when using a single buffer
// returns false if using multiple buffers and a re-allocation is needed (call ggml_gallocr_reserve_n first to set the node buffers)
GGML_API bool ggml_gallocr_alloc_graph(ggml_gallocr_t galloc, struct ggml_cgraph * graph);

GGML_API size_t ggml_gallocr_get_buffer_size(ggml_gallocr_t galloc, int buffer_id);

// Debug: get allocator state info
GGML_API void ggml_gallocr_get_n_nodes_leafs(ggml_gallocr_t galloc, int * n_nodes, int * n_leafs);
GGML_API void ggml_gallocr_get_node_alloc(ggml_gallocr_t galloc, int idx, int * buffer_id, size_t * offset, size_t * size_max);

// Per-plan allocator state save/restore (for multi-plan switching without fragmentation)
// Returns sizes needed for buffers (call with NULL to get size, then allocate and call again)
GGML_API void ggml_gallocr_get_state_sizes(ggml_gallocr_t galloc, size_t * node_size, size_t * leaf_size);
// Save allocator state to caller-provided buffers
GGML_API void ggml_gallocr_save_state(ggml_gallocr_t galloc, 
                                       void * node_buf, void * leaf_buf,
                                       int * n_nodes, int * n_leafs);
// Restore allocator state from caller-provided buffers
GGML_API void ggml_gallocr_restore_state(ggml_gallocr_t galloc,
                                          const void * node_buf, size_t node_size,
                                          const void * leaf_buf, size_t leaf_size,
                                          int n_nodes, int n_leafs);
// Utils
// Create a buffer and allocate all the tensors in a ggml_context
GGML_API struct ggml_backend_buffer * ggml_backend_alloc_ctx_tensors_from_buft(struct ggml_context * ctx, ggml_backend_buffer_type_t buft);
GGML_API struct ggml_backend_buffer * ggml_backend_alloc_ctx_tensors(struct ggml_context * ctx, ggml_backend_t backend);

GGML_API ggml_backend_buffer_type_t ggml_gallocr_get_buft(ggml_gallocr_t galloc, int buffer_id);																														
#ifdef  __cplusplus
}
#endif
