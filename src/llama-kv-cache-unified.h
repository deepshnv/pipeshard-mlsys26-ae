#pragma once

#include "llama-batch.h"
#include "llama-graph.h"
#include "llama-kv-cells.h"
#include "llama-memory.h"
#include "llama-kv-cache-pipe-shard.h"

#include <unordered_map>
#include <vector>

struct llama_cparams;
struct llama_hparams;
struct llama_model;
struct llama_context;

//
// llama_kv_cache_unified
//

class llama_kv_cache_unified : public llama_memory_i {
public:
    static uint32_t get_padding(const llama_cparams & cparams);

    // this callback is used to filter out layers that should not be included in the cache
    using layer_filter_cb = std::function<bool(int32_t il)>;

    struct defrag_info {
        bool empty() const {
            return ids.empty();
        }

        // contains information about which cell moves where:
        //  - cell i moves to ids[i]
        //  - if ids[i] == i || ids[i] == ids.size(), then cell i is not moved
        std::vector<uint32_t> ids;
    };

    struct stream_copy_info {
        bool empty() const {
            assert(ssrc.size() == sdst.size());
            return ssrc.empty();
        }

        std::vector<uint32_t> ssrc;
        std::vector<uint32_t> sdst;
    };

    // for each ubatch, create a slot_info that contains information about where the ubatch should be inserted in the
    //   KV cells. for example, cell indices for each token, such that: token[i] -> goes to cells[idxs[i]]
    struct slot_info {
        // data for ggml_set_rows
        using idx_vec_t = std::vector<uint32_t>;

        // number of streams: ns = s1 - s0 + 1
        llama_seq_id s0;
        llama_seq_id s1;

        std::vector<llama_seq_id> strm; // [ns]
        std::vector<idx_vec_t>    idxs; // [ns]

        uint32_t head() const {
            GGML_ASSERT(idxs.size() == 1);
            GGML_ASSERT(!idxs[0].empty());

            return idxs[0][0];
        }

        void resize(size_t n) {
            strm.resize(n);
            idxs.resize(n);
        }

        size_t size() const {
            GGML_ASSERT(idxs.size() == strm.size());
            GGML_ASSERT(!idxs.empty());

            return idxs[0].size();
        }

        size_t n_stream() const {
            return strm.size();
        }

        bool empty() const {
            return idxs.empty();
        }

        void clear() {
            idxs.clear();
        }
    };

    using slot_info_vec_t = std::vector<slot_info>;

    llama_kv_cache_unified(
            const llama_model &  model,
              layer_filter_cb && filter,
                    ggml_type    type_k,
                    ggml_type    type_v,
                         bool    v_trans,
                         bool    offload,
                         bool    unified,
                     uint32_t    kv_size,
                     uint32_t    n_seq_max,
                     uint32_t    n_pad,
                     uint32_t    n_swa,
               llama_swa_type    swa_type);

    ~llama_kv_cache_unified() = default;

    //
    // llama_memory_i
    //

    llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;

    llama_memory_context_ptr init_full() override;

    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;

    bool get_can_shift() const override;

    void clear(bool data) override;

    bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) override;
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id)                                                          override;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;

    llama_pos seq_pos_min(llama_seq_id seq_id) const override;
    llama_pos seq_pos_max(llama_seq_id seq_id) const override;

    // state write/load

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1)       override;

    //
    // llama_kv_cache_unified specific API
    //

    uint32_t get_size()     const;
    uint32_t get_n_stream() const;

    bool get_has_shift() const;

    //
    // graph_build API
    //

    uint32_t get_n_kv() const;

    // TODO: temporary
    bool get_supports_set_rows() const;

    // get views of the current state of the cache
    ggml_tensor * get_k(ggml_context * ctx, int32_t il, uint32_t n_kv, const slot_info & sinfo) const;
    ggml_tensor * get_v(ggml_context * ctx, int32_t il, uint32_t n_kv, const slot_info & sinfo) const;

    // store k_cur and v_cur in the cache based on the provided head location
    ggml_tensor * cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il, const slot_info & sinfo) const;
    ggml_tensor * cpy_v(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs, int32_t il, const slot_info & sinfo) const;

    //
    // preparation API
    //

    // find places for the provided ubatches in the cache, returns the slot infos
    // return empty vector on failure
    slot_info_vec_t prepare(const std::vector<llama_ubatch> & ubatches);

    bool update(llama_context * lctx, bool do_shift, const defrag_info & dinfo, const stream_copy_info & sc_info);

    // find a slot of kv cells that can hold the ubatch
    // if cont == true, then the slot must be continuous
    // return empty slot_info on failure
    slot_info find_slot(const llama_ubatch & ubatch, bool cont) const;

    // emplace the ubatch context into slot: [sinfo.idxs[0...ubatch.n_tokens - 1]]
    void apply_ubatch(const slot_info & sinfo, const llama_ubatch & ubatch);

    //
    // input API
    //

    ggml_tensor * build_input_k_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const;
    ggml_tensor * build_input_v_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const;

    void set_input_k_idxs(ggml_tensor * dst, const llama_ubatch * ubatch, const slot_info & sinfo) const;
    void set_input_v_idxs(ggml_tensor * dst, const llama_ubatch * ubatch, const slot_info & sinfo) const;

    void set_input_k_shift(ggml_tensor * dst) const;

    void set_input_kq_mask   (ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const;
    void set_input_pos_bucket(ggml_tensor * dst, const llama_ubatch * ubatch) const;

    // for pipe_shard
	llama_kv_cache_pipe_shard * get_pipe_shard() { return pipe_shard_kv.get(); }
    void activate_cpu_kv(int layer_num);
    void activate_gpu_kv(int layer_num);

    ggml_tensor * get_k_l_tensor(int layer_num);
    ggml_tensor * get_v_l_tensor(int layer_num);
    ggml_tensor * get_k_l_cpu_tensor(int layer_num);
    ggml_tensor * get_v_l_cpu_tensor(int layer_num);
    ggml_tensor * get_k_l_gpu_tensor(int layer_num);
    ggml_tensor * get_v_l_gpu_tensor(int layer_num);

    size_t max_layer_size_k_bytes() const;
    size_t max_layer_size_v_bytes() const;

    size_t size_k_bytes() const;
    size_t size_v_bytes() const;

    // Get active cells for all sequences (stream 0 by default)
    const std::set<uint32_t> & get_active_cells(uint32_t stream = 0) const { return v_cells[stream].get_used_cells(); }

    // Get cells for a specific sequence
    std::vector<uint32_t> get_cells_for_seq(llama_seq_id seq_id) const {
        const uint32_t stream = seq_to_stream[seq_id];
        return v_cells[stream].get_cells_for_seq(seq_id);
    }

    // Get cells for multiple sequences (union)
    std::vector<uint32_t> get_cells_for_seqs(const std::vector<llama_seq_id> & seq_ids) const {
        std::set<uint32_t> cell_set;
        for (llama_seq_id seq_id : seq_ids) {
            auto cells = get_cells_for_seq(seq_id);
            cell_set.insert(cells.begin(), cells.end());
        }
        return std::vector<uint32_t>(cell_set.begin(), cell_set.end());
    }

    // Get row size in bytes for K tensor at a layer
    size_t get_k_row_bytes(int layer_id) const {
        const int32_t ikv = map_layer_ids.at(layer_id);
        return ggml_row_size(layers[ikv].k->type, hparams.n_embd_k_gqa(layer_id));
    }

    // Get row size in bytes for V tensor at a layer
    size_t get_v_row_bytes(int layer_id) const {
        const int32_t ikv = map_layer_ids.at(layer_id);
        if (v_trans) {
            // V is transposed: row = one position across all heads
            return ggml_row_size(layers[ikv].v->type, hparams.n_embd_head_v);
        }
        return ggml_row_size(layers[ikv].v->type, hparams.n_embd_v_gqa(layer_id));
    }

    // Get the stream ID for a given sequence ID
    uint32_t get_stream_for_seq(llama_seq_id seq_id) const {
        GGML_ASSERT(seq_id >= 0 && (size_t)seq_id < seq_to_stream.size());
        return seq_to_stream[seq_id];
    }

private:
	friend struct llama_kv_cache_pipe_shard;
    const llama_model & model;
    const llama_hparams & hparams;

    struct kv_layer {
        // layer index in the model
        // note: can be different from the layer index in the KV cache
        uint32_t il;

        ggml_tensor * k;
        ggml_tensor * v;

        std::vector<ggml_tensor *> k_stream;
        std::vector<ggml_tensor *> v_stream;
    };

    bool v_trans = true;  // the value tensor is transposed

    const uint32_t n_seq_max = 1;
    const uint32_t n_stream  = 1;

    // required padding
    const uint32_t n_pad = 1;

    // SWA
    const uint32_t n_swa = 0;

    // env: LLAMA_KV_CACHE_DEBUG
    int debug = 0;

    // env: LLAMA_SET_ROWS (temporary)
    // ref: https://github.com/ggml-org/llama.cpp/pull/14285
    bool supports_set_rows = true;

    const llama_swa_type swa_type = LLAMA_SWA_TYPE_NONE;

    std::vector<ggml_context_ptr>        ctxs;
    std::vector<ggml_backend_buffer_ptr> bufs;

    // the current index from where we start searching for a free slot in the ring buffer of KV cells (see find_slot())
    // note: this is not part of the KV state and it's only used to speed-up the find_slot() method
    std::vector<uint32_t> v_heads;

    std::vector<llama_kv_cells_unified> v_cells;

    // maps from a sequence id to a stream id
    std::vector<uint32_t> seq_to_stream;

    // pending stream copies that will be applied during the next update
    stream_copy_info sc_info;

    std::vector<kv_layer> layers;
	
    std::unique_ptr<llama_kv_cache_pipe_shard> pipe_shard_kv;

    // model layer id -> KV cache layer id
    std::unordered_map<int32_t, int32_t> map_layer_ids;

    // return non-empty vector if cells have been moved
    defrag_info defrag_prepare(int32_t n_max_nodes) const;

    size_t total_size() const;

    bool is_masked_swa(llama_pos p0, llama_pos p1) const;

    ggml_tensor * build_rope_shift(
            const llama_cparams & cparams,
                   ggml_context * ctx,
                    ggml_tensor * cur,
                    ggml_tensor * shift,
                    ggml_tensor * factors,
                          float   freq_base,
                          float   freq_scale) const;

    ggml_cgraph * build_graph_shift(
               llm_graph_result * res,
                  llama_context * lctx) const;

    ggml_cgraph * build_graph_defrag(
               llm_graph_result * res,
                  llama_context * lctx,
              const defrag_info & dinfo) const;

    struct cell_ranges_t {
        uint32_t strm;

        std::vector<std::pair<uint32_t, uint32_t>> data; // ranges, from inclusive, to exclusive
    };

    void state_write_meta(llama_io_write_i & io, const cell_ranges_t & cr, llama_seq_id seq_id = -1) const;
    void state_write_data(llama_io_write_i & io, const cell_ranges_t & cr) const;

    bool state_read_meta(llama_io_read_i & io, uint32_t strm, uint32_t cell_count, llama_seq_id dest_seq_id = -1);
    bool state_read_data(llama_io_read_i & io, uint32_t strm, uint32_t cell_count);
};

class llama_kv_cache_unified_context : public llama_memory_context_i {
public:
    // some shorthands
    using slot_info_vec_t  = llama_kv_cache_unified::slot_info_vec_t;
    using defrag_info      = llama_kv_cache_unified::defrag_info;
    using stream_copy_info = llama_kv_cache_unified::stream_copy_info;

    // used for errors
    llama_kv_cache_unified_context(llama_memory_status status);

    // used to create a full-cache context
    llama_kv_cache_unified_context(
            llama_kv_cache_unified * kv);

    // used to create an update context
    llama_kv_cache_unified_context(
            llama_kv_cache_unified * kv,
            llama_context * lctx,
            bool do_shift,
            defrag_info dinfo,
            stream_copy_info sc_info);

    // used to create a batch procesing context from a batch
    llama_kv_cache_unified_context(
            llama_kv_cache_unified * kv,
            slot_info_vec_t sinfos,
            std::vector<llama_ubatch> ubatches);

    virtual ~llama_kv_cache_unified_context();

    //
    // llama_memory_context_i
    //

    bool next()  override;
    bool apply() override;

    llama_memory_status  get_status() const override;
    const llama_ubatch & get_ubatch() const override;

    //
    // llama_kv_cache_unified_context specific API
    //

    uint32_t get_n_kv() const;

    // TODO: temporary
    bool get_supports_set_rows() const;

    // get views of the current state of the cache
    ggml_tensor * get_k(ggml_context * ctx, int32_t il) const;
    ggml_tensor * get_v(ggml_context * ctx, int32_t il) const;

    // store k_cur and v_cur in the cache based on the provided head location
    ggml_tensor * cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il) const;
    ggml_tensor * cpy_v(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs, int32_t il) const;

    ggml_tensor * build_input_k_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const;
    ggml_tensor * build_input_v_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const;

    void set_input_k_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const;
    void set_input_v_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const;

    void set_input_k_shift   (ggml_tensor * dst) const;
    void set_input_kq_mask   (ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const;
    void set_input_pos_bucket(ggml_tensor * dst, const llama_ubatch * ubatch) const;

    // Get the cell indices that will be written by the current ubatch
    std::vector<uint32_t> get_current_write_cells() const {
        if (i_cur >= sinfos.size()) {
            return {};
        }

        std::vector<uint32_t> cells;
        const auto & sinfo = sinfos[i_cur];
        for (const auto & idx_vec : sinfo.idxs) {
            cells.insert(cells.end(), idx_vec.begin(), idx_vec.end());
        }
        return cells;
    }

    // Get all sequences in the current ubatch
    std::vector<llama_seq_id> get_current_seq_ids() const {
        if (i_cur >= ubatches.size()) {
            return {};
        }

        const auto & ubatch = ubatches[i_cur];
        std::set<llama_seq_id> seq_set;
        for (int i = 0; i < ubatch.n_tokens; ++i) {
            for (int s = 0; s < ubatch.n_seq_id[i]; ++s) {
                seq_set.insert(ubatch.seq_id[i][s]);
            }
        }
        return std::vector<llama_seq_id>(seq_set.begin(), seq_set.end());
    }

    std::vector<uint32_t> get_write_cells() const override { return get_current_write_cells(); }

    std::vector<llama_seq_id> get_seq_ids() const override { return get_current_seq_ids(); }

    std::unordered_map<llama_seq_id, std::vector<uint32_t>> get_write_cells_per_seq() const override {
        if (i_cur >= sinfos.size()) {
            return {};
        }
        const auto & sinfo = sinfos[i_cur];
        std::unordered_map<llama_seq_id, std::vector<uint32_t>> result;
        for (size_t i = 0; i < sinfo.strm.size(); ++i) {
            if (!sinfo.idxs[i].empty()) {
                result[sinfo.strm[i]] = sinfo.idxs[i];
            }
        }
        return result;
    }

private:
    llama_memory_status status;

    llama_kv_cache_unified * kv;
    llama_context * lctx;

    //
    // update context
    //

    bool do_shift = false;

    defrag_info dinfo;

    stream_copy_info sc_info;

    //
    // batch processing context
    //

    // the index of the cur ubatch to process
    size_t i_cur = 0;

    slot_info_vec_t sinfos;

    std::vector<llama_ubatch> ubatches;

    //
    // data needed for building the compute graph for the current ubatch:
    //

    // a heuristic, to avoid attending the full cache if it is not yet utilized
    // as the cache gets filled, the benefit from this heuristic disappears
    int32_t n_kv;
};
