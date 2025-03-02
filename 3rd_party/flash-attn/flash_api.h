#pragma once
#include <torch/extension.h>
#include <torch/python.h>
#include <torch/nn/functional.h>
#include <vector>
#include <optional>
#include <cuda.h>
#include <vector>

#ifdef OLD_GENERATOR_PATH
#include <ATen/CUDAGeneratorImpl.h>
#else
#include <ATen/cuda/CUDAGeneratorImpl.h>
#endif

#include <ATen/cuda/CUDAGraphsUtils.cuh> // For at::cuda::philox::unpack

#define CHECK_DEVICE(x) TORCH_CHECK(x.is_cuda(), #x " must be on CUDA")
#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

// API-0: mha_fwd in `3rd_party/flash-attention/csrc/flash_attn/flash_api.cpp`
// do not use this to get rid of at interface
// ERROR: `mha_fwd` symbol not found
std::vector<at::Tensor>
mha_fwd(at::Tensor &q,                            // batch_size x seqlen_q x num_heads x round_multiple(head_size, 8)
        const at::Tensor &k,                      // batch_size x seqlen_k x num_heads_k x round_multiple(head_size, 8)
        const at::Tensor &v,                      // batch_size x seqlen_k x num_heads_k x round_multiple(head_size, 8)
        std::optional<at::Tensor> &out_,          // batch_size x seqlen_q x num_heads x round_multiple(head_size, 8)
        std::optional<at::Tensor> &alibi_slopes_, // num_heads or batch_size x num_heads
        const float p_dropout,
        const float softmax_scale,
        bool is_causal,
        int window_size_left,
        int window_size_right,
        const float softcap,
        const bool return_softmax,
        std::optional<at::Generator> gen_);
        
std::vector<at::Tensor>
mha_bwd(const at::Tensor &dout,  // batch_size x seqlen_q x num_heads, x multiple_of(head_size_og, 8)
        const at::Tensor &q,   // batch_size x seqlen_q x num_heads x head_size
        const at::Tensor &k,   // batch_size x seqlen_k x num_heads_k x head_size
        const at::Tensor &v,   // batch_size x seqlen_k x num_heads_k x head_size
        const at::Tensor &out,   // batch_size x seqlen_q x num_heads x head_size
        const at::Tensor &softmax_lse,     // b x h x seqlen_q
        std::optional<at::Tensor> &dq_,   // batch_size x seqlen_q x num_heads x head_size
        std::optional<at::Tensor> &dk_,   // batch_size x seqlen_k x num_heads_k x head_size
        std::optional<at::Tensor> &dv_,   // batch_size x seqlen_k x num_heads_k x head_size
        std::optional<at::Tensor> &alibi_slopes_, // num_heads or batch_size x num_heads
        const float p_dropout,         // probability to drop
        const float softmax_scale,
        const bool is_causal,
        int window_size_left,
        int window_size_right,
        const float softcap,
        const bool deterministic,
        std::optional<at::Generator> gen_,
        std::optional<at::Tensor> &rng_state);
// API-1: run_mha_fwd and num_splits_heuristic (adapted from ZhiLight open-source engine)
// plus: set_params_fprop (need for `test/test_run_mha_fwd.cpp`)

constexpr int TOTAL_DIM = 0;
constexpr int H_DIM = 1;
constexpr int D_DIM = 2;

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Qkv_params
{
        using index_t = int64_t;
        // The QKV matrices.
        void *__restrict__ q_ptr;
        void *__restrict__ k_ptr;
        void *__restrict__ v_ptr;

        // The stride between rows of the Q, K and V matrices.
        index_t q_batch_stride;
        index_t k_batch_stride;
        index_t v_batch_stride;
        index_t q_row_stride;
        index_t k_row_stride;
        index_t v_row_stride;
        index_t q_head_stride;
        index_t k_head_stride;
        index_t v_head_stride;

        // The number of heads.
        int h, h_k;
        // In the case of multi-query and grouped-query attention (MQA/GQA), nheads_k could be
        // different from nheads (query).
        int h_h_k_ratio; // precompute h / h_k,
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Flash_fwd_params : public Qkv_params
{

        // The O matrix (output).
        void *__restrict__ o_ptr;
        void *__restrict__ oaccum_ptr;

        // The stride between rows of O.
        index_t o_batch_stride;
        index_t o_row_stride;
        index_t o_head_stride;

        // The pointer to the P matrix.
        void *__restrict__ p_ptr;

        // The pointer to the softmax sum.
        void *__restrict__ softmax_lse_ptr;
        void *__restrict__ softmax_lseaccum_ptr;

        // The dimensions.
        int b, seqlen_q, seqlen_k, seqlen_knew, d, seqlen_q_rounded, seqlen_k_rounded, d_rounded, rotary_dim, total_q;

        // The scaling factors for the kernel.
        float scale_softmax;
        float scale_softmax_log2;

        // array of length b+1 holding starting offset of each sequence.
        int *__restrict__ cu_seqlens_q;
        int *__restrict__ cu_seqlens_k;
        int *__restrict__ leftpad_k;

        // If provided, the actual length of each k sequence.
        int *__restrict__ seqused_k;

        int *__restrict__ blockmask;

        // The K_new and V_new matrices.
        void *__restrict__ knew_ptr;
        void *__restrict__ vnew_ptr;

        // The stride between rows of the Q, K and V matrices.
        index_t knew_batch_stride;
        index_t vnew_batch_stride;
        index_t knew_row_stride;
        index_t vnew_row_stride;
        index_t knew_head_stride;
        index_t vnew_head_stride;

        // The cos and sin matrices for rotary embedding.
        void *__restrict__ rotary_cos_ptr;
        void *__restrict__ rotary_sin_ptr;

        // The indices to index into the KV cache.
        int *__restrict__ cache_batch_idx;

        // Paged KV cache
        int *__restrict__ block_table;
        index_t block_table_batch_stride;
        int page_block_size;

        // The dropout probability (probability of keeping an activation).
        float p_dropout;
        // uint32_t p_dropout_in_uint;
        // uint16_t p_dropout_in_uint16_t;
        uint8_t p_dropout_in_uint8_t;

        // Scale factor of 1 / (1 - p_dropout).
        float rp_dropout;
        float scale_softmax_rp_dropout;

        // Local window size
        int window_size_left, window_size_right;
        float softcap;

        // Random state.
        at::PhiloxCudaState philox_args;

        // Pointer to the RNG seed (idx 0) and offset (idx 1).
        uint64_t *rng_state;

        bool is_bf16;
        bool is_causal;

        // If is_seqlens_k_cumulative, then seqlen_k is cu_seqlens_k[bidb + 1] - cu_seqlens_k[bidb].
        // Otherwise it's cu_seqlens_k[bidb], i.e., we use cu_seqlens_k to store the sequence lengths of K.
        bool is_seqlens_k_cumulative;

        bool is_rotary_interleaved;

        int num_splits; // For split-KV version

        void *__restrict__ alibi_slopes_ptr;
        index_t alibi_slopes_batch_stride;

        bool unpadded_lse;            // For varlen paths: LSE is in [nheads, total_seqlen_q] format instead of [b, nheads, seqlen_q].
        bool seqlenq_ngroups_swapped; // q has been transposed from (b, 1, (nheads_kv ngroups), d) to (b, ngroups, nheads_kv, d).
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Flash_bwd_params : public Flash_fwd_params
{

        // The dO and dQKV matrices.
        void *__restrict__ do_ptr;
        void *__restrict__ dq_ptr;
        void *__restrict__ dk_ptr;
        void *__restrict__ dv_ptr;

        // To accumulate dQ
        void *__restrict__ dq_accum_ptr;
        void *__restrict__ dk_accum_ptr;
        void *__restrict__ dv_accum_ptr;

        // // To accumulate dK and dV in case we're splitting the bwd along seqlen_q
        // dimension void *__restrict__ dk_accum_ptr; void *__restrict__
        // dv_accum_ptr;

        // The stride between rows of the dO, dQ, dK and dV matrices.
        // TD [2022-04-16]: We're using 32-bit indexing to save registers.
        // The code probably won't work for arrays larger than 2GB.
        index_t do_batch_stride;
        index_t do_row_stride;
        index_t do_head_stride;
        index_t dq_batch_stride;
        index_t dk_batch_stride;
        index_t dv_batch_stride;
        index_t dq_row_stride;
        index_t dk_row_stride;
        index_t dv_row_stride;
        index_t dq_head_stride;
        index_t dk_head_stride;
        index_t dv_head_stride;

        // The pointer to the softmax d sum.
        void *__restrict__ dsoftmax_sum;

        bool deterministic;
        index_t dq_accum_split_stride;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

void run_mha_fwd(Flash_fwd_params &params, cudaStream_t stream, bool force_split_kernel = false);

// Find the number of splits that maximizes the occupancy. For example, if we have
// batch * n_heads = 48 and we have 108 SMs, having 2 splits (efficiency = 0.89) is
// better than having 3 splits (efficiency = 0.67). However, we also don't want too many
// splits as that would incur more HBM reads/writes.
// So we find the best efficiency, then find the smallest number of splits that gets 85%
// of the best efficiency.
inline int num_splits_heuristic(int batch_nheads_mblocks, int num_SMs, int num_n_blocks, int max_splits)
{
        // If we have enough to almost fill the SMs, then just use 1 split
        if (batch_nheads_mblocks >= 0.8f * num_SMs)
        {
                return 1;
        }
        max_splits = std::min({max_splits, num_SMs, num_n_blocks});
        float max_efficiency = 0.f;
        std::vector<float> efficiency;
        efficiency.reserve(max_splits);
        auto ceildiv = [](int a, int b)
        { return (a + b - 1) / b; };
        // Some splits are not eligible. For example, if we have 64 blocks and choose 11 splits,
        // we'll have 6 * 10 + 4 blocks. If we choose 12 splits, we'll have 6 * 11 + (-2) blocks
        // (i.e. it's 11 splits anyway).
        // So we check if the number of blocks per split is the same as the previous num_splits.
        auto is_split_eligible = [&ceildiv, &num_n_blocks](int num_splits)
        {
                return num_splits == 1 || ceildiv(num_n_blocks, num_splits) != ceildiv(num_n_blocks, num_splits - 1);
        };
        for (int num_splits = 1; num_splits <= max_splits; num_splits++)
        {
                if (!is_split_eligible(num_splits))
                {
                        efficiency.push_back(0.f);
                }
                else
                {
                        float n_waves = float(batch_nheads_mblocks * num_splits) / num_SMs;
                        float eff = n_waves / ceil(n_waves);
                        // printf("num_splits = %d, eff = %f\n", num_splits, eff);
                        if (eff > max_efficiency)
                        {
                                max_efficiency = eff;
                        }
                        efficiency.push_back(eff);
                }
        }
        for (int num_splits = 1; num_splits <= max_splits; num_splits++)
        {
                if (!is_split_eligible(num_splits))
                {
                        continue;
                }
                if (efficiency[num_splits - 1] >= 0.85 * max_efficiency)
                {
                        // printf("num_splits chosen = %d\n", num_splits);
                        return num_splits;
                }
        }
        return 1;
}

void set_params_fprop(Flash_fwd_params &params,
                      // sizes
                      const size_t b,
                      const size_t seqlen_q,
                      const size_t seqlen_k,
                      const size_t seqlen_q_rounded,
                      const size_t seqlen_k_rounded,
                      const size_t h,
                      const size_t h_k,
                      const size_t d,
                      const size_t d_rounded,
                      // device pointers
                      const at::Tensor q,
                      const at::Tensor k,
                      const at::Tensor v,
                      at::Tensor out,
                      void *cu_seqlens_q_d,
                      void *cu_seqlens_k_d,
                      void *seqused_k,
                      void *p_d,
                      void *softmax_lse_d,
                      float p_dropout,
                      float softmax_scale,
                      int window_size_left,
                      int window_size_right,
                      const float softcap,
                      bool seqlenq_ngroups_swapped = false,
                      const bool unpadded_lse = false);

// API-2: run_mha_bwd
// plus: set_params_dgrad (need for `test/test_run_mha_bwd.cpp`)
void run_mha_bwd(Flash_bwd_params &params, cudaStream_t stream);

void set_params_dgrad(Flash_bwd_params &params,
                      // sizes
                      const size_t b,
                      const size_t seqlen_q,
                      const size_t seqlen_k,
                      const size_t seqlen_q_rounded,
                      const size_t seqlen_k_rounded,
                      const size_t h,
                      const size_t h_k,
                      const size_t d,
                      const size_t d_rounded,
                      // device pointers
                      const at::Tensor q,
                      const at::Tensor k,
                      const at::Tensor v,
                      const at::Tensor out,
                      const at::Tensor dout,
                      at::Tensor dq,
                      at::Tensor dk,
                      at::Tensor dv,
                      void *cu_seqlens_q_d,
                      void *cu_seqlens_k_d,
                      void *dq_accum_d,
                      void *dk_accum_d,
                      void *dv_accum_d,
                      void *softmax_lse_d,
                      void *dsoftmax_sum_d,
                      float p_dropout,
                      float softmax_scale,
                      int window_size_left,
                      int window_size_right,
                      const float softcap,
                      bool deterministic,
                      const bool unpadded_lse);


// some other helpers
void set_params_alibi(Flash_fwd_params &params, std::optional<at::Tensor> &alibi_slopes_, int batch_size, int num_heads);

std::tuple<at::Tensor, at::Tensor> set_params_splitkv(Flash_fwd_params &params, const int batch_size,
    const int num_heads, const int head_size, const int max_seqlen_k, const int max_seqlen_q,
    const int head_size_rounded, const float p_dropout,
    const int num_splits, const int num_sm, struct c10::TensorOptions opts);
