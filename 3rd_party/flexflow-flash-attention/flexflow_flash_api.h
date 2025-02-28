// here we define the interface exposed by the custom-built flash-attention library to FlexFlow

// usage: include in flexflow and compile flexflow with the custom-built flash-attention library
#pragma once

#include <cuda.h>
#include <vector>

#ifdef OLD_GENERATOR_PATH
#include <ATen/CUDAGeneratorImpl.h>
#else
#include <ATen/cuda/CUDAGeneratorImpl.h>
#endif

#include <ATen/cuda/CUDAGraphsUtils.cuh> // For at::cuda::philox::unpack

namespace flash
{
        // TODO: remove torch interface
        // torch tensor: data_type, device (should be on device), shape, strides, data_ptr
        // You should define your own tensor type in Flexflow
        std::vector<at::Tensor>
        mha_fwd(at::Tensor &q,                            // batch_size x seqlen_q x num_heads x round_multiple(head_size, 8)
                const at::Tensor &k,                      // batch_size x seqlen_k x num_heads_k x round_multiple(head_size, 8)
                const at::Tensor &v,                      // batch_size x seqlen_k x num_heads_k x round_multiple(head_size, 8)
                std::optional<at::Tensor> &out_,          // batch_size x seqlen_q x num_heads x round_multiple(head_size, 8)
                std::optional<at::Tensor> &alibi_slopes_, // num_heads or batch_size x num_heads
                const float p_dropout,
                const float softmax_scale,
                bool is_causal,
                int window_size_left,  // -1 for no window
                int window_size_right, // -1 for no window
                const float softcap,
                const bool return_softmax,
                std::optional<at::Generator> gen_);

        std::vector<at::Tensor>
        mha_bwd(const at::Tensor &dout,                   // batch_size x seqlen_q x num_heads, x multiple_of(head_size_og, 8)
                const at::Tensor &q,                      // batch_size x seqlen_q x num_heads x head_size
                const at::Tensor &k,                      // batch_size x seqlen_k x num_heads_k x head_size
                const at::Tensor &v,                      // batch_size x seqlen_k x num_heads_k x head_size
                const at::Tensor &out,                    // batch_size x seqlen_q x num_heads x head_size
                const at::Tensor &softmax_lse,            // b x h x seqlen_q
                std::optional<at::Tensor> &dq_,           // batch_size x seqlen_q x num_heads x head_size
                std::optional<at::Tensor> &dk_,           // batch_size x seqlen_k x num_heads_k x head_size
                std::optional<at::Tensor> &dv_,           // batch_size x seqlen_k x num_heads_k x head_size
                std::optional<at::Tensor> &alibi_slopes_, // num_heads or batch_size x num_heads
                const float p_dropout,                    // probability to drop
                const float softmax_scale,
                const bool is_causal,
                int window_size_left,
                int window_size_right,
                const float softcap,
                const bool deterministic,
                std::optional<at::Generator> gen_,
                std::optional<at::Tensor> &rng_state);
}
