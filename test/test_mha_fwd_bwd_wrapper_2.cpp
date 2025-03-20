#include <torch/torch.h>
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAGeneratorImpl.h> // For at::Generator and at::PhiloxCudaState
// #include <ATen/cuda/CUDAContext.h>       // for at::cuda::CUDAStreamGuard

#include <iostream>
#include "flash_api.h"

void torch_attention_forward_matmul(
    at::Tensor const &q, // [batch_size, seqlen_q, num_heads, head_size]
    at::Tensor const &k, // [batch_size, seqlen_k, num_heads_k, head_size]
    at::Tensor const &v, // [batch_size, seqlen_k, num_heads_k, head_size]
    bool is_causal)
{
    // Get shape
    auto batch_size = q.size(0);
    auto seqlen_q = q.size(1);
    auto num_heads = q.size(2);
    auto head_size = q.size(3);
    auto seqlen_k = k.size(1);
    auto num_heads_k = k.size(2);
    assert(head_size == k.size(3));

    // Clone input tensors to avoid modifying them
    auto q_compute = q.clone();
    auto k_compute = k.clone();
    auto v_compute = v.clone();

    // Permute Q, K, V to [batch_size, num_heads, seqlen, head_size]
    q_compute = q_compute.permute({0, 2, 1, 3});
    k_compute = k_compute.permute({0, 2, 1, 3});
    v_compute = v_compute.permute({0, 2, 1, 3});

    // Compute attention scores: q * k_t
    // q: [batch_size, num_heads, seqlen_q, head_size]
    // k: [batch_size, num_heads_k, seqlen_k, head_size]
    // k_t: [batch_size, num_heads_k, head_size, seqlen_k]
    auto k_t = k_compute.transpose(-2, -1);
    // scores: [batch_size, num_heads, seqlen_q, seqlen_k]
    float softmax_scale = 1.0 / std::sqrt(head_size);
    auto scores = torch::matmul(q_compute, k_t) * softmax_scale;

    // Handle causal mask properly
    if (is_causal)
    {
        // Fill the upper triangle with -inf (excluding the diagonal)
        auto mask = torch::zeros({seqlen_q, seqlen_k}, q.options());
        mask = torch::triu(mask, /*diagonal=*/1).masked_fill(torch::triu(torch::ones({seqlen_q, seqlen_k}, q.options()), 1) == 1, -std::numeric_limits<float>::infinity());
        scores = scores + mask.unsqueeze(0).unsqueeze(0);
    }

    // Compute softmax
    auto max_scores = std::get<0>(scores.max(-1, true)); // Max per row for stability
    auto exp_scores = (scores - max_scores).exp();
    auto sum_exp_scores = exp_scores.sum(-1, true);
    auto attn_weights = exp_scores / sum_exp_scores;

    // get softmax_lse
    auto torch_softmax_lse = torch::log(sum_exp_scores);
    torch_softmax_lse = torch_softmax_lse.squeeze(-1);
    // add max scores to softmax_lse
    torch_softmax_lse = torch_softmax_lse + max_scores.squeeze(-1);

    // Compute attention output
    auto torch_out = torch::matmul(attn_weights, v_compute);

    // Reorder torch_out to [batch_size, seqlen_q, num_heads, head_size]
    torch_out = torch_out.permute({0, 2, 1, 3});

    torch::save(torch_out.clone().detach(), "manual_out_cpp.pt");
    torch::save(torch_softmax_lse.clone().detach(), "manual_softmax_lse_cpp.pt");
}

// Convert cudaStream_t to at::cuda::CUDAStream
at::cuda::CUDAStream convertCudaStream(cudaStream_t raw_stream)
{
    return at::cuda::getStreamFromExternal(raw_stream, at::cuda::current_device());
}

std::vector<at::Tensor> _wrapper_mha_fwd_2(at::Tensor &q,                            // batch_size x seqlen_q x num_heads x round_multiple(head_size, 8)
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
                                           std::optional<at::Generator> gen_,
                                           cudaStream_t stream)
{
    auto torch_stream = convertCudaStream(stream);
    at::cuda::CUDAStreamGuard guard(torch_stream);

    std::vector<at::Tensor> mha_fwd_output;
    mha_fwd_output = flash::mha_fwd(q, k, v, out_, alibi_slopes_,
                                    p_dropout, softmax_scale, is_causal,
                                    window_size_left, window_size_right,
                                    softcap, return_softmax, gen_);
    return mha_fwd_output;
}

std::vector<at::Tensor> _wrapper_mha_bwd_2(const at::Tensor &dout,                   // batch_size x seqlen_q x num_heads, x multiple_of(head_size_og, 8)
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
                                           std::optional<at::Tensor> &rng_state,
                                           cudaStream_t stream)
{
    auto torch_stream = convertCudaStream(stream);
    at::cuda::CUDAStreamGuard guard(torch_stream);

    auto mha_bwd_output = flash::mha_bwd(dout, q, k, v, out, softmax_lse, dq_, dk_, dv_, alibi_slopes_,
                                         p_dropout, softmax_scale, is_causal,
                                         window_size_left, window_size_right,
                                         softcap, deterministic, gen_, rng_state);
    return mha_bwd_output;
}

// To pass raw data pointer (on-device) to at::Tensor interface without copying
// Refer this: torch::from_blob
// https://pytorch.org/cppdocs/api/function_namespacetorch_1ad7fb2a7759ef8c9443b489ddde494787.html
// raw data should stay valid during the lifetime of the tensor
// copy or ownership tranfer: use tensor = tensor.clone();
int main()
{
    torch::manual_seed(42);

    try
    {
        std::cout << "CUDA available: " << torch::cuda::is_available() << std::endl;
        std::cout << "CUDNN available: " << torch::cuda::cudnn_is_available() << std::endl;

        int batch_size = 1, seqlen_q = 64, seqlen_k = 64;
        int num_heads = 8, num_heads_k = 8, head_size = 128;
        float p_dropout = 0.0f, softmax_scale = 1.0f / sqrt(head_size), softcap = 0.0f;
        bool return_softmax = false, is_causal = true;
        int window_size_left = -1, window_size_right = -1;
        // add alibi slopes. ALiBi slopes must have dtype fp32
        // at::Tensor alibi_slopes = at::randn({batch_size, num_heads}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        std::optional<at::Tensor> alibi_slopes_ = std::nullopt;
        std::optional<at::Tensor> out_ = std::nullopt;
        std::optional<at::Generator> gen_ = std::nullopt;

        // q size: (batch_size, seqlen_q, num_heads, head_size)
        // k size: (batch_size, seqlen_k, num_heads_k, head_size)
        // v size: (batch_size, seqlen_k, num_heads_k, head_size)
        at::Tensor q = torch::randn({batch_size, seqlen_q, num_heads, head_size},
                                    torch::dtype(torch::kFloat16).device(torch::kCUDA));
        at::Tensor k = torch::randn({batch_size, seqlen_k, num_heads_k, head_size},
                                    torch::dtype(torch::kFloat16).device(torch::kCUDA));
        at::Tensor v = torch::randn({batch_size, seqlen_k, num_heads_k, head_size},
                                    torch::dtype(torch::kFloat16).device(torch::kCUDA));

        const auto sizes = q.sizes();
        std::cout << "Q: " << q.sizes() << std::endl;
        std::cout << "K: " << k.sizes() << std::endl;
        std::cout << "V: " << v.sizes() << std::endl;
        torch::save(q.clone().detach(), "q.pt");
        torch::save(k.clone().detach(), "k.pt");
        torch::save(v.clone().detach(), "v.pt");

        at::Tensor dq, dk, dv;
        std::optional<at::Tensor> dq_ = std::nullopt;
        std::optional<at::Tensor> dk_ = std::nullopt;
        std::optional<at::Tensor> dv_ = std::nullopt;
        bool deterministic = false;

        // test _wrapper_2 interface
        at::Tensor dout;
        {
            auto stream = at::cuda::getCurrentCUDAStream().stream();
            auto q_1 = q.clone().detach();
            auto k_1 = k.clone().detach();
            auto v_1 = v.clone().detach();
            std::optional<at::Tensor> dq_ = torch::empty_like(q);
            std::optional<at::Tensor> dk_ = torch::empty_like(k);
            std::optional<at::Tensor> dv_ = torch::empty_like(v);

            // test mha_fwd_2 interface
            auto mha_fwd_output = _wrapper_mha_fwd_2(q_1, k_1, v_1, out_, alibi_slopes_,
                                                     p_dropout, softmax_scale, is_causal,
                                                     window_size_left, window_size_right,
                                                     softcap, return_softmax, gen_, stream);
            auto mha_fwd_out = mha_fwd_output[0];
            auto mha_fwd_softmax_lse = mha_fwd_output[1];
            auto S_dmask = mha_fwd_output[2];
            std::optional<at::Tensor> rng_state = mha_fwd_output[3];

            // save tensor to file
            torch::save(mha_fwd_out.clone().detach(), "run_mha_fwd_out_cpp.pt");
            torch::save(mha_fwd_softmax_lse.clone().detach(), "run_mha_fwd_softmax_lse_cpp.pt");

            dout = torch::randn_like(mha_fwd_out);
            torch::save(dout.clone().detach(), "dout.pt");

            // test mha_bwd_2 interface
            auto mha_bwd_output = _wrapper_mha_bwd_2(dout, q_1, k_1, v_1, mha_fwd_out, mha_fwd_softmax_lse, dq_, dk_, dv_, alibi_slopes_,
                                                     p_dropout, softmax_scale, is_causal,
                                                     window_size_left, window_size_right,
                                                     softcap, deterministic, gen_, rng_state, stream);
            dq_ = mha_bwd_output[0];
            dk_ = mha_bwd_output[1];
            dv_ = mha_bwd_output[2];
            torch::save(dq_.value().clone().detach(), "run_mha_bwd_dq_cpp.pt");
            torch::save(dk_.value().clone().detach(), "run_mha_bwd_dk_cpp.pt");
            torch::save(dv_.value().clone().detach(), "run_mha_bwd_dv_cpp.pt");

            // test my handwritten attn interface
            torch_attention_forward_matmul(q, k, v, is_causal);
        }

        // // Trim dq, dk, dv (equivalent to `dq[..., :dout.shape[-1]]` in Python)
        // if (pad_size > 0)
        // {
        //     dq = dq.index({"...", torch::indexing::Slice(0, head_size)});
        //     dk = dk.index({"...", torch::indexing::Slice(0, head_size)});
        //     dv = dv.index({"...", torch::indexing::Slice(0, head_size)});
        // }

        // test mha_fwd and mha_bwd interface
        { // Forward pass: mha_fwd
            std::vector<at::Tensor> mha_fwd_output;
            mha_fwd_output = flash::mha_fwd(q, k, v, out_, alibi_slopes_,
                                            p_dropout, softmax_scale, is_causal,
                                            window_size_left, window_size_right,
                                            softcap, return_softmax, gen_);
            auto mha_fwd_out = mha_fwd_output[0];
            auto mha_fwd_softmax_lse = mha_fwd_output[1];
            auto S_dmask = mha_fwd_output[2];
            std::optional<at::Tensor> rng_state = mha_fwd_output[3];
            // save tensor to file
            torch::save(mha_fwd_out.clone().detach(), "mha_fwd_out_cpp.pt");
            torch::save(mha_fwd_softmax_lse.clone().detach(), "mha_fwd_softmax_lse_cpp.pt");

            // Backward pass: mha_bwd
            // at::Tensor dout = torch::randn_like(mha_fwd_out);
            std::optional<at::Tensor> dq_ = torch::empty_like(q);
            std::optional<at::Tensor> dk_ = torch::empty_like(k);
            std::optional<at::Tensor> dv_ = torch::empty_like(v);
            bool deterministic = false;
            std::vector<at::Tensor> mha_bwd_output;
            flash::mha_bwd(dout, q, k, v, mha_fwd_out, mha_fwd_softmax_lse, dq_, dk_, dv_, alibi_slopes_,
                           p_dropout, softmax_scale, is_causal,
                           window_size_left, window_size_right,
                           softcap, deterministic, gen_, rng_state);
            torch::save(dq_.value().clone().detach(), "mha_bwd_dq_cpp.pt");
            torch::save(dk_.value().clone().detach(), "mha_bwd_dk_cpp.pt");
            torch::save(dv_.value().clone().detach(), "mha_bwd_dv_cpp.pt");
        }
    }
    catch (const c10::Error &e)
    {
        std::cerr << "Torch error: " << e.what() << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Standard exception: " << e.what() << std::endl;
    }

    return 0;
}