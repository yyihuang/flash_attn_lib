#include <torch/torch.h>
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAGeneratorImpl.h> // For at::Generator and at::PhiloxCudaState

#include <iostream>
#include "flash_api.h"

void torch_attention_forward_matmul(
    at::Tensor const &q, // shape: [batch_size, seqlen_q, num_heads, head_size]
    at::Tensor const
        &k, // shape: [batch_size, seqlen_k, num_heads_k, head_size]
    at::Tensor const
        &v, // shape: [batch_size, seqlen_k, num_heads_k, head_size]
    at::Tensor const
        &out, // shape: [batch_size, seqlen_q, num_heads, head_size]
    at::Tensor const
        &softmax_lse, // shape: [batch_size, num_heads, seqlen_q, 1]
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
        // std::cout << "mask: " << mask << std::endl;
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

    // closeness check: out vs torch_out
    auto out_fp32 = out.to(torch::kFloat32);
    auto torch_out_fp32 = torch_out.to(torch::kFloat32);
    auto diff = (out_fp32 - torch_out_fp32);
    auto max_diff = diff.abs().max().item<float>();
    std::cout << "Max difference between Flash Attention and PyTorch attention "
                 "outputs: "
              << max_diff << std::endl;
    torch::save(out.clone().detach(), "run_mha_fwd_out_cpp.pt");
    torch::save(torch_out.clone().detach(), "manual_out_cpp.pt");

    if (max_diff > 1e-3)
    {
        // print the shape of out and torch_out
        std::cout << "flash_out.shape: " << out.sizes() << std::endl;
        std::cout << "torch_out.shape: " << torch_out.sizes() << std::endl;
        std::cout << "Warning: Large difference detected in attention outputs!"
                  << std::endl;
    }

    // closeness check: softmax_lse vs torch_softmax_lse
    std::cout << "softmax_lse.shape: " << softmax_lse.sizes() << std::endl;
    std::cout << "torch_softmax_lse.shape: " << torch_softmax_lse.sizes() << std::endl;
    auto diff_softmax_lse = (softmax_lse - torch_softmax_lse);
    auto max_diff_softmax_lse = diff_softmax_lse.abs().max().item<float>();
    std::cout << "Max difference between Flash Attention and PyTorch attention "
                 "softmax_lse: "
              << max_diff_softmax_lse << std::endl;
    torch::save(torch_softmax_lse.clone().detach(), "manual_softmax_lse_cpp.pt");
    torch::save(softmax_lse.clone().detach(), "run_mha_fwd_softmax_lse_cpp.pt");

    if (max_diff_softmax_lse > 1e-3)
    {
        std::cout << "Warning: Large difference detected in softmax_lse!"
                  << std::endl;
    }
}

std::vector<at::Tensor> _wrapper_mha_fwd_1(at::Tensor &q,                            // batch_size x seqlen_q x num_heads x round_multiple(head_size, 8)
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
    // Otherwise the kernel will be launched from cuda:0 device
    at::cuda::CUDAGuard device_guard{q.device()};

    auto [cc_major, cc_minor] = flash::get_compute_capability(flash::get_current_device());
    bool is_sm8x_min = cc_major >= 8;
    TORCH_CHECK(is_sm8x_min, "FlashAttention only supports Ampere GPUs or newer.");

    auto q_dtype = q.dtype();
    TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16,
                "FlashAttention only support fp16 and bf16 data type");
    TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
    TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");

    CHECK_DEVICE(q);
    CHECK_DEVICE(k);
    CHECK_DEVICE(v);

    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");

    const auto sizes = q.sizes();

    const int batch_size = sizes[0];
    int seqlen_q = sizes[1];
    int num_heads = sizes[2];
    const int head_size = sizes[3];
    const int seqlen_k = k.size(1);
    const int num_heads_k = k.size(2);
    TORCH_CHECK(batch_size > 0, "batch size must be positive");
    TORCH_CHECK(head_size <= 256, "FlashAttention forward only supports head dimension at most 256");
    TORCH_CHECK(head_size % 8 == 0, "query, key, value, and out_ must have a head_size that is a multiple of 8");
    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");

    if (softcap > 0.f)
    {
        TORCH_CHECK(p_dropout == 0.f, "Softcapping does not support dropout for now");
    }

    if (window_size_left >= seqlen_k)
    {
        window_size_left = -1;
    }
    if (window_size_right >= seqlen_k)
    {
        window_size_right = -1;
    }

    // causal=true is the same as causal=false in this case
    if (seqlen_q == 1 && !alibi_slopes_.has_value())
    {
        is_causal = false;
    }
    if (is_causal)
    {
        window_size_right = 0;
    }

    // Faster to transpose q from (b, 1, (nheads_kv ngroups), d) to (b, ngroups, nheads_kv, d) in this case
    // H/t Daniel Haziza
    const int seqlenq_ngroups_swapped = seqlen_q == 1 && num_heads > num_heads_k && window_size_left < 0 && window_size_right < 0 && p_dropout == 0.f && head_size % 8 == 0 && !alibi_slopes_.has_value();
    const int ngroups = num_heads / num_heads_k;
    if (seqlenq_ngroups_swapped)
    {
        q = q.reshape({batch_size, num_heads_k, ngroups, head_size}).transpose(1, 2);
        seqlen_q = ngroups;
        num_heads = num_heads_k;
    }

    CHECK_SHAPE(q, batch_size, seqlen_q, num_heads, head_size);
    CHECK_SHAPE(k, batch_size, seqlen_k, num_heads_k, head_size);
    CHECK_SHAPE(v, batch_size, seqlen_k, num_heads_k, head_size);

    at::Tensor out;
    if (out_.has_value())
    {
        out = out_.value();
        TORCH_CHECK(out.dtype() == q_dtype, "Output must have the same dtype as inputs");
        CHECK_DEVICE(out);
        TORCH_CHECK(out.stride(-1) == 1, "Output tensor must have contiguous last dimension");
        CHECK_SHAPE(out, batch_size, sizes[1], sizes[2], head_size);
        if (seqlenq_ngroups_swapped)
        {
            out = out.reshape({batch_size, num_heads_k, ngroups, head_size}).transpose(1, 2);
        }
    }
    else
    {
        out = torch::empty_like(q);
    }

    auto round_multiple = [](int x, int m)
    { return (x + m - 1) / m * m; };
    const int head_size_rounded = head_size <= 192 ? round_multiple(head_size, 32) : 256;
    const int seqlen_q_rounded = round_multiple(seqlen_q, 128);
    const int seqlen_k_rounded = round_multiple(seqlen_k, 128);

    auto opts = q.options();

    auto softmax_lse = torch::empty({batch_size, num_heads, seqlen_q}, opts.dtype(at::kFloat));
    at::Tensor p;
    // Only return softmax if there's dropout to reduce compilation time
    if (return_softmax)
    {
        TORCH_CHECK(p_dropout > 0.0f, "return_softmax is only supported when p_dropout > 0.0");
        p = torch::empty({batch_size, num_heads, seqlen_q_rounded, seqlen_k_rounded}, opts);
    }
    else
    {
        p = torch::empty({0}, opts);
    }

    flash::Flash_fwd_params params;
    flash::set_params_fprop(params,
                     batch_size,
                     seqlen_q, seqlen_k,
                     seqlen_q_rounded, seqlen_k_rounded,
                     num_heads, num_heads_k,
                     head_size, head_size_rounded,
                     q, k, v, out,
                     /*cu_seqlens_q_d=*/nullptr,
                     /*cu_seqlens_k_d=*/nullptr,
                     /*seqused_k=*/nullptr,
                     return_softmax ? p.data_ptr() : nullptr,
                     softmax_lse.data_ptr(),
                     p_dropout,
                     softmax_scale,
                     window_size_left,
                     window_size_right,
                     softcap);

    // Keep references to these tensors to extend their lifetime
    at::Tensor softmax_lse_accum, out_accum;
    std::tie(softmax_lse_accum, out_accum) = flash::set_params_splitkv(
        params, batch_size, num_heads, head_size, seqlen_k, seqlen_q,
        head_size_rounded, p_dropout, /*num_splits*/ 0, flash::get_num_sm(flash::get_current_device()), opts);

    // number of times random will be generated per thread, to offset philox counter in thc random
    // state
    // We use a custom RNG that increases the offset by batch_size * nheads * 32.
    int64_t counter_offset = params.b * params.h * 32;
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto rng_state = torch::empty({2}, options.dtype(torch::kInt64));
    // Forward kernel will populate memory with the seed and offset.
    params.rng_state = reinterpret_cast<uint64_t *>(rng_state.data_ptr());

    if (p_dropout > 0.0)
    {
        auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
            gen_, at::cuda::detail::getDefaultCUDAGenerator());
        // See Note [Acquire lock when using random generators]
        std::lock_guard<std::mutex> lock(gen->mutex_);
        params.philox_args = gen->philox_cuda_state(counter_offset);
    }

    set_params_alibi(params, alibi_slopes_, batch_size, num_heads);

    if (seqlen_k > 0)
    {
        // auto stream = at::cuda::getCurrentCUDAStream().stream();
        flash::run_mha_fwd(params, stream);
    }
    else
    {
        // If seqlen_k == 0, then we have an empty tensor. We need to set the output to 0.
        out.zero_();
        softmax_lse.fill_(std::numeric_limits<float>::infinity());
    }

    if (seqlenq_ngroups_swapped)
    {
        out = out.transpose(1, 2).reshape({batch_size, 1, num_heads_k * seqlen_q, head_size});
        q = q.transpose(1, 2).reshape({batch_size, 1, num_heads_k * seqlen_q, head_size});
        softmax_lse = softmax_lse.reshape({batch_size, num_heads_k * seqlen_q, 1});
    }
    return {out, softmax_lse, p, rng_state};
}

std::vector<at::Tensor> _wrapper_mha_bwd_1(const at::Tensor &dout,                   // batch_size x seqlen_q x num_heads, x multiple_of(head_size_og, 8)
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
    if (is_causal)
    {
        window_size_right = 0;
    }

    // Otherwise the kernel will be launched from cuda:0 device
    at::cuda::CUDAGuard device_guard{q.device()};

    auto [cc_major, cc_minor] = flash::get_compute_capability(flash::get_current_device());
    bool is_sm8x_min = cc_major >= 8;
    TORCH_CHECK(is_sm8x_min, "FlashAttention only supports Ampere GPUs or newer.");

    bool is_dropout = p_dropout > 0.0;
    // auto stream = at::cuda::getCurrentCUDAStream().stream();

    auto q_dtype = q.dtype();
    TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16,
                "FlashAttention only support fp16 and bf16 data type");
    TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
    TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");
    TORCH_CHECK(out.dtype() == q_dtype, "query and out must have the same dtype");
    TORCH_CHECK(dout.dtype() == q_dtype, "query and dout must have the same dtype");

    CHECK_DEVICE(q);
    CHECK_DEVICE(k);
    CHECK_DEVICE(v);
    CHECK_DEVICE(out);
    CHECK_DEVICE(dout);
    CHECK_DEVICE(softmax_lse);

    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(out.stride(-1) == 1, "out tensor must have contiguous last dimension");
    TORCH_CHECK(dout.stride(-1) == 1, "dout tensor must have contiguous last dimension");

    const auto sizes = q.sizes();

    const int batch_size = sizes[0];
    const int seqlen_q = sizes[1];
    const int num_heads = sizes[2];
    const int head_size = sizes[3];
    const int seqlen_k = k.size(1);
    const int num_heads_k = k.size(2);
    TORCH_CHECK(batch_size > 0, "batch size must be positive");
    TORCH_CHECK(head_size % 8 == 0, "head_size should be a multiple of 8");
    TORCH_CHECK(head_size <= 256, "FlashAttention backward only supports head dimension at most 256");
    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");

    auto round_multiple = [](int x, int m)
    { return (x + m - 1) / m * m; };
    const int head_size_rounded = head_size <= 192 ? round_multiple(head_size, 32) : 256;
    const int seqlen_q_rounded = round_multiple(seqlen_q, 128);
    const int seqlen_k_rounded = round_multiple(seqlen_k, 128);

    if (softcap > 0.f)
    {
        TORCH_CHECK(p_dropout == 0.f, "Softcapping does not support dropout for now");
    }

    if (window_size_left >= seqlen_k)
    {
        window_size_left = -1;
    }
    if (window_size_right >= seqlen_k)
    {
        window_size_right = -1;
    }

    CHECK_SHAPE(q, batch_size, seqlen_q, num_heads, head_size);
    CHECK_SHAPE(k, batch_size, seqlen_k, num_heads_k, head_size);
    CHECK_SHAPE(v, batch_size, seqlen_k, num_heads_k, head_size);
    CHECK_SHAPE(out, batch_size, seqlen_q, num_heads, head_size);
    CHECK_SHAPE(dout, batch_size, seqlen_q, num_heads, head_size);

    at::Tensor dq, dk, dv;
    if (dq_.has_value())
    {
        dq = dq_.value();
        TORCH_CHECK(dq.dtype() == q_dtype, "dq must have the same dtype as q");
        CHECK_DEVICE(dq);
        TORCH_CHECK(dq.stride(-1) == 1, "dq must have contiguous last dimension");
        CHECK_SHAPE(dq, batch_size, seqlen_q, num_heads, head_size);
    }
    else
    {
        dq = torch::empty_like(q);
    }
    if (dk_.has_value())
    {
        dk = dk_.value();
        TORCH_CHECK(dk.dtype() == q_dtype, "dk must have the same dtype as q");
        CHECK_DEVICE(dk);
        TORCH_CHECK(dk.stride(-1) == 1, "dk must have contiguous last dimension");
        CHECK_SHAPE(dk, batch_size, seqlen_k, num_heads_k, head_size);
    }
    else
    {
        dk = torch::empty_like(k);
    }
    if (dv_.has_value())
    {
        dv = dv_.value();
        TORCH_CHECK(dv.dtype() == q_dtype, "dv must have the same dtype as q");
        CHECK_DEVICE(dv);
        TORCH_CHECK(dv.stride(-1) == 1, "dv must have contiguous last dimension");
        CHECK_SHAPE(dv, batch_size, seqlen_k, num_heads_k, head_size);
    }
    else
    {
        dv = torch::empty_like(v);
    }

    // bool loop = seqlen_k > blocksize_c;
    // TODO: change later, for now set to true for simplicity
    bool loop = true;

    auto opts = q.options();
    auto softmax_d = torch::empty({batch_size, num_heads, seqlen_q_rounded}, opts.dtype(at::kFloat));
    at::Tensor dq_accum;
    at::Tensor dk_accum, dv_accum;
    if (loop)
    {
        if (!deterministic)
        {
            dq_accum = torch::empty({batch_size, seqlen_q_rounded, num_heads, head_size_rounded}, opts.dtype(at::kFloat));
        }
        else
        {
            const int nsplits = (flash::get_num_sm(flash::get_current_device()) + batch_size * num_heads - 1) / (batch_size * num_heads);
            dq_accum = torch::zeros({nsplits, batch_size, seqlen_q_rounded, num_heads, head_size_rounded}, opts.dtype(at::kFloat));
        }
        // dk_accum = torch::empty({batch_size, num_heads_k, seqlen_k_rounded, head_size_rounded}, opts.dtype(at::kFloat));
        // dv_accum = torch::empty({batch_size, num_heads_k, seqlen_k_rounded, head_size_rounded}, opts.dtype(at::kFloat));
    }

    at::Tensor dk_expanded, dv_expanded;
    if (num_heads_k != num_heads)
    { // MQA / GQA
        dk_expanded = torch::empty({batch_size, seqlen_k, num_heads, head_size}, opts);
        dv_expanded = torch::empty({batch_size, seqlen_k, num_heads, head_size}, opts);
    }
    else
    {
        dk_expanded = dk;
        dv_expanded = dv;
    }

    flash::Flash_bwd_params params;

    flash::set_params_dgrad(params,
                     batch_size,
                     seqlen_q, seqlen_k,
                     seqlen_q_rounded, seqlen_k_rounded,
                     num_heads, num_heads_k,
                     head_size, head_size_rounded,
                     q, k, v, out,
                     dout, dq, dk_expanded, dv_expanded,
                     nullptr,
                     nullptr,
                     loop ? dq_accum.data_ptr() : nullptr,
                     // loop ? dk_accum.data_ptr() : nullptr,
                     // loop ? dv_accum.data_ptr() : nullptr,
                     nullptr,
                     nullptr,
                     softmax_lse.data_ptr(),
                     softmax_d.data_ptr(),
                     p_dropout,
                     softmax_scale,
                     window_size_left,
                     window_size_right,
                     softcap,
                     deterministic,
                     /*unpadded_lse*/ false);
    params.dq_accum_split_stride = !deterministic ? 0 : dq_accum.stride(0);

    auto launch = &flash::run_mha_bwd;

    auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
        gen_, at::cuda::detail::getDefaultCUDAGenerator());

    // We use a custom RNG that increases the offset by batch_size * nheads * 32.
    int64_t counter_offset = params.b * params.h * 32;

    if (rng_state.has_value())
    {
        params.rng_state = reinterpret_cast<uint64_t *>(rng_state.value().data_ptr());
    }
    else if (is_dropout)
    {
        // See Note [Acquire lock when using random generators]
        std::lock_guard<std::mutex> lock(gen->mutex_);
        params.philox_args = gen->philox_cuda_state(counter_offset);
        auto seeds = at::cuda::philox::unpack(params.philox_args);
        params.rng_state[0] = std::get<0>(seeds);
        params.rng_state[1] = std::get<1>(seeds);
    }

    flash::set_params_alibi(params, alibi_slopes_, batch_size, num_heads);

    if (seqlen_q > 0)
    {
        launch(params, stream);
    }
    else
    {
        // If seqlen_q == 0, then we have an empty tensor. We need to set the output to 0.
        dk_expanded.zero_();
        dv_expanded.zero_();
        softmax_d.zero_();
    }

    // For MQA/GQA we need to sum dK and dV across the groups
    if (num_heads_k != num_heads)
    {
        at::sum_out(dk, at::reshape(dk_expanded, {batch_size, seqlen_k, num_heads_k, num_heads / num_heads_k, head_size}), {3});
        at::sum_out(dv, at::reshape(dv_expanded, {batch_size, seqlen_k, num_heads_k, num_heads / num_heads_k, head_size}), {3});
    }

    return {dq, dk, dv, softmax_d};
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

        std::cout << "dq: " << dq.sizes() << std::endl;
        std::cout << "dk: " << dk.sizes() << std::endl;
        std::cout << "dv: " << dv.sizes() << std::endl;
        torch::save(dq.clone().detach(), "run_mha_bwd_dq_cpp.pt");
        torch::save(dk.clone().detach(), "run_mha_bwd_dk_cpp.pt");
        torch::save(dv.clone().detach(), "run_mha_bwd_dv_cpp.pt");

        // test _wrapper_1 interface
        at::Tensor dout;
        {
            auto stream = at::cuda::getCurrentCUDAStream().stream();
            auto q_1 = q.clone().detach();
            auto k_1 = k.clone().detach();
            auto v_1 = v.clone().detach();
            std::optional<at::Tensor> dq_1 = std::nullopt;
            std::optional<at::Tensor> dk_1 = std::nullopt;
            std::optional<at::Tensor> dv_1 = std::nullopt;

            // test mha_fwd_1 interface
            auto mha_fwd_output = _wrapper_mha_fwd_1(q_1, k_1, v_1, out_, alibi_slopes_,
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

            // dout = torch::randn_like(mha_fwd_out);
            // torch::save(dout.clone().detach(), "run_mha_bwd_dout_cpp.pt");

            // // test mha_bwd_1 interface
            // auto mha_bwd_output = _wrapper_mha_bwd_1(dout, q_1, k_1, v_1, mha_fwd_out, mha_fwd_softmax_lse, dq_1, dk_1, dv_1, alibi_slopes_,
            //                                         p_dropout, softmax_scale, is_causal,
            //                                         window_size_left, window_size_right,
            //                                         softcap, deterministic, gen_, rng_state, stream);
            // torch::save(dq_1.value().clone().detach(), "run_mha_bwd_dq_cpp.pt");
            // torch::save(dk_1.value().clone().detach(), "run_mha_bwd_dk_cpp.pt");
            // torch::save(dv_1.value().clone().detach(), "run_mha_bwd_dv_cpp.pt");

            // // test my handwritten attn interface
            // torch_attention_forward_matmul(q, k, v, mha_fwd_out, mha_fwd_softmax_lse, is_causal);
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