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
        std::cout << "mask: " << mask << std::endl;
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

        // ========================================================================================================
        // Forward Pass: Allocate input tensors
        // ========================================================================================================
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
        std::cout << "Q Tensor Shape: " << q.sizes() << std::endl;
        std::cout << "K Tensor Shape: " << k.sizes() << std::endl;
        std::cout << "V Tensor Shape: " << v.sizes() << std::endl;
        torch::save(q.clone().detach(), "q.pt");
        torch::save(k.clone().detach(), "k.pt");
        torch::save(v.clone().detach(), "v.pt");

        // Forward pass
        std::vector<at::Tensor> mha_fwd_output;
        mha_fwd_output = flash::mha_fwd(q, k, v, out_, alibi_slopes_,
                                        p_dropout, softmax_scale, is_causal,
                                        window_size_left, window_size_right,
                                        softcap, return_softmax, gen_);
        auto mha_fwd_out = mha_fwd_output[0];
        auto mha_fwd_softmax_lse = mha_fwd_output[1];
        // save tensor to file
        torch::save(mha_fwd_out.clone().detach(), "mha_fwd_out_cpp.pt");
        torch::save(mha_fwd_softmax_lse.clone().detach(), "mha_fwd_softmax_lse_cpp.pt");

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

        flash::Flash_fwd_params fwd_params;
        flash::set_params_fprop(fwd_params,
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
            fwd_params, batch_size, num_heads, head_size, seqlen_k, seqlen_q,
            head_size_rounded, p_dropout, /*num_splits*/ 0, flash::get_num_sm(flash::get_current_device()), opts);

        // number of times random will be generated per thread, to offset philox counter in thc random
        // state
        // We use a custom RNG that increases the offset by batch_size * nheads * 32.
        int64_t fwd_counter_offset = fwd_params.b * fwd_params.h * 32;
        auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
        auto rng_state = torch::empty({2}, options.dtype(torch::kInt64));
        // Forward kernel will populate memory with the seed and offset.
        fwd_params.rng_state = reinterpret_cast<uint64_t *>(rng_state.data_ptr());

        if (p_dropout > 0.0)
        {
            auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
                gen_, at::cuda::detail::getDefaultCUDAGenerator());
            // See Note [Acquire lock when using random generators]
            std::lock_guard<std::mutex> lock(gen->mutex_);
            fwd_params.philox_args = gen->philox_cuda_state(fwd_counter_offset);
        }

        set_params_alibi(fwd_params, alibi_slopes_, batch_size, num_heads);

        if (seqlen_k > 0)
        {
            auto stream = at::cuda::getCurrentCUDAStream().stream();
            flash::run_mha_fwd(fwd_params, stream);
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

        // out should have shape (batch_size, seqlen_q, num_heads, head_size)
        // std::cout << "Output Tensor Shape: " << out.sizes() << std::endl;
        // std::cout << "out: " << out << std::endl;

        // std::cout << "Softmax LSE Shape: " << softmax_lse.sizes() << std::endl;
        // std::cout << "softmax_lse: " << softmax_lse << std::endl;

        torch_attention_forward_matmul(q, k, v, out, softmax_lse, is_causal);

        // ========================================================================================================
        // Backward Pass: Create dout, ensuring alignment with PyTorch behavior
        // ========================================================================================================
        at::Tensor dq, dk, dv;
        at::Tensor dout = torch::randn_like(out);
        std::optional<at::Tensor> dq_ = std::nullopt;
        std::optional<at::Tensor> dk_ = std::nullopt;
        std::optional<at::Tensor> dv_ = std::nullopt;
        bool deterministic = false;

        if (dq_.has_value())
        {
            dq = dq_.value();
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

        auto bwd_opts = q.options();
        auto softmax_d = torch::empty({batch_size, num_heads, seqlen_q_rounded}, bwd_opts.dtype(at::kFloat));
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
                dq_accum = torch::zeros({nsplits, batch_size, seqlen_q_rounded, num_heads, head_size_rounded}, bwd_opts.dtype(at::kFloat));
            }
            // dk_accum = torch::empty({batch_size, num_heads_k, seqlen_k_rounded, head_size_rounded}, bwd_opts.dtype(at::kFloat));
            // dv_accum = torch::empty({batch_size, num_heads_k, seqlen_k_rounded, head_size_rounded}, bwd_opts.dtype(at::kFloat));
        }

        at::Tensor dk_expanded, dv_expanded;
        if (num_heads_k != num_heads)
        { // MQA / GQA
            dk_expanded = torch::empty({batch_size, seqlen_k, num_heads, head_size}, bwd_opts);
            dv_expanded = torch::empty({batch_size, seqlen_k, num_heads, head_size}, bwd_opts);
        }
        else
        {
            dk_expanded = dk;
            dv_expanded = dv;
        }

        flash::Flash_bwd_params bwd_params;

        flash::set_params_dgrad(bwd_params,
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
        bwd_params.dq_accum_split_stride = !deterministic ? 0 : dq_accum.stride(0);

        auto launch = &flash::run_mha_bwd;

        auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
            gen_, at::cuda::detail::getDefaultCUDAGenerator());

        // We use a custom RNG that increases the offset by batch_size * nheads * 32.
        int64_t bwd_counter_offset = bwd_params.b * bwd_params.h * 32;

        // if (rng_state.has_value())
        // {
        bwd_params.rng_state = reinterpret_cast<uint64_t *>(rng_state.data_ptr());
        // }
        // else if (is_dropout)
        // {
        //     // See Note [Acquire lock when using random generators]
        //     std::lock_guard<std::mutex> lock(gen->mutex_);
        //     bwd_params.philox_args = gen->philox_cuda_state(bwd_counter_offset);
        //     auto seeds = at::cuda::philox::unpack(bwd_params.philox_args);
        //     bwd_params.rng_state[0] = std::get<0>(seeds);
        //     bwd_params.rng_state[1] = std::get<1>(seeds);
        // }

        set_params_alibi(bwd_params, alibi_slopes_, batch_size, num_heads);

        auto stream = at::cuda::getCurrentCUDAStream().stream();

        if (seqlen_q > 0)
        {
            launch(bwd_params, stream);
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

        // // Trim dq, dk, dv (equivalent to `dq[..., :dout.shape[-1]]` in Python)
        // if (pad_size > 0)
        // {
        //     dq = dq.index({"...", torch::indexing::Slice(0, head_size)});
        //     dk = dk.index({"...", torch::indexing::Slice(0, head_size)});
        //     dv = dv.index({"...", torch::indexing::Slice(0, head_size)});
        // }

        // Print the size of dq, dk, dv and their values
        // std::cout << "dq Tensor Shape: " << dq.sizes() << std::endl;
        // std::cout << "dq: " << dq << std::endl;
        // std::cout << "dk Tensor Shape: " << dk.sizes() << std::endl;
        // std::cout << "dk: " << dk << std::endl;
        // std::cout << "dv Tensor Shape: " << dv.sizes() << std::endl;
        // std::cout << "dv: " << dv << std::endl;
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