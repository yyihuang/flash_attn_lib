#include <torch/torch.h>
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAGeneratorImpl.h> // For at::Generator and at::PhiloxCudaState

#include <iostream>
#include "flash_api.h"

#define CHECK_CUDA_ERRORS()                                                                                               \
    {                                                                                                                     \
        cudaError_t err = cudaGetLastError();                                                                             \
        if (err != cudaSuccess)                                                                                           \
        {                                                                                                                 \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE);                                                                                           \
        }                                                                                                                 \
    }

#define CHECK_CUDA(call)                                                    \
    do                                                                      \
    {                                                                       \
        cudaError_t status_ = call;                                         \
        if (status_ != cudaSuccess)                                         \
        {                                                                   \
            fprintf(stderr, "CUDA error (%s:%d): %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(status_));                           \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

#define CHECK_DEVICE(x) TORCH_CHECK(x.is_cuda(), #x " must be on CUDA")
#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

inline int get_current_device()
{
    int device;
    CHECK_CUDA(cudaGetDevice(&device));
    return device;
}

inline int get_num_sm(int device)
{
    int multiprocessor_count;
    CHECK_CUDA(cudaDeviceGetAttribute(&multiprocessor_count, cudaDevAttrMultiProcessorCount, device));
    return multiprocessor_count;
}

// why this could not be linked????
std::tuple<at::Tensor, at::Tensor> set_params_splitkv(Flash_fwd_params &params, const int batch_size,
                                                      const int num_heads, const int head_size, const int max_seqlen_k, const int max_seqlen_q,
                                                      const int head_size_rounded, const float p_dropout,
                                                      const int num_splits, const int num_sm, struct c10::TensorOptions opts)
{

    // This needs to match with run_mha_fwd_splitkv_dispatch
    const int block_n = head_size <= 64 ? 256 : (head_size <= 128 ? 128 : 64);
    const int num_n_blocks = (max_seqlen_k + block_n - 1) / block_n;
    // Technically kBlockM = 64 only for the splitKV kernels, not the standard kernel.
    // In any case we don't expect seqlen_q to be larger than 64 for inference.
    const int num_m_blocks = (max_seqlen_q + 64 - 1) / 64;
    params.num_splits = num_splits;
    at::Tensor softmax_lse_accum;
    at::Tensor out_accum;

    if (p_dropout == 0.0f)
    { // SplitKV is not implemented for dropout
        if (num_splits < 1)
        {
            // We multiply number of SMs by 2 to hard-code the fact that we're using 128 threads per block.
            params.num_splits = num_splits_heuristic(batch_size * num_heads * num_m_blocks, num_sm * 2, num_n_blocks, 128);
        }
        if (params.num_splits > 1)
        {
            softmax_lse_accum = torch::empty({params.num_splits, batch_size, num_heads, max_seqlen_q}, opts.dtype(at::kFloat));
            out_accum = torch::empty({params.num_splits, batch_size, num_heads, max_seqlen_q, head_size_rounded}, opts.dtype(at::kFloat));
            params.softmax_lseaccum_ptr = softmax_lse_accum.data_ptr();
            params.oaccum_ptr = out_accum.data_ptr();
        }
        TORCH_CHECK(params.num_splits <= 128, "num_splits > 128 not supported");
    }

    return std::make_tuple(softmax_lse_accum, out_accum);
}

void print_tensor_values(const at::Tensor &tensor, const std::string &name, int num_elements = 5)
{
    at::Tensor tensor_cpu = tensor.to(torch::kCPU, torch::kFloat); // Convert to float32 for printing
    std::cout << name << " (first " << num_elements << " values): ";

    if (tensor.dim() == 4)
    { // Standard Q, K, V, Output tensors
        auto accessor = tensor_cpu.accessor<float, 4>();
        for (int i = 0; i < num_elements && i < tensor_cpu.numel(); ++i)
        {
            std::cout << std::fixed << std::setprecision(5) << accessor[0][0][0][i] << " ";
        }
    }
    else if (tensor.dim() == 3)
    { // Softmax LSE (batch_size, num_heads, seqlen_q)
        auto accessor = tensor_cpu.accessor<float, 3>();
        for (int i = 0; i < num_elements && i < tensor_cpu.numel(); ++i)
        {
            std::cout << std::fixed << std::setprecision(5) << accessor[0][0][i] << " ";
        }
    }
    else
    {
        std::cerr << "Unsupported tensor dimension: " << tensor.dim() << std::endl;
    }

    std::cout << std::endl;
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

        int batch_size = 2, seqlen_q = 128, seqlen_k = 128;
        int num_heads = 8, num_heads_k = 8, head_size = 128;
        float p_dropout = 0.1f, softmax_scale = 1.0f, softcap = 0.0f;
        bool return_softmax = true, is_causal = true;
        int window_size_left = -1, window_size_right = -1;
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

        Flash_fwd_params fwd_params;
        set_params_fprop(fwd_params,
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
        std::tie(softmax_lse_accum, out_accum) = set_params_splitkv(
            fwd_params, batch_size, num_heads, head_size, seqlen_k, seqlen_q,
            head_size_rounded, p_dropout, /*num_splits*/ 0, get_num_sm(get_current_device()), opts);

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
            run_mha_fwd(fwd_params, stream);
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

        std::cout << "Output Tensor Shape: " << out.sizes() << std::endl;
        std::cout << "Softmax LSE Shape: " << softmax_lse.sizes() << std::endl;

        print_tensor_values(out, "Output Tensor (after Forward)");
        print_tensor_values(softmax_lse, "Softmax LSE");

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
                const int nsplits = (get_num_sm(get_current_device()) + batch_size * num_heads - 1) / (batch_size * num_heads);
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

        Flash_bwd_params bwd_params;

        set_params_dgrad(bwd_params,
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

        auto launch = &run_mha_bwd;

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

        // Print the size of dq, dk, dv
        std::cout << "dq Tensor Shape: " << dq.sizes() << std::endl;
        std::cout << "dk Tensor Shape: " << dk.sizes() << std::endl;
        std::cout << "dv Tensor Shape: " << dv.sizes() << std::endl;

        // Print output gradients
        print_tensor_values(dq, "dQ (Gradient of Q)");
        print_tensor_values(dk, "dK (Gradient of K)");
        print_tensor_values(dv, "dV (Gradient of V)");
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