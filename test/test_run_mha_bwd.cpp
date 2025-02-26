#include <torch/torch.h>
#include <torch/extension.h>
#include <iostream>
#include <iomanip> // For formatted printing
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

int main()
{
    try
    {
        std::cout << "CUDA available: " << torch::cuda::is_available() << std::endl;
        std::cout << "CUDNN available: " << torch::cuda::cudnn_is_available() << std::endl;

        // Define tensor dimensions
        int batch_size = 2;
        int seqlen_q = 16;
        int seqlen_k = 16;
        int num_heads = 8;
        int num_heads_k = 8; // Same as num_heads for standard MHA
        int head_size = 64;

        // Ensure head_size is a multiple of 8
        if (head_size % 8 != 0)
        {
            throw std::runtime_error("head_size must be a multiple of 8");
        }

        // Create input tensors on CUDA with half precision (float16)
        at::Tensor q = torch::randn({batch_size, seqlen_q, num_heads, head_size},
                                    torch::dtype(torch::kFloat16).device(torch::kCUDA));
        at::Tensor k = torch::randn({batch_size, seqlen_k, num_heads_k, head_size},
                                    torch::dtype(torch::kFloat16).device(torch::kCUDA));
        at::Tensor v = torch::randn({batch_size, seqlen_k, num_heads_k, head_size},
                                    torch::dtype(torch::kFloat16).device(torch::kCUDA));

        // Output tensor for forward pass
        at::Tensor out = torch::empty_like(q);

        // Softmax LSE tensor (log-sum-exp)
        at::Tensor softmax_lse = torch::empty({batch_size, num_heads, seqlen_q},
                                              torch::dtype(torch::kFloat).device(torch::kCUDA));

        // Checking in original code
        {
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
        }

        // Forward pass
        std::cout << "Running FlashAttention Forward..." << std::endl;
        Flash_fwd_params fwd_params;
        set_params_fprop(fwd_params,
                         batch_size, seqlen_q, seqlen_k,
                         seqlen_q, seqlen_k, // Rounded lengths
                         num_heads, num_heads_k,
                         head_size, head_size, // Rounded head_size
                         q, k, v, out,
                         nullptr, nullptr, // cu_seqlens_q_d, cu_seqlens_k_d
                         nullptr, nullptr, // seqused_k, dropout mask
                         softmax_lse.data_ptr(),
                         0.0f,                        // p_dropout = 0
                         1.0f / std::sqrt(head_size), // softmax scale
                         -1, -1,                      // window_size_left, window_size_right (no local attention)
                         1.0f                         // softcap
        );

        auto stream = at::cuda::getCurrentCUDAStream().stream();
        run_mha_fwd(fwd_params, stream);
        torch::cuda::synchronize();

        // Print forward output
        print_tensor_values(out, "Output Tensor (after Forward)");
        print_tensor_values(softmax_lse, "Softmax LSE");

        // Backward pass preparation
        at::Tensor dout = torch::randn({batch_size, seqlen_q, num_heads, head_size},
                                       torch::dtype(torch::kFloat16).device(torch::kCUDA));

        at::Tensor dq = torch::empty_like(q);
        at::Tensor dk = torch::empty_like(k);
        at::Tensor dv = torch::empty_like(v);

        {
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
        }

        // Run backward pass
        Flash_bwd_params bwd_params;
        set_params_dgrad(bwd_params,
                         batch_size,
                         seqlen_q, seqlen_k,
                         seqlen_q, seqlen_k, // Rounded lengths
                         num_heads, num_heads_k,
                         head_size, head_size, // Rounded head_size
                         q, k, v, out,
                         dout, dq, dk, dv,
                         nullptr, nullptr,            // cu_seqlens_q_d, cu_seqlens_k_d
                         nullptr, nullptr, nullptr,   // dq_accum_d, dk_accum_d, dv_accum_d
                         softmax_lse.data_ptr(),      // softmax_lse_d
                         nullptr,                     // dsoftmax_sum_d
                         0.0f,                        // p_dropout = 0
                         1.0f / std::sqrt(head_size), // softmax scale
                         -1, -1,                      // window_size_left, window_size_right (no local attention)
                         1.0f,                        // softcap
                         false,                       // deterministic
                         false                        // unpadded_lse
        );

        std::cout << "Running FlashAttention Backward..." << std::endl;
        torch::cuda::synchronize();
        run_mha_bwd(bwd_params, stream);
        torch::cuda::synchronize();
        CHECK_CUDA_ERRORS();

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