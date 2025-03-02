#include <torch/torch.h>
#include <torch/extension.h>
#include <iostream>
#include "flash_api.h"
#include <iomanip> // For formatted printing

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
        int head_size = 128;

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

        // Print input tensor values before running FlashAttention
        print_tensor_values(q, "Q Input");
        print_tensor_values(k, "K Input");
        print_tensor_values(v, "V Input");

        // Output tensor
        at::Tensor out = torch::empty_like(q);

        // Print output tensor values before running FlashAttention
        print_tensor_values(out, "Output Tensor (before FlashAttention)");

        // Softmax LSE tensor (log-sum-exp)
        at::Tensor softmax_lse = torch::empty({batch_size, num_heads, seqlen_q},
                                              torch::dtype(torch::kFloat).device(torch::kCUDA));

        // Create an empty tensor for dropout mask (if required)
        at::Tensor p = torch::empty({0}, q.options());

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

        // FlashAttention forward parameters
        Flash_fwd_params params;
        set_params_fprop(params,
                         batch_size,
                         seqlen_q, seqlen_k,
                         seqlen_q, seqlen_k, // Rounded lengths (not rounded in this test)
                         num_heads, num_heads_k,
                         head_size, head_size, // Rounded head_size (not rounded in this test)
                         q, k, v, out,
                         /*cu_seqlens_q_d=*/nullptr,
                         /*cu_seqlens_k_d=*/nullptr,
                         /*seqused_k=*/nullptr,
                         p.data_ptr(),
                         softmax_lse.data_ptr(),
                         0.0,                        // p_dropout = 0
                         1.0 / std::sqrt(head_size), // softmax scale
                         -1, -1,                     // window_size_left, window_size_right (no local attention)
                         1.0                         // softcap
        );

        // Get CUDA stream
        auto stream = at::cuda::getCurrentCUDAStream().stream();

        // Run the MHA forward function
        std::cout << "Running FlashAttention..." << std::endl;
        run_mha_fwd(params, stream);

        // Synchronize to ensure computation is complete
        torch::cuda::synchronize();

        // Print output shape
        std::cout << "Output Tensor Shape: " << out.sizes() << std::endl;
        std::cout << "Softmax LSE Shape: " << softmax_lse.sizes() << std::endl;

        // Print output tensor values after running FlashAttention
        print_tensor_values(out, "Output Tensor (after FlashAttention)");

        // Print softmax_lse values
        print_tensor_values(softmax_lse, "Softmax LSE");
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
