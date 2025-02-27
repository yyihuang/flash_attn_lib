#include <torch/torch.h>
#include <torch/extension.h>
#include <cuda_runtime.h>

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

        int batch_size = 2, seqlen_q = 16, seqlen_k = 16;
        int num_heads = 8, num_heads_k = 8, head_size = 64;

        if (head_size % 8 != 0)
            throw std::runtime_error("head_size must be a multiple of 8");

        // Forward Pass: Allocate input tensors
        // q size: (batch_size, seqlen_q, num_heads, head_size)
        // k size: (batch_size, seqlen_k, num_heads_k, head_size)
        // v size: (batch_size, seqlen_k, num_heads_k, head_size)
        at::Tensor q = torch::randn({batch_size, seqlen_q, num_heads, head_size},
                                    torch::dtype(torch::kFloat16).device(torch::kCUDA));
        at::Tensor k = torch::randn({batch_size, seqlen_k, num_heads_k, head_size},
                                    torch::dtype(torch::kFloat16).device(torch::kCUDA));
        at::Tensor v = torch::randn({batch_size, seqlen_k, num_heads_k, head_size},
                                    torch::dtype(torch::kFloat16).device(torch::kCUDA));
        std::cout << "Q Tensor Shape: " << q.sizes() << std::endl;
        std::cout << "K Tensor Shape: " << k.sizes() << std::endl;
        std::cout << "V Tensor Shape: " << v.sizes() << std::endl;

        // Forward pass output tensors
        at::Tensor out = torch::empty_like(q);
        at::Tensor softmax_lse = torch::empty({batch_size, num_heads, seqlen_q},
                                              torch::dtype(torch::kFloat).device(torch::kCUDA));

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
                         -1, -1,                      // window_size_left, window_size_right
                         1.0f                         // softcap
        );

        auto stream = at::cuda::getCurrentCUDAStream().stream();
        run_mha_fwd(fwd_params, stream);
        torch::cuda::synchronize();

        // Store `out` (Python: out_padded) for backward pass
        // out size: (batch_size, seqlen_q, num_heads, head_size)

        std::cout << "Output Tensor Shape: " << out.sizes() << std::endl;
        std::cout << "Softmax LSE Shape: " << softmax_lse.sizes() << std::endl;

        print_tensor_values(out, "Output Tensor (after Forward)");
        print_tensor_values(softmax_lse, "Softmax LSE");

        // Backward Pass: Create dout, ensuring alignment with PyTorch behavior
        at::Tensor dout = torch::randn({batch_size, seqlen_q, num_heads, head_size},
                                       torch::dtype(torch::kFloat16).device(torch::kCUDA));

        // Handle padding for dout_padded (Python equivalent)
        int pad_size = head_size % 8 == 0 ? 0 : (8 - head_size % 8);
        at::Tensor dout_padded = dout;
        if (pad_size > 0)
        {
            dout_padded = torch::cat({dout, torch::zeros({batch_size, seqlen_q, num_heads, pad_size},
                                                         torch::dtype(torch::kFloat16).device(torch::kCUDA))},
                                     -1);
        }

        // print the size of dout_padded
        std::cout << "dout_padded Tensor Shape: " << dout_padded.sizes() << std::endl;

        // Allocate gradient tensors
        at::Tensor dq = torch::empty_like(q);
        at::Tensor dk = torch::empty_like(k);
        at::Tensor dv = torch::empty_like(v);

        std::cout << "Running FlashAttention Backward..." << std::endl;

        // Set backward parameters
        Flash_bwd_params bwd_params;
        set_params_dgrad(bwd_params,
                         batch_size, seqlen_q, seqlen_k,
                         seqlen_q, seqlen_k, // Rounded lengths
                         num_heads, num_heads_k,
                         head_size, head_size, // Rounded head_size
                         q, k, v, out,
                         dout_padded, dq, dk, dv,
                         nullptr, nullptr, // cu_seqlens_q_d, cu_seqlens_k_d
                         nullptr, nullptr, nullptr, // dq_accum_d, dk_accum_d, dv_accum_d
                         softmax_lse.data_ptr(),
                         nullptr,                     // dsoftmax_sum_d
                         0.0f,                        // p_dropout = 0
                         1.0f / std::sqrt(head_size), // softmax scale
                         -1, -1,                      // window_size_left, window_size_right
                         1.0f,                        // softcap
                         false,                       // deterministic
                         false                        // unpadded_lse
        );

        run_mha_bwd(bwd_params, stream);
        torch::cuda::synchronize();
        CHECK_CUDA_ERRORS();

        // Trim dq, dk, dv (equivalent to `dq[..., :dout.shape[-1]]` in Python)
        if (pad_size > 0)
        {
            dq = dq.index({"...", torch::indexing::Slice(0, head_size)});
            dk = dk.index({"...", torch::indexing::Slice(0, head_size)});
            dv = dv.index({"...", torch::indexing::Slice(0, head_size)});
        }

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