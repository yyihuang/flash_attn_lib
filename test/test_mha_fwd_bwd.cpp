#include <torch/torch.h>
#include <torch/extension.h>
#include <iostream>
#include <vector>
#include "flash_api.h"



int main()
{
    int batch_size = 2, seqlen_q = 16, seqlen_k = 16;
    int num_heads = 8, head_size = 32; // hdim=32

    // Ensure FP16 (half precision)
    at::Tensor q = torch::randn({batch_size, seqlen_q, num_heads, head_size},
                                torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA).requires_grad(true));

    at::Tensor k = torch::randn({batch_size, seqlen_k, num_heads, head_size},
                                torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA).requires_grad(true));

    at::Tensor v = torch::randn({batch_size, seqlen_k, num_heads, head_size},
                                torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA).requires_grad(true));

    std::optional<at::Tensor> out_;
    std::optional<at::Tensor> alibi_slopes_;

    float p_dropout = 0.1;
    float softmax_scale = 1.0 / sqrt(head_size);
    bool is_causal = false;
    int window_size_left = -1, window_size_right = -1;
    float softcap = 0.0;
    bool return_softmax = true; // return for bwd
    std::optional<at::Generator> gen_;

    // Forward pass
    std::vector<at::Tensor> output;
    try
    {
        output = mha_fwd(q, k, v, out_, alibi_slopes_,
                                p_dropout, softmax_scale, is_causal,
                                window_size_left, window_size_right,
                                softcap, return_softmax, gen_);

        torch::cuda::synchronize();
    }
    catch (const std::exception &e)
    {
        std::cerr << "MHA forward pass failed: " << e.what() << std::endl;
        return 1;
    }

    if (output.empty())
    {
        std::cerr << "MHA forward returned an empty output!" << std::endl;
        return 1;
    }

    at::Tensor out = output[0];
    at::Tensor softmax_lse = output[1];
    at::Tensor p = output[2];
    std::optional<at::Tensor> rng_state = output[3];

    // Print forward results
    std::cout << "MHA forward output shape: " << out.sizes() << std::endl;
    std::cout << "MHA forward output values:\n"
              << out << std::endl;

    std::cout << "MHA forward lse shape: " << softmax_lse.sizes() << std::endl;
    std::cout << "MHA forward lse values:\n"
              << softmax_lse << std::endl;

    // Create dout (random gradient tensor)
    at::Tensor dout = torch::randn_like(out, torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA));

    std::optional<at::Tensor> dq_, dk_, dv_;

    bool deterministic = false;

    // Backward pass
    if (softmax_lse.device().is_cpu())
    {
        std::cout << "Moving softmax_lse to CUDA..." << std::endl;
        softmax_lse = softmax_lse.to(torch::kCUDA);
    }

    std::vector<at::Tensor> grad_output;
    try
    {
        grad_output = mha_bwd(dout, q, k, v, out, softmax_lse,
                                     dq_, dk_, dv_, alibi_slopes_,
                                     p_dropout, softmax_scale, is_causal,
                                     window_size_left, window_size_right,
                                     softcap, deterministic, gen_, rng_state);

        torch::cuda::synchronize();
    }
    catch (const std::exception &e)
    {
        std::cerr << "MHA backward pass failed: " << e.what() << std::endl;
        return 1;
    }

    if (grad_output.empty())
    {
        std::cerr << "MHA backward returned an empty output!" << std::endl;
        return 1;
    }

    at::Tensor dq = grad_output[0];
    at::Tensor dk = grad_output[1];
    at::Tensor dv = grad_output[2];

    // Print backward results
    std::cout << "Gradients of q shape: " << dq.sizes() << std::endl;
    std::cout << "dq values:\n"
              << dq << std::endl;

    std::cout << "Gradients of k shape: " << dk.sizes() << std::endl;
    std::cout << "dk values:\n"
              << dk << std::endl;

    std::cout << "Gradients of v shape: " << dv.sizes() << std::endl;
    std::cout << "dv values:\n"
              << dv << std::endl;

    return 0;
}
