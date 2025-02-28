#include <torch/torch.h>
#include <torch/extension.h>
#include <iostream>
#include <vector>
#include "../../3rd_party/flexflow-flash-attention/flexflow_flash_api.h"
using namespace flash;

// extern std::vector<at::Tensor> mha_fwd(
//     at::Tensor &q, 
//     const at::Tensor &k, 
//     const at::Tensor &v, 
//     std::optional<at::Tensor> &out_,
//     std::optional<at::Tensor> &alibi_slopes_,
//     const float p_dropout,
//     const float softmax_scale,
//     bool is_causal,
//     int window_size_left,
//     int window_size_right,
//     const float softcap,
//     const bool return_softmax,
//     std::optional<at::Generator> gen_
// );


int main() {
    int batch_size = 2, seqlen_q = 16, seqlen_k = 16;
    int num_heads = 8, head_size = 32;  // hdim=32

    // Ensure FP16 (half precision)
    at::Tensor q = torch::randn({batch_size, seqlen_q, num_heads, head_size}, 
                                 torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA));

    at::Tensor k = torch::randn({batch_size, seqlen_k, num_heads, head_size}, 
                                 torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA));

    at::Tensor v = torch::randn({batch_size, seqlen_k, num_heads, head_size}, 
                                 torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA));

    std::optional<at::Tensor> out_;
    std::optional<at::Tensor> alibi_slopes_;

    float p_dropout = 0.0;
    float softmax_scale = 1.0 / sqrt(head_size);
    bool is_causal = false;
    int window_size_left = -1, window_size_right = -1;
    float softcap = 1.0;
    bool return_softmax = false;  // Ensure softmax LSE is NOT returned
    std::optional<at::Generator> gen_;

    // Ensure the function is called correctly with FP16 tensors
    std::vector<at::Tensor> output;
    try {
        output = flash::mha_fwd(q, k, v, out_, alibi_slopes_,
                                p_dropout, softmax_scale, is_causal,
                                window_size_left, window_size_right,
                                softcap, return_softmax, gen_);
        
        // Ensure CUDA synchronization
        torch::cuda::synchronize();
    } catch (const std::exception &e) {
        std::cerr << "MHA forward pass failed: " << e.what() << std::endl;
        return 1;
    }

    if (output.empty()) {
        std::cerr << "MHA forward returned an empty output!" << std::endl;
        return 1;
    }

    // Output tensor
    std::cout << "MHA forward output shape: " << output[0].sizes() << std::endl;
    std::cout << "MHA forward output values:\n" << output[0] << std::endl;

    return 0;
}
