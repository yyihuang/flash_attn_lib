#include <torch/torch.h>
#include <torch/extension.h>
#include <iostream>
#include <vector>
#include "flash_api.h"

int main() {
    // Define tensor dimensions
    int batch_size = 2, seqlen_q = 16, seqlen_k = 16;
    int num_heads = 8, head_size = 64;

    // Create input tensors
    at::Tensor q = torch::randn({batch_size, seqlen_q, num_heads, head_size}, torch::kCUDA);
    at::Tensor k = torch::randn({batch_size, seqlen_k, num_heads, head_size}, torch::kCUDA);
    at::Tensor v = torch::randn({batch_size, seqlen_k, num_heads, head_size}, torch::kCUDA);

    std::optional<at::Tensor> out_;
    std::optional<at::Tensor> alibi_slopes_;

    float p_dropout = 0.0; // disable dropout and thus torch dependency here?
    float softmax_scale = 1.0 / sqrt(head_size);
    bool is_causal = false;
    int window_size_left = -1, window_size_right = -1;
    float softcap = 1.0;
    bool return_softmax = false;
    std::optional<at::Generator> gen_;

    // Call mha_fwd
    std::vector<at::Tensor> output = mha_fwd(q, k, v, out_, alibi_slopes_,
                                             p_dropout, softmax_scale, is_causal,
                                             window_size_left, window_size_right,
                                             softcap, return_softmax, gen_);

    std::cout << "MHA forward output shape: " << output[0].sizes() << std::endl;

    return 0;
}