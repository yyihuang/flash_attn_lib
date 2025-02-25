#include <torch/torch.h>
#include <torch/extension.h>
#include <iostream>
#include <vector>
#include "flash_api.h"
#include <dlfcn.h>

int main()
{
    // Define tensor dimensions
    int batch_size = 2, seqlen_q = 16, seqlen_k = 16;
    int num_heads = 8, head_size = 64;

    // Create input tensors
    at::Tensor q = torch::randn({batch_size, seqlen_q, num_heads, head_size}, torch::kHalf);
    at::Tensor k = torch::randn({batch_size, seqlen_k, num_heads, head_size}, torch::kHalf);
    at::Tensor v = torch::randn({batch_size, seqlen_k, num_heads, head_size}, torch::kHalf);

    std::optional<at::Tensor> out_;
    std::optional<at::Tensor> alibi_slopes_;

    float p_dropout = 0.0; // disable dropout and thus torch dependency here?
    float softmax_scale = 1.0 / sqrt(head_size);
    bool is_causal = false;
    int window_size_left = -1, window_size_right = -1;
    float softcap = 1.0;
    bool return_softmax = false;
    std::optional<at::Generator> gen_;

    // Test if mha_fwd is linked
    try {
        std::cout << "CUDA available: " << torch::cuda::is_available() << std::endl;
        std::cout << "CUDNN available: " << torch::cuda::cudnn_is_available() << std::endl;

        // Test a simple tensor creation
        at::Tensor test_tensor = torch::randn({2, 2}, torch::kCUDA);
        std::cout << "Tensor on CUDA:\n" << test_tensor << std::endl;
        
        // Verify FlashAttention symbol is available
        void* symbol = dlsym(RTLD_DEFAULT, "mha_fwd");
        if (symbol) {
            std::cout << "mha_fwd symbol found!" << std::endl;
        } else {
            std::cerr << "Error: mha_fwd symbol NOT found!" << std::endl;
        }

    } catch (const c10::Error& e) {
        std::cerr << "Torch error: " << e.what() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Standard exception: " << e.what() << std::endl;
    }

    // Call mha_fwd (API-0)
    try
    {
        std::vector<at::Tensor> output = mha_fwd(q, k, v, out_, alibi_slopes_,
                                                 p_dropout, softmax_scale, is_causal,
                                                 window_size_left, window_size_right,
                                                 softcap, return_softmax, gen_);

        std::cout << "MHA forward output shape: " << output[0].sizes() << std::endl;
    }
    catch (const c10::Error &e)
    {
        std::cerr << "Torch error: " << e.what() << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Standard exception: " << e.what() << std::endl;
    }
    catch (...)
    {
        std::cerr << "Unknown error occurred!" << std::endl;
    }
    std::cout << "CUDA tensors created successfully!\n"; // test if everything other than mha is working

    return 0;
}
