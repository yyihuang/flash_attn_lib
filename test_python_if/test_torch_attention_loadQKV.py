import torch
import math
from flash_attn import flash_attn_func

# load q, k, v tensor from file
# compute attention from my kernel, torch attention, and flash attention library

# q: [batch_size, seqlen_q, num_heads, head_size]
# k: [batch_size, seqlen_k, num_heads_k, head_size]
# v: [batch_size, seqlen_v, num_heads_k, head_size]
# num_heads_k != num_heads when MQA or GroupedAttention

def torch_manual_attention_fwd(q, k, v, is_causal):
    # q: [batch_size, seqlen_q, num_heads, head_size]
    # k: [batch_size, seqlen_k, num_heads_k, head_size]
    # v: [batch_size, seqlen_v, num_heads_k, head_size]

    batch_size = q.shape[0]
    seqlen_q = q.shape[1]
    num_heads = q.shape[2]
    head_size = q.shape[-1]
    seqlen_k = k.shape[1]

    # Define scaling factor
    scaling_factor = 1.0 / math.sqrt(head_size)

    # q_permuted: [BATCH_SIZE, num_heads, seqlen_q, head_size]
    q_permuted = q.permute(0, 2, 1, 3)

    # k_t: [BATCH_SIZE, num_heads, head_size, seqlen_k]
    k_t = k.permute(0, 2, 3, 1)

    # v_permuted: [BATCH_SIZE, num_heads, seqlen_v, head_size]
    v_permuted = v.permute(0, 2, 1, 3)

    # Compute attention weights
    attn_scores = q_permuted @ k_t * scaling_factor  # Use multiplication instead of division

    # save the first checkpoint tensor
    torch.save(attn_scores, "manual_attn_scores.pt")

    # Apply the causal mask by setting masked positions to a large negative value
    # This will make them effectively zero after softmax
    if is_causal:
        causal_mask = torch.tril(torch.ones(seqlen_q, seqlen_k, device=device))
        masked_attn_scores = attn_scores.masked_fill(causal_mask.reshape(1, 1, seqlen_q, seqlen_k) == 0, float('-inf'))
    else:
        masked_attn_scores = attn_scores

    # Calculate softmax log-sum-exp (LSE) manually
    # This is a numerically stable way to calculate softmax
    # First, find the maximum value along the seqlen_k dimension (for numerical stability)
    max_attn_scores = torch.max(masked_attn_scores, dim=-1, keepdim=True)[0]

    # Calculate exp(attn_scores - max_attn_scores) to prevent overflow
    exp_attn_scores = torch.exp(masked_attn_scores - max_attn_scores)

    # Sum along the seqlen_k dimension
    sum_exp_attn_scores = torch.sum(exp_attn_scores, dim=-1, keepdim=True)

    # Calculate log(sum(exp(attn_scores - max_attn_scores))) + max_attn_scores
    # This is the softmax log-sum-exp (LSE)
    manual_softmax_lse = torch.log(sum_exp_attn_scores) + max_attn_scores

    # Remove the keepdim added for calculation
    manual_softmax_lse = manual_softmax_lse.squeeze(-1)
    # print("manual_softmax_lse.shape:", manual_softmax_lse.shape)

    # softmax along last dimension
    attn_p = torch.softmax(masked_attn_scores, dim=-1)

    # print("attn_p.shape: [BATCH_SIZE, num_heads, seqlen_q, seqlen_k]", attn_p.shape)

    # Compute attention output
    attn_out = attn_p @ v_permuted

    # print("attn_out.shape: [BATCH_SIZE, num_heads, seqlen_q, head_size]", attn_out.shape)

    return attn_out, manual_softmax_lse

def torch_built_in_attention_fwd(q, k, v, is_causal):
    # Prepare k and v in the correct format for scaled_dot_product_attention
    q_permuted = q.permute(0, 2, 1, 3)
    k_permuted = k.permute(0, 2, 1, 3)
    v_permuted = v.permute(0, 2, 1, 3)
    
    attn_out_torch = torch.nn.functional.scaled_dot_product_attention(
        q_permuted, k_permuted, v_permuted, 
        is_causal=is_causal,
    )
    return attn_out_torch

def flash_attention_fwd(q, k, v, is_causal):
    # q: [batch_size, seqlen_q, num_heads, head_size]
    # k: [batch_size, seqlen_k, num_heads_k, head_size]
    # v: [batch_size, seqlen_v, num_heads_k, head_size]

    head_size = q.shape[-1]

    # Define scaling factor
    scaling_factor = 1.0 / math.sqrt(head_size)

    # Default softmax scale in flash_attn_func is 1/sqrt(head_size)
    flash_attn_out, flash_softmax_lse, S_dmask = flash_attn_func(
        q, k, v, 
        causal=is_causal,
        softmax_scale=scaling_factor,  # Explicitly pass the scaling factor
        return_attn_probs=True
    )
    
    # Permute to match our manual implementation's output shape [batch_size, num_heads, seqlen_q, head_size]
    flash_attn_out_permuted = flash_attn_out.permute(0, 2, 1, 3)
    
    # print("flash_attn_out.shape (after permute): [BATCH_SIZE, num_heads, seqlen_q, head_size]", flash_attn_out_permuted.shape)
    # print("flash_softmax_lse.shape:", flash_softmax_lse.shape)
    return flash_attn_out_permuted, flash_softmax_lse

def torch_manual_attention_bwd(dout, q, k, v, out, softmax_lse, dq_, dk_, dv_, alibi_slopes_, p_dropout, softmax_scale, is_causal, window_size_left, window_size_right, softcap, deterministic, gen_, rng_state):
    pass

def flash_attention_bwd(dout, q, k, v, out, softmax_lse, dq_, dk_, dv_, alibi_slopes_, p_dropout, softmax_scale, is_causal, window_size_left, window_size_right, softcap, deterministic, gen_, rng_state):
    pass

def attention_out_closeness_test(manual_out_py, touch_out_py, flash_out_py, manual_out_cpp, mha_out_cpp, run_mha_fwd_out_cpp, atol_val, rtol_val):
    # check shape
    print("manual_out_py.shape:", manual_out_py.shape)
    print("touch_out_py.shape:", touch_out_py.shape)
    print("flash_out_py.shape:", flash_out_py.shape)
    print("manual_out_cpp.shape:", manual_out_cpp.shape)
    print("mha_out_cpp.shape:", mha_out_cpp.shape)
    print("run_mha_fwd_out_cpp.shape:", run_mha_fwd_out_cpp.shape)

    # store name and tensors to a dictionary
    tensors_dict = {
        "manual_out_py": manual_out_py,
        "touch_out_py": touch_out_py,
        "flash_out_py": flash_out_py,
        "manual_out_cpp": manual_out_cpp,
        "mha_out_cpp": mha_out_cpp,
    }

    # compare all the tensors in the dictionary
    for key, value in tensors_dict.items():
        print(f"{key}.shape:", value.shape)
        # compare value with run_mha_fwd_out_cpp
        comparison_result = torch.allclose(value, run_mha_fwd_out_cpp, atol=atol_val, rtol=rtol_val)
        print(f"{key} == run_mha_fwd_out_cpp:", comparison_result)

        if not comparison_result:
            max_diff = torch.max(torch.abs(value - run_mha_fwd_out_cpp))
            print("Maximum absolute difference:", max_diff.item())

def softmax_lse_closeness_test(manual_softmax_lse_py, flash_softmax_lse_py, manual_softmax_lse_cpp, mha_fwd_softmax_lse_cpp, run_mha_fwd_softmax_lse_cpp, atol_val, rtol_val):
    # convert all tensors to float32
    manual_softmax_lse_py = manual_softmax_lse_py.to(torch.float32)
    flash_softmax_lse_py = flash_softmax_lse_py.to(torch.float32)
    manual_softmax_lse_cpp = manual_softmax_lse_cpp.to(torch.float32)
    mha_fwd_softmax_lse_cpp = mha_fwd_softmax_lse_cpp.to(torch.float32)
    run_mha_fwd_softmax_lse_cpp = run_mha_fwd_softmax_lse_cpp.to(torch.float32)

    # check shape
    print("manual_softmax_lse_py.shape:", manual_softmax_lse_py.shape)
    print("flash_softmax_lse_py.shape:", flash_softmax_lse_py.shape)
    print("manual_softmax_lse_cpp.shape:", manual_softmax_lse_cpp.shape)
    print("mha_fwd_softmax_lse_cpp.shape:", mha_fwd_softmax_lse_cpp.shape)
    print("run_mha_fwd_softmax_lse_cpp.shape:", run_mha_fwd_softmax_lse_cpp.shape)

    # store name and tensors to a dictionary
    tensors_dict = {
        "manual_softmax_lse_py": manual_softmax_lse_py,
        "flash_softmax_lse_py": flash_softmax_lse_py,
        "manual_softmax_lse_cpp": manual_softmax_lse_cpp,
        "mha_fwd_softmax_lse_cpp": mha_fwd_softmax_lse_cpp,
    }

    # compare all the tensors in the dictionary
    for key, value in tensors_dict.items():
        print(f"{key}.shape:", value.shape)
        # compare value with run_mha_fwd_softmax_lse_cpp
        comparison_result = torch.allclose(value, run_mha_fwd_softmax_lse_cpp, atol=atol_val, rtol=rtol_val)
        print(f"{key} == run_mha_fwd_softmax_lse_cpp:", comparison_result)

        if not comparison_result:
            max_diff = torch.max(torch.abs(value - run_mha_fwd_softmax_lse_cpp))
            print("Maximum absolute difference:", max_diff.item())

if __name__ == "__main__":

    BATCH_SIZE = 1
    is_causal = True

    # Check if CUDA is available
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")
    print(f"Using device: {device}")

    # Set common tolerance levels for all comparisons
    atol_val = 1e-3
    rtol_val = 1e-3
    print(f"Using comparison tolerances: atol={atol_val}, rtol={rtol_val}")

    # load q, k, v tensor from file
    root_path = "/home/yingyih/workspace/flash-attn-integration/flash_attention_lib/build"
    q = torch.jit.load(root_path + "/q.pt")
    q = list(q.parameters())[0]
    k = torch.jit.load(root_path + "/k.pt")
    k = list(k.parameters())[0]
    v = torch.jit.load(root_path + "/v.pt")
    v = list(v.parameters())[0]

    # load forward pass output from cpp
    run_mha_fwd_out_cpp = torch.jit.load(root_path + "/run_mha_fwd_out_cpp.pt")
    run_mha_fwd_out_cpp = list(run_mha_fwd_out_cpp.parameters())[0]
    run_mha_fwd_out_cpp = run_mha_fwd_out_cpp.permute(0, 2, 1, 3)

    mha_fwd_out_cpp = torch.jit.load(root_path + "/mha_fwd_out_cpp.pt")
    mha_fwd_out_cpp = list(mha_fwd_out_cpp.parameters())[0]
    mha_fwd_out_cpp = mha_fwd_out_cpp.permute(0, 2, 1, 3)

    manual_out_cpp = torch.jit.load(root_path + "/manual_out_cpp.pt")
    manual_out_cpp = list(manual_out_cpp.parameters())[0]
    manual_out_cpp = manual_out_cpp.permute(0, 2, 1, 3)

    run_mha_fwd_softmax_lse_cpp = torch.jit.load(root_path + "/run_mha_fwd_softmax_lse_cpp.pt")
    run_mha_fwd_softmax_lse_cpp = list(run_mha_fwd_softmax_lse_cpp.parameters())[0]
    
    mha_fwd_softmax_lse_cpp = torch.jit.load(root_path + "/mha_fwd_softmax_lse_cpp.pt")
    mha_fwd_softmax_lse_cpp = list(mha_fwd_softmax_lse_cpp.parameters())[0]

    manual_softmax_lse_cpp = torch.jit.load(root_path + "/manual_softmax_lse_cpp.pt")
    manual_softmax_lse_cpp = list(manual_softmax_lse_cpp.parameters())[0]

    # load backward pass output from cpp
    mha_bwd_dq_cpp = torch.jit.load(root_path + "/mha_bwd_dq_cpp.pt")
    mha_bwd_dq_cpp = list(mha_bwd_dq_cpp.parameters())[0]

    mha_bwd_dk_cpp = torch.jit.load(root_path + "/mha_bwd_dk_cpp.pt")
    mha_bwd_dk_cpp = list(mha_bwd_dk_cpp.parameters())[0]

    mha_bwd_dv_cpp = torch.jit.load(root_path + "/mha_bwd_dv_cpp.pt")
    mha_bwd_dv_cpp = list(mha_bwd_dv_cpp.parameters())[0]

    run_mha_bwd_dq_cpp = torch.jit.load(root_path + "/run_mha_bwd_dq_cpp.pt")
    run_mha_bwd_dq_cpp = list(run_mha_bwd_dq_cpp.parameters())[0]

    run_mha_bwd_dk_cpp = torch.jit.load(root_path + "/run_mha_bwd_dk_cpp.pt")
    run_mha_bwd_dk_cpp = list(run_mha_bwd_dk_cpp.parameters())[0]

    run_mha_bwd_dv_cpp = torch.jit.load(root_path + "/run_mha_bwd_dv_cpp.pt")
    run_mha_bwd_dv_cpp = list(run_mha_bwd_dv_cpp.parameters())[0]
    

    # print the shape of q, k, v, out_load_flash, out_load_manual
    print("q.shape:", q.shape)
    print("k.shape:", k.shape)
    print("v.shape:", v.shape)

    manual_attn_out, manual_softmax_lse = torch_manual_attention_fwd(q, k, v, is_causal)
    torch_attn_out = torch_built_in_attention_fwd(q, k, v, is_causal)    
    flash_attn_out, flash_softmax_lse = flash_attention_fwd(q, k, v, is_causal)

    # compare attention out in forward pass
    attention_out_closeness_test(manual_attn_out, torch_attn_out, flash_attn_out, manual_out_cpp, mha_fwd_out_cpp, run_mha_fwd_out_cpp, atol_val, rtol_val)
    
    # compare softmax_lse in forward pass
    softmax_lse_closeness_test(manual_softmax_lse, flash_softmax_lse, manual_softmax_lse_cpp, mha_fwd_softmax_lse_cpp, run_mha_fwd_softmax_lse_cpp, atol_val, rtol_val)    
    
    # compare backward pass output from manual, torch, and flash attention in py

'''
(flash) [yingyih@catalyst-0-15 flash_attention_lib]$ python3 /home/yingyih/workspace/flash-attn-integration/flash_attention_lib/test_python_if/test_torch_attention_loadQKV.py
Using device: cuda
Using comparison tolerances: atol=0.001, rtol=0.001
q.shape: torch.Size([1, 64, 8, 128])
k.shape: torch.Size([1, 64, 8, 128])
v.shape: torch.Size([1, 64, 8, 128])
manual_out_py.shape: torch.Size([1, 8, 64, 128])
touch_out_py.shape: torch.Size([1, 8, 64, 128])
flash_out_py.shape: torch.Size([1, 8, 64, 128])
manual_out_cpp.shape: torch.Size([1, 8, 64, 128])
mha_out_cpp.shape: torch.Size([1, 8, 64, 128])
run_mha_fwd_out_cpp.shape: torch.Size([1, 8, 64, 128])
manual_out_py.shape: torch.Size([1, 8, 64, 128])
manual_out_py == run_mha_fwd_out_cpp: False
Maximum absolute difference: 0.001953125
touch_out_py.shape: torch.Size([1, 8, 64, 128])
touch_out_py == run_mha_fwd_out_cpp: True
flash_out_py.shape: torch.Size([1, 8, 64, 128])
flash_out_py == run_mha_fwd_out_cpp: True
manual_out_cpp.shape: torch.Size([1, 8, 64, 128])
manual_out_cpp == run_mha_fwd_out_cpp: False
Maximum absolute difference: 0.001953125
mha_out_cpp.shape: torch.Size([1, 8, 64, 128])
mha_out_cpp == run_mha_fwd_out_cpp: True
manual_softmax_lse_py.shape: torch.Size([1, 8, 64])
flash_softmax_lse_py.shape: torch.Size([1, 8, 64])
manual_softmax_lse_cpp.shape: torch.Size([1, 8, 64])
mha_fwd_softmax_lse_cpp.shape: torch.Size([1, 8, 64])
run_mha_fwd_softmax_lse_cpp.shape: torch.Size([1, 8, 64])
manual_softmax_lse_py.shape: torch.Size([1, 8, 64])
manual_softmax_lse_py == run_mha_fwd_softmax_lse_cpp: True
flash_softmax_lse_py.shape: torch.Size([1, 8, 64])
flash_softmax_lse_py == run_mha_fwd_softmax_lse_cpp: True
manual_softmax_lse_cpp.shape: torch.Size([1, 8, 64])
manual_softmax_lse_cpp == run_mha_fwd_softmax_lse_cpp: True
mha_fwd_softmax_lse_cpp.shape: torch.Size([1, 8, 64])
mha_fwd_softmax_lse_cpp == run_mha_fwd_softmax_lse_cpp: True
'''


