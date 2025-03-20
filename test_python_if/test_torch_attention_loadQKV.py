import torch
import math
from flash_attn import flash_attn_func

# q: [batch_size, seqlen_q, num_heads, head_size]
# k: [batch_size, seqlen_k, num_heads_k, head_size]
# v: [batch_size, seqlen_v, num_heads_k, head_size]
# num_heads_k != num_heads when MQA or GroupedAttention

def torch_manual_attention(q, k, v, is_causal):
    # q: [batch_size, seqlen_q, num_heads, head_size]
    # k: [batch_size, seqlen_k, num_heads_k, head_size]
    # v: [batch_size, seqlen_v, num_heads_k, head_size]

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

def torch_built_in_attention(q, k, v, is_causal):
    # Prepare k and v in the correct format for scaled_dot_product_attention
    q_permuted = q.permute(0, 2, 1, 3)
    k_permuted = k.permute(0, 2, 1, 3)
    v_permuted = v.permute(0, 2, 1, 3)
    
    attn_out_torch = torch.nn.functional.scaled_dot_product_attention(
        q_permuted, k_permuted, v_permuted, 
        is_causal=is_causal,
    )
    return attn_out_torch

def flash_attention(q, k, v, is_causal):
    # q: [batch_size, seqlen_q, num_heads, head_size]
    # k: [batch_size, seqlen_k, num_heads_k, head_size]
    # v: [batch_size, seqlen_v, num_heads_k, head_size]

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

def attn_out_closeness(manual_attn_out, torch_attn_out, flash_attn_out):
    # compare the results: manual vs torch
    comparison_result = torch.allclose(manual_attn_out, torch_attn_out, atol=atol_val, rtol=rtol_val)
    print("manual_attn_out == torch_attn_out:", comparison_result)

    if not comparison_result:
        max_diff = torch.max(torch.abs(manual_attn_out - torch_attn_out))
        print("Maximum absolute difference:", max_diff.item())

    # compare the results: manual vs flash
    comparison_result = torch.allclose(manual_attn_out, flash_attn_out, atol=atol_val, rtol=rtol_val)
    print("manual_attn_out == flash_attn_out:", comparison_result)

    if not comparison_result:
        max_diff = torch.max(torch.abs(manual_attn_out - flash_attn_out))
        print("Maximum absolute difference:", max_diff.item())

    # compare the results: torch vs flash
    comparison_result = torch.allclose(torch_attn_out, flash_attn_out, atol=atol_val, rtol=rtol_val)
    print("torch_attn_out == flash_attn_out:", comparison_result)

    if not comparison_result:
        max_diff = torch.max(torch.abs(torch_attn_out - flash_attn_out))
        print("Maximum absolute difference:", max_diff.item())
    
def softmax_lse_closeness(manual_softmax_lse, flash_softmax_lse):
    # convert to float32
    manual_softmax_lse = manual_softmax_lse.to(torch.float32)
    flash_softmax_lse = flash_softmax_lse.to(torch.float32)

    comparison_result = torch.allclose(manual_softmax_lse, flash_softmax_lse, atol=atol_val, rtol=rtol_val)
    print("manual_softmax_lse == flash_softmax_lse:", comparison_result)

    if not comparison_result:
        max_diff = torch.max(torch.abs(manual_softmax_lse - flash_softmax_lse))
        print("Maximum absolute difference:", max_diff.item())

def loaded_out_closeness(out_load_flash, out_load_manual, manual_attn_out, flash_attn_out):
    # compare flash_attn_out: python vs cpp
    comparison_result = torch.allclose(out_load_flash, flash_attn_out, atol=atol_val, rtol=rtol_val)
    print("out_load_flash == flash_attn_out:", comparison_result)

    if not comparison_result:
        max_diff = torch.max(torch.abs(out_load_flash - flash_attn_out))
        print("Maximum absolute difference:", max_diff.item())

    # compare manual_attn_out: python vs cpp
    comparison_result = torch.allclose(out_load_manual, manual_attn_out, atol=atol_val, rtol=rtol_val)
    print("out_load_manual == manual_attn_out:", comparison_result)

    if not comparison_result:
        max_diff = torch.max(torch.abs(out_load_flash - manual_attn_out))
        print("Maximum absolute difference:", max_diff.item())
    
    # compare cpp out: flash vs manual
    comparison_result = torch.allclose(out_load_flash, out_load_manual, atol=atol_val, rtol=rtol_val)
    print("out_load_flash == out_load_manual:", comparison_result)

    if not comparison_result:
        max_diff = torch.max(torch.abs(out_load_flash - out_load_manual))
        print("Maximum absolute difference:", max_diff.item())

if __name__ == "__main__":

    BATCH_SIZE = 1
    seqlen_q = 64
    seqlen_k = 64
    num_heads = 16
    head_size = 128
    is_causal = True

    # Check if CUDA is available
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")
    print(f"Using device: {device}")

    # Set common tolerance levels for all comparisons
    atol_val = 1e-2 
    rtol_val = 1e-2
    print(f"Using comparison tolerances: atol={atol_val}, rtol={rtol_val}")

    # load q, k, v tensor from file
    root_path = "/home/yingyih/workspace/flash-attn-integration/flash_attention_lib/build"
    q = torch.jit.load(root_path + "/q.pt")
    q = list(q.parameters())[0]
    k = torch.jit.load(root_path + "/k.pt")
    k = list(k.parameters())[0]
    v = torch.jit.load(root_path + "/v.pt")
    v = list(v.parameters())[0]
    out_load_flash = torch.jit.load(root_path + "/flash_out.pt")
    out_load_flash = list(out_load_flash.parameters())[0]
    out_load_flash = out_load_flash.permute(0, 2, 1, 3)
    out_load_manual = torch.jit.load(root_path + "/torch_out.pt")
    out_load_manual = list(out_load_manual.parameters())[0]
    out_load_manual = out_load_manual.permute(0, 2, 1, 3)
    # print the shape of q, k, v, out_load_flash, out_load_manual
    print("q.shape:", q.shape)
    print("k.shape:", k.shape)
    print("v.shape:", v.shape)
    print("out_load_flash.shape:", out_load_flash.shape)
    print("out_load_manual.shape:", out_load_manual.shape)

    manual_attn_out, manual_softmax_lse = torch_manual_attention(q, k, v, is_causal)
    torch_attn_out = torch_built_in_attention(q, k, v, is_causal)    
    flash_attn_out, flash_softmax_lse = flash_attention(q, k, v, is_causal)

    print("attn_out_load_flash[0][0][0][0]:", out_load_flash[0][0][0][0])
    print("attn_out_load_manual[0][0][0][0]:", out_load_manual[0][0][0][0])
    print("attn_out_manual[0][0][0][0]:", manual_attn_out[0][0][0][0])
    print("attn_out_torch[0][0][0][0]:", torch_attn_out[0][0][0][0])
    print("attn_out_flash[0][0][0][0]:", flash_attn_out[0][0][0][0])

    print("manual_attn_out.shape:", manual_attn_out.shape)
    print("manual_softmax_lse.shape:", manual_softmax_lse.shape)

    attn_out_closeness(manual_attn_out, torch_attn_out, flash_attn_out)
    softmax_lse_closeness(manual_softmax_lse, flash_softmax_lse)

    loaded_out_closeness(out_load_flash, out_load_manual, manual_attn_out, flash_attn_out)



