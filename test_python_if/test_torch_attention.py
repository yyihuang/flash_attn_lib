import torch
import math
from flash_attn import flash_attn_func

# Random q, k, v
# q: [batch_size, seqlen_q, num_heads, head_size]
# k: [batch_size, seqlen_k, num_heads_k, head_size]
# v: [batch_size, seqlen_v, num_heads_k, head_size]
# num_heads_k != num_heads when MQA or GroupedAttention

BATCH_SIZE = 1
seqlen_q = 10
seqlen_k = 10
num_heads = 12
head_size = 64
is_causal = False

# Check if CUDA is available
cuda_available = torch.cuda.is_available()
device = torch.device("cuda" if cuda_available else "cpu")
print(f"Using device: {device}")

# Set the data type to be used consistently for all tensors
# Using bfloat16 as it has better numerical stability than float16 while saving memory compared to float32
dtype = torch.bfloat16
print(f"Using dtype: {dtype}")

# set seed
torch.manual_seed(42)

# Create tensors on the appropriate device - using the same data type for all tensors
q = torch.randn(BATCH_SIZE, seqlen_q, num_heads, head_size, device=device, dtype=dtype)
k = torch.randn(BATCH_SIZE, seqlen_k, num_heads, head_size, device=device, dtype=dtype)
v = torch.randn(BATCH_SIZE, seqlen_k, num_heads, head_size, device=device, dtype=dtype)

# Define scaling factor
scaling_factor = 1.0 / math.sqrt(head_size)

# q: [BATCH_SIZE, num_heads, seqlen_q, head_size]
q_permuted = q.permute(0, 2, 1, 3)

# k_t: [BATCH_SIZE, num_heads, head_size, seqlen_k]
k_t = k.permute(0, 2, 3, 1)

# Compute attention weights
attn_scores = q_permuted @ k_t * scaling_factor  # Use multiplication instead of division

print("q_permuted.shape: [BATCH_SIZE, num_heads, seqlen_q, head_size]", q_permuted.shape)
print("k_t.shape: [BATCH_SIZE, num_heads, head_size, seqlen_k]", k_t.shape)
print("attn_scores.shape: [BATCH_SIZE, num_heads, seqlen_q, seqlen_k]", attn_scores.shape)


# Apply the causal mask by setting masked positions to a large negative value
# This will make them effectively zero after softmax
if is_causal:
    causal_mask = torch.tril(torch.ones(seqlen_q, seqlen_k, device=device))
    masked_attn_scores = attn_scores.masked_fill(causal_mask.reshape(1, 1, seqlen_q, seqlen_k) == 0, float('-inf'))
else:
    masked_attn_scores = attn_scores

# softmax along last dimension
attn_p = torch.softmax(masked_attn_scores, dim=-1)

print("attn_p.shape: [BATCH_SIZE, num_heads, seqlen_q, seqlen_k]", attn_p.shape)

# Compute attention output
v_permuted = v.permute(0, 2, 1, 3)
attn_out = attn_p @ v_permuted

print("attn_out.shape: [BATCH_SIZE, num_heads, seqlen_q, head_size]", attn_out.shape)

# Prepare k and v in the correct format for scaled_dot_product_attention
# k needs to be [BATCH_SIZE, num_heads, seqlen_k, head_size]
k_permuted = k.permute(0, 2, 1, 3)

# Set common tolerance levels for all comparisons
atol_val = 1e-2 
rtol_val = 1e-2
print(f"Using comparison tolerances: atol={atol_val}, rtol={rtol_val}")

# compare the results with torch.nn.functional.scaled_dot_product_attention
# Pass the attn_mask to ensure causal masking
attn_out_torch = torch.nn.functional.scaled_dot_product_attention(
    q_permuted, k_permuted, v_permuted, 
    attn_mask=None, 
    is_causal=is_causal,
    scale=scaling_factor  # Explicitly pass the same scaling factor
)

print("attn_out_torch.shape: [BATCH_SIZE, num_heads, seqlen_q, head_size]", attn_out_torch.shape)

# compare the results: manual vs torch
comparison_result = torch.allclose(attn_out, attn_out_torch, atol=atol_val, rtol=rtol_val)
print("attn_out == attn_out_torch:", comparison_result)

# If still not close, check the maximum difference
if not comparison_result:
    max_diff = torch.max(torch.abs(attn_out - attn_out_torch))
    print("Maximum absolute difference:", max_diff.item())
    
    # Show a sample of differences
    print("\nSample differences (first 3 positions, first head):")
    for i in range(min(3, seqlen_q)):
        for j in range(min(3, head_size)):
            manual_val = attn_out[0, 0, i, j].item()
            torch_val = attn_out_torch[0, 0, i, j].item()
            diff = abs(manual_val - torch_val)
            print(f"Position [{i},{j}]: Manual={manual_val:.6f}, Torch={torch_val:.6f}, Diff={diff:.6f}")

# Flash Attention is only supported on CUDA devices
if cuda_available:
    # The format for flash_attn_func is [batch_size, seqlen, num_heads, head_size]
    # For Flash Attention, we use the tensors as is since they're already in bfloat16
    q_flash = q.clone().contiguous()  # Original shape: [batch_size, seqlen_q, num_heads, head_size]
    k_flash = k.clone().contiguous()  # Original shape: [batch_size, seqlen_k, num_heads, head_size]
    v_flash = v.clone().contiguous()  # Original shape: [batch_size, seqlen_k, num_heads, head_size]
    
    # Default softmax scale in flash_attn_func is 1/sqrt(head_size)
    flash_attn_out = flash_attn_func(
        q_flash, k_flash, v_flash, 
        causal=is_causal,
        softmax_scale=scaling_factor,  # Explicitly pass the scaling factor
        return_attn_probs=False
    )
    
    # Permute to match our manual implementation's output shape [batch_size, num_heads, seqlen_q, head_size]
    flash_attn_out_permuted = flash_attn_out.permute(0, 2, 1, 3)
    
    print("flash_attn_out.shape (after permute): [BATCH_SIZE, num_heads, seqlen_q, head_size]", flash_attn_out_permuted.shape)
    
    # compare the results: manual vs flash attention
    comparison_result_flash = torch.allclose(attn_out, flash_attn_out_permuted, atol=atol_val, rtol=rtol_val)
    print("attn_out == flash_attn_out:", comparison_result_flash)
    
    if not comparison_result_flash:
        max_diff_flash = torch.max(torch.abs(attn_out - flash_attn_out_permuted))
        print("Maximum absolute difference with Flash Attention:", max_diff_flash.item())
    
    # compare the results: torch vs flash attention
    comparison_result_torch_flash = torch.allclose(attn_out_torch, flash_attn_out_permuted, atol=atol_val, rtol=rtol_val)
    print("attn_out_torch == flash_attn_out:", comparison_result_torch_flash)
    
    if not comparison_result_torch_flash:
        max_diff_torch_flash = torch.max(torch.abs(attn_out_torch - flash_attn_out_permuted))
        print("Maximum absolute difference between torch and Flash Attention:", max_diff_torch_flash.item())
            
        

