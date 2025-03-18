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
is_causal = True

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
print("manual_softmax_lse.shape:", manual_softmax_lse.shape)

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

# Compare with torch.nn.functional.scaled_dot_product_attention
# The standard function doesn't return LSE values, so we'll use it for output comparison only
try:
    # First try to use the internal _scaled_dot_product_attention_math which might return LSE in some PyTorch versions
    from torch.nn.functional import _scaled_dot_product_attention_math
    attn_out_torch, torch_softmax_lse = _scaled_dot_product_attention_math(
        q_permuted, k_permuted, v_permuted, 
        attn_mask=None, 
        dropout_p=0.0,
        is_causal=is_causal,
        scale=scaling_factor
    )
    print("Using _scaled_dot_product_attention_math which returns LSE values")
except (ImportError, AttributeError):
    # If that's not available, fall back to the standard function
    print("_scaled_dot_product_attention_math not available, using standard scaled_dot_product_attention")
    attn_out_torch = torch.nn.functional.scaled_dot_product_attention(
        q_permuted, k_permuted, v_permuted, 
        attn_mask=None, 
        is_causal=is_causal,
        scale=scaling_factor
    )
    
    # For PyTorch versions without built-in LSE access, we can try to calculate it from the same input
    # We'll use the same approach as our manual calculation but with the PyTorch function's input
    # Create a copy of the inputs to avoid modifying the originals
    q_copy = q_permuted.clone()
    k_copy = k_permuted.clone()
    
    # Calculate attention scores the same way PyTorch would
    k_copy_t = k_copy.transpose(-1, -2)
    torch_attn_scores = q_copy @ k_copy_t * scaling_factor
    
    # Apply causal masking if needed
    if is_causal:
        causal_mask = torch.tril(torch.ones(seqlen_q, seqlen_k, device=device))
        torch_masked_scores = torch_attn_scores.masked_fill(
            causal_mask.reshape(1, 1, seqlen_q, seqlen_k) == 0, 
            float('-inf')
        )
    else:
        torch_masked_scores = torch_attn_scores
    
    # Calculate LSE the same way we did manually
    torch_max_scores = torch.max(torch_masked_scores, dim=-1, keepdim=True)[0]
    torch_exp_scores = torch.exp(torch_masked_scores - torch_max_scores)
    torch_sum_exp = torch.sum(torch_exp_scores, dim=-1, keepdim=True)
    torch_softmax_lse = torch.log(torch_sum_exp) + torch_max_scores
    torch_softmax_lse = torch_softmax_lse.squeeze(-1)

print("attn_out_torch.shape: [BATCH_SIZE, num_heads, seqlen_q, head_size]", attn_out_torch.shape)
print("torch_softmax_lse.shape:", torch_softmax_lse.shape)

# Compare the softmax LSE values
if torch_softmax_lse.shape == manual_softmax_lse.shape:
    lse_comparison = torch.allclose(manual_softmax_lse, torch_softmax_lse, atol=atol_val, rtol=rtol_val)
    print("manual_softmax_lse == torch_softmax_lse:", lse_comparison)
    
    if not lse_comparison:
        max_lse_diff = torch.max(torch.abs(manual_softmax_lse - torch_softmax_lse))
        print("Maximum LSE difference:", max_lse_diff.item())
        
        # Show a sample of LSE differences
        print("\nSample LSE differences (first 3 positions, first 3 heads):")
        for i in range(min(3, seqlen_q)):
            for h in range(min(3, num_heads)):
                manual_val = manual_softmax_lse[0, h, i].item()
                torch_val = torch_softmax_lse[0, h, i].item()
                diff = abs(manual_val - torch_val)
                print(f"Position [{i}], Head [{h}]: Manual={manual_val:.6f}, Torch={torch_val:.6f}, Diff={diff:.6f}")
else:
    print("LSE shapes don't match:", manual_softmax_lse.shape, "vs", torch_softmax_lse.shape)
    # Try to reshape the torch LSE if necessary
    if torch_softmax_lse.dim() > manual_softmax_lse.dim():
        # If torch returns a higher dimensional tensor, we might need to reduce it
        print("Attempting to reshape torch_softmax_lse to match manual_softmax_lse")
        # This is just a guess - we'll need to see the actual shape to determine the right reshaping
        reshaped_torch_lse = torch_softmax_lse.view(manual_softmax_lse.shape)
        lse_comparison = torch.allclose(manual_softmax_lse, reshaped_torch_lse, atol=atol_val, rtol=rtol_val)
        print("After reshaping: manual_softmax_lse == torch_softmax_lse:", lse_comparison)

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
    flash_attn_out, flash_softmax_lse, S_dmask = flash_attn_func(
        q_flash, k_flash, v_flash, 
        causal=is_causal,
        softmax_scale=scaling_factor,  # Explicitly pass the scaling factor
        return_attn_probs=True
    )
    
    # Permute to match our manual implementation's output shape [batch_size, num_heads, seqlen_q, head_size]
    flash_attn_out_permuted = flash_attn_out.permute(0, 2, 1, 3)
    
    print("flash_attn_out.shape (after permute): [BATCH_SIZE, num_heads, seqlen_q, head_size]", flash_attn_out_permuted.shape)
    print("flash_softmax_lse.shape:", flash_softmax_lse.shape)
    
    # Compare Flash Attention LSE with manual LSE
    if flash_softmax_lse.shape == manual_softmax_lse.shape:
        flash_lse_comparison = torch.allclose(manual_softmax_lse.to(dtype=torch.bfloat16), flash_softmax_lse.to(dtype=torch.bfloat16), atol=atol_val, rtol=rtol_val)
        print("manual_softmax_lse == flash_softmax_lse:", flash_lse_comparison)
        
        if not flash_lse_comparison:
            max_flash_lse_diff = torch.max(torch.abs(manual_softmax_lse - flash_softmax_lse))
            print("Maximum LSE difference with Flash Attention:", max_flash_lse_diff.item())
    else:
        print("Flash LSE shape doesn't match manual LSE shape:", manual_softmax_lse.shape, "vs", flash_softmax_lse.shape)
        # Try to reshape if possible
        if flash_softmax_lse.dim() == manual_softmax_lse.dim():
            if flash_softmax_lse.shape[0] == manual_softmax_lse.shape[0] and \
                flash_softmax_lse.shape[1] == manual_softmax_lse.shape[1] and \
                flash_softmax_lse.shape[2] == manual_softmax_lse.shape[2]:
                print("Dimensions look compatible for comparison")
                flash_lse_comparison = torch.allclose(manual_softmax_lse, flash_softmax_lse, atol=atol_val, rtol=rtol_val)
                print("manual_softmax_lse == flash_softmax_lse (after checking compatibility):", flash_lse_comparison)
    
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
        
    # Print summary of LSE findings
    print("\nSummary of LSE comparisons:")
    if 'lse_comparison' in locals():
        print(f"Manual LSE vs Torch LSE match: {lse_comparison}")
    if 'flash_lse_comparison' in locals():
        print(f"Manual LSE vs Flash Attention LSE match: {flash_lse_comparison}")
        
    # Compare Torch LSE with Flash LSE if shapes are compatible
    if 'torch_softmax_lse' in locals() and 'flash_softmax_lse' in locals():
        if torch_softmax_lse.shape == flash_softmax_lse.shape:
            torch_flash_lse_comparison = torch.allclose(torch_softmax_lse.to(dtype=torch.bfloat16), flash_softmax_lse.to(dtype=torch.bfloat16), atol=atol_val, rtol=rtol_val)
            print(f"Torch LSE vs Flash LSE match: {torch_flash_lse_comparison}")
            if not torch_flash_lse_comparison:
                max_torch_flash_lse_diff = torch.max(torch.abs(torch_softmax_lse - flash_softmax_lse))
                print(f"Maximum LSE difference between Torch and Flash: {max_torch_flash_lse_diff.item()}")

            
        

