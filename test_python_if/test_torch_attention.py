import torch
import math
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

# set seed
torch.manual_seed(42)

q = torch.randn(BATCH_SIZE, seqlen_q, num_heads, head_size)
k = torch.randn(BATCH_SIZE, seqlen_k, num_heads, head_size)
v = torch.randn(BATCH_SIZE, seqlen_k, num_heads, head_size)

# q: [BATCH_SIZE, num_heads, seqlen_q, head_size]
q = q.permute(0, 2, 1, 3)

# k_t: [BATCH_SIZE, num_heads, head_size, seqlen_k]
k_t = k.permute(0, 2, 3, 1)

# Compute attention weights
attn_scores = q @ k_t / math.sqrt(head_size)

print("q.shape: [BATCH_SIZE, num_heads, seqlen_q, head_size]", q.shape)
print("k_t.shape: [BATCH_SIZE, num_heads, head_size, seqlen_k]", k_t.shape)
print("attn_scores.shape: [BATCH_SIZE, num_heads, seqlen_q, seqlen_k]", attn_scores.shape)

# Create a causal mask
# In a causal mask, each position can only attend to previous positions and itself
# So we create a lower triangular matrix with 1s (allowed connections) and 0s (masked connections)
causal_mask = torch.tril(torch.ones(seqlen_q, seqlen_k)).to(q.device) if is_causal else None

# Apply the causal mask by setting masked positions to a large negative value
# This will make them effectively zero after softmax
if is_causal:
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

# Create the same causal mask for torch.nn.functional.scaled_dot_product_attention
# For this function, the mask should be True where attention is allowed
# and False where it should be masked
attn_mask = torch.tril(torch.ones(seqlen_q, seqlen_k, dtype=torch.bool, device=q.device)) if is_causal else None

# compare the results with torch.nn.functional.scaled_dot_product_attention
# Pass the attn_mask to ensure causal masking
attn_out_torch = torch.nn.functional.scaled_dot_product_attention(
    q, k_permuted, v_permuted, attn_mask=attn_mask, is_causal=is_causal
)

print("attn_out_torch.shape: [BATCH_SIZE, num_heads, seqlen_q, head_size]", attn_out_torch.shape)

# compare the results
comparison_result = torch.allclose(attn_out, attn_out_torch, atol=1e-5, rtol=1e-5)
print("attn_out == attn_out_torch:", comparison_result)

# If still not close, check the maximum difference
if not comparison_result:
    max_diff = torch.max(torch.abs(attn_out - attn_out_torch))
    print("Maximum absolute difference:", max_diff.item())
    
    # Try with is_causal=True parameter instead
    attn_out_torch_v2 = torch.nn.functional.scaled_dot_product_attention(
        q, k_permuted, v_permuted, attn_mask=None, is_causal=True
    )
    comparison_result_v2 = torch.allclose(attn_out, attn_out_torch_v2, atol=1e-4, rtol=1e-4)
    print("Using is_causal=True: attn_out == attn_out_torch_v2:", comparison_result_v2)

