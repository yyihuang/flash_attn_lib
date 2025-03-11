import torch
from flash_attn import flash_attn_func, flash_attn_qkvpacked_func

# Define tensor dimensions
batch_size = 2
seqlen_q = 16
seqlen_k = 16
num_heads = 8
head_size = 64

# Create input tensors
q = torch.randn(batch_size, seqlen_q, num_heads, head_size, device="cuda", dtype=torch.float16)
k = torch.randn(batch_size, seqlen_k, num_heads, head_size, device="cuda", dtype=torch.float16)
v = torch.randn(batch_size, seqlen_k, num_heads, head_size, device="cuda", dtype=torch.float16)

# Set parameters
dropout_p = 0.0  # No dropout during inference
softmax_scale = 1.0 / (head_size ** 0.5)
causal = True  # decoder 
window_size = (-1, 0)  # Full attention

# Run FlashAttention
try:
    output = flash_attn_func(q, k, v, dropout_p=dropout_p, softmax_scale=softmax_scale,
                             causal=causal, window_size=window_size)

    print("FlashAttention output shape:", output.shape)
except Exception as e:
    print("Error running FlashAttention:", e)