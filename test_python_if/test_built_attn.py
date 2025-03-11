import torch
from flash_attn import flash_attn_func, flash_attn_qkvpacked_func

# Define tensor dimensions
batch_size = 1
seqlen_q = 16
seqlen_k = 256
num_heads = 16
num_kv_heads = 8
head_size = 128

# 0. test setup
# set torch parameters
torch.manual_seed(2025)
torch.cuda.manual_seed(2025)
device = "cuda" if torch.cuda.is_available() else "cpu"
# set attention parameters
dropout_p = 0.0  # No dropout during inference
softmax_scale = 1.0 / (head_size ** 0.5)
causal = True  # decoder 
window_size = (-1, 0)  # causal attention


# 1. vanilla tensor
# Create input tensors
q = torch.randn(batch_size, num_heads, seqlen_q, head_size, device=device, dtype=torch.float16, requires_grad=True)
k = torch.randn(batch_size, num_kv_heads, seqlen_k, head_size, device=device, dtype=torch.float16, requires_grad=True)
v = torch.randn(batch_size, num_kv_heads, seqlen_k, head_size, device=device, dtype=torch.float16, requires_grad=True)


# Run FlashAttention and get the gradient
try:
    output = flash_attn_func(q, k, v, dropout_p=dropout_p, softmax_scale=softmax_scale,
                             causal=causal, window_size=window_size)
    loss = output.sum()
    loss.backward()

    # Check if gradients are computed
    assert q.grad is not None, "Gradient for q is missing"
    assert k.grad is not None, "Gradient for k is missing"
    assert v.grad is not None, "Gradient for v is missing"

    print("FlashAttention output shape:", output.shape)
except Exception as e:
    print("Error running FlashAttention:", e)

# 2. tensor of different layout
# construct 3 new tensors with the same values as q, k, v on each dimension, but the shapes are:
# q: [head_size, num_heads, seqlen_q]
# k: [head_size, num_kv_heads, seqlen_k]
# v: [head_size, num_kv_heads, seqlen_k]

# remove the batch dimension
q = q.squeeze(0)  # [head_size, num_heads, seqlen_q]
k = k.squeeze(0)  # [head_size, num_kv_heads, seqlen_k]
v = v.squeeze(0)  # [head_size, num_kv_heads, seqlen_k]

# todo: test un-contiguous
q_new = q.permute(2, 0, 1).contiguous()  # [head_size, num_heads, seqlen_q]
k_new = k.permute(2, 0, 1).contiguous()  # [head_size, num_kv_heads, seqlen_k]
v_new = v.permute(2, 0, 1).contiguous()  # [head_size, num_kv_heads, seqlen_k]

# add the batch dimension back
q_new = q_new.unsqueeze(0)  # [1, head_size, num_heads, seqlen_q]
k_new = k_new.unsqueeze(0)  # [1, head_size, num_kv_heads, seqlen_k]
v_new = v_new.unsqueeze(0)  # [1, head_size, num_kv_heads, seqlen_k]

# Run FlashAttention and get the gradient
try:
    output_new = flash_attn_func(q_new, k_new, v_new, dropout_p=dropout_p, softmax_scale=softmax_scale,
                             causal=causal, window_size=window_size)
    loss_new = output_new.sum()
    loss_new.backward() 

    # Check if gradients are computed
    assert q_new.grad is not None, "Gradient for q_new is missing"
    assert k_new.grad is not None, "Gradient for k_new is missing"
    assert v_new.grad is not None, "Gradient for v_new is missing"  

    print("FlashAttention output shape:", output.shape)
except Exception as e:
    print("Error running FlashAttention:", e)   

# check if the output and gradient are the same as the original tensor
assert torch.allclose(output, output_new)
assert torch.allclose(q.grad, q_new.grad)
assert torch.allclose(k.grad, k_new.grad)
assert torch.allclose(v.grad, v_new.grad)






