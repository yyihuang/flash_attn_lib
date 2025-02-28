import torch
from flash_attn import flash_attn_func

def test_flash_attn():
    # Batch size, sequence length, number of heads, head dimension
    B, N, H, D = 2, 16, 8, 64

    # Initialize input tensors
    q = torch.randn(B, N, H, D, dtype=torch.float16, device="cuda", requires_grad=True)
    k = torch.randn(B, N, H, D, dtype=torch.float16, device="cuda", requires_grad=True)
    v = torch.randn(B, N, H, D, dtype=torch.float16, device="cuda", requires_grad=True)

    # Forward pass
    out = flash_attn_func(q, k, v, dropout_p=0.0, causal=True)

    print(f"q shape: {q.shape}")  # (B, N, H, D)
    print(f"k shape: {k.shape}")  # (B, N, H, D)
    print(f"v shape: {v.shape}")  # (B, N, H, D)
    print(f"out shape: {out.shape}")  # (B, N, H, D)

    # Backward pass
    dout = torch.randn_like(out)  # Simulated gradient for backpropagation
    out.backward(dout)

    # Print gradients
    print(f"dq shape: {q.grad.shape}")  # (B, N, H, D)
    print(f"dk shape: {k.grad.shape}")  # (B, N, H, D)
    print(f"dv shape: {v.grad.shape}")  # (B, N, H, D)
    print(f"dout shape: {dout.shape}")  # (B, N, H, D)

if __name__ == "__main__":
    test_flash_attn()