"""
A test file to illustrate how to use the Python interface of the original file.
"""

import torch
from flash_attn import (
    flash_attn_qkvpacked_func,
    flash_attn_kvpacked_func,
    flash_attn_func
)

# Set seeds for reproducibility
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

def attention_ref(q, k, v, causal=False):
    """Reference attention implementation"""
    scores = torch.einsum('bqhd,bkhd->bhqk', q, k) / (q.size(-1) ** 0.5)
    if causal:
        mask = torch.triu(torch.ones_like(scores, dtype=torch.bool), diagonal=1)
        scores.masked_fill_(mask, float('-inf'))
    attn = torch.softmax(scores, dim=-1)
    return torch.einsum('bhqk,bkhd->bqhd', attn, v)

def test_qkvpacked_interface():
    """Example 1: QKV Packed Interface"""
    print("\nTesting QKV Packed Interface:")
    batch, seqlen, nheads, d = 2, 128, 8, 64
    qkv = torch.randn(batch, seqlen, 3, nheads, d, device='cuda', requires_grad=True)
    
    # Flash Attention
    flash_out = flash_attn_qkvpacked_func(qkv, causal=True)
    
    # Reference
    q, k, v = qkv.unbind(2)
    ref_out = attention_ref(q, k, v, causal=True)
    
    # Check forward
    assert torch.allclose(flash_out, ref_out, atol=1e-3), "Forward pass mismatch"
    
    # Check backward
    grad = torch.randn_like(flash_out)
    flash_out.backward(grad)
    flash_grad = qkv.grad.clone()
    qkv.grad = None
    ref_out.backward(grad)
    ref_grad = qkv.grad.clone()
    assert torch.allclose(flash_grad, ref_grad, atol=1e-3), "Backward pass mismatch"
    print("✅ QKV Packed passed!")

def test_kvpacked_interface():
    """Example 2: KV Packed Interface (supports MQA/GQA)"""
    print("\nTesting KV Packed Interface:")
    batch, seqlen_q, seqlen_k = 2, 128, 256
    nheads_q, nheads_k, d = 8, 2, 64
    
    q = torch.randn(batch, seqlen_q, nheads_q, d, device='cuda', requires_grad=True)
    kv = torch.randn(batch, seqlen_k, 2, nheads_k, d, device='cuda', requires_grad=True)
    
    # Flash Attention (supports multi-query/grouped-query)
    flash_out = flash_attn_kvpacked_func(q, kv)
    
    # Reference (repeat heads for comparison)
    k, v = kv.unbind(2)
    k_rep = k.repeat_interleave(nheads_q//nheads_k, dim=2)
    v_rep = v.repeat_interleave(nheads_q//nheads_k, dim=2)
    ref_out = attention_ref(q, k_rep, v_rep)
    
    # Check forward
    assert torch.allclose(flash_out, ref_out, atol=1e-3), "Forward pass mismatch"
    
    # Check backward
    grad = torch.randn_like(flash_out)
    flash_out.backward(grad)
    flash_grad_q, flash_grad_kv = q.grad.clone(), kv.grad.clone()
    q.grad = kv.grad = None
    ref_out.backward(grad)
    ref_grad_q, ref_grad_kv = q.grad.clone(), kv.grad.clone()
    assert torch.allclose(flash_grad_q, ref_grad_q, atol=1e-3), "Q grad mismatch"
    assert torch.allclose(flash_grad_kv, ref_grad_kv, atol=1e-3), "KV grad mismatch"
    print("✅ KV Packed passed!")

def test_standard_interface():
    """Example 3: Standard Interface (separate Q/K/V)"""
    print("\nTesting Standard Interface:")
    batch, seqlen_q, seqlen_k = 2, 128, 256
    nheads, d = 8, 64
    
    q = torch.randn(batch, seqlen_q, nheads, d, device='cuda', requires_grad=True)
    k = torch.randn(batch, seqlen_k, nheads, d, device='cuda', requires_grad=True)
    v = torch.randn(batch, seqlen_k, nheads, d, device='cuda', requires_grad=True)
    
    # Flash Attention with causal masking
    flash_out = flash_attn_func(q, k, v, causal=True)
    
    # Reference
    ref_out = attention_ref(q, k, v, causal=True)
    
    # Check forward
    assert torch.allclose(flash_out, ref_out, atol=1e-3), "Forward pass mismatch"
    
    # Check backward
    grad = torch.randn_like(flash_out)
    flash_out.backward(grad)
    flash_grads = (q.grad.clone(), k.grad.clone(), v.grad.clone())
    q.grad = k.grad = v.grad = None
    ref_out.backward(grad)
    ref_grads = (q.grad.clone(), k.grad.clone(), v.grad.clone())
    for fg, rg in zip(flash_grads, ref_grads):
        assert torch.allclose(fg, rg, atol=1e-3), "Grad mismatch"
    print("✅ Standard interface passed!")

if __name__ == "__main__":
    test_qkvpacked_interface()
    test_kvpacked_interface()
    test_standard_interface()
    print("\nAll interfaces verified!")