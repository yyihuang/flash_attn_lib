import torch
print(torch.version.cuda)          # CUDA version PyTorch/libtorch was compiled with
print(torch.cuda.is_available())   # Checks if CUDA is usable
