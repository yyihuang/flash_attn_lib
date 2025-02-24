# Base Image: Ubuntu 20.04 + CUDA 12.4.1 + cuDNN + NCCL
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu20.04

# Install system dependencies: remove linux-tools-common linux-tools-`uname -r` linux-cloud-tools-`uname -r` if not needed
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     sudo wget curl inetutils-ping gdb git gnupg2 vim ca-certificates \
#     linux-tools-common linux-tools-`uname -r` linux-cloud-tools-`uname -r` \
#     && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y --no-install-recommends \
    sudo wget curl inetutils-ping gdb git gnupg2 vim ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda and Python 3.10
RUN wget -c https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /usr/local/miniconda && \
    rm -rf /tmp/miniconda.sh && \
    /usr/local/miniconda/bin/conda create -n py310 python=3.10 -y 

# Set environment variables for Conda
ENV PATH="/usr/local/miniconda/envs/py310/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64"
ENV NVIDIA_DISABLE_REQUIRE=1
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Install CMake 3.30.1
RUN pip install --no-cache-dir cmake==3.30.1

# Set working directory
WORKDIR /workspace

# Download and install FlashAttention 2.7.4.post1 for CUDA 12 and Torch 2.2
RUN wget -O /tmp/flash_attn-2.7.4.post1+cu12torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl \
    https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.0.post2/flash_attn-2.7.0.post2+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl && \
    pip install /tmp/flash_attn-2.7.4.post1+cu12torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl --no-build-isolation && \
    rm -rf /tmp/flash_attn-2.7.4.post1+cu12torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# Install PyTorch (CUDA 12, Torch 2.2)
# RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
RUN pip install --no-cache-dir ninja

# Set up CMake environment variables
ENV CMAKE_PREFIX_PATH="/usr/local/miniconda/envs/py310/lib/python3.10/site-packages/torch/"

# Uncomment if you need to build from source
# COPY . /workspace/flash_attention_lib
# WORKDIR /workspace/flash_attention_lib
# RUN mkdir -p build && cd build && cmake .. && make -j$(nproc)

# Set default working directory
WORKDIR /app

# Default to interactive mode
CMD ["bash"]