# Use NVIDIA's base image with CUDA 12.2
FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

# Set non-interactive mode
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt update && apt install -y \
    build-essential \
    gcc-9 g++-9 \
    cmake \
    python3-dev python3-pip python3-venv \
    python3.10-dev \
    git wget curl \
    ninja-build \
    libtinfo-dev \
    libncurses5-dev \
    libncursesw5-dev \
    && rm -rf /var/lib/apt/lists/*

# Set default GCC version to 9
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 100 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 100

# Install Miniconda
WORKDIR /root
RUN curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o Miniconda3.sh && \
    bash Miniconda3.sh -b -p /opt/conda && \
    rm Miniconda3.sh
ENV PATH="/opt/conda/bin:$PATH"

# Create a conda environment
RUN conda create -n flash python=3.10 -y && \
    echo "conda activate flash" >> ~/.bashrc
ENV CONDA_DEFAULT_ENV=flash
ENV PATH="/opt/conda/envs/flash/bin:$PATH"

# Install PyTorch (CUDA 12.1, works with CUDA 12.2)
RUN pip install --upgrade pip && \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install FlashAttention via pip
RUN pip install flash-attn --no-build-isolation

# Set working directory inside the container
WORKDIR /workspace/flash_attention_lib

# Create build directory and run CMake
# RUN mkdir -p build && cd build && \
#     cmake .. && \
#     make -j$(nproc)

# Set default command
CMD ["bash"]