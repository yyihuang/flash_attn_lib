# Use NVIDIA's base image with CUDA 12.2
FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

# Set non-interactive mode
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt update && apt install -y \
    build-essential \
    gcc-9 g++-9 \
    cmake \
    python3.10 python3.10-dev python3-pip python3-venv \
    git wget curl \
    ninja-build \
    libtinfo-dev \
    libncurses5-dev \
    libncursesw5-dev \
    && rm -rf /var/lib/apt/lists/*

# Set default GCC version to 9 (ensures compatibility with cxx11abi=False)
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 100 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 100
ENV CC=/usr/bin/gcc-9
ENV CXX=/usr/bin/g++-9

# Install Miniconda
WORKDIR /root
RUN curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o Miniconda3.sh && \
    bash Miniconda3.sh -b -p /opt/conda && \
    rm Miniconda3.sh
ENV PATH="/opt/conda/bin:$PATH"

# Create a conda environment with Python 3.10
RUN conda create -n flash python=3.10 -y && \
    echo "conda activate flash" >> ~/.bashrc
ENV CONDA_DEFAULT_ENV=flash
ENV PATH="/opt/conda/envs/flash/bin:$PATH"

# Install PyTorch (CUDA 12, Torch 2.2, cxx11abi=False)
RUN pip install --upgrade pip && \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install FlashAttention from the exact wheel (without renaming)
RUN wget -P /tmp/ https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl && \
    pip install /tmp/flash_attn-2.7.4.post1+cu12torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl --no-build-isolation

# Set working directory inside the container
WORKDIR /workspace/flash_attention_lib

# Set environment variables for CMake
ENV CMAKE_PREFIX_PATH="/opt/conda/envs/flash/lib/python3.10/site-packages/torch/"

# Create build directory and run CMake
# RUN mkdir -p build && cd build && \
#     cmake .. && \
#     make -j$(nproc)

# Set default command
CMD ["bash"]