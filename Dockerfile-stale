# Use NVIDIA's CUDA 12.4 base image
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

# Set environment variables for CUDA
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# Set user information (change this UID/GID as needed)
ARG USER_ID=2711822
ARG GROUP_ID=2711822
ARG USERNAME=flashuser

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    unzip \
    pkg-config \
    libssl-dev \
    libffi-dev \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    gcc-11 g++-11 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set GCC 11 as default
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 100

# Upgrade pip and install common Python packages
RUN python3.10 -m pip install --upgrade pip setuptools wheel \
    && python3.10 -m pip install numpy scipy pandas tqdm packaging

# Install PyTorch (CUDA 12.4 compatible)
RUN python3.10 -m pip install torch torchvision torchaudio

# Set Python 3.10 as default
RUN ln -sf /usr/bin/python3.10 /usr/bin/python3
RUN ln -sf /usr/bin/python3.10 /usr/bin/python

# Set up a user with the same UID/GID as the host
RUN groupadd --gid ${GROUP_ID} ${USERNAME} \
    && useradd --uid ${USER_ID} --gid ${GROUP_ID} --create-home --shell /bin/bash ${USERNAME}

# Ensure user has permissions in /workspace
RUN mkdir -p /workspace && chown -R ${USERNAME}:${USERNAME} /workspace

# Switch to the user
USER ${USERNAME}
WORKDIR /workspace

# Set up Torch_DIR for CMake (Fix for ENV error)
RUN echo "export Torch_DIR=$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)')" >> ~/.bashrc
RUN echo "export CMAKE_PREFIX_PATH=\$Torch_DIR:\$CMAKE_PREFIX_PATH" >> ~/.bashrc

# Ensure Python finds user-installed packages
ENV PYTHONPATH="/home/${USERNAME}/.local/lib/python3.10/site-packages:${PYTHONPATH}"

# Verify installations
RUN python --version && gcc --version && nvcc --version && whoami

# Entry point
CMD ["/bin/bash"]