ARG ff_environment_base_image=flexflow-environment-cuda-12.1
FROM ${ff_environment_base_image}

LABEL org.opencontainers.image.source=https://github.com/flexflow/flexflow-serve
LABEL org.opencontainers.image.description="flexflow-serve environment container"

SHELL ["/bin/bash", "-c"]

# Install basic dependencies
RUN apt-get update && apt-get install -y --no-install-recommends wget sudo binutils git zlib1g-dev lsb-release nano gdb libhdf5-dev jq openssh-client && \
    rm -rf /var/lib/apt/lists/* /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/nvidia-ml.list && \
	apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends software-properties-common && \
    apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends build-essential apt-utils \
    ca-certificates libssl-dev curl unzip htop ninja-build && DEBIAN_FRONTEND=noninteractive \
        apt-get install -y software-properties-common && \
        add-apt-repository ppa:ubuntu-toolchain-r/test && \
        apt-get update -y && \
        apt-get upgrade -y libstdc++6

# Install Python3 with Miniconda
ARG python_version "latest"
RUN rm -rf /opt/conda && \
    MINICONDA_SCRIPT_NAME=Miniconda3-latest-Linux-x86_64.sh; \
    if [ "$python_version" != "3.8" ] && [ "$python_version" != "3.9" ] && [ "$python_version" != "3.10" ] && [ "$python_version" != "3.11" ] && [ "$python_version" != "3.12" ] && [ "$python_version" != "latest" ]; then \
        echo "python_version '${python_version}' is not supported, please choose among {3.8, 3.9, 3.10, 3.11, 3.12 or latest (default)}"; \
        exit 1; \
    fi; \
    if [ "${python_version}" = "3.8" ]; then \
        MINICONDA_SCRIPT_NAME=Miniconda3-py38_23.5.2-0-Linux-x86_64.sh; \
    elif [ "${python_version}" = "3.9" ]; then \
        MINICONDA_SCRIPT_NAME=Miniconda3-py39_23.5.2-0-Linux-x86_64.sh; \
    elif [ "${python_version}" = "3.10" ]; then \
        MINICONDA_SCRIPT_NAME=Miniconda3-py310_23.5.2-0-Linux-x86_64.sh; \
    elif [ "${python_version}" = "3.11" ]; then \
        MINICONDA_SCRIPT_NAME=Miniconda3-py311_23.5.2-0-Linux-x86_64.sh; \
    elif [ "${python_version}" = "3.12" ]; then \
        MINICONDA_SCRIPT_NAME=Miniconda3-py312_25.1.1-2-Linux-x86_64.sh; \
    elif [ "${python_version}" = "latest" ]; then \
        MINICONDA_SCRIPT_NAME=Miniconda3-latest-Linux-x86_64.sh; \
    fi; \
    wget -c -q https://repo.continuum.io/miniconda/${MINICONDA_SCRIPT_NAME} && \
        mv ./${MINICONDA_SCRIPT_NAME} ~/${MINICONDA_SCRIPT_NAME} && \
        chmod +x ~/${MINICONDA_SCRIPT_NAME} && \
        bash ~/${MINICONDA_SCRIPT_NAME} -b -p /opt/conda && \
        rm ~/${MINICONDA_SCRIPT_NAME} && \
	/opt/conda/bin/conda config --set solver classic && \
        /opt/conda/bin/conda upgrade --all && \
        /opt/conda/bin/conda install conda-build conda-verify && \
        /opt/conda/bin/conda clean -ya && \
    ln -sf /usr/lib/x86_64-linux-gnu/libstdc++.so.6 /opt/conda/lib/libstdc++.so.6

# set MAKEFLAGS to speedup any dependency that uses make
ARG N_BUILD_CORES
ENV MAKEFLAGS "${MAKEFLAGS} -j${N_BUILD_CORES}"

# Set env vars
ENV PATH /opt/conda/bin:$PATH
ENV CUDNN_DIR /usr/local/cuda
ENV CUDA_DIR /usr/local/cuda
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:$PATH
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH
RUN echo "export CUDA_HOME=${CUDA_HOME}" >> ~/.bashrc && \
    echo "export PATH=${CUDA_HOME}/bin:$PATH" >> ~/.bashrc && \
    echo "export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH" >> ~/.bashrc

# GPU-specific dependencies
ARG FF_GPU_BACKEND "cuda"
ARG cuda_version ""
ARG hip_version "5.6"

# Install hip dependencies if FF_GPU_BACKEND is hip_cuda or hip_rocm
# Note that amd's docs say to also install the `hip-runtime-nvidia` package. This
# package attempts to re-install cuda even though cuda is already installed
# in the container. It also attempts to install packages for a graphical install.
# For our container, we don't need `hip-runtime-nvidia`
RUN  if [ "$FF_GPU_BACKEND" = "hip_cuda" ] || [ "$FF_GPU_BACKEND" = "hip_rocm" ]; then \
        echo "FF_GPU_BACKEND: ${FF_GPU_BACKEND}. Installing HIP dependencies"; \
        # Check that hip_version is one of 5.3,5.4,5.5,5.6
        if [ "$hip_version" != "5.3" ] && [ "$hip_version" != "5.4" ] && [ "$hip_version" != "5.5" ] && [ "$hip_version" != "5.6" ]; then \
            echo "hip_version '${hip_version}' is not supported, please choose among {5.3, 5.4, 5.5, 5.6}"; \
            exit 1; \
        fi; \
        # Compute script name and url given the version
        AMD_GPU_SCRIPT_NAME=amdgpu-install_5.6.50600-1_all.deb; \
        if [ "$hip_version" = "5.3" ]; then \
            AMD_GPU_SCRIPT_NAME=amdgpu-install_5.3.50300-1_all.deb; \
        elif [ "$hip_version" = "5.4" ]; then \
            AMD_GPU_SCRIPT_NAME=amdgpu-install_5.4.50400-1_all.deb; \
        elif [ "$hip_version" = "5.5" ]; then \
            AMD_GPU_SCRIPT_NAME=amdgpu-install_5.5.50500-1_all.deb; \
        fi; \
        ubuntu_codename=$(lsb_release -cs); \
        AMD_GPU_SCRIPT_URL="https://repo.radeon.com/amdgpu-install/${hip_version}/ubuntu/${ubuntu_codename}/${AMD_GPU_SCRIPT_NAME}"; \
        # Download and install AMD GPU software with ROCM and HIP support
        wget $AMD_GPU_SCRIPT_URL; \
        apt-get install -y ./${AMD_GPU_SCRIPT_NAME}; \
        rm ./${AMD_GPU_SCRIPT_NAME}; \
        amdgpu-install -y --usecase=hip,rocm --no-dkms; \
        apt-get install -y hip-dev hipblas miopen-hip rocm-hip-sdk rocm-device-libs; \
    else \
        echo "FF_GPU_BACKEND: ${FF_GPU_BACKEND}. Skipping installing HIP dependencies"; \
    fi
RUN rm -rf /var/lib/apt/lists/*

# Install python packages and other dependencies
RUN conda install -c conda-forge cmake make cmake-build-extension numpy pandas
# Install Pytorch
COPY install_pytorch.sh /usr/local/bin/install_pytorch.sh
RUN if [ "$FF_GPU_BACKEND" == "cuda" ] ; then \
        echo "FF_GPU_BACKEND: ${FF_GPU_BACKEND}. Installing PyTorch with CUDA"; \
        pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121; \
    else \
        echo "FF_GPU_BACKEND: ${FF_GPU_BACKEND}. Installing CPU-only PyTorch"; \
        pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu; \
    fi && \
    PYTORCH_PATH=$(python3 -c "import torch; print(torch.__path__[0])") && \
    echo "${PYTORCH_PATH}/lib" >> /etc/ld.so.conf.d/pytorch.conf && \
    ldconfig
RUN rm /usr/local/bin/install_pytorch.sh
# Install Flash Attention
# RUN pip3 install packaging ninja
# RUN pip3 install flash-attn --no-build-isolation --no-cache-dir
# Various dependencies
RUN pip3 install transformers>=4.47.1 sentencepiece einops
RUN pip3 install tensorflow notebook
# PEFT-related
RUN pip3 install scipy bitsandbytes datasets accelerate loralib triton peft pytest
RUN pip3 install streamlit

# Install Rust
RUN rm -rf /root/.cargo /root/.rustup && \
    mkdir -p /root/.cargo /root/.rustup && \
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path --profile minimal --default-toolchain stable && \
    rm -rf /root/.rustup/tmp/ && \
    rm -rf /root/.rustup/toolchains/*/share/zsh && \
    rm -rf /root/.rustup/toolchains/*/etc/bash_completion.d && \
    chmod -R a+w /root/.rustup /root/.cargo

ENV PATH="/root/.cargo/bin:${PATH}"
ENV RUSTUP_HOME=/root/.rustup
ENV CARGO_HOME=/root/.cargo

ENTRYPOINT ["/bin/bash"]
