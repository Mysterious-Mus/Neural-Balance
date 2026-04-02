FROM nvidia/cuda:13.0.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3.11-dev python3.11-distutils python3-pip \
        gcc nvtop htop git vim curl wget jq ca-certificates \
        xvfb x11vnc chafa \
        xdotool x11-apps \
        imagemagick netpbm && \
    ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip

# vLLM bundles its own PyTorch; install the cu130 wheel so CUDA 13.0 is used throughout
RUN pip install https://github.com/vllm-project/vllm/releases/download/v0.16.0/vllm-0.16.0+cu130-cp38-abi3-manylinux_2_35_x86_64.whl \
        --extra-index-url https://download.pytorch.org/whl/cu130

WORKDIR /workspace
COPY .devcontainer/requirements-dev.txt /workspace/requirements-dev.txt
RUN pip install --no-cache-dir -r /workspace/requirements-dev.txt

# Install Playwright browser and system deps for computer-use agent
RUN playwright install chromium
