FROM anywinter4079/pytorch:2.10.0-cu128

WORKDIR /root

ADD https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb /tmp/cuda-keyring.deb

RUN dpkg -i /tmp/cuda-keyring.deb \
 && apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends cuda-toolkit-13-2=13.2.2-1 \
 && rm -f /tmp/cuda-keyring.deb \
 && rm -rf /var/lib/apt/lists/*

ENV CUDA_VERSION=13.2.2
ENV CUDA_HOME=/usr/local/cuda-13.2
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64"

RUN python -m pip install --no-cache-dir --upgrade torch==2.13.0+cu132 \
 --index-url https://download.pytorch.org/whl/cu132 \
 && python -m pip install --no-cache-dir --no-deps liger-kernel==0.8.4

RUN nvcc --version \
 && python -c "import torch, triton; from importlib.metadata import version; print('torch:', torch.__version__, 'CUDA:', torch.version.cuda, 'triton:', triton.__version__, 'liger-kernel:', version('liger-kernel')); assert torch.version.cuda == '13.2'"
