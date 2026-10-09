FROM anywinter4079/pytorch:2.13.0-cu132-devel

WORKDIR /root

RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends cuda-toolkit-13-4=13.4.2-1 \
 && rm -rf /var/lib/apt/lists/*

ENV CUDA_VERSION=13.4.2
ENV CUDA_HOME=/usr/local/cuda-13.4
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/nvidia/lib:/usr/local/nvidia/lib64"

RUN python -m pip install --no-cache-dir --pre --upgrade torch==2.16.0.dev20261009+cu134 \
 --index-url https://download.pytorch.org/whl/nightly/cu134 \
 && python -m pip install --no-cache-dir --no-deps liger-kernel==0.8.4

RUN nvcc --version \
 && python -c "import torch, triton; from importlib.metadata import version; print('torch:', torch.__version__, 'CUDA:', torch.version.cuda, 'triton:', triton.__version__, 'liger-kernel:', version('liger-kernel')); assert torch.__version__ == '2.16.0.dev20261009+cu134'; assert torch.version.cuda == '13.4'"
