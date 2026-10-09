# gpt-moe

## Build and push the CUDA 13.2 image

Run locally with Docker Desktop running. This extends `anywinter4079/pytorch:2.10.0-cu128`, keeps its Python 3.11 and data packages, and installs the full CUDA Toolkit 13.2.2, PyTorch 2.13.0+cu132, and Liger 0.8.4. The default working directory is `/root`.

```bash
mkdir -p docker_build && \
cat > docker_build/Dockerfile <<'EOF'
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
EOF
```

Log in and select the existing builder:

```bash
docker login
docker buildx use amd64-builder
```

If the builder does not exist, create it instead:

```bash
docker buildx create --name amd64-builder --driver docker-container --use
```

Build and push under a new tag:

```bash
docker buildx build --platform linux/amd64 \
  -t anywinter4079/pytorch:2.13.0-cu132-devel \
  ./docker_build --push
```

## Build and push the CUDA 13.4 image

Run locally with Docker Desktop running. This extends the CUDA 13.2 image above, keeps its Python 3.11 and data packages, and installs the full CUDA Toolkit 13.4.2, PyTorch nightly 2.16.0.dev20261009+cu134, and Liger 0.8.4. The nightly version is PyTorch; Python stays at 3.11. The separate build directory and image tag let you keep both images.

```bash
mkdir -p docker_build/cu134 && \
cat > docker_build/cu134/Dockerfile <<'EOF'
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
EOF
```

The library path avoids inheriting the older toolkit's runtime libraries; PyTorch uses the CUDA libraries installed with its wheel. The full 13.4 toolkit remains available through `CUDA_HOME` and `PATH`.

Log in and select (or create) `amd64-builder` as above, then build and push:

```bash
docker buildx build --platform linux/amd64 \
  -t anywinter4079/pytorch:2.16.0.dev20261009-cu134-devel \
  ./docker_build/cu134 --push
```

Use `anywinter4079/pytorch:2.16.0.dev20261009-cu134-devel` in the new instance template. The built image contains the installed nightly; starting a container does not reinstall it. Rebuilding later still requires the pinned nightly to remain available in the [PyTorch nightly index](https://download.pytorch.org/whl/nightly/cu134/torch/).

## Run

```
git clone https://github.com/Any-Winter-4079/gpt-moe.git && cd gpt-moe/ && python data/fineweb-npy.py
```
