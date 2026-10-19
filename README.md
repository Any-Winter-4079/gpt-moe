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

Run locally with Docker Desktop running. This extends the CUDA 13.2 image above, keeps its Python 3.11 and data packages, and installs the full CUDA Toolkit 13.4.2, PyTorch nightly 2.16.0.dev20261009+cu134, Liger 0.8.4, and Transformer Engine 2.20.2 with CUDA 13 and PyTorch support. The nightly version is PyTorch; Python stays at 3.11. The separate build directory and image tag let you keep both images.

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

RUN TE_CUDNN_ROOT="$(python -c 'from importlib.metadata import distribution; d = distribution("nvidia-cudnn-cu13"); print(next(d.locate_file(p) for p in d.files if p.name == "cudnn_version.h").parent.parent)')" \
 && ln -sfn "$TE_CUDNN_ROOT" /opt/te-cudnn

ENV CUDNN_PATH=/opt/te-cudnn
ENV CUDNN_HOME=/opt/te-cudnn
ENV LD_LIBRARY_PATH="/opt/te-cudnn/lib:${LD_LIBRARY_PATH}"

RUN NVTE_BUILD_USE_NVIDIA_WHEELS=0 \
    CPPFLAGS="-I${CUDNN_PATH}/include ${CPPFLAGS:-}" \
    python -m pip install --no-cache-dir --no-build-isolation \
    "transformer_engine[pytorch,core_cu13]==2.20.2"

RUN nvcc --version \
 && python -c "import torch, triton; import transformer_engine.pytorch; from importlib.metadata import version; print('torch:', torch.__version__, 'CUDA:', torch.version.cuda, 'triton:', triton.__version__, 'liger-kernel:', version('liger-kernel'), 'transformer-engine:', version('transformer-engine'), 'cuDNN:', torch.backends.cudnn.version()); assert torch.__version__ == '2.16.0.dev20261009+cu134'; assert torch.version.cuda == '13.4'"
EOF
```

The library path avoids inheriting the older toolkit's runtime libraries; PyTorch uses the CUDA libraries installed with its wheel. The `/opt/te-cudnn` symlink points to the installed CUDA 13 cuDNN package, and its headers and libraries are selected explicitly for Transformer Engine compilation and runtime. The build verifies that Transformer Engine imports. The full 13.4 toolkit remains available through `CUDA_HOME` and `PATH`.

Log in and select (or create) `amd64-builder` as above, then build and push:

```bash
docker buildx build --platform linux/amd64 \
  -t anywinter4079/pytorch:2.16.0.dev20261009-cu134-devel \
  ./docker_build/cu134 --push
```

Use `anywinter4079/pytorch:2.16.0.dev20261009-cu134-devel` in the new instance template. The built image contains the installed nightly; starting a container does not reinstall it. Rebuilding later still requires the pinned nightly to remain available in the [PyTorch nightly index](https://download.pytorch.org/whl/nightly/cu134/torch/).

## Build and push the experimental CUDA 13.4 DeepGEMM image

This extends the CUDA 13.4 image above with [DeepGEMM revision b64107f](https://github.com/deepseek-ai/DeepGEMM/tree/b64107f2b9599ca76445b7f62eedf66bae1d095b), its pinned submodules, and two SM120 API patches: GPU offsets for grouped FP8 weight gradients, and FP32 grouped FP8 outputs for the projection experiment. It includes GCC 13, the C++20 standard library, and `libdw-dev`; the compiler selections persist in the image for DeepGEMM's runtime kernel compilation. Ubuntu's repository helper runs explicitly with Python 3.10, while training continues to use Python 3.11.

Run from the repository root with Docker Desktop running. Select `amd64-builder` as above, then build and push under a separate tag:

```bash
docker buildx build --platform linux/amd64 \
  -f docker_build/cu134-deepgemm/Dockerfile \
  -t anywinter4079/pytorch:2.16.0.dev20261009-cu134-deepgemm-experimental \
  ./docker_build --push
```

Use `anywinter4079/pytorch:2.16.0.dev20261009-cu134-deepgemm-experimental` in the instance template. The source checkout remains at `/opt/DeepGEMM` for inspecting or rebuilding the installed dependency. The image build checks the required headers and Python imports; GPU kernels compile during the first training warmup on the instance.

The ordered code experiments and their application instructions are in [experiments/README.md](experiments/README.md).

## Run

```
git clone https://github.com/Any-Winter-4079/gpt-moe.git && cd gpt-moe/ && python data/fineweb-npy.py
```
