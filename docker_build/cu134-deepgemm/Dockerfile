FROM anywinter4079/pytorch:2.16.0.dev20261009-cu134-devel

WORKDIR /root

RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    ca-certificates git ninja-build python3.10 python3-apt software-properties-common libdw-dev \
 && /usr/bin/python3.10 /usr/bin/add-apt-repository -y ppa:ubuntu-toolchain-r/test \
 && apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends gcc-13 g++-13 \
 && rm -rf /var/lib/apt/lists/*

ENV CC=/usr/bin/gcc-13
ENV CXX=/usr/bin/g++-13
ENV NVCC_CCBIN=/usr/bin/g++-13
ENV DG_JIT_NVCC_COMPILER=/usr/local/cuda-13.4/bin/nvcc
ENV DG_JIT_CPP_STANDARD=20
ENV DEEPGEMM_REVISION=b64107f2b9599ca76445b7f62eedf66bae1d095b

COPY cu134/deepgemm-sm120-psum.patch /tmp/deepgemm-sm120-psum.patch
COPY cu134-deepgemm/deepgemm-sm120-fp32-output.patch /tmp/deepgemm-sm120-fp32-output.patch

RUN git clone --branch nv_dev https://github.com/deepseek-ai/DeepGEMM.git /opt/DeepGEMM \
 && git -C /opt/DeepGEMM checkout "${DEEPGEMM_REVISION}" \
 && git -C /opt/DeepGEMM submodule update --init --recursive \
 && git -C /opt/DeepGEMM apply --check /tmp/deepgemm-sm120-psum.patch \
 && git -C /opt/DeepGEMM apply /tmp/deepgemm-sm120-psum.patch \
 && git -C /opt/DeepGEMM apply --check /tmp/deepgemm-sm120-fp32-output.patch \
 && git -C /opt/DeepGEMM apply /tmp/deepgemm-sm120-fp32-output.patch \
 && printf '#include <format>\n#include <elfutils/libdwfl.h>\n' | "${CXX}" -std=c++20 -x c++ -fsyntax-only - \
 && DG_FORCE_BUILD=1 python -m pip install --no-cache-dir --no-build-isolation --no-deps /opt/DeepGEMM \
 && rm -f /tmp/deepgemm-sm120-psum.patch /tmp/deepgemm-sm120-fp32-output.patch

RUN nvcc --version \
 && "${CXX}" --version \
 && python -c "import torch, triton, deep_gemm; import transformer_engine.pytorch; from importlib.metadata import version; print('torch:', torch.__version__, 'CUDA:', torch.version.cuda, 'triton:', triton.__version__, 'transformer-engine:', version('transformer-engine'), 'deep-gemm:', version('deep-gemm')); assert torch.__version__ == '2.16.0.dev20261009+cu134'; assert torch.version.cuda == '13.4'"
