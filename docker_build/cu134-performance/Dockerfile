FROM anywinter4079/pytorch:2.16.0.dev20261009-cu134-deepgemm-experimental

WORKDIR /workspace

ENV CUTLASS_REVISION=2c85f497547eb441f0fe80bfb0fb2e6f9add1719
ENV CUTLASS_PATH=/opt/cutlass
ENV FA4_REVISION=e7a6402a9ff099de5eb628ce207a88a3f0aa5378

RUN python -m pip install --no-cache-dir --no-deps --only-binary=:all: \
    setuptools==80.9.0 setuptools-scm==9.2.2 wheel==0.45.1 packaging==25.0 \
    "nvidia-cutlass-dsl[cu13]==4.8.1" \
    nvidia-cutlass-dsl-libs-core==4.8.1 nvidia-cutlass-dsl-libs-base==4.8.1 \
    nvidia-cutlass-dsl-libs-cu12==4.8.1 nvidia-cutlass-dsl-libs-cu13==4.8.1 \
    quack-kernels==0.6.5 apache-tvm-ffi==0.1.11 torch-c-dlpack-ext==0.1.5 \
    einops==0.8.2 typing_extensions==4.16.0 \
    cuda-bindings==13.4.4 cuda-pathfinder==1.8.4 protobuf==7.36.2

RUN git clone --depth 1 --filter=blob:none --no-checkout --branch v4.8.1 \
    https://github.com/NVIDIA/cutlass.git "${CUTLASS_PATH}" \
 && git -C "${CUTLASS_PATH}" sparse-checkout set include tools/util/include \
 && git -C "${CUTLASS_PATH}" checkout --detach "${CUTLASS_REVISION}" \
 && test -f "${CUTLASS_PATH}/include/cutlass/gemm/collective/builders/sm120_blockscaled_mma_builder.inl" \
 && test -f "${CUTLASS_PATH}/tools/util/include/cutlass/util/packed_stride.hpp"

ARG FA4_SM120_FLEX=0
ARG FA4_DEVICE_WINDOW=0

RUN case "${FA4_SM120_FLEX}:${FA4_DEVICE_WINDOW}" in \
      0:0|1:0|0:1) ;; \
      *) echo "FA4_SM120_FLEX and FA4_DEVICE_WINDOW must be 0 or 1 and cannot both be 1" >&2; exit 1 ;; \
    esac

COPY cu134-fa4/fa4-sm120-local-window.patch /tmp/fa4-sm120-local-window.patch
COPY cu134-fa4/fa4-sm120-flex.patch /tmp/fa4-sm120-flex.patch
COPY cu134-fa4/fa4-sm120-device-window.patch /tmp/fa4-sm120-device-window.patch

RUN git clone --depth 1 --filter=blob:none --no-checkout --branch fa4-v4.0.0.beta36 \
    https://github.com/Dao-AILab/flash-attention.git /opt/flash-attention \
 && git -C /opt/flash-attention sparse-checkout set flash_attn/cute \
 && git -C /opt/flash-attention checkout --detach "${FA4_REVISION}" \
 && git -C /opt/flash-attention apply --check /tmp/fa4-sm120-local-window.patch \
 && git -C /opt/flash-attention apply /tmp/fa4-sm120-local-window.patch \
 && if [ "${FA4_SM120_FLEX}" = 1 ]; then \
      git -C /opt/flash-attention apply --check --include='flash_attn/*' /tmp/fa4-sm120-flex.patch \
   && git -C /opt/flash-attention apply --include='flash_attn/*' /tmp/fa4-sm120-flex.patch \
   && FA4_TORCH_ROOT="$(python -c 'from importlib.metadata import distribution; print(distribution("torch").locate_file(""))')" \
   && git -C "${FA4_TORCH_ROOT}" apply --check --include='torch/*' /tmp/fa4-sm120-flex.patch \
   && git -C "${FA4_TORCH_ROOT}" apply --include='torch/*' /tmp/fa4-sm120-flex.patch; \
    elif [ "${FA4_DEVICE_WINDOW}" = 1 ]; then \
      git -C /opt/flash-attention apply --check /tmp/fa4-sm120-device-window.patch \
   && git -C /opt/flash-attention apply /tmp/fa4-sm120-device-window.patch; fi \
 && SETUPTOOLS_SCM_PRETEND_VERSION_FOR_FLASH_ATTN_4=4.0.0b36 \
    python -m pip install --no-cache-dir --no-build-isolation --no-deps \
    "/opt/flash-attention/flash_attn/cute[cu13]" \
 && rm -f /tmp/fa4-sm120-local-window.patch /tmp/fa4-sm120-flex.patch /tmp/fa4-sm120-device-window.patch

RUN nvcc --version \
 && "${CXX}" --version \
 && python -c "import inspect; import torch, triton, deep_gemm; import transformer_engine.pytorch; import cutlass; from flash_attn.cute import flash_attn_func; from flash_attn.cute.interface import _flash_attn_fwd, _flash_attn_bwd; import flash_attn.cute.interface as fa4_interface; from importlib.metadata import version; device_window = all('window_size_tensor' in inspect.signature(fn).parameters for fn in (_flash_attn_fwd, _flash_attn_bwd)); print('torch:', torch.__version__, 'CUDA:', torch.version.cuda, 'triton:', triton.__version__, 'transformer-engine:', version('transformer-engine'), 'deep-gemm:', version('deep-gemm'), 'flash-attn-4:', version('flash-attn-4'), 'nvidia-cutlass-dsl:', version('nvidia-cutlass-dsl'), 'SM120 Flex:', getattr(fa4_interface, 'SM120_FLEX_SUPPORT', False), 'device window:', device_window); assert torch.__version__ == '2.16.0.dev20261009+cu134'; assert torch.version.cuda == '13.4'; assert version('flash-attn-4') == '4.0.0b36'; assert version('nvidia-cutlass-dsl') == '4.8.1'; assert getattr(fa4_interface, 'SM120_FLEX_SUPPORT', False) == bool(int('${FA4_SM120_FLEX}')); assert device_window == bool(int('${FA4_DEVICE_WINDOW}'))"
