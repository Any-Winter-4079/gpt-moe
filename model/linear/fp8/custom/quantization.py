import triton
import triton.language as tl


@triton.jit
def _inverse_scale(amax):
    # E4M3 with current, power-of-two scales
    scale = tl.exp2(tl.ceil(tl.log2(amax / 448.0)))
    return tl.where(amax == 0, 1.0, tl.maximum(scale, 2.0 ** -126))
