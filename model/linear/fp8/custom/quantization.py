import triton
import triton.language as tl


@triton.jit
def _inverse_scale(amax):
    # round the positive FP32 ratio upward by clearing its mantissa after a carry
    ratio = amax / 448.0
    bits = ratio.to(tl.uint32, bitcast=True)
    rounded = (bits + 0x007fffff) & 0x7f800000
    rounded = tl.where(bits > 0x7f800000, 0, rounded)
    scale = tl.maximum(rounded, 0x00800000).to(tl.float32, bitcast=True)
    return tl.where(amax == 0, 1.0, scale)
