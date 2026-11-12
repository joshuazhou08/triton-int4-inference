"""Official TorchAO affine primitives configured for symmetric groupwise INT4.

This adapter handles tensor layout and calls TorchAO for parameter selection,
quantization, and dequantization. It does not call our custom quantizer.
Codes remain unpacked int8; this is a numerical reference, not a packed kernel.
Tested API: torch 2.8.0 and torchao 0.13.0.
"""

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F
from torchao.quantization.quant_primitives import (
    MappingType,
    choose_qparams_affine,
    dequantize_affine,
    quantize_affine,
)


_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


@dataclass(frozen=True)
class TorchAOGroupwiseInt4:
    """Padded integer codes and TorchAO's unmodified quantization parameters.

    qweight has shape [out_features, padded_in_features]. Scales and zero
    points have one element per group. in_features records the unpadded size.
    """

    qweight: Tensor
    scales: Tensor
    zero_points: Tensor
    group_size: int
    in_features: int
    dtype: torch.dtype


@torch.no_grad()
def quantize_groupwise(weight: Tensor, group_size: int = 128) -> TorchAOGroupwiseInt4:
    """Call TorchAO's symmetric affine quantizer with bounds [-7, 7].

    Group size must be at least 2. Quantization arithmetic and scales use
    FP32. TorchAO's default FP32 epsilon floor is retained, including for
    zero groups; unlike our custom implementation, their scale is not 1.
    """
    if isinstance(group_size, bool) or not isinstance(group_size, int) or group_size < 2:
        raise ValueError("TorchAO reference group_size must be an integer >= 2")
    if weight.ndim != 2 or any(size == 0 for size in weight.shape):
        raise ValueError("weight must be a nonempty [out_features, in_features] matrix")
    if weight.dtype not in _FLOAT_DTYPES:
        raise TypeError("weight must use float16, bfloat16, or float32")
    if not torch.isfinite(weight).all().item():
        raise ValueError("weight must contain only finite values")

    # Padding is adapter glue; all quantization math is delegated to TorchAO.
    padded = F.pad(weight.float(), (0, (-weight.shape[1]) % group_size)).contiguous()
    block_size = (1, group_size)
    scales, zero_points = choose_qparams_affine(
        padded, MappingType.SYMMETRIC, block_size, torch.int8,
        quant_min=-7, quant_max=7,
        scale_dtype=torch.float32, zero_point_dtype=torch.int32,
    )
    codes = quantize_affine(
        padded, block_size, scales, zero_points, torch.int8,
        quant_min=-7, quant_max=7,
    )
    return TorchAOGroupwiseInt4(
        codes, scales, zero_points, group_size, weight.shape[1], weight.dtype,
    )


@torch.no_grad()
def dequantize_groupwise(
    quantized: TorchAOGroupwiseInt4, *, dtype: torch.dtype | None = None
) -> Tensor:
    """Use TorchAO to decode in FP32, crop padding, then cast the result."""
    dtype = quantized.dtype if dtype is None else dtype
    if dtype not in _FLOAT_DTYPES:
        raise TypeError("output dtype must be float16, bfloat16, or float32")
    decoded = dequantize_affine(
        quantized.qweight, (1, quantized.group_size),
        quantized.scales, quantized.zero_points, torch.int8,
        quant_min=-7, quant_max=7, output_dtype=torch.float32,
    )
    return decoded[:, :quantized.in_features].to(dtype).contiguous()


@torch.no_grad()
def linear(inputs: Tensor, quantized: TorchAOGroupwiseInt4, bias: Tensor | None = None) -> Tensor:
    """Evaluate a floating-point linear layer using TorchAO-decoded weights."""
    if inputs.dtype not in _FLOAT_DTYPES:
        raise TypeError("inputs must use float16, bfloat16, or float32")
    if inputs.ndim == 0 or inputs.shape[-1] != quantized.in_features:
        raise ValueError("inputs' last dimension must match weight in_features")
    if inputs.device != quantized.qweight.device:
        raise ValueError("inputs and quantized weights must be on the same device")
    if bias is not None:
        if bias.shape != (quantized.qweight.shape[0],):
            raise ValueError("bias must have shape [out_features]")
        if bias.dtype not in _FLOAT_DTYPES:
            raise TypeError("bias must use float16, bfloat16, or float32")
        if bias.device != inputs.device:
            raise ValueError("bias and inputs must be on the same device")
        bias = bias.to(dtype=inputs.dtype)
    return F.linear(inputs, dequantize_groupwise(quantized, dtype=inputs.dtype), bias)
