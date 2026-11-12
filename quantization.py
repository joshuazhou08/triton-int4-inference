"""PyTorch correctness reference for symmetric groupwise INT4 weights.

Weights follow nn.Linear's [out_features, in_features] layout. Integer codes
are stored unpacked in int8; this module does not implement packed storage or
a fast inference kernel. Quantization is inference-only, without autograd.
"""

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F


_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


@dataclass(frozen=True)
class GroupwiseInt4:
    """Unpacked codes and FP32 scales, on the same device as the input weights.

    qweight: [out_features, in_features], int8 values in [-7, 7].
    scales: [out_features, ceil(in_features / group_size)], float32.
    dtype: original floating-point weight dtype used by default on decoding.
    """

    qweight: Tensor
    scales: Tensor
    group_size: int
    dtype: torch.dtype


@torch.no_grad()
def quantize_groupwise(weight: Tensor, group_size: int = 128) -> GroupwiseInt4:
    """Quantize each contiguous input-feature group independently per row.

    For each group, s = max(abs(w)) / 7 and q = round(w / s).
    Rounding uses PyTorch's ties-to-even convention. The symmetric [-7, 7]
    range leaves the signed INT4 code -8 unused, avoiding asymmetric clipping.
    All-zero groups use s = 1. A partial final group is padded internally with
    zeros, but the returned codes retain the original weight shape.
    """
    if isinstance(group_size, bool) or not isinstance(group_size, int) or group_size <= 0:
        raise ValueError("group_size must be a positive integer")
    if weight.ndim != 2 or any(size == 0 for size in weight.shape):
        raise ValueError("weight must be a nonempty [out_features, in_features] matrix")
    if weight.dtype not in _FLOAT_DTYPES:
        raise TypeError("weight must use float16, bfloat16, or float32")
    if not torch.isfinite(weight).all().item():
        raise ValueError("weight must contain only finite values")

    rows, columns = weight.shape
    padding = (-columns) % group_size
    groups = F.pad(weight.float(), (0, padding)).reshape(rows, -1, group_size)
    absmax = groups.abs().amax(dim=-1)
    # Keep nonzero scales representable, even for FP32 subnormal inputs.
    scales = (absmax / 7).clamp_min(torch.finfo(torch.float32).tiny)
    scales = torch.where(absmax == 0, torch.ones_like(scales), scales)
    codes = torch.round(groups / scales.unsqueeze(-1)).clamp(-7, 7)
    qweight = codes.to(torch.int8).reshape(rows, -1)[:, :columns].contiguous()
    return GroupwiseInt4(qweight, scales, group_size, weight.dtype)


@torch.no_grad()
def dequantize_groupwise(
    quantized: GroupwiseInt4, *, dtype: torch.dtype | None = None
) -> Tensor:
    """Reconstruct the full floating-point weight matrix (reference only).

    Rescaling happens in FP32 before casting to the requested output dtype.
    """
    dtype = quantized.dtype if dtype is None else dtype
    if dtype not in _FLOAT_DTYPES:
        raise TypeError("output dtype must be float16, bfloat16, or float32")
    columns = quantized.qweight.shape[1]
    group_indices = torch.arange(columns, device=quantized.qweight.device)
    group_indices = group_indices // quantized.group_size
    scales = quantized.scales[:, group_indices]
    return (quantized.qweight.float() * scales).to(dtype)


@torch.no_grad()
def linear_reference(
    inputs: Tensor, quantized: GroupwiseInt4, bias: Tensor | None = None
) -> Tensor:
    """Compute inputs @ dequantized_weight.T + bias without quantizing inputs.

    Inputs may have any leading dimensions, including a single input vector.
    Weight and bias are cast to the activation dtype; all tensors must already
    be on the same device. This deliberately materializes the decoded weights
    on every call, so it is a correctness baseline rather than a speedup.
    """
    if inputs.dtype not in _FLOAT_DTYPES:
        raise TypeError("inputs must use float16, bfloat16, or float32")
    if inputs.ndim == 0 or inputs.shape[-1] != quantized.qweight.shape[1]:
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
    weight = dequantize_groupwise(quantized, dtype=inputs.dtype)
    return F.linear(inputs, weight, bias)
