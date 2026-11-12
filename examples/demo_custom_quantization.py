"""Demonstrate only our custom quantizer; not an official PyTorch demo."""

import argparse

import torch
from torch.nn import functional as F

from quantization.custom_symmetric import dequantize_groupwise, linear, quantize_groupwise


def report_error(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    delta = actual.float() - expected.float()
    relative = delta.norm() / expected.float().norm().clamp_min(1e-12)
    print(
        f"{name}: MSE={delta.square().mean().item():.6g}, "
        f"max_abs={delta.abs().max().item():.6g}, relative_L2={relative.item():.4%}"
    )


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--dtype", default="float16", choices=("float16", "float32", "bfloat16"))
    parser.add_argument("--group-size", type=int, default=128)
    args = parser.parse_args()
    torch.manual_seed(221)
    dtype = getattr(torch, args.dtype)
    weight = torch.randn(256, 512, device=args.device, dtype=dtype) / (512**0.5)
    inputs = torch.randn(4, 512, device=args.device, dtype=dtype)
    bias = torch.randn(256, device=args.device, dtype=dtype)

    quantized = quantize_groupwise(weight, args.group_size)
    reconstructed = dequantize_groupwise(quantized)
    actual = linear(inputs, quantized, bias)
    expected = F.linear(inputs, weight, bias)
    print(f"Device={args.device}, dtype={args.dtype}, group_size={args.group_size}")
    report_error("Weights", reconstructed, weight)
    report_error("Linear output", actual, expected)
    print("Custom quantizer: unpacked int8 codes, full floating-point reconstruction.")


if __name__ == "__main__":
    main()
