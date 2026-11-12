"""Compare our custom quantizer with official TorchAO on identical inputs."""

import argparse

import torch
import torchao
from torch.nn import functional as F

from examples.demo_custom_quantization import report_error
from quantization import custom_symmetric, torchao_reference


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--dtype", default="float16", choices=("float16", "float32", "bfloat16"))
    parser.add_argument("--group-size", type=int, default=128)
    args = parser.parse_args()
    if args.group_size < 2:
        parser.error("--group-size must be at least 2 for the TorchAO reference")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable in this environment")

    torch.manual_seed(221)
    dtype = getattr(torch, args.dtype)
    weight = torch.randn(256, 512, device=args.device, dtype=dtype) / (512**0.5)
    inputs = torch.randn(4, 512, device=args.device, dtype=dtype)
    bias = torch.randn(256, device=args.device, dtype=dtype)
    fp_output = F.linear(inputs, weight, bias)

    custom = custom_symmetric.quantize_groupwise(weight, args.group_size)
    official = torchao_reference.quantize_groupwise(weight, args.group_size)
    custom_weight = custom_symmetric.dequantize_groupwise(custom)
    official_weight = torchao_reference.dequantize_groupwise(official)
    custom_output = custom_symmetric.linear(inputs, custom, bias)
    official_output = torchao_reference.linear(inputs, official, bias)

    print(f"torch={torch.__version__}, torchao={torchao.__version__}")
    print(f"device={args.device}, dtype={args.dtype}, group_size={args.group_size}")
    report_error("Custom weights vs original", custom_weight, weight)
    report_error("TorchAO weights vs original", official_weight, weight)
    report_error("Custom output vs floating-point baseline", custom_output, fp_output)
    report_error("TorchAO output vs floating-point baseline", official_output, fp_output)
    report_error("Custom weights vs TorchAO", custom_weight, official_weight)
    report_error("Custom output vs TorchAO", custom_output, official_output)
    mismatches = (custom.qweight != official.qweight[:, :weight.shape[1]]).sum().item()
    print(f"Integer code mismatches: {mismatches}/{weight.numel()}")
    print("Both paths use unpacked int8 codes and full floating-point reconstruction.")
    print("This compares numerical results, not packed-kernel speed or memory savings.")


if __name__ == "__main__":
    main()
