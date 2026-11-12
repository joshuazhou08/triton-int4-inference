# Motivation

## Batching

For a simple neural network lyaer: $z = Wx + b$, suppose weight matrix $W \in \mathbb{R}^{n \times n}$. For usual 16 bit floating point, loading in W requires $2n^2$ bytes. Usually, this weight matrix is too large to fit into the fast on chip cache. It usually resides in GPU VRAM (slower, larger), and is read back and forth to the chip (CPU) cache. Additionally, matrix multiplication for $Wx$ requires $2n^2$ FLOPs (floating point operations), as for each row of W, we compute:
$$y_i = \sum_{j = 1}^{n} W_{ij}x_j$$

It is clear that there are (n - 1) additions and n multplications for each row. Since there are n rows, we end up with

$$ n(n + n - 1) \approx 2n^2 $$

Assume we have B inputs: $\{x_1, x_2, ...,\ x_B\}$. Without batching, we read the weight matrix back and forth six times, and the **arithmetic intensity** is then:

$$I = \frac{\text{FLOPS}}{\text{bytes transferred}} \approx \frac{2n^2}{2n^2} = \boxed{ 1 \ \text{FLOP/byte}}$$

This is quite low for a modern GPU. Batching helps as we can do matrix-matrix multiplication:

$$Y = WX, \ X \in \mathbb{R} ^ {n \times B}$$

In doing so, every time we transfer the bytes needed for W, we perform the operations needed for B of the inputs. Hence (ignoring activations) we get:

$$I \approx \boxed{B \ \text{FLOPs/byte}}$$

### Small Batch Sizes

For token inference at small batch sizes, loading model weights repeatedly from VRAM to on chip cache can limit performance as seen by the measure of arithmetic intensity given batch size B. We can reduce the precision we operate at, but this can hurt accuracy.

## Our Proposal

We can store the model's weights as 4-bit integers while keeping the activations as floating-point numbers during inference. This should quadruple (16 divided by 4) our arithmetic intensity, ignoring bias.

Consider layer:
$$ Y = XW + b $$

- X: **Activations**: The input values to the layer, represented using FP16.
- W: **Weights**: The learned parameters of the neurla network, stored as 4 bit integers instead of FP16.
- b: **Bias**: Bias. Stored as FP16. We will only quantize the weights as bias is much smaller.

### Quantizing the weights.

Suppose original weights:

$$
\begin{bmatrix}
0.27 & -0.83\\
1.42 & 0.51
\end{bmatrix}
$$

We can approximate each weight as $ w_i \approx sq_i $ where $q_i$ is a signed 4 bit integer in [-8, 7].

$$
q_i = \text{clip}(\text{round}(\frac{w_i}{s}, -8, 7))
$$

For example, using $s = 0.2$:

$$
0.27 \approx 0.2(1) = 0.2
$$

$$
-0.83 \approx 0.2(-4) = -0.8
$$

Notice that in quantizing, we do lose some accuracy. Larger s results in more loss of accuracy. Hence, we can try to group weights together and maybe even have an "offset" to 0 mean groups of weights to allow for smaller s:

$$
\boxed{\hat{w_i} = s_gq_i + b_g} \ \ \text{(g indexes the weight group)}
$$

**A big part of this project than also becomes finding an algorithm to get a good bias to allow for smallest s while reducing clipping.**

- We can also try techniques like mixed precision, where we find the numbers that are massive outliers and save them in a separate full precision sparse matrix, and quantize the other elements.

# Implementation

1. Compare our custom symmetric groupwise quantizer against official TorchAO quantization functions (see below).

2. We will write out own custom Triton GPU kernel.
   - Load packed 4-bit weights
   - Unpack and rescale them and add bias within the kernel
   - Multiply with floating-point activations
   - Output floating-point activations
   - We never recosntruct entire weight matrix in GPU memory!
3. Replace selected linear layers in a pretrained Transformer with custom kernel and evaluate inference accuracy and speed. We can also benchmark our implementation against standard floating point matrix multplication and our pytorch reference.

# Project structure

```text
quantization/
    custom_symmetric.py          Our hand-written quantization math
    torchao_reference.py         Adapter calling official TorchAO quantizers
examples/
    demo_custom_quantization.py  Demonstrates only our custom implementation
    compare_quantizers.py        Compares custom, TorchAO, and floating-point results
tests/
    test_custom_symmetric.py
    test_torchao_reference.py
requirements.txt                Pinned PyTorch/TorchAO dependencies
```

The former `quantization.py` and `demo_quantization.py` were our own code,
not official PyTorch examples. They have been moved and renamed as shown above.
Import an implementation explicitly; both expose `quantize_groupwise`,
`dequantize_groupwise`, and `linear`.

## Custom quantizer and official reference

Both use weights shaped `[out_features, in_features]`, with contiguous groups
along each row (default size 128), and compute `Y = X @ weight.T + bias`.

- **Custom:** implements `scale = max(abs(weights)) / 7`, rounding, clipping to
  `[-7, 7]`, and reconstruction ourselves using torch tensor operations.
- **TorchAO reference:** calls the official `choose_qparams_affine`,
  `quantize_affine`, and `dequantize_affine` functions with
  `MappingType.SYMMETRIC`, block size `(1, group_size)`, and explicit bounds
  `[-7, 7]`. It never calls our custom quantizer. See the
  [TorchAO primitives source for the pinned release](https://github.com/pytorch/ao/blob/v0.13.0/torchao/quantization/quant_primitives.py).

The bounds deliberately match our custom algorithm, leaving INT4 code `-8`
unused. This is an **official-library numerical reference configured for our
experiment**, not TorchAO's default `Int4WeightOnlyConfig` or its packed inference
kernel. Both paths use FP32 scales and quantization arithmetic, support partial
final groups, and accept FP16, BF16, or FP32 weights and activations.
The TorchAO adapter requires group size >= 2; the custom implementation also
supports size 1. The adapter retains padded codes internally and crops decoded
weights to their original shape.

Small differences are expected: TorchAO retains its default FP32 epsilon floor
for scales (`~1.19e-7`), whereas our custom code uses the smallest normal FP32
value for nonzero groups and scale 1 for zero groups. Both reconstruct zero
groups as zero, but very small nonzero weights can differ. Floating-point
rounding near quantization thresholds can also produce different integer codes.
The comparison reports these differences rather than assuming exact equality.

## Storage and computation

**4-bit values currently use 8-bit storage:** each code occupies one byte in
an unpacked `torch.int8` tensor. Scales add four bytes per group; the TorchAO
reference also retains its INT32 zero points and any padding. Each `linear`
function reconstructs the full floating-point weights before multiplication.
Activations and bias remain floating point, and decoding uses the activation
dtype. Both implementations are inference-only and do not track gradients.

Actual 4-bit storage will pack **two weights per byte**. The future Triton kernel
will unpack and rescale inside the kernel. These numerical references do not
measure that kernel's speed or memory savings.

## Setup and usage

Use Python 3.10–3.13. Dependencies pin PyTorch 2.8.0 and TorchAO 0.13.0, a
[compatible release pair](https://github.com/pytorch/ao/issues/2919).
Run commands from the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v

# Official TorchAO vs our custom implementation, using identical inputs:
python -m examples.compare_quantizers

# Our custom implementation alone:
python -m examples.demo_custom_quantization

# Requires a CUDA-enabled PyTorch installation and an accessible GPU:
python -m examples.compare_quantizers --device cuda --group-size 64
```

```python
import torch
from quantization import torchao_reference

layer = torch.nn.Linear(512, 256, dtype=torch.float16)
x = torch.randn(4, 512, dtype=torch.float16)
quantized = torchao_reference.quantize_groupwise(layer.weight, group_size=128)
weight_hat = torchao_reference.dequantize_groupwise(quantized)
y = torchao_reference.linear(x, quantized, layer.bias)
```

Keep weights, activations, and bias on the same device. To use our implementation,
import `custom_symmetric` instead. The comparison prints dependency versions,
weight/output errors against the original floating-point layer, errors between
the two quantizers, and integer code mismatches, using a fixed random seed.

# Evaluation

Tests cover hand-calculated results, rounding, partial groups, noncontiguous
weights, zero/tiny groups, reconstruction error, and floating-point linear
outputs. They also compare our algorithm against actual TorchAO calls. CUDA
tests run when available. Model accuracy, packed storage, and optimized inference
timing remain future work.
