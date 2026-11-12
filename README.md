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

1. We use a symmetric groupwise quantization reference implemented with PyTorch tensor operations (see below).

2. We will write out own custom Triton GPU kernel.
   - Load packed 4-bit weights
   - Unpack and rescale them and add bias within the kernel
   - Multiply with floating-point activations
   - Output floating-point activations
   - We never recosntruct entire weight matrix in GPU memory!
3. Replace selected linear layers in a pretrained Transformer with custom kernel and evaluate inference accuracy and speed. We can also benchmark our implementation against standard floating point matrix multplication and our pytorch reference.

# PyTorch reference

`quantization.py` implements the initial correctness baseline. It uses PyTorch's
linear-layer layout, `weight.shape == [out_features, in_features]`, so the layer
operation is `Y = X @ weight.T + bias`.

Each output row is split into contiguous groups along `in_features` (default
group size: 128). For each group:

$$s_g = \max_i |w_i| / 7, \qquad q_i = \operatorname{clip}(\operatorname{round}(w_i/s_g), -7, 7), \qquad \hat w_i = s_g q_i.$$

This baseline uses the symmetric subset `[-7, 7]` of signed INT4; the code `-8`
is unused. Rounding follows [PyTorch's ties-to-even convention](https://docs.pytorch.org/docs/stable/generated/torch.round.html).
Scales and quantization arithmetic use FP32. All-zero groups use scale 1;
nonzero scales are bounded below by the smallest normal FP32 value to avoid
underflow. A short final group is supported without changing the returned
weight shape. Group offsets and outlier handling are future extensions.

Quantized weights have **4-bit values but currently use 8-bit storage**:
each code in `[-7, 7]` occupies one byte in a `torch.int8` tensor, plus four
bytes per group for scales. `linear_reference` reconstructs floating-point
weights on every call before applying [PyTorch's linear operation](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.linear.html);
the matrix multiplication uses floating-point weights, not int8 weights.
This reference measures quantization error. Actual 4-bit storage will pack
**two weights per byte**, with the future Triton kernel unpacking and rescaling
them inside the kernel.
Activations and layer bias remain floating point; decoded weights and bias use
the activation dtype. FP16, BF16, and FP32 are supported. The functions are
inference-only and do not track gradients.

## Setup and usage

Requires Python 3.10+ and PyTorch. Install a CUDA-enabled PyTorch build if running
on a GPU; the reference and its tests also run on CPU.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python demo_quantization.py
# With a CUDA-enabled PyTorch installation:
python demo_quantization.py --device cuda --group-size 64
```

```python
import torch
from quantization import quantize_groupwise, dequantize_groupwise, linear_reference

layer = torch.nn.Linear(512, 256, dtype=torch.float16)
x = torch.randn(4, 512, dtype=torch.float16)
quantized = quantize_groupwise(layer.weight, group_size=128)
weight_hat = dequantize_groupwise(quantized)
y = linear_reference(x, quantized, layer.bias)
```

Keep activations, weights, and bias on the same device. The returned
`GroupwiseInt4` holds `qweight`, `scales`, `group_size`, and the original `dtype`.
The demo reports weight reconstruction and linear-output MSE, maximum absolute
error, and relative L2 error against the unquantized layer with a fixed seed.

# Evaluation

The unit tests cover known quantization results, rounding, independent row
groups, partial groups, zeros, the half-scale reconstruction error bound,
noncontiguous weights, and linear outputs with and without bias. CUDA checks
run when available. The synthetic demo measures numerical error only; model
accuracy and inference timing remain to be evaluated after kernel integration.
