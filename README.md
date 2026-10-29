# Motivation

## Batching

For a simple neural network lyaer: $z = Wx + b$, suppose weight matrix $W \in \mathbb{R}^{n \times n}$. For usual 16 bit floating point, loading in W requires $2n^2$ bytes. Usually, this weight matrix is too large to fit into the fast on chip cache. It usually resides in GPU VRAM (slower, larger), and is read back and forth to the chip (CPU) cache. Additionally, matrix multiplication for $Wx$ requires $2n^2$ FLOPs (floating point operations), as for each row of W, we compute:
$$y_i = \sum_{j = 1}^{n} W_{ij}x_j$$

It is clear that there are n additions and n - multplications for each row. Since there are n rows, we end up with

$$ n(n + n - 1) \approx 2n^2 $$

Assume we have B inputs: $\{x_1, x_2, ...,\ x_B\}$. Without batching, we read the weight matrix back and forth six times, and the **arithmetic intensity** is then:

$$I = \frac{\text{FLOPS}}{\text{bytes transferred}} \approx \frac{2n^2}{2n^2} = \boxed{ 1 \ \text{FLOP/byte}}$$

This is quite low for a modern GPU. Batching helps as we can do matrix-matrix multiplication:

$$Y = WX, \ X \in \mathbb{R} ^ {n \times B}$$

In doing so, every time we transfer the bytes needed for W, we perform the operations needed for B of the inputs. Hence (ignoring activations) we get:

$$I \approx \boxed{B \ \text{FLOPs/byte}}$$

### Small Batch Sizes

For token inference at small batch sizes, loading model weights repeatedly from VRAM to on chip cache can limit performance as seen by the measure of arithmetic intensity given batch size B. We can either reduce the precision of the weights, but this can hurt accuracy.

## Our Proposal

We can store the model's weights as 4-bit integers while keeping the activations as floating-point numbers during inference.

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

1. We will use Pytorch's symmetric groupwise quantization as a reference.

2. We will write out own custom Triton GPU kernel.
   - Load packed 4-bit weights
   - Unpack and rescale them and add bias within the kernel
   - Multiply with floating-point activations
   - Output floating-point activations
   - We never recosntruct entire weight matrix in GPU memory!
3. Replace selected linear layers in a pretrained Transformer with custom kernel and evaluate inference accuracy and speed. We can also benchmark our implementation against standard floating point matrix multplication and our pytorch reference.
