import math
import unittest

import torch
from torch.nn import functional as F

from quantization import dequantize_groupwise, linear_reference, quantize_groupwise


class QuantizationTests(unittest.TestCase):
    def test_known_groups_and_rounding(self):
        weight = torch.tensor([[-7., -2.5, 0.5, 7., -14., -3., 3., 14.]])
        quantized = quantize_groupwise(weight, 4)
        torch.testing.assert_close(quantized.scales, torch.tensor([[1., 2.]]))
        torch.testing.assert_close(
            quantized.qweight,
            torch.tensor([[-7, -2, 0, 7, -7, -2, 2, 7]], dtype=torch.int8),
        )
        torch.testing.assert_close(
            dequantize_groupwise(quantized),
            torch.tensor([[-7., -2., 0., 7., -14., -4., 4., 14.]]),
        )

    def test_partial_groups_are_independent_per_row(self):
        weight = torch.tensor([[7., 0., 14.], [70., 0., 140.]])
        quantized = quantize_groupwise(weight, 2)
        torch.testing.assert_close(quantized.scales, torch.tensor([[1., 2.], [10., 20.]]))
        self.assertEqual(quantized.qweight.shape, weight.shape)
        torch.testing.assert_close(dequantize_groupwise(quantized), weight)

    def test_zero_groups_and_subnormals(self):
        weight = torch.tensor([[0., 0., 0.], [1e-44, -1e-44, 0.]])
        quantized = quantize_groupwise(weight, 2)
        self.assertTrue(torch.isfinite(quantized.scales).all())
        self.assertTrue((quantized.scales > 0).all())
        torch.testing.assert_close(quantized.scales[0], torch.ones(2))
        torch.testing.assert_close(dequantize_groupwise(quantized)[0], weight[0])

    def test_error_bound_noncontiguous_and_group_sizes(self):
        torch.manual_seed(221)
        weight = torch.randn(13, 5).T
        self.assertFalse(weight.is_contiguous())
        for group_size in (1, 4, 13, 32):
            with self.subTest(group_size=group_size):
                quantized = quantize_groupwise(weight, group_size)
                scales = quantized.scales[:, torch.arange(13) // group_size]
                error = (dequantize_groupwise(quantized) - weight).abs()
                self.assertTrue((error <= scales / 2 + 1e-6).all())
                self.assertEqual(quantized.qweight.dtype, torch.int8)
                self.assertTrue((quantized.qweight.abs() <= 7).all())

    def check_linear(self, device, dtype):
        # Exactly representable weights isolate the linear operation from
        # quantization error and check orientation, broadcasting, and bias.
        weight = torch.tensor([[7., -7., 0.], [14., 0., -14.]], device=device, dtype=dtype)
        bias = torch.tensor([0.5, -0.5], device=device, dtype=dtype)
        quantized = quantize_groupwise(weight, 2)
        self.assertEqual(quantized.scales.dtype, torch.float32)
        self.assertEqual(dequantize_groupwise(quantized).dtype, dtype)
        self.assertEqual(quantized.qweight.device, weight.device)
        for shape in ((3,), (4, 3), (2, 4, 3)):
            inputs = torch.arange(math.prod(shape), device=device, dtype=dtype)
            inputs = inputs.reshape(shape) / 8
            for layer_bias in (None, bias):
                actual = linear_reference(inputs, quantized, layer_bias)
                self.assertEqual(actual.dtype, dtype)
                torch.testing.assert_close(actual, F.linear(inputs, weight, layer_bias))

    def test_linear_cpu(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            with self.subTest(dtype=dtype):
                self.check_linear("cpu", dtype)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_linear_cuda(self):
        for dtype in (torch.float16, torch.float32):
            with self.subTest(dtype=dtype):
                self.check_linear("cuda", dtype)

    def test_quantization_detaches_and_preserves_source(self):
        weight = torch.randn(3, 9, requires_grad=True)
        original = weight.detach().clone()
        quantized = quantize_groupwise(weight, 4)
        torch.testing.assert_close(weight, original)
        self.assertFalse(quantized.scales.requires_grad)
        self.assertFalse(dequantize_groupwise(quantized).requires_grad)
        self.assertEqual(dequantize_groupwise(quantized, dtype=torch.float16).dtype, torch.float16)

    def test_linear_with_lossy_weights(self):
        # Hand-computed decoded weights verify linear integration when the
        # quantization actually changes values, including a partial group.
        weight = torch.tensor([[7., 0.3, -2.6], [14., 1.2, 0.]])
        quantized = quantize_groupwise(weight, 2)
        inputs = torch.tensor([[1., 2., 3.], [-1., 0., 1.]], dtype=torch.float16)
        bias = torch.tensor([0.5, -0.5])
        decoded = torch.tensor([[7., 0., -2.6], [14., 2., 0.]], dtype=torch.float16)
        expected = F.linear(inputs, decoded, bias.half())
        torch.testing.assert_close(linear_reference(inputs, quantized, bias), expected)

    def test_invalid_inputs(self):
        for group_size in (0, -1, 2.5, True):
            with self.subTest(group_size=group_size), self.assertRaises(ValueError):
                quantize_groupwise(torch.ones(2, 3), group_size)
        invalid_weights = (
            torch.ones(3), torch.ones(0, 3),
            torch.tensor([[float("nan")]]), torch.tensor([[float("inf")]]),
        )
        for weight in invalid_weights:
            with self.subTest(weight=weight), self.assertRaises(ValueError):
                quantize_groupwise(weight)
        with self.assertRaises(TypeError):
            quantize_groupwise(torch.ones(2, 3, dtype=torch.int32))
        quantized = quantize_groupwise(torch.ones(2, 3))
        with self.assertRaises(ValueError):
            linear_reference(torch.ones(4, 5), quantized)
        with self.assertRaises(ValueError):
            linear_reference(torch.ones(4, 3), quantized, torch.ones(3))
        with self.assertRaises(TypeError):
            linear_reference(torch.ones(4, 3, dtype=torch.int32), quantized)
        with self.assertRaises(TypeError):
            dequantize_groupwise(quantized, dtype=torch.int8)


if __name__ == "__main__":
    unittest.main()
