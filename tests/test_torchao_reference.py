import unittest

import torch

from quantization import custom_symmetric, torchao_reference


class TorchAOReferenceTests(unittest.TestCase):
    def test_known_codes_scales_and_reconstruction(self):
        weight = torch.tensor([[-7., -2.5, 0.5, 7., -14., -3., 3., 14.]])
        result = torchao_reference.quantize_groupwise(weight, 4)
        torch.testing.assert_close(result.scales, torch.tensor([[1., 2.]]))
        self.assertEqual(result.zero_points.dtype, torch.int32)
        self.assertTrue((result.zero_points == 0).all())
        torch.testing.assert_close(
            result.qweight,
            torch.tensor([[-7, -2, 0, 7, -7, -2, 2, 7]], dtype=torch.int8),
        )
        torch.testing.assert_close(
            torchao_reference.dequantize_groupwise(result),
            torch.tensor([[-7., -2., 0., 7., -14., -4., 4., 14.]]),
        )

    def test_zero_and_tiny_groups_keep_official_epsilon_policy(self):
        weight = torch.tensor([[0., 0.], [1e-9, -1e-9]])
        official = torchao_reference.quantize_groupwise(weight, 2)
        custom = custom_symmetric.quantize_groupwise(weight, 2)
        torch.testing.assert_close(
            official.scales, torch.full_like(official.scales, torch.finfo(torch.float32).eps),
            rtol=0, atol=0,
        )
        torch.testing.assert_close(
            torchao_reference.dequantize_groupwise(official), torch.zeros_like(weight),
            rtol=0, atol=0,
        )
        self.assertEqual(custom.scales[0, 0].item(), 1.)
        self.assertTrue((custom.qweight[1] != 0).all())

    def test_matches_custom_for_ordinary_weights_and_partial_groups(self):
        torch.manual_seed(221)
        weight = torch.randn(13, 5).T
        for group_size in (2, 4, 13, 32, 128):
            with self.subTest(group_size=group_size):
                custom = custom_symmetric.quantize_groupwise(weight, group_size)
                official = torchao_reference.quantize_groupwise(weight, group_size)
                torch.testing.assert_close(
                    official.qweight[:, :13], custom.qweight, rtol=0, atol=0,
                )
                torch.testing.assert_close(
                    official.scales.reshape_as(custom.scales), custom.scales,
                )
                torch.testing.assert_close(
                    torchao_reference.dequantize_groupwise(official),
                    custom_symmetric.dequantize_groupwise(custom),
                )

    def check_linear(self, device, dtype):
        weight = torch.tensor([[7., 0.3, -2.6], [14., 1.2, 0.]], device=device)
        result = torchao_reference.quantize_groupwise(weight, 2)
        inputs = torch.tensor([1., 2., 3.], device=device, dtype=dtype)
        bias = torch.tensor([0.5, -0.5], device=device)
        decoded = torch.tensor([[7., 0., -2.6], [14., 2., 0.]], device=device, dtype=dtype)
        for shape in ((3,), (4, 3), (2, 4, 3)):
            x = inputs.expand(shape)
            for b in (None, bias):
                expected = x @ decoded.T
                if b is not None:
                    # Compute together to match the linear operation's bias rounding.
                    expected = torch.nn.functional.linear(x, decoded, b.to(dtype))
                actual = torchao_reference.linear(x, result, b)
                self.assertEqual(actual.dtype, dtype)
                self.assertEqual(actual.device, inputs.device)
                torch.testing.assert_close(actual, expected)

    def test_linear_cpu(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            with self.subTest(dtype=dtype):
                self.check_linear("cpu", dtype)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_linear_cuda(self):
        self.check_linear("cuda", torch.float16)

    def test_invalid_inputs_and_no_mutation(self):
        for size in (0, 1, -1, True, 2.5):
            with self.assertRaises(ValueError):
                torchao_reference.quantize_groupwise(torch.ones(2, 3), size)
        for weight in (torch.ones(3), torch.ones(0, 3), torch.tensor([[float("nan")]])):
            with self.assertRaises(ValueError):
                torchao_reference.quantize_groupwise(weight)
        with self.assertRaises(TypeError):
            torchao_reference.quantize_groupwise(torch.ones(2, 3, dtype=torch.int8))
        weight = torch.randn(2, 3, requires_grad=True)
        original = weight.detach().clone()
        result = torchao_reference.quantize_groupwise(weight, 2)
        torch.testing.assert_close(weight, original)
        self.assertFalse(result.scales.requires_grad)
        self.assertFalse(torchao_reference.dequantize_groupwise(result).requires_grad)
        with self.assertRaises(ValueError):
            torchao_reference.linear(torch.ones(2, 4), result)
        with self.assertRaises(ValueError):
            torchao_reference.linear(torch.ones(2, 3), result, torch.ones(3))


if __name__ == "__main__":
    unittest.main()
