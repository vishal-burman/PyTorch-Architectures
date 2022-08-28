import unittest

import torch
from .model import LeNet


class LeNetTestCase(unittest.TestCase):
    def setUp(self):
        self.model = LeNet(num_classes=2, grayscale=False)

    def test_parameter_count(self):
        params_count = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        self.assertEqual(params_count, 546854)

    def test_forward_pass(self):
        pixel_values_sample = torch.rand(3, 3, 32, 32)
        labels_sample = torch.ones(3, dtype=torch.long)
        loss, logits = self.model(
            pixel_values=pixel_values_sample, labels=labels_sample
        )
        self.assertEqual(logits.dim(), 2)
        self.assertEqual(logits.size(0), pixel_values_sample.size(0))
        self.assertEqual(logits.size(1), 2)
        self.assertIsNotNone(loss)
        self.assertTrue(type(loss), torch.Tensor)

    def tearDown(self):
        del self.model
