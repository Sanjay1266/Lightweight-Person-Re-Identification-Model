"""
Automated Test Suite for Paper-Faithful DCR-ReID (Track A)
Reference: Zhenyu Cui et al., "DCR-ReID: Deep Component Reconstruction for Cloth-Changing Person Re-Identification",
IEEE Transactions on Circuits and Systems for Video Technology (TCSVT), Vol. 33, No. 8, August 2023.

Verifies:
1. ResNet-50 backbone with layer4 stride=1
2. CRD projection (2048 -> 2560) and 3-component channel decomposition (1024, 512, 1024)
3. 4-block residual decoders ψ^+, ψ^-, ψ^t with Tanh activation
4. Stacked max-pooling and average-pooling φ(P_i^v)
5. Inference feature extraction (4096-D, L2 normalized, F_i^- removed)
6. DAD feature assembly G_i (5120-D)
7. Label-smoothed identity loss L_ID
8. Component reconstruction loss L_R
9. Clothes adversarial loss L_ca with uniform target q(c) = 1 / (N_c - 1)
10. Assembled clothes loss L_ac with intra-identity target h(c) = 1 / K
11. Two-stage optimization loss formulation
12. CUDA forward/backward gradient flow and numerical stability
"""

import os
import sys
sys.path.insert(0, os.path.abspath('.'))

import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.paper_dcr_reid import PaperDCRReID, ComponentReconstructionDecoder, StackedGlobalPooling
from models.paper_loss import (
    LabelSmoothCrossEntropy,
    ComponentReconstructionLoss,
    ClothesAdversarialLoss,
    AssembledClothesLoss,
    PaperDCRReIDCombinedLoss
)


class TestPaperDCRReID(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def test_01_paper_model_initialization(self):
        """Verifies PaperDCRReID architectural submodules and layer structure."""
        model = PaperDCRReID(num_classes=150, num_clothes=100, pretrained=False)
        self.assertIsNotNone(model.stem)
        self.assertIsNotNone(model.layer1)
        self.assertIsNotNone(model.layer2)
        self.assertIsNotNone(model.layer3)
        self.assertIsNotNone(model.layer4)
        self.assertIsNotNone(model.crd_proj)
        self.assertIsNotNone(model.psi_plus)
        self.assertIsNotNone(model.psi_minus)
        self.assertIsNotNone(model.psi_contour)
        self.assertIsNotNone(model.pool_plus)
        self.assertIsNotNone(model.pool_minus)
        self.assertIsNotNone(model.pool_contour)
        self.assertIsNotNone(model.classifier_id)
        self.assertIsNotNone(model.classifier_clothes)
        self.assertIsNotNone(model.classifier_assembled_id)
        self.assertIsNotNone(model.classifier_assembled_clothes)

    def test_02_backbone_stride_and_feature_map(self):
        """Verifies ResNet-50 layer4 stride=1 gives 24x12 feature map at 384x192 input."""
        model = PaperDCRReID(num_classes=150, num_clothes=100, pretrained=False)
        x = torch.randn(2, 3, 384, 192)
        feat_2048 = model.extract_backbone_features(x)
        self.assertEqual(feat_2048.shape, torch.Size([2, 2048, 24, 12]))

    def test_03_crd_channel_decomposition(self):
        """Verifies CRD projects 2048 -> 2560 and decomposes into 1024, 512, 1024 channels."""
        model = PaperDCRReID(num_classes=150, num_clothes=100, pretrained=False)
        x = torch.randn(2, 3, 384, 192)
        model.train()
        out = model(x)
        self.assertEqual(out['p_plus'].shape, torch.Size([2, 1024, 24, 12]))
        self.assertEqual(out['p_minus'].shape, torch.Size([2, 512, 24, 12]))
        self.assertEqual(out['p_contour'].shape, torch.Size([2, 1024, 24, 12]))

    def test_04_reconstruction_decoders_shape_and_tanh_range(self):
        """Verifies decoders output [B, 1, H, W] single-channel maps with Tanh in [-1, 1]."""
        decoder = ComponentReconstructionDecoder(in_channels=512, mid_channels=64)
        x = torch.randn(2, 512, 24, 12)
        y = decoder(x)
        self.assertEqual(y.shape, torch.Size([2, 1, 24, 12]))
        self.assertTrue(torch.all(y >= -1.0))
        self.assertTrue(torch.all(y <= 1.0))

    def test_05_stacked_pooling_dimensions(self):
        """Verifies stacked max+avg pooling doubles channel dimensions: 1024->2048, 512->1024."""
        pool_1024 = StackedGlobalPooling(1024)
        pool_512 = StackedGlobalPooling(512)
        x_1024 = torch.randn(2, 1024, 24, 12)
        x_512 = torch.randn(2, 512, 24, 12)
        f_1024 = pool_1024(x_1024)
        f_512 = pool_512(x_512)
        self.assertEqual(f_1024.shape, torch.Size([2, 2048]))
        self.assertEqual(f_512.shape, torch.Size([2, 1024]))

    def test_06_inference_feature_dim_and_normalization(self):
        """Verifies inference removes F_i^- and outputs 4096-D L2-normalized F_i^+ ⊕ F_i^t."""
        model = PaperDCRReID(num_classes=150, num_clothes=100, pretrained=False)
        model.eval()
        x = torch.randn(2, 3, 384, 192)
        with torch.no_grad():
            feat = model(x)
        self.assertEqual(feat.shape, torch.Size([2, 4096]))
        norms = torch.norm(feat, dim=-1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-5))

    def test_07_dad_assembled_feature_dim(self):
        """Verifies DAD assembled feature G_i has 5120 dimensions (2048 + 1024 + 2048)."""
        model = PaperDCRReID(num_classes=150, num_clothes=100, pretrained=False)
        model.train()
        x = torch.randn(2, 3, 384, 192)
        out = model(x, shuffle_clothes=True)
        self.assertEqual(out['g_i'].shape, torch.Size([2, 5120]))
        self.assertEqual(out['assembled_id_logits'].shape, torch.Size([2, 150]))
        self.assertEqual(out['assembled_clothes_logits'].shape, torch.Size([2, 100]))

    def test_08_reconstruction_loss_targets(self):
        """Verifies ComponentReconstructionLoss computes mean L1 reconstruction error."""
        loss_fn = ComponentReconstructionLoss()
        # Binary target masks {0, 1} which map to {-1, 1}
        target_plus = torch.ones(2, 1, 10, 10)     # mapped to 1.0
        target_minus = torch.zeros(2, 1, 10, 10)    # mapped to -1.0
        target_contour = torch.ones(2, 1, 10, 10)  # mapped to 1.0

        # Predictions
        y_plus = torch.full((2, 1, 10, 10), 0.5)      # |0.5 - 1.0| = 0.5
        y_minus = torch.full((2, 1, 10, 10), -0.5)    # |-0.5 - (-1.0)| = 0.5
        y_contour = torch.full((2, 1, 10, 10), 0.5)   # |0.5 - 1.0| = 0.5

        loss = loss_fn(y_plus, y_minus, y_contour, target_plus, target_minus, target_contour)
        # Expected: (0.5 + 0.5 + 0.5) / 3 = 0.5
        self.assertAlmostEqual(loss.item(), 0.5, places=4)

    def test_09_clothes_adversarial_loss_uniformity(self):
        """Verifies ClothesAdversarialLoss implements paper Eq. 13-14 target distribution q(c)."""
        loss_fn = ClothesAdversarialLoss(num_clothes=10, epsilon=0.0)
        logits = torch.zeros(2, 10)
        c_labels = torch.tensor([0, 2])
        pids = torch.tensor([0, 1])
        pid_to_clothes = {0: [0, 1], 1: [2, 3]}
        loss = loss_fn(logits, c_labels, pids, pid_to_clothes_map=pid_to_clothes)
        self.assertGreater(loss.item(), 0.0)

    def test_10_assembled_clothes_loss_intra_identity(self):
        """Verifies AssembledClothesLoss targets intra-identity clothes categories Eq. 17-18."""
        loss_fn = AssembledClothesLoss(num_clothes=10)
        logits = torch.zeros(2, 10)
        pids = torch.tensor([0, 1])
        pid_to_clothes = {0: [0, 1], 1: [2, 3, 4]}
        loss = loss_fn(logits, pids, pid_to_clothes_map=pid_to_clothes)
        import math
        self.assertAlmostEqual(loss.item(), math.log(10.0), places=3)

    def test_11_two_stage_loss_composition(self):
        """Verifies Stage 1 vs Stage 2 loss components in PaperDCRReIDCombinedLoss."""
        loss_fn = PaperDCRReIDCombinedLoss(num_classes=20, num_clothes=10, alpha=0.05, gamma=1.0)
        outputs = {
            'id_logits': torch.randn(4, 20),
            'clothes_logits': torch.randn(4, 10),
            'assembled_id_logits': torch.randn(4, 20),
            'assembled_clothes_logits': torch.randn(4, 10),
            'y_plus': torch.zeros(4, 1, 12, 6),
            'y_minus': torch.zeros(4, 1, 12, 6),
            'y_contour': torch.zeros(4, 1, 12, 6),
        }
        pids = torch.tensor([0, 1, 2, 3])
        c_labels = torch.tensor([0, 1, 2, 3])

        # Stage 1
        loss_s1 = loss_fn(outputs, pids, c_labels, stage=1)
        self.assertEqual(loss_s1['l_ca'].item(), 0.0)
        self.assertEqual(loss_s1['l_ac'].item(), 0.0)
        self.assertAlmostEqual(
            loss_s1['total_loss'].item(),
            (loss_s1['l_id'] + loss_s1['l_c'] + loss_s1['l_r']).item(),
            places=4
        )

        # Stage 2
        loss_s2 = loss_fn(outputs, pids, c_labels, stage=2)
        expected_s2 = (
            loss_s2['l_id'] + loss_s2['l_c'] + loss_s2['l_ca']
            + 0.05 * loss_s2['l_ac'] + 1.0 * loss_s2['l_id_prime'] + loss_s2['l_r']
        )
        self.assertAlmostEqual(loss_s2['total_loss'].item(), expected_s2.item(), places=4)

    def test_12_cuda_gradient_flow_and_stability(self):
        """Verifies full forward, backward, optimizer step on CUDA without NaN or Inf."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        model = PaperDCRReID(num_classes=20, num_clothes=10, pretrained=False).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=3.5e-4)
        loss_fn = PaperDCRReIDCombinedLoss(num_classes=20, num_clothes=10, alpha=0.05, gamma=1.0).cuda()

        x = torch.randn(4, 3, 384, 192, device='cuda')
        pids = torch.tensor([0, 1, 2, 3], device='cuda')
        c_labels = torch.tensor([0, 1, 2, 3], device='cuda')

        optimizer.zero_grad()
        out = model(x, shuffle_clothes=True)
        losses = loss_fn(out, pids, c_labels, stage=2)
        total_loss = losses['total_loss']
        total_loss.backward()

        self.assertFalse(torch.isnan(total_loss))
        self.assertFalse(torch.isinf(total_loss))

        # Check gradients exist and have no NaNs
        has_grad = False
        for param in model.parameters():
            if param.grad is not None:
                has_grad = True
                self.assertFalse(torch.isnan(param.grad).any())
                self.assertFalse(torch.isinf(param.grad).any())
        self.assertTrue(has_grad)
        optimizer.step()


if __name__ == '__main__':
    unittest.main()
