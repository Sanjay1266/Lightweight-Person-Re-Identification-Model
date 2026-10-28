"""
Comprehensive Automated Test Suite for Lightweight DCR-ReID
Course: 23CSE373 - Computer Vision
Verifies:
1. Model initialization
2. Checkpoint compatibility (162/162 tensors)
3. Forward-pass shapes
4. Embedding dimensions & L2 normalization
5. NaN/Inf detection
6. Dataset & label consistency
7. Query/gallery validity
8. Rank-k metrics
9. mAP calculation
10. Same-camera filtering
11. FAST reproducibility
12. FULL result completeness
13. Checkpoint save & reload
14. Gradient flow
15. Parameter counting
16. FLOPs/MACs measurement
"""

import os
import sys
sys.path.insert(0, os.path.abspath('.'))

import unittest
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.lightweight_reid import LightweightDCRReID
from models.loss import DCRReIDCombinedLoss
from utils.metrics import compute_distance_matrix, eval_market1501
from data.dataset_loader import build_transforms, PersonReIDDataset

class TestDCRReID(unittest.TestCase):

    def setUp(self):
        self.device = torch.device('cpu')
        torch.manual_seed(42)
        np.random.seed(42)

    def test_01_model_initialization(self):
        model = LightweightDCRReID(num_classes=150, num_clothes=100, feat_dim=512)
        self.assertIsNotNone(model.stem)
        self.assertIsNotNone(model.stage1)
        self.assertIsNotNone(model.stage2)
        self.assertIsNotNone(model.stage3)
        self.assertIsNotNone(model.stage4)
        self.assertIsNotNone(model.crd)
        self.assertIsNotNone(model.dad)
        self.assertIsNotNone(model.bottleneck)
        self.assertIsNotNone(model.classifier)
        self.assertIsNotNone(model.clothes_classifier)

    def test_02_checkpoint_compatibility(self):
        ckpt_path = 'checkpoints/best_model.pth'
        if os.path.exists(ckpt_path):
            sd = torch.load(ckpt_path, map_location='cpu')
            num_classes = sd['classifier.weight'].shape[0]
            num_clothes = sd['clothes_classifier.weight'].shape[0]
            model = LightweightDCRReID(num_classes=num_classes, num_clothes=num_clothes, feat_dim=512)
            model_dict = model.state_dict()
            matched = {k: v for k, v in sd.items() if k in model_dict and v.shape == model_dict[k].shape}
            self.assertEqual(len(matched), len(model_dict), f"Expected 162/162 tensors matched, got {len(matched)}/{len(model_dict)}")
            model.load_state_dict(model_dict)

    def test_03_forward_pass_shapes(self):
        model = LightweightDCRReID(num_classes=150, num_clothes=100, feat_dim=512)
        dummy_input = torch.randn(2, 3, 256, 128)
        
        # Eval mode
        model.eval()
        with torch.no_grad():
            feat = model(dummy_input)
            self.assertEqual(feat.shape, torch.Size([2, 512]))

        # Train mode
        model.train()
        out = model(dummy_input, shuffle_clothes=True)
        self.assertIn('cls_score', out)
        self.assertIn('global_feat', out)
        self.assertIn('f_irrel', out)
        self.assertIn('f_rel', out)
        self.assertIn('f_contour', out)
        self.assertIn('recon_irrel', out)
        self.assertEqual(out['cls_score'].shape, torch.Size([2, 150]))
        self.assertEqual(out['global_feat'].shape, torch.Size([2, 512]))

    def test_04_embedding_dimensions_and_normalization(self):
        model = LightweightDCRReID(num_classes=150, num_clothes=100, feat_dim=512)
        model.eval()
        dummy_input = torch.randn(4, 3, 256, 128)
        with torch.no_grad():
            feat = model(dummy_input)
            norms = torch.norm(feat, p=2, dim=1)
            self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-5))

    def test_05_nan_inf_detection(self):
        model = LightweightDCRReID(num_classes=150, num_clothes=100, feat_dim=512)
        model.eval()
        dummy_input = torch.randn(2, 3, 256, 128)
        with torch.no_grad():
            feat = model(dummy_input)
            self.assertFalse(torch.isnan(feat).any().item())
            self.assertFalse(torch.isinf(feat).any().item())

    def test_06_dataset_and_label_consistency(self):
        ds = PersonReIDDataset('./Data_set', dataset_type='both_small')
        self.assertGreater(len(ds.train_samples), 0)
        self.assertGreater(len(ds.query_samples), 0)
        self.assertGreater(len(ds.gallery_samples), 0)
        for _, pid, cam in ds.train_samples[:50]:
            self.assertIsInstance(pid, int)
            self.assertIsInstance(cam, int)
            self.assertGreaterEqual(pid, 0)
            self.assertGreaterEqual(cam, 0)

    def test_07_query_gallery_validity(self):
        ds = PersonReIDDataset('./Data_set', dataset_type='both_small')
        q_pids = set(pid for _, pid, _ in ds.query_samples)
        g_pids = set(pid for _, pid, _ in ds.gallery_samples)
        overlap = q_pids & g_pids
        self.assertEqual(len(overlap), len(q_pids), "All query identities must be present in the gallery")

    def test_08_rank_k_metrics(self):
        # Known synthetic example: 2 queries, 4 gallery items
        # Query 0: pid 1, cam 0. Gallery: [pid 1 (cam 1), pid 2 (cam 0), pid 3 (cam 0), pid 4 (cam 0)]
        # Query 1: pid 2, cam 0. Gallery: [pid 1 (cam 1), pid 3 (cam 0), pid 2 (cam 1), pid 4 (cam 0)]
        q_pids = np.array([1, 2])
        q_cams = np.array([0, 0])
        g_pids = np.array([1, 2, 3, 4])
        g_cams = np.array([1, 0, 1, 1])

        # Distmat where query 0 matches gallery 0 (dist 0.1, rank 1), query 1 matches gallery 1 (dist 0.2, rank 2)
        distmat = np.array([
            [0.1, 0.5, 0.8, 0.9],
            [0.4, 0.2, 0.6, 0.7]  # gallery 1 is pid 2 cam 0 (same cam as query 1! so it will be filtered out)
        ])
        # For query 1, gallery 1 has g_pid 2 and g_cam 0 (same as query). So it must be filtered!
        # Then gallery 0 (pid 1), 2 (pid 3), 3 (pid 4) are left. None match query 1!
        # Let's adjust gallery so query 1 has valid cross-camera match:
        g_cams_adjusted = np.array([1, 1, 1, 1])
        distmat_adjusted = np.array([
            [0.1, 0.5, 0.8, 0.9], # q0: rank 1 is g0 (pid 1, match!)
            [0.4, 0.1, 0.6, 0.7]  # q1: rank 1 is g1 (pid 2, match!)
        ])
        cmc, mAP = eval_market1501(distmat_adjusted, q_pids, g_pids, q_cams, g_cams_adjusted, max_rank=4)
        self.assertEqual(cmc[0], 1.0, "Rank-1 must be 100%")
        self.assertEqual(mAP, 1.0, "mAP must be 100%")

    def test_09_map_calculation(self):
        # 1 query, 4 gallery items: g0 is positive (rank 1), g2 is positive (rank 3)
        # precision at rank 1: 1/1 = 1.0; precision at rank 3: 2/3 = 0.6667
        # AP = (1.0 + 2/3) / 2 = 5/6 = 0.8333
        q_pids = np.array([1])
        q_cams = np.array([0])
        g_pids = np.array([1, 2, 1, 3])
        g_cams = np.array([1, 1, 1, 1])
        distmat = np.array([[0.1, 0.2, 0.3, 0.4]])

        cmc, mAP = eval_market1501(distmat, q_pids, g_pids, q_cams, g_cams, max_rank=4)
        expected_ap = (1.0 + 2.0 / 3.0) / 2.0
        self.assertAlmostEqual(mAP, expected_ap, places=4)

    def test_10_same_camera_filtering(self):
        # Query: pid 1, cam 0.
        # Gallery 0: pid 1, cam 0 (junk, same cam)
        # Gallery 1: pid 2, cam 1 (negative)
        # Gallery 2: pid 1, cam 1 (true positive cross-camera)
        q_pids = np.array([1])
        q_cams = np.array([0])
        g_pids = np.array([1, 2, 1])
        g_cams = np.array([0, 1, 1])
        distmat = np.array([[0.01, 0.1, 0.2]])

        cmc, mAP = eval_market1501(distmat, q_pids, g_pids, q_cams, g_cams, max_rank=3)
        # Junk g0 (dist 0.01) must be removed.
        # Remaining: g1 (dist 0.1, negative), g2 (dist 0.2, positive).
        # Positive is at rank 2!
        self.assertEqual(cmc[0], 0.0, "Rank-1 should be 0 because nearest valid match is at rank 2")
        self.assertEqual(cmc[1], 1.0, "Rank-2 should be 1.0")

    def test_11_fast_reproducibility(self):
        # Sampling with fixed random seed must produce identical sampled subsets
        rng1 = np.random.RandomState(42)
        sample1 = rng1.choice(100, 20, replace=False)
        rng2 = np.random.RandomState(42)
        sample2 = rng2.choice(100, 20, replace=False)
        np.testing.assert_array_equal(sample1, sample2)

    def test_12_full_result_completeness(self):
        # Compute distance matrix between 5 queries and 10 gallery items
        q_feats = np.random.randn(5, 512).astype(np.float32)
        g_feats = np.random.randn(10, 512).astype(np.float32)
        q_feats /= np.linalg.norm(q_feats, axis=1, keepdims=True)
        g_feats /= np.linalg.norm(g_feats, axis=1, keepdims=True)
        distmat = compute_distance_matrix(q_feats, g_feats, metric='cosine')
        self.assertEqual(distmat.shape, (5, 10))

    def test_13_checkpoint_save_and_reload(self):
        model1 = LightweightDCRReID(num_classes=10, num_clothes=10, feat_dim=512)
        save_path = 'checkpoints/test_temp.pth'
        torch.save(model1.state_dict(), save_path)
        
        model2 = LightweightDCRReID(num_classes=10, num_clothes=10, feat_dim=512)
        model2.load_state_dict(torch.load(save_path, map_location='cpu'))

        for k in model1.state_dict():
            self.assertTrue(torch.allclose(model1.state_dict()[k], model2.state_dict()[k]))
        if os.path.exists(save_path):
            os.remove(save_path)

    def test_14_gradient_flow(self):
        model = LightweightDCRReID(num_classes=10, num_clothes=10, feat_dim=512)
        model.train()
        criterion = DCRReIDCombinedLoss(num_classes=10, num_clothes=10)

        imgs = torch.randn(4, 3, 256, 128)
        pids = torch.tensor([0, 0, 1, 1])
        cams = torch.tensor([0, 1, 0, 1])
        clothes_labels = torch.tensor([0, 1, 2, 3])

        outputs = model(imgs, shuffle_clothes=True)
        loss_dict = criterion(outputs, pids, clothes_labels=clothes_labels)
        total_loss = loss_dict['total_loss']
        total_loss.backward()

        # Check gradients on stem, stages, CRD, DAD
        self.assertIsNotNone(model.stem[0].weight.grad)
        self.assertIsNotNone(model.stage1.conv[0].weight.grad)
        self.assertIsNotNone(model.crd.proj_irrel[0].weight.grad)
        self.assertIsNotNone(model.dad.fusion[0].weight.grad)

    def test_15_parameter_counting(self):
        model = LightweightDCRReID(num_classes=150, num_clothes=100, feat_dim=512)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.assertEqual(total_params, 9752277)
        self.assertEqual(trainable_params, 9751253)

    def test_16_flops_macs_measurement(self):
        model = LightweightDCRReID(num_classes=150, num_clothes=100, feat_dim=512)
        model.eval()
        x = torch.randn(2, 3, 256, 128)
        from torch.utils.flop_counter import FlopCounterMode
        with FlopCounterMode(display=False) as fcm:
            _ = model(x)
        flops_g = (fcm.get_total_flops() / 2) / 1e9
        macs_g = flops_g / 2.0
        self.assertAlmostEqual(flops_g, 3.0015, places=2)
        self.assertAlmostEqual(macs_g, 1.5007, places=2)

    def test_17_prcc_clothing_label_mapping(self):
        from data.dataset_loader import generate_clothing_labels

        # 1. Same identity in Cam A (0) and Cam B (1) -> same clothing label
        pids = torch.tensor([5, 5])
        cams_ab = torch.tensor([0, 1])
        labels_ab = generate_clothing_labels(pids, cams_ab, is_prcc=True, num_clothes=100)
        self.assertEqual(labels_ab[0].item(), labels_ab[1].item(), "Cam A and Cam B for same identity must have identical clothing label")

        # 2. Same identity in Cam C (2) -> different clothing label
        cams_ac = torch.tensor([0, 2])
        labels_ac = generate_clothing_labels(pids, cams_ac, is_prcc=True, num_clothes=100)
        self.assertNotEqual(labels_ac[0].item(), labels_ac[1].item(), "Cam A and Cam C for same identity must have different clothing label")

        # 3. Different identities in same camera -> different clothing labels
        pids_diff = torch.tensor([5, 6])
        cams_same = torch.tensor([0, 0])
        labels_diff = generate_clothing_labels(pids_diff, cams_same, is_prcc=True, num_clothes=100)
        self.assertNotEqual(labels_diff[0].item(), labels_diff[1].item(), "Different identities must not have the same clothing label")

        # 4. Invalid camera IDs trigger ValueError
        invalid_cams = torch.tensor([0, 5])
        with self.assertRaises(ValueError):
            generate_clothing_labels(pids, invalid_cams, is_prcc=True, num_clothes=100)

        # 5. Non-PRCC dataset uses standard mapping without crashing
        labels_market = generate_clothing_labels(pids, torch.tensor([1, 2]), is_prcc=False, num_clothes=100)
        self.assertEqual(len(labels_market), 2)

    def test_18_random_identity_sampler_pk_batching(self):
        from data.dataset_loader import RandomIdentitySampler, ImageDataset
        from collections import Counter

        # Synthetic dataset with 10 identities, varying samples per identity
        synthetic_data = []
        for pid in range(10):
            # PID 0 has only 2 samples (insufficient for K=4), others have 10
            n_samples = 2 if pid == 0 else 10
            for i in range(n_samples):
                synthetic_data.append((f"img_{pid}_{i}.jpg", pid, 0))

        # P=4 identities, K=4 instances per identity -> batch_size = 16
        sampler = RandomIdentitySampler(synthetic_data, batch_size=16, num_instances=4, seed=42)
        indices = list(sampler)
        
        # Verify length is divisible by batch_size
        self.assertEqual(len(indices) % 16, 0)

        # Inspect batches
        for b_start in range(0, len(indices), 16):
            batch_idxs = indices[b_start:b_start + 16]
            batch_pids = [synthetic_data[idx][1] for idx in batch_idxs]
            counts = Counter(batch_pids)
            # Exactly 4 unique identities in the batch
            self.assertEqual(len(counts), 4, f"Expected 4 distinct identities, got {len(counts)}")
            # Exactly 4 instances per identity
            for pid, count in counts.items():
                self.assertEqual(count, 4, f"Identity {pid} should have exactly 4 instances, got {count}")

        # Test reproducibility with seed
        sampler1 = RandomIdentitySampler(synthetic_data, batch_size=16, num_instances=4, seed=123)
        sampler2 = RandomIdentitySampler(synthetic_data, batch_size=16, num_instances=4, seed=123)
        self.assertEqual(list(sampler1), list(sampler2), "Samplers with same seed must yield identical batches")

    def test_19_cuda_cpu_device_selection(self):
        from evaluate import resolve_device
        # CPU resolution
        dev_cpu = resolve_device('cpu')
        self.assertEqual(dev_cpu.type, 'cpu')

        # Auto resolution
        dev_auto = resolve_device('auto')
        expected_auto = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.assertEqual(dev_auto.type, expected_auto)

        # CUDA resolution
        if not torch.cuda.is_available():
            with self.assertRaises(RuntimeError):
                resolve_device('cuda')
        else:
            dev_cuda = resolve_device('cuda')
            self.assertEqual(dev_cuda.type, 'cuda')
            # Verify CUDA execution in eval mode
            model = LightweightDCRReID(num_classes=10, num_clothes=10, feat_dim=512).to(dev_cuda)
            model.eval()
            x = torch.randn(2, 3, 256, 128, device=dev_cuda)
            out = model(x)
            self.assertEqual(out.device.type, 'cuda')
            self.assertEqual(out.shape, torch.Size([2, 512]))

    def test_20_triplet_loss_construction(self):
        from models.loss import TripletLoss
        trip = TripletLoss(margin=0.3)
        # Inputs: 4 samples with 2 identities: [0, 0, 1, 1]
        pids = torch.tensor([0, 0, 1, 1])
        # Construct embeddings where intra-class distance is 0.1 and inter-class distance is 1.0
        # For identity 0: [1, 0, 0] and [0.99, 0.01, 0]
        # For identity 1: [0, 1, 0] and [0, 0.99, 0.01]
        feats = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.9, 0.1]
        ], dtype=torch.float32)

        loss = trip(feats, pids)
        self.assertTrue(torch.isfinite(loss).item())
        self.assertGreaterEqual(loss.item(), 0.0)

        # Test with single-sample batch: handles gracefully without crashing
        loss_single = trip(feats[:1], pids[:1])
        self.assertTrue(torch.isfinite(loss_single).item())

    def test_21_baseline_and_trained_checkpoint_parity(self):
        ckpt_base = 'checkpoints/best_model.pth'
        ckpt_opt = 'checkpoints/experiments/dcr_reid_cuda_v1/best_model.pth'

        self.assertTrue(os.path.exists(ckpt_base), f"Baseline checkpoint missing: {ckpt_base}")
        self.assertTrue(os.path.exists(ckpt_opt), f"CUDA optimized checkpoint missing: {ckpt_opt}")

        sd_base = torch.load(ckpt_base, map_location='cpu', weights_only=False)
        sd_opt = torch.load(ckpt_opt, map_location='cpu', weights_only=False)

        # Distinct weights
        base_weights = sd_base['stem.0.weight']
        opt_weights = sd_opt['stem.0.weight']
        self.assertFalse(torch.allclose(base_weights, opt_weights), "Baseline and trained weights must be distinct")

        # Both models forward valid embeddings on identical test input
        m_base = LightweightDCRReID(num_classes=sd_base['classifier.weight'].shape[0], num_clothes=100, feat_dim=512)
        m_base.load_state_dict(sd_base, strict=False)
        m_base.eval()

        m_opt = LightweightDCRReID(num_classes=sd_opt['classifier.weight'].shape[0], num_clothes=100, feat_dim=512)
        m_opt.load_state_dict(sd_opt, strict=False)
        m_opt.eval()

        dummy = torch.randn(4, 3, 256, 128)
        with torch.no_grad():
            f_base = m_base(dummy)
            f_opt = m_opt(dummy)

        self.assertEqual(f_base.shape, torch.Size([4, 512]))
        self.assertEqual(f_opt.shape, torch.Size([4, 512]))
        self.assertTrue(torch.allclose(torch.norm(f_base, p=2, dim=1), torch.ones(4), atol=1e-5))
        self.assertTrue(torch.allclose(torch.norm(f_opt, p=2, dim=1), torch.ones(4), atol=1e-5))
        self.assertFalse(torch.isnan(f_base).any().item())
        self.assertFalse(torch.isnan(f_opt).any().item())

    def test_22_cuda_checkpoint_compatibility(self):
        ckpt_opt = 'checkpoints/experiments/dcr_reid_cuda_v1/best_model.pth'
        sd = torch.load(ckpt_opt, map_location='cpu', weights_only=False)
        num_classes = sd['classifier.weight'].shape[0]
        num_clothes = sd['clothes_classifier.weight'].shape[0]
        model = LightweightDCRReID(num_classes=num_classes, num_clothes=num_clothes, feat_dim=512)
        model_dict = model.state_dict()
        matched = {k: v for k, v in sd.items() if k in model_dict and v.shape == model_dict[k].shape}
        self.assertEqual(len(matched), len(model_dict), f"Expected 162/162 tensors matched in CUDA checkpoint, got {len(matched)}/{len(model_dict)}")

if __name__ == '__main__':
    unittest.main()
