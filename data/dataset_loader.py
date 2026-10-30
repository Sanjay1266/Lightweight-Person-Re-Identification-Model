import os
import re
import glob
from collections import defaultdict
from PIL import Image
try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    import torchvision.transforms as T
except ImportError:
    pass

class PersonReIDDataset:
    """
    Parses Person Re-Identification dataset folders:
    - Market-1501 format: bounding_box_train, bounding_box_test, query (XXXX_cY_fZ.jpg)
    - PRCC format: rgb/train, rgb/test/A (query), rgb/test/B (same-clothes gallery), rgb/test/C (cloth-changing gallery)
    """
    def __init__(self, data_dir, dataset_type='both_small'):
        """
        data_dir: Path to root Data_set folder
        dataset_type: 'with_bag', 'without_bag', 'both_small', 'both_large', 'prcc', 'prcc_cc', 'prcc_sc', 'prcc_all'
        """
        self.data_dir = data_dir
        self.dataset_type = dataset_type
        self.is_prcc = dataset_type.lower().startswith('prcc')

        if self.is_prcc:
            self._load_prcc()
        else:
            self.root_dir = os.path.join(data_dir, dataset_type)
            self.train_dir = os.path.join(self.root_dir, 'bounding_box_train')
            self.test_dir = os.path.join(self.root_dir, 'bounding_box_test')
            self.query_dir = os.path.join(self.root_dir, 'query')

            self._check_before_run()

            self.train_samples = self._process_dir(self.train_dir, relabel=True)
            self.gallery_samples = self._process_dir(self.test_dir, relabel=False)
            self.query_samples = self._process_dir(self.query_dir, relabel=False)

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self._get_stats(self.train_samples)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self._get_stats(self.gallery_samples)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self._get_stats(self.query_samples)

    def _load_prcc(self):
        # Resolve PRCC base path
        if os.path.exists(os.path.join(self.data_dir, 'prcc', 'rgb')):
            self.root_dir = os.path.join(self.data_dir, 'prcc')
        elif os.path.exists(os.path.join(self.data_dir, 'rgb')):
            self.root_dir = self.data_dir
        else:
            self.root_dir = os.path.join(self.data_dir, self.dataset_type)

        rgb_dir = os.path.join(self.root_dir, 'rgb') if os.path.exists(os.path.join(self.root_dir, 'rgb')) else self.root_dir
        self.train_dir = os.path.join(rgb_dir, 'train')
        self.test_dir = os.path.join(rgb_dir, 'test')
        self.query_dir = os.path.join(self.test_dir, 'A')

        if not os.path.exists(self.train_dir):
            raise RuntimeError(f"PRCC train dir '{self.train_dir}' is not available")
        if not os.path.exists(self.test_dir):
            raise RuntimeError(f"PRCC test dir '{self.test_dir}' is not available")

        # 1. Train set (150 identities)
        self.train_samples = []
        pid_container = set()
        for pid_str in sorted(os.listdir(self.train_dir)):
            p_dir = os.path.join(self.train_dir, pid_str)
            if not os.path.isdir(p_dir):
                continue
            pid_container.add(int(pid_str))

        pid2label = {pid: label for label, pid in enumerate(sorted(pid_container))}
        for pid_str in sorted(os.listdir(self.train_dir)):
            p_dir = os.path.join(self.train_dir, pid_str)
            if not os.path.isdir(p_dir):
                continue
            pid = pid2label[int(pid_str)]
            for fname in sorted(os.listdir(p_dir)):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    camid = {'A': 0, 'B': 1, 'C': 2}.get(fname[0].upper(), 0)
                    self.train_samples.append((os.path.join(p_dir, fname), pid, camid))

        # 2. Query set (Camera A - 71 identities)
        self.query_samples = []
        cam_a_dir = os.path.join(self.test_dir, 'A')
        for pid_str in sorted(os.listdir(cam_a_dir)):
            p_dir = os.path.join(cam_a_dir, pid_str)
            if not os.path.isdir(p_dir):
                continue
            pid = int(pid_str)
            for fname in sorted(os.listdir(p_dir)):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.query_samples.append((os.path.join(p_dir, fname), pid, 0))

        # 3. Gallery set based on PRCC protocol:
        # prcc_cc (Cloth-Changing): Query Cam A vs Gallery Cam C (different clothes)
        # prcc_sc (Same-Clothes): Query Cam A vs Gallery Cam B (same clothes)
        # prcc_all: Query Cam A vs Gallery Cam B + C
        self.gallery_samples = []
        subset_name = self.dataset_type.lower()
        if subset_name in ['prcc', 'prcc_cc']:
            cams = [('C', 2)]
        elif subset_name == 'prcc_sc':
            cams = [('B', 1)]
        else:  # prcc_all
            cams = [('B', 1), ('C', 2)]

        for cam_letter, camid in cams:
            cam_dir = os.path.join(self.test_dir, cam_letter)
            if not os.path.exists(cam_dir):
                continue
            for pid_str in sorted(os.listdir(cam_dir)):
                p_dir = os.path.join(cam_dir, pid_str)
                if not os.path.isdir(p_dir):
                    continue
                pid = int(pid_str)
                for fname in sorted(os.listdir(p_dir)):
                    if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                        self.gallery_samples.append((os.path.join(p_dir, fname), pid, camid))

    def _check_before_run(self):
        if not os.path.exists(self.root_dir):
            raise RuntimeError(f"'{self.root_dir}' is not available")
        if not os.path.exists(self.train_dir):
            raise RuntimeError(f"'{self.train_dir}' is not available")
        if not os.path.exists(self.test_dir):
            raise RuntimeError(f"'{self.test_dir}' is not available")
        if not os.path.exists(self.query_dir):
            raise RuntimeError(f"'{self.query_dir}' is not available")

    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(os.path.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d+)')

        pid_container = set()
        for img_path in img_paths:
            fname = os.path.basename(img_path)
            res = pattern.search(fname)
            if res:
                pid, _ = map(int, res.groups())
                if pid == -1:
                    continue  # junk images
                pid_container.add(pid)

        pid2label = {pid: label for label, pid in enumerate(sorted(pid_container))}

        # Check minimum camera index in dataset
        min_cam = 100
        for img_path in img_paths:
            fname = os.path.basename(img_path)
            res = pattern.search(fname)
            if res:
                _, c_val = map(int, res.groups())
                min_cam = min(min_cam, c_val)

        data = []
        for img_path in img_paths:
            fname = os.path.basename(img_path)
            res = pattern.search(fname)
            if res:
                pid, camid = map(int, res.groups())
                if pid == -1:
                    continue  # junk images
                assert 0 <= pid <= 9999
                assert 0 <= camid <= 10
                if min_cam > 0:
                    camid -= 1  # 0-indexed if starting from 1
                if relabel:
                    pid = pid2label[pid]
                data.append((img_path, pid, camid))

        return data

    def _get_stats(self, samples):
        pids = set()
        cams = set()
        for _, pid, camid in samples:
            pids.add(pid)
            cams.add(camid)
        return len(pids), len(samples), len(cams)

    def print_dataset_summary(self):
        print("=> Dataset Summary Loaded:")
        print(f"   Root Path: {self.root_dir}")
        print(f"   Train   : {self.num_train_pids} PIDs | {self.num_train_imgs} Images | {self.num_train_cams} Cameras")
        print(f"   Gallery : {self.num_gallery_pids} PIDs | {self.num_gallery_imgs} Images | {self.num_gallery_cams} Cameras")
        print(f"   Query   : {self.num_query_pids} PIDs | {self.num_query_imgs} Images | {self.num_query_cams} Cameras")


class ImageDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid = self.dataset[index]
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, pid, camid, img_path


def build_transforms(height=256, width=128, is_train=True):
    """
    Builds data transforms tailored for Person Re-ID.
    Height=256, Width=128 is standard aspect ratio for person Re-ID images.
    """
    if is_train:
        transform = T.Compose([
            T.Resize((height, width)),
            T.RandomHorizontalFlip(p=0.5),
            T.Pad(10),
            T.RandomCrop((height, width)),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            T.RandomErasing(p=0.5, value=[0.485, 0.456, 0.406])
        ])
    else:
        transform = T.Compose([
            T.Resize((height, width)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    return transform


import copy
import numpy as np

class RandomIdentitySampler:
    """
    Random Identity (P x K) Sampler for Person Re-ID.
    Randomly samples P identities, and K instances per identity,
    forming a batch of P * K samples.
    Ensures that every selected identity has K samples in the batch.
    """
    def __init__(self, data_source, batch_size=16, num_instances=4, seed=None):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = batch_size // num_instances
        self.seed = seed
        self.rng = np.random.RandomState(seed) if seed is not None else np.random

        self.index_dic = defaultdict(list)
        for index, (_, pid, _) in enumerate(data_source):
            self.index_dic[pid].append(index)
        self.pids = sorted(list(self.index_dic.keys()))

        # Estimate epoch length
        self.length = 0
        for pid in self.pids:
            num = len(self.index_dic[pid])
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)
        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = self.rng.choice(idxs, size=self.num_instances, replace=True).tolist()
            else:
                self.rng.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = self.rng.choice(avai_pids, self.num_pids_per_batch, replace=False).tolist()
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        return iter(final_idxs)

    def __len__(self):
        return self.length


def generate_clothing_labels(pids, cams, is_prcc=True, num_clothes=300):
    """
    Generates validated clothing condition labels.
    For PRCC:
      - Camera A (camid=0): same-clothes condition -> clothing index 0
      - Camera B (camid=1): same-clothes condition -> clothing index 0
      - Camera C (camid=2): cloth-changing condition -> clothing index 1
      Formula: clothes_label = (pid * 2 + clothes_condition) % num_clothes
      where clothes_condition is 0 for Cams A/B, and 1 for Cam C.
      Explicit validation: cams not in {0, 1, 2} raises ValueError.
    For Market-1501:
      - Viewpoint/camera condition mapping: (cams % 2)
    """
    if is_prcc:
        # Validate camera IDs
        invalid_mask = (cams < 0) | (cams > 2)
        if invalid_mask.any():
            invalid_cams = cams[invalid_mask].tolist()
            raise ValueError(f"Invalid PRCC camera ID detected: {invalid_cams}. PRCC only supports cameras 0 (A), 1 (B), and 2 (C).")

        # Explicit mapping: 0 for Cam A (0) and B (1); 1 for Cam C (2)
        clothes_condition = (cams == 2).long()
        raw_labels = pids * 2 + clothes_condition
        if num_clothes is not None and num_clothes < 300:
            clothes_labels = raw_labels % num_clothes
        else:
            clothes_labels = raw_labels
    else:
        clothes_labels = (pids * 2 + (cams % 2)).long() % num_clothes

    return clothes_labels


def get_dataloaders(data_dir, dataset_type='both_small', batch_size=32, num_workers=0, use_pk_sampler=False, num_instances=4, height=256, width=128, img_size=None):
    if img_size is not None:
        height, width = img_size
    dataset = PersonReIDDataset(data_dir=data_dir, dataset_type=dataset_type)

    transform_train = build_transforms(height=height, width=width, is_train=True)
    transform_test = build_transforms(height=height, width=width, is_train=False)

    if use_pk_sampler:
        sampler = RandomIdentitySampler(dataset.train_samples, batch_size=batch_size, num_instances=num_instances)
        train_loader = DataLoader(
            ImageDataset(dataset.train_samples, transform=transform_train),
            batch_size=batch_size, sampler=sampler, num_workers=num_workers, drop_last=True
        )
    else:
        train_loader = DataLoader(
            ImageDataset(dataset.train_samples, transform=transform_train),
            batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True
        )

    query_loader = DataLoader(
        ImageDataset(dataset.query_samples, transform=transform_test),
        batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    gallery_loader = DataLoader(
        ImageDataset(dataset.gallery_samples, transform=transform_test),
        batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return dataset, train_loader, query_loader, gallery_loader
