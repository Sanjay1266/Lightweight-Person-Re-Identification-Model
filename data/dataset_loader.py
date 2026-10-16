import os
import re
import glob
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
                camid -= 1  # 0-indexed
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


def get_dataloaders(data_dir, dataset_type='both_small', batch_size=32, num_workers=0):
    dataset = PersonReIDDataset(data_dir=data_dir, dataset_type=dataset_type)

    transform_train = build_transforms(is_train=True)
    transform_test = build_transforms(is_train=False)

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
