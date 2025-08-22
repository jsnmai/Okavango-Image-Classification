from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
import random
from typing import Tuple

# ===============================================================
# Class for Partitions (Train/Val/Test)
# ===============================================================
class DataPartition(Dataset):
    """
    Initialize the dataset with a DataFrame and transform pipeline.
    Args:
        df (pd.DataFrame): DataFrame with image paths and labels.
        label_columns (list): List of all label names.
        transform: Data augmentation pipeline.
    """
    def __init__(self, df, label_columns, transform=None):
        self.label_columns = label_columns
        self.transform = transform
        self.img_paths = df["image_path"].tolist() # List of image paths
        self.labels = df[label_columns].to_numpy(dtype=np.float32) # 2-D Array of of shape (N_samples, N_labels)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert("RGB") # Retrieve image
        if self.transform:                        # Apply transformations to image
            img = self.transform(img)
        label_vector = torch.from_numpy(self.labels[idx]) # Retrieve label vector for the given sample
        return img, label_vector

# ===============================================================
# Additional Image Augmentations
# ===============================================================
class MixupCutmixWrapper:
    """
    Wraps a dataset so that every time __getitem__ is called
    we MAY return a MixUp or CutMix composite of two images.
    Works for multi-label (labels are float32 in {0,1}).
    """
    def __init__(
        self,
        dataset,
        mixup_alpha: float = 0.4,
        cutmix_alpha: float = 1.0,
        prob_mixup: float = 0.3,
        prob_cutmix: float = 0.3,
    ):
        self.ds = dataset
        self.m_alpha = mixup_alpha
        self.c_alpha = cutmix_alpha
        self.pm = prob_mixup
        self.pc = prob_cutmix

    def __len__(self): return len(self.ds)

    def _rand_bbox(self, size: Tuple[int,int], lam: float):
        """Return bounding-box coords for CutMix."""
        H, W = size
        cut_rat = (1. - lam) ** 0.5
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        cx = random.randint(0, W)
        cy = random.randint(0, H)
        x1 = np.clip(cx - cut_w // 2, 0, W)
        y1 = np.clip(cy - cut_h // 2, 0, H)
        x2 = np.clip(cx + cut_w // 2, 0, W)
        y2 = np.clip(cy + cut_h // 2, 0, H)
        return x1, y1, x2, y2

    def __getitem__(self, idx):
        img1, y1 = self.ds[idx]
        r = random.random()

        # ---------- MixUp ---------- #
        if r < self.pm:
            lam = np.random.beta(self.m_alpha, self.m_alpha)
            j = random.randint(0, len(self.ds)-1)
            img2, y2 = self.ds[j]
            img = lam*img1 + (1-lam)*img2
            y = lam*y1 + (1-lam)*y2
            return img, y

        # ---------- CutMix ---------- #
        if r < self.pm + self.pc:
            lam = np.random.beta(self.c_alpha, self.c_alpha)
            j = random.randint(0, len(self.ds)-1)
            img2, y2 = self.ds[j]
            img1 = img1.clone() 
            _, H, W = img1.shape
            x1,y1b,x2,y2b = self._rand_bbox((H,W), lam)
            img1[:, y1b:y2b, x1:x2] = img2[:, y1b:y2b, x1:x2]
            lam = 1 - (x2-x1)*(y2b-y1b)/(H*W)   # adjusted lambda
            y = lam*y1 + (1-lam)*y2
            return img1, y

        return img1, y1
    
class RandomHorizontalRoll:
    def __init__(self, p=0.5): self.p = p
    def __call__(self, img: Image.Image):
        if random.random() < self.p:
            w, h = img.size
            dx = random.randint(0, w - 1)
            return img.transform(img.size, Image.AFFINE, (1, 0, -dx, 0, 1, 0))
        return img

