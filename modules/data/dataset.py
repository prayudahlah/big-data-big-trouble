import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from modules.utils.config import IMG_SIZE, MEAN, STD, NUM_CLASSES, PROJECT_ROOT
from modules.utils.load_data import load_train

_CLASS_TO_IDX = {"0_Recyclable": 0, "1_Electronic": 1, "2_Organic": 2}


class TrashDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        self.class_weights = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(PROJECT_ROOT / row["path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = _CLASS_TO_IDX[row["label"]]
        return img, label


def get_transforms(augment: bool = True):
    resize = transforms.Resize((IMG_SIZE, IMG_SIZE))
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(MEAN, STD)

    if augment:
        return transforms.Compose([
            transforms.Resize((int(IMG_SIZE * 1.14), int(IMG_SIZE * 1.14))),
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            to_tensor,
            normalize,
        ])
    return transforms.Compose([resize, to_tensor, normalize])


def get_dataloaders(batch_size=32, num_workers=4, val_split=0.2):
    df = load_train()
    y = df["label"].map(_CLASS_TO_IDX).values

    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_split, random_state=42)
    train_idx, val_idx = next(sss.split(df, y))

    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_val = df.iloc[val_idx].reset_index(drop=True)

    train_ds = TrashDataset(df_train, transform=get_transforms(augment=True))
    val_ds = TrashDataset(df_val, transform=get_transforms(augment=False))

    counts = np.bincount([_CLASS_TO_IDX[l] for l in df_train["label"]], minlength=NUM_CLASSES)
    class_weights = torch.FloatTensor(counts.sum() / (NUM_CLASSES * counts))
    val_ds.class_weights = class_weights

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader, val_ds
