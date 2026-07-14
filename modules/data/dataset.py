from pathlib import Path

import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from modules.utils.config import CLASS_LABELS, IMG_SIZE, MEAN, NUM_CLASSES, PROJECT_ROOT, SEED, STD
from modules.utils.load_data import load_train

_LABEL_TO_IDX = {name: i for i, name in enumerate(CLASS_LABELS)}


class TrashDataset(Dataset):
    def __init__(self, df, transform=None):
        self.paths = df["path"].values
        self.labels = df["label"].map(_LABEL_TO_IDX).values
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(PROJECT_ROOT / self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


def _get_train_transform(img_size=IMG_SIZE):
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomGrayscale(p=0.1),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def _get_val_transform(img_size=IMG_SIZE):
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def _compute_class_weights(labels, num_classes=NUM_CLASSES):
    labels = torch.tensor(labels, dtype=torch.long)
    counts = torch.zeros(num_classes)
    for c in range(num_classes):
        counts[c] = (labels == c).sum()
    weights = 1.0 / counts.float()
    weights = weights / weights.sum() * num_classes
    return weights


def get_dataloaders(
    batch_size,
    num_workers=4,
    val_split=0.2,
    img_size=IMG_SIZE,
):
    df = load_train()

    train_df, val_df = train_test_split(
        df,
        test_size=val_split,
        stratify=df["label"],
        random_state=SEED,
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    train_ds = TrashDataset(train_df, transform=_get_train_transform(img_size))
    val_ds = TrashDataset(val_df, transform=_get_val_transform(img_size))

    train_labels = train_df["label"].map(_LABEL_TO_IDX).values
    val_ds.class_weights = _compute_class_weights(train_labels.copy())

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, val_ds


class TestDataset(Dataset):
    def __init__(self, df, transform=None):
        self.paths = df["path"].values
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(PROJECT_ROOT / self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, 0  # dummy label


def get_test_loader(
    batch_size,
    num_workers=4,
    img_size=IMG_SIZE,
):
    from modules.utils.load_data import load_test

    df = load_test()
    test_ds = TestDataset(df, transform=_get_val_transform(img_size))
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return test_loader


def _get_train_transform_v2(img_size=IMG_SIZE):
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.3, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 3.0)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
        transforms.RandomErasing(p=0.25),
    ])


def get_dataloaders_v2(
    batch_size,
    num_workers=4,
    val_split=0.2,
    img_size=IMG_SIZE,
    oversample=True,
):
    df = load_train()

    train_df, val_df = train_test_split(
        df,
        test_size=val_split,
        stratify=df["label"],
        random_state=SEED,
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    train_ds = TrashDataset(train_df, transform=_get_train_transform_v2(img_size))
    val_ds = TrashDataset(val_df, transform=_get_val_transform(img_size))

    train_labels = train_df["label"].map(_LABEL_TO_IDX).values
    val_ds.class_weights = _compute_class_weights(train_labels.copy())

    if oversample:
        class_counts = torch.bincount(torch.tensor(train_labels))
        weights = 1.0 / class_counts[train_labels].float()
        sampler = torch.utils.data.WeightedRandomSampler(
            weights, len(weights), replacement=True,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, val_ds


def _get_train_transform_v3(img_size=IMG_SIZE):
    from timm.data.auto_augment import augmix_ops

    aa_ops = augmix_ops(magnitude=9)

    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.3, 1.0)),
        transforms.RandomHorizontalFlip(),
        aa_ops,
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
        transforms.RandomErasing(p=0.3),
    ])


def get_dataloaders_v3(
    batch_size,
    num_workers=4,
    val_split=0.2,
    img_size=IMG_SIZE,
    oversample=True,
    use_mixup=False,
    mixup_alpha=0.2,
):
    from modules.utils.load_data import load_train

    df = load_train()

    train_df, val_df = train_test_split(
        df,
        test_size=val_split,
        stratify=df["label"],
        random_state=SEED,
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    train_ds = TrashDataset(train_df, transform=_get_train_transform_v3(img_size))
    val_ds = TrashDataset(val_df, transform=_get_val_transform(img_size))

    train_labels = train_df["label"].map(_LABEL_TO_IDX).values
    val_ds.class_weights = _compute_class_weights(train_labels.copy())

    if oversample:
        class_counts = torch.bincount(torch.tensor(train_labels))
        weights = 1.0 / class_counts[train_labels].float()
        sampler = torch.utils.data.WeightedRandomSampler(
            weights, len(weights), replacement=True,
        )
        shuffle = False
    else:
        sampler = None
        shuffle = True

    if use_mixup:
        from modules.training.collator import MixUpCollator
        collate_fn = MixUpCollator(alpha=mixup_alpha, num_classes=NUM_CLASSES)
    else:
        collate_fn = None

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False if sampler else shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, val_ds
