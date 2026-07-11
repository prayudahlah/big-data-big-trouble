import torch
import numpy as np


class MixUpCollator:
    def __init__(self, alpha=0.2, num_classes=3):
        self.alpha = alpha
        self.num_classes = num_classes

    def __call__(self, batch):
        images, labels = zip(*batch)
        images = torch.stack(images, dim=0)
        labels = torch.tensor(labels, dtype=torch.long)

        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
            batch_size = images.size(0)
            index = torch.randperm(batch_size)
            mixed_images = lam * images + (1 - lam) * images[index]
            return mixed_images, (labels, labels[index], lam)

        return images, (labels, labels, 1.0)
