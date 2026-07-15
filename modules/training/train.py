import json
from copy import deepcopy

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from modules.training.evaluate import compute_metrics
from modules.utils.config import RESULTS


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self._register(model)

    def _register(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().to(param.device)

    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone().to(param.device)

    def apply_shadow(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone().to(param.device)
                param.data = self.shadow[name].to(param.device)

    def restore(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name].to(param.device)
        self.backup = {}


def fit(
    model,
    train_loader,
    val_loader,
    name="model",
    encoder_name=None,
    accumulation_steps=1,
    epochs_head=10,
    epochs_finetune=20,
    lr_head=1e-3,
    lr_finetune=1e-4,
    patience=10,
    class_weights=None,
    criterion=None,
    device="cuda",
    use_ema=True,
    ema_decay=0.999,
):
    model = model.to(device)
    criterion = criterion or nn.CrossEntropyLoss(
        weight=class_weights.to(device) if class_weights is not None else None
    )

    best_val_f1 = 0.0
    best_epoch = 0
    best_state = None
    best_ema_state = None
    ema_helper = None
    history = {"train_loss": [], "val_f1": []}

    def run_epoch(loader, phase, optimizer=None, scaler=None, ema=None):
        is_train = phase == "train"
        model.train() if is_train else model.eval()
        total_loss = 0.0
        all_preds, all_labels = [], []
        stream = tqdm(loader, desc=f"{name} {phase}", leave=False)

        for i, batch in enumerate(stream):
            inputs, targets = batch
            if isinstance(targets, tuple):
                targets_a, targets_b, lam = targets
            else:
                targets_a = targets_b = targets
                lam = 1.0
            inputs, targets_a, targets_b = inputs.to(device), targets_a.to(device), targets_b.to(device)

            if is_train:
                with autocast(device_type=device):
                    outputs = model(inputs)
                    loss = (lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)) / accumulation_steps

                scaler.scale(loss).backward()

                if (i + 1) % accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    if ema is not None:
                        ema.update(model)

                total_loss += loss.item() * accumulation_steps
            else:
                with torch.no_grad(), autocast(device_type=device):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets_a)
                total_loss += loss.item()

            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(targets_a.cpu().numpy().tolist())

            if is_train:
                stream.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(loader)
        f1_macro, f1_per_class, prec, rec = compute_metrics(all_labels, all_preds)
        return avg_loss, f1_macro, all_preds, all_labels

    # Phase 1
    print(f"\n=== {name}: Phase 1 — Head Only ===")
    model.freeze_encoder()

    optimizer = AdamW(model.classifier.parameters(), lr=lr_head, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs_head)
    scaler = GradScaler(device=device)
    if use_ema:
        ema_helper = EMA(model, decay=ema_decay)

    epochs_no_improve = 0
    for epoch in range(epochs_head):
        train_loss, _, _, _ = run_epoch(train_loader, "train", optimizer, scaler, ema_helper)

        if use_ema and ema_helper is not None:
            ema_helper.apply_shadow(model)
        val_loss, val_f1, _, _ = run_epoch(val_loader, "val")
        if use_ema and ema_helper is not None:
            ema_helper.restore(model)

        history["train_loss"].append(train_loss)
        history["val_f1"].append(val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            best_state = model.state_dict()
            if use_ema and ema_helper is not None:
                ema_helper.apply_shadow(model)
                best_ema_state = deepcopy(model.state_dict())
                ema_helper.restore(model)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        print(
            f"  E{epoch+1:02d}: train_loss={train_loss:.4f}  "
            f"val_f1={val_f1:.4f}  best={best_val_f1:.4f}"
        )
        scheduler.step()

        if epochs_no_improve >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    # Phase 2
    print(f"\n=== {name}: Phase 2 — Fine-tune All ===")
    model.unfreeze_encoder()
    if best_state is None:
        best_state = deepcopy(model.state_dict())
    model.load_state_dict(best_state)

    param_groups = [
        {"params": model.encoder.parameters(), "lr": lr_finetune},
        {"params": model.classifier.parameters(), "lr": lr_finetune * 10},
    ]
    optimizer = AdamW(param_groups, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs_finetune)
    scaler = GradScaler(device=device)
    if use_ema:
        ema_helper = EMA(model, decay=ema_decay)

    for epoch in range(epochs_finetune):
        train_loss, _, _, _ = run_epoch(train_loader, "train", optimizer, scaler, ema_helper)

        if use_ema and ema_helper is not None:
            ema_helper.apply_shadow(model)
        val_loss, val_f1, _, _ = run_epoch(val_loader, "val")
        if use_ema and ema_helper is not None:
            ema_helper.restore(model)

        history["train_loss"].append(train_loss)
        history["val_f1"].append(val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch + epochs_head + 1
            best_state = model.state_dict()
            if use_ema and ema_helper is not None:
                ema_helper.apply_shadow(model)
                best_ema_state = deepcopy(model.state_dict())
                ema_helper.restore(model)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        print(
            f"  E{epoch+1+epochs_head:02d}: train_loss={train_loss:.4f}  "
            f"val_f1={val_f1:.4f}  best={best_val_f1:.4f}"
        )
        scheduler.step()

        if epochs_no_improve >= patience:
            print(f"  Early stopping at epoch {epoch+1+epochs_head}")
            break

    # Final evaluation (best regular state)
    model.load_state_dict(best_state)
    _, _, all_preds, all_labels = run_epoch(val_loader, "val")
    f1_macro, f1_per_class, precision_per_class, recall_per_class = compute_metrics(
        all_labels, all_preds
    )

    # Final evaluation (best EMA state)
    ema_f1_macro = -1.0
    if use_ema and best_ema_state is not None:
        model.load_state_dict(best_ema_state)
        _, ema_f1_macro, _, _ = run_epoch(val_loader, "val")

    result = {
        "name": name,
        "encoder_name": encoder_name or name,
        "best_val_f1": best_val_f1,
        "best_epoch": best_epoch + 1,
        "ema_val_f1": ema_f1_macro,
        "f1_per_class": f1_per_class,
        "precision_per_class": precision_per_class,
        "recall_per_class": recall_per_class,
        "history": history,
    }

    torch.save(best_state, RESULTS / f"{name}.pt")
    if use_ema and best_ema_state is not None:
        torch.save(best_ema_state, RESULTS / f"{name}_ema.pt")
    with open(RESULTS / f"{name}.json", "w") as f:
        save_result = {k: v for k, v in result.items() if k != "history"}
        json.dump(save_result, f, indent=2)

    return result


def fit_progressive(
    model,
    progressive_loaders,
    name="model",
    encoder_name=None,
    accumulation_steps=1,
    epochs_head=10,
    epochs_finetune=20,
    lr_head=1e-3,
    lr_finetune=1e-4,
    patience=10,
    class_weights=None,
    criterion=None,
    device="cuda",
):
    model = model.to(device)
    best_val_f1 = 0.0
    best_state = None
    best_epoch = 0
    total_epochs = 0

    for stage_idx, (train_loader, val_loader) in enumerate(progressive_loaders):
        img_size = getattr(train_loader.dataset, "img_size", 224)
        print(f"\n=== Progressive Stage {stage_idx + 1}: {img_size}x{img_size} ===")

        result = fit(
            model,
            train_loader,
            val_loader,
            name=f"{name}_stage{stage_idx + 1}",
            encoder_name=encoder_name,
            accumulation_steps=accumulation_steps,
            epochs_head=epochs_head,
            epochs_finetune=epochs_finetune,
            lr_head=lr_head,
            lr_finetune=lr_finetune,
            patience=patience,
            class_weights=class_weights,
            criterion=criterion,
            device=device,
        )

        total_epochs += result["best_epoch"]

        if result["best_val_f1"] > best_val_f1:
            best_val_f1 = result["best_val_f1"]
            best_state = deepcopy(model.state_dict())
            best_epoch = total_epochs

    model.load_state_dict(best_state)
    _, _, all_preds, all_labels = _run_epoch(model, val_loader, criterion, device, "val")

    final_result = {
        "name": name,
        "encoder_name": encoder_name or name,
        "best_val_f1": best_val_f1,
        "best_epoch": best_epoch,
    }

    torch.save(best_state, RESULTS / f"{name}.pt")
    with open(RESULTS / f"{name}.json", "w") as f:
        save_result = {k: v for k, v in final_result.items()}
        json.dump(save_result, f, indent=2)

    return final_result


def _run_epoch(model, loader, criterion, device, phase="val"):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(targets.cpu().numpy().tolist())
    avg_loss = total_loss / len(loader)
    f1_macro, f1_per_class, prec, rec = compute_metrics(all_labels, all_preds)
    return avg_loss, f1_macro, all_preds, all_labels
