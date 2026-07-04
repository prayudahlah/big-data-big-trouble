import json

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from modules.training.evaluate import compute_metrics
from modules.utils.config import RESULTS


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
):
    model = model.to(device)
    criterion = criterion or nn.CrossEntropyLoss(
        weight=class_weights.to(device) if class_weights is not None else None
    )

    best_val_f1 = 0.0
    best_epoch = 0
    best_state = None
    history = {"train_loss": [], "val_f1": []}

    def run_epoch(loader, phase, optimizer=None, scaler=None):
        is_train = phase == "train"
        model.train() if is_train else model.eval()
        total_loss = 0.0
        all_preds, all_labels = [], []
        stream = tqdm(loader, desc=f"{name} {phase}", leave=False)

        for i, (inputs, targets) in enumerate(stream):
            inputs, targets = inputs.to(device), targets.to(device)

            if is_train:
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets) / accumulation_steps

                scaler.scale(loss).backward()

                if (i + 1) % accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                total_loss += loss.item() * accumulation_steps
            else:
                with torch.no_grad(), autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                total_loss += loss.item()

            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(targets.cpu().numpy().tolist())

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
    scaler = GradScaler()

    epochs_no_improve = 0
    for epoch in range(epochs_head):
        train_loss, _, _, _ = run_epoch(train_loader, "train", optimizer, scaler)
        val_loss, val_f1, _, _ = run_epoch(val_loader, "val")

        history["train_loss"].append(train_loss)
        history["val_f1"].append(val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            best_state = model.state_dict()
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
    model.load_state_dict(best_state)

    param_groups = [
        {"params": model.encoder.parameters(), "lr": lr_finetune},
        {"params": model.classifier.parameters(), "lr": lr_finetune * 10},
    ]
    optimizer = AdamW(param_groups, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs_finetune)
    scaler = GradScaler()

    for epoch in range(epochs_finetune):
        train_loss, _, _, _ = run_epoch(train_loader, "train", optimizer, scaler)
        val_loss, val_f1, _, _ = run_epoch(val_loader, "val")

        history["train_loss"].append(train_loss)
        history["val_f1"].append(val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch + epochs_head + 1
            best_state = model.state_dict()
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

    # Final evaluation
    model.load_state_dict(best_state)
    _, _, all_preds, all_labels = run_epoch(val_loader, "val")
    f1_macro, f1_per_class, precision_per_class, recall_per_class = compute_metrics(
        all_labels, all_preds
    )

    result = {
        "name": name,
        "encoder_name": encoder_name or name,
        "best_val_f1": best_val_f1,
        "best_epoch": best_epoch + 1,
        "f1_per_class": f1_per_class,
        "precision_per_class": precision_per_class,
        "recall_per_class": recall_per_class,
        "history": history,
    }

    torch.save(best_state, RESULTS / f"{name}.pt")
    with open(RESULTS / f"{name}.json", "w") as f:
        save_result = {k: v for k, v in result.items() if k != "history"}
        json.dump(save_result, f, indent=2)

    return result
