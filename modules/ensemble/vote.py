import json

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

from modules.models.factory import TrashClassifier
from modules.utils.config import NUM_CLASSES, RESULTS


def load_top_models(results_dir=RESULTS, num_models=5, device="cuda"):
    json_files = sorted(results_dir.glob("*.json"))
    records = []
    for f in json_files:
        with open(f) as fh:
            records.append(json.load(fh))

    records.sort(key=lambda r: r["best_val_f1"], reverse=True)
    records = records[:num_models]

    models = []
    for rec in records:
        m = TrashClassifier(rec["encoder_name"], num_classes=NUM_CLASSES).to(device)
        pt_path = results_dir / f"{rec['name']}.pt"
        m.load_state_dict(torch.load(pt_path, map_location=device))
        m.eval()
        models.append(m)

    return models, records


def soft_voting(results_dir=RESULTS, num_models=5, device="cuda"):
    return load_top_models(results_dir, num_models, device)


@torch.inference_mode()
def predict_ensemble(models, test_loader, device="cuda"):
    return predict_weighted_ensemble(models, test_loader, weights=None, device=device)


@torch.inference_mode()
def predict_weighted_ensemble(models, test_loader, weights=None, device="cuda"):
    all_probs = []
    for inputs, _ in tqdm(test_loader, desc="Ensemble inference"):
        inputs = inputs.to(device)
        logits = torch.stack([m(inputs) for m in models])
        probs = F.softmax(logits, dim=-1)
        if weights is not None:
            w = torch.tensor(weights, device=device).view(-1, 1, 1)
            probs = (probs * w).sum(dim=0) / w.sum()
        else:
            probs = probs.mean(dim=0)
        all_probs.append(probs.cpu())
    return torch.cat(all_probs)


@torch.inference_mode()
def predict_stacking(models, test_loader, val_loader, device="cuda"):
    def extract_features(loader):
        all_feats, all_labels = [], []
        for inputs, targets in tqdm(loader, desc="Extracting features"):
            inputs = inputs.to(device)
            logits = torch.stack([m(inputs) for m in models])
            probs = F.softmax(logits, dim=-1)
            flat = probs.permute(1, 0, 2).reshape(len(inputs), -1)
            all_feats.append(flat.cpu().numpy())
            all_labels.append(targets.cpu().numpy())
        return np.concatenate(all_feats), np.concatenate(all_labels)

    X_val, y_val = extract_features(val_loader)
    meta = LogisticRegression(multi_class="multinomial", max_iter=1000)
    meta.fit(X_val, y_val)

    X_test, _ = extract_features(test_loader)
    preds = meta.predict(X_test)

    probs = torch.from_numpy(meta.predict_proba(X_test))
    return probs, meta


def generate_submission(preds_or_probs, output_path="submission.csv", as_ints=False):
    if isinstance(preds_or_probs, (np.ndarray, torch.Tensor)):
        if preds_or_probs.ndim == 2:
            preds = preds_or_probs.argmax(axis=1)
        else:
            preds = preds_or_probs
    else:
        preds = preds_or_probs
    if isinstance(preds, torch.Tensor):
        preds = preds.tolist()
    elif isinstance(preds, np.ndarray):
        preds = preds.tolist()
    preds = [int(p) for p in preds]

    if as_ints:
        df = pd.DataFrame({"id": range(1, len(preds) + 1), "predicted": preds})
    else:
        from modules.utils.config import CLASS_LABELS
        label_names = [CLASS_LABELS[p] for p in preds]
        df = pd.DataFrame({"id": range(1, len(preds) + 1), "predicted": label_names})
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
    return df


def tune_thresholds(val_probs, val_labels, n_classes=3, step=0.05):
    from sklearn.metrics import f1_score

    best_thresh = np.ones(n_classes) * 0.5
    best_f1 = 0.0
    grid = np.arange(step, 1.0, step)
    for c in range(n_classes):
        for t in grid:
            thresh = best_thresh.copy()
            thresh[c] = t
            adjusted = val_probs / thresh
            preds = adjusted.argmax(axis=1)
            f1 = f1_score(val_labels, preds, average="macro")
            if f1 > best_f1:
                best_f1 = f1
                best_thresh[c] = t
    return best_thresh


def apply_threshold(probs, thresholds):
    adjusted = probs / np.array(thresholds)
    return adjusted.argmax(axis=1)
