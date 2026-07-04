import json

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from modules.models.factory import TrashClassifier
from modules.utils.config import NUM_CLASSES, RESULTS


def soft_voting(results_dir=RESULTS, num_models=5, device="cuda"):
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


@torch.inference_mode()
def predict_ensemble(models, test_loader, device="cuda"):
    all_probs = []
    for inputs, _ in tqdm(test_loader, desc="Ensemble inference"):
        inputs = inputs.to(device)
        logits = torch.stack([m(inputs) for m in models])
        probs = F.softmax(logits, dim=-1).mean(dim=0)
        all_probs.append(probs.cpu())
    return torch.cat(all_probs)


def generate_submission(pred_probs, output_path="submission.csv"):
    preds = pred_probs.argmax(dim=1).tolist()
    from modules.utils.config import CLASS_LABELS

    label_names = [CLASS_LABELS[p] for p in preds]
    df = pd.DataFrame({"id": range(1, len(preds) + 1), "predicted": label_names})
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
    return df
