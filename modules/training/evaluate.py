from sklearn.metrics import f1_score, precision_score, recall_score


def compute_metrics(y_true, y_pred, num_classes=3):
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_per_class = f1_score(
        y_true, y_pred, average=None, labels=list(range(num_classes)), zero_division=0
    ).tolist()
    precision_per_class = precision_score(
        y_true, y_pred, average=None, labels=list(range(num_classes)), zero_division=0
    ).tolist()
    recall_per_class = recall_score(
        y_true, y_pred, average=None, labels=list(range(num_classes)), zero_division=0
    ).tolist()
    return f1_macro, f1_per_class, precision_per_class, recall_per_class
