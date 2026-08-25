import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    cohen_kappa_score,
    matthews_corrcoef
)
import config
from dataset import (
    create_dataloaders,
    set_seed
)
from model import (
    create_model
)
def evaluate_model(
    model,
    test_loader
):
    model.eval()
    all_labels = []
    all_predictions = []
    all_probabilities = []
    with torch.no_grad():
        for images, labels in tqdm(
            test_loader,
            desc="Testing"
        ):
            images = images.to(
                config.DEVICE,
                non_blocking=True
            )
            outputs = model(
                images
            )
            probabilities = F.softmax(
                outputs,
                dim=1
            )
            predictions = torch.argmax(
                probabilities,
                dim=1
            )
            all_labels.extend(
                labels.numpy()
            )
            all_predictions.extend(
                predictions.cpu().numpy()
            )
            # Probability of CRACK.
            # crack = class 0
            all_probabilities.extend(
                probabilities[:, 0]
                .cpu()
                .numpy()
            )
    return (
        np.array(all_labels),
        np.array(all_predictions),
        np.array(all_probabilities)
    )
def calculate_specificity(
    y_true,
    y_pred
):
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )
    tn, fp, fn, tp = cm.ravel()
    specificity = (
        tn /
        (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )
    return specificity
def main():
    set_seed()
    print("\nLoading dataset...")
    (
        train_loader,
        val_loader,
        test_loader,
        train_dataset,
        val_dataset,
        test_dataset
    ) = create_dataloaders()
    print("\nLoading model...")
    model = create_model()
    checkpoint = torch.load(
        config.BEST_MODEL_PATH,
        map_location=config.DEVICE,
        weights_only=False
    )
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
    model.eval()
    print(
        "Best validation F1:",
        checkpoint.get(
            "best_val_f1",
            "N/A"
        )
    )
    print("\nGenerating predictions...")
    (
        y_true,
        y_pred,
        y_prob
    ) = evaluate_model(
        model,
        test_loader
    )
    accuracy = accuracy_score(
        y_true,
        y_pred
    )
    precision = precision_score(
        y_true,
        y_pred,
        pos_label=config.POSITIVE_CLASS_INDEX,
        zero_division=0
    )
    recall = recall_score(
        y_true,
        y_pred,
        pos_label=config.POSITIVE_CLASS_INDEX,
        zero_division=0
    )
    f1 = f1_score(
        y_true,
        y_pred,
        pos_label=config.POSITIVE_CLASS_INDEX,
        zero_division=0
    )
    specificity = calculate_specificity(
        y_true,
        y_pred
    )
    auc_score = roc_auc_score(
        y_true,
        y_prob
    )
    kappa = cohen_kappa_score(
        y_true,
        y_pred
    )
    mcc = matthews_corrcoef(
        y_true,
        y_pred
    )
    print("\n")
    print("=" * 70)
    print("TEST RESULTS")
    print("=" * 70)
    print(
        f"Accuracy       : "
        f"{accuracy * 100:.2f}%"
    )
    print(
        f"Precision      : "
        f"{precision:.4f}"
    )
    print(
        f"Recall         : "
        f"{recall:.4f}"
    )
    print(
        f"F1 Score       : "
        f"{f1:.4f}"
    )
    print(
        f"Specificity    : "
        f"{specificity:.4f}"
    )
    print(
        f"ROC-AUC        : "
        f"{auc_score:.4f}"
    )
    print(
        f"Cohen Kappa    : "
        f"{kappa:.4f}"
    )
    print(
        f"MCC            : "
        f"{mcc:.4f}"
    )
    print("=" * 70)
    metrics = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1,
        "Specificity": specificity,
        "ROC_AUC": auc_score,
        "Cohen_Kappa": kappa,
        "MCC": mcc
    }
    metrics_df = pd.DataFrame(
        [metrics]
    )
    metrics_df.to_csv(
        config.METRICS_FILE,
        index=False
    )
    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=config.CLASS_NAMES,
        output_dict=True,
        zero_division=0
    )
    report_df = pd.DataFrame(
        report
    ).transpose()
    report_df.to_csv(
        config.CLASSIFICATION_REPORT_FILE
    )
    prediction_df = pd.DataFrame({
        "true_label": y_true,
        "predicted_label": y_pred,
        "true_class": [
            config.CLASS_NAMES[x]
            for x in y_true
        ],
        "predicted_class": [
            config.CLASS_NAMES[x]
            for x in y_pred
        ],
        "crack_probability": y_prob,
        "no_crack_probability": 1.0 - y_prob
    })
    prediction_df.to_csv(
        config.PREDICTIONS_FILE,
        index=False
    )
    print(
        "\nEvaluation files saved successfully."
    )
    print(
        f"Metrics: "
        f"{config.METRICS_FILE}"
    )
    print(
        f"Report: "
        f"{config.CLASSIFICATION_REPORT_FILE}"
    )
    print(
        f"Predictions: "
        f"{config.PREDICTIONS_FILE}"
    )