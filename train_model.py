"""Train or refresh the local model and print its holdout metrics."""

from model_backend import train_and_save


if __name__ == "__main__":
    _, metadata = train_and_save()
    metrics = metadata["metrics"]
    print(f"Model: {metadata['model_name']}")
    print(f"Holdout accuracy: {metrics['accuracy']:.2%}")
    print(f"Anomaly precision: {metrics['precision_anomaly']:.2%}")
    print(f"Anomaly recall: {metrics['recall_anomaly']:.2%}")
    print(f"Specificity: {metrics['specificity']:.2%}")
    print(f"F1 score: {metrics['f1_anomaly']:.2%}")
    print(f"ROC-AUC: {metrics['roc_auc']:.2%}")
    print(f"Average precision: {metrics['average_precision']:.2%}")
    print(f"Training rows: {metadata['training_rows']:,}")
    print("Report: txt.md")
