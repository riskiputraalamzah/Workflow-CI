import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn

from mlflow.models.signature import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)


DATA_DIR = "breast_cancer_preprocessing"
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
ARTIFACT_DIR = "artifacts"


def load_data():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    X_train = train_df.drop(columns=["target"])
    y_train = train_df["target"]

    X_test = test_df.drop(columns=["target"])
    y_test = test_df["target"]

    return X_train, X_test, y_train, y_test


def save_artifacts(model, feature_names, y_test, y_pred, metrics):
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    cm = confusion_matrix(y_test, y_pred)
    display = ConfusionMatrixDisplay(confusion_matrix=cm)

    fig, ax = plt.subplots(figsize=(6, 5))
    display.plot(ax=ax)
    plt.title("Confusion Matrix - Workflow CI Model")
    plt.tight_layout()
    confusion_matrix_path = os.path.join(ARTIFACT_DIR, "confusion_matrix.png")
    plt.savefig(confusion_matrix_path)
    plt.close()

    report_path = os.path.join(ARTIFACT_DIR, "classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as file:
        file.write(classification_report(y_test, y_pred))

    feature_importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_
    }).sort_values(by="importance", ascending=False)

    feature_importance_path = os.path.join(ARTIFACT_DIR, "feature_importance.csv")
    feature_importance_df.to_csv(feature_importance_path, index=False)

    metrics_path = os.path.join(ARTIFACT_DIR, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    return [
        confusion_matrix_path,
        report_path,
        feature_importance_path,
        metrics_path,
    ]


def main():
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("SMSML_Workflow_CI_Riski_Alamzah")

    X_train, X_test, y_train, y_test = load_data()

    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=1,
        random_state=42
    )

    with mlflow.start_run(run_name="workflow_ci_random_forest") as run:
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="weighted"),
            "recall": recall_score(y_test, y_pred, average="weighted"),
            "f1_score": f1_score(y_test, y_pred, average="weighted"),
            "roc_auc": roc_auc_score(y_test, y_proba),
        }

        params = {
            "n_estimators": 150,
            "max_depth": "None",
            "min_samples_split": 5,
            "min_samples_leaf": 1,
            "random_state": 42,
        }

        mlflow.log_params(params)

        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        artifact_paths = save_artifacts(
            model=model,
            feature_names=X_train.columns,
            y_test=y_test,
            y_pred=y_pred,
            metrics=metrics,
        )

        for artifact_path in artifact_paths:
            mlflow.log_artifact(artifact_path, artifact_path="evaluation")

        signature = infer_signature(X_train, model.predict(X_train))
        input_example = X_train.head(5)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
        )

        run_id_path = os.path.join(ARTIFACT_DIR, "run_id.txt")
        with open(run_id_path, "w", encoding="utf-8") as file:
            file.write(run.info.run_id)

        mlflow.log_artifact(run_id_path, artifact_path="run_info")

        print("Workflow CI training selesai.")
        print(f"Run ID: {run.info.run_id}")
        print(f"Tracking URI: {tracking_uri}")
        print("Metrics:")
        for metric_name, metric_value in metrics.items():
            print(f"{metric_name}: {metric_value:.4f}")


if __name__ == "__main__":
    main()