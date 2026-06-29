"""
Intro to MLOps — hands-on: track your first experiment with MLflow.

Trains a logistic-regression classifier on the breast-cancer dataset and logs
the parameters, metrics, the trained model, and a confusion-matrix plot to a
local MLflow store. Run it twice with different `C` values, then compare the
runs in the MLflow UI:

    python train.py
    # edit C below to e.g. 0.01, then:
    python train.py
    mlflow ui --backend-store-uri sqlite:///mlflow.db
    # open http://127.0.0.1:5000

Companion tutorial: https://jienweng.github.io/notes/intro-to-mlops-tutorial/
Slides:            https://jienweng.github.io/slides/2026/intro-to-mlops/
"""

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score
from sklearn.model_selection import train_test_split

# Where to store runs: a local SQLite db (metadata) + ./mlruns (artifacts).
# SQLite is the friction-free local backend recommended on MLflow 3.x.
mlflow.set_tracking_uri("sqlite:///mlflow.db")
# Group related runs under a named experiment.
mlflow.set_experiment("intro-to-mlops")

# --- data ---
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# A hyperparameter we want to track and vary between runs.
C = 1.0  # inverse regularisation strength

with mlflow.start_run(run_name=f"logreg-C={C}"):
    # --- train ---
    model = LogisticRegression(C=C, max_iter=10_000)
    model.fit(X_train, y_train)

    # --- evaluate ---
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    # --- log: parameters, metrics, the model itself ---
    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_param("C", C)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1", f1)
    mlflow.sklearn.log_model(model, name="model")

    # --- log: a plot as an artifact ---
    ConfusionMatrixDisplay.from_predictions(y_test, preds)
    plt.title(f"Confusion matrix (C={C})")
    plt.savefig("confusion_matrix.png", bbox_inches="tight")
    mlflow.log_artifact("confusion_matrix.png")

    print(f"Logged run: accuracy={acc:.4f}, f1={f1:.4f}")
