# Intro to MLOps — hands-on

Runnable companion code for the tutorial **["An Introduction to MLOps"](https://jienweng.github.io/notes/intro-to-mlops-tutorial/)** and its [slides](https://jienweng.github.io/slides/2026/intro-to-mlops/).

In about ten minutes you will track a real machine-learning experiment with [MLflow](https://mlflow.org), compare two runs, and inspect them in a web UI — the smallest useful version of an MLOps workflow.

## What it does

[`train.py`](train.py) trains a logistic-regression classifier on the scikit-learn breast-cancer dataset and logs, for each run:

- **parameters** (`model`, `C`)
- **metrics** (`accuracy`, `f1`)
- the **trained model** (in MLflow's model format)
- a **confusion-matrix plot** as an artifact

Runs are stored locally in a SQLite database (`mlflow.db`) plus an `mlruns/` folder — no server to set up.

## Setup

Requires Python 3.9+.

```shell
python -m venv .venv && source .venv/bin/activate   # optional but recommended
python -m pip install -r requirements.txt
```

## Run it

Two ways — pick one. Either runs the same hands-on.

**A. As a script**

```shell
# 1. First run (default C=1.0)
python train.py

# 2. Edit C in train.py to e.g. 0.01, then run again to get a second run
python train.py

# 3. Compare runs in the MLflow UI
mlflow ui --backend-store-uri sqlite:///mlflow.db
# open http://127.0.0.1:5000
```

**B. As a notebook**

```shell
jupyter notebook notebook.ipynb
```

Run all cells, change `C` (e.g. to `0.01`), run again, then start the UI as above.

## The hands-on

In the MLflow UI, open the `intro-to-mlops` experiment, tick both runs and click **Compare** to see parameters and metrics side by side, and open a run to view its confusion-matrix plot and download the model. That "which settings gave the best score, and exactly how?" comparison is the whole point.

## Serve it: MLflow → BentoML

[`service.py`](service.py) loads the most recent tracked run and wraps its model in a minimal [BentoML](https://www.bentoml.com) API — no run ID to copy by hand.

```shell
python -m pip install bentoml   # already in requirements.txt
bentoml serve service:MLopsService
# open http://127.0.0.1:3000 for interactive docs
```

Call it:

```shell
curl -X POST http://127.0.0.1:3000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [[14.0, 20.0, 90.0, 600.0, 0.1, 0.1, 0.1, 0.05, 0.2, 0.06, 0.4, 1.0, 3.0, 40.0, 0.006, 0.02, 0.03, 0.01, 0.02, 0.003, 16.0, 25.0, 105.0, 900.0, 0.14, 0.25, 0.3, 0.12, 0.3, 0.08]]}'
```

## Going further

- Add `mlflow.sklearn.autolog()` before `fit()` to capture params/metrics/model in one line.
- Commit your changes to git so each run is tied to a code version.
- Register your best model in the MLflow Model Registry, instead of always serving the latest run.
- Automate the run in CI, then monitor the served model's predictions.

## License & attribution

Licensed under [CC BY-SA 4.0](LICENSE). This is an adaptation of *"An introduction to MLOps"* by [Alexandre Boucaud](https://aboucaud.github.io/slides/2023/lsst-france-mlops) (CC BY-SA 4.0); released under the same license. Adapted by [Jien Weng](https://jienweng.github.io).
