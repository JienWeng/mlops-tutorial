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

## Going further

- Add `mlflow.sklearn.autolog()` before `fit()` to capture params/metrics/model in one line.
- Commit your changes to git so each run is tied to a code version.
- Register your best model in the MLflow Model Registry.
- Automate the run in CI, then serve and monitor the model.

## License & attribution

Licensed under [CC BY-SA 4.0](LICENSE). This is an adaptation of *"An introduction to MLOps"* by [Alexandre Boucaud](https://aboucaud.github.io/slides/2023/lsst-france-mlops) (CC BY-SA 4.0); released under the same license. Adapted by [Jien Weng](https://jienweng.github.io).
