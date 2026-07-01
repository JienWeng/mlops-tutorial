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

**C. A second, simpler notebook — churn + autolog**

[`notebook2.ipynb`](notebook2.ipynb) predicts customer **churn** from a CSV
([`churn.csv`](churn.csv)) with a Random Forest, and uses one line —
`mlflow.sklearn.autolog()` — to capture parameters, metrics, and the model
automatically (no manual `log_param`/`log_metric`).

```shell
jupyter notebook notebook2.ipynb
```

Run all cells, change `n_estimators`/`max_depth`, run again, then compare the
runs in the `churn-prediction` experiment in the UI. It also includes an optional
`GridSearchCV` section — autolog is set to `max_tuning_runs=None` so it logs a
child run for **every** parameter combination (the default, `5`, only keeps the
top 5, which is fine for a bigger search but would hide most of this small grid).

**D. Manage models — the Model Registry**

[`notebook3.ipynb`](notebook3.ipynb) takes the best run from notebooks 1 and 2
and registers each in the **MLflow Model Registry** — the standard way to version
models and mark which one is in production. It shows the modern workflow: register
a version, tag it, point the `@champion` **alias** at it, load with
`models:/<name>@champion` (no run IDs in your code), then promote a `@challenger`
by moving the alias. Run notebooks 1 and 2 first, then run all cells.

```shell
jupyter notebook notebook3.ipynb
```

**E. Model decay & monitoring**

[`notebook4.ipynb`](notebook4.ipynb) takes the churn model from notebook 2 and
simulates 12 weeks of production traffic that **drifts** starting week 6
(customers skew newer, file more support tickets). Each week's accuracy is
logged to MLflow as a step-indexed metric — the monitoring dashboard is just
that curve — a threshold flags the decay, and the simplest real fix, retraining
on the recent drifted data, recovers most of the lost accuracy. Run notebook 2
first (for `churn.csv` — already included — and matching feature columns).

```shell
jupyter notebook notebook4.ipynb
```

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
- Register your best model in the MLflow Model Registry, instead of always serving the latest run — see [`notebook3.ipynb`](notebook3.ipynb).
- Automate the run in CI.
- Monitor the served model for decay and retrain when it drops — see [`notebook4.ipynb`](notebook4.ipynb).

## License & attribution

Licensed under [CC BY-SA 4.0](LICENSE). This is an adaptation of *"An introduction to MLOps"* by [Alexandre Boucaud](https://aboucaud.github.io/slides/2023/lsst-france-mlops) (CC BY-SA 4.0); released under the same license. Adapted by [Jien Weng](https://jienweng.github.io).
