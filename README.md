# Diagnosing and Fixing Operational Problems

## How to diagnose and fix problems in a production deployed code?

To understand a model's functionality and find its underlying problems, we need to take care of the following things:

* Periodic training
    * Figure out an optimal retraining strategy.
* Monitor model performance
* Analyse
    * Clear visibility of the model helps us and guides the model performance.

## Summary

This tutorial explores data drift monitoring using the Bike Sharing Dataset. We will:

* Use **evidently** and **mlflow** libraries.
* Calculate data drift for the model.
* Use **mlflow Tracking** for the training experiments indicating data drift.
* Explore the results using **mlflow UI**.

## Tutorial: Data Drift / mlflow

In this tutorial we will use the [Bike Sharing Dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip).

You can use the following code snippet (from `train.py`) to analyze this data before starting this tutorial.

```python
content = requests.get("https://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip").content

with zipfile.ZipFile(io.BytesIO(content)) as arc:
    raw_data = pd.read_csv(arc.open("day.csv"), header=0, sep=',', parse_dates=['dteday'], index_col='dteday')

# observe data structure
raw_data.tail()
```

## Run in the Udacity workspace

The workspace includes the code and required dependencies. It opens in
`/workspace/cd0583-diagnose-and-fix/`, the repository root. No cloning or
package installation is needed.

1. Start the workspace and select **Terminal > New Terminal** (under **More**
   if the Terminal menu is hidden).
2. Run the experiment script:

   ```bash
   python train.py
   ```

   This records six monthly drift runs in `mlflow.db` and saves four interactive
   charts (`fig1.html` through `fig4.html`) in `images/`. Download the HTML files
   and open them in your browser. PNG export and Kaleido are not required.
3. Click **Toggle Panel** in the upper-right corner of the IDE and select
   **Ports** in the bottom panel. If port `5000` is not listed, select
   **Forward a Port** or **Add Port**, enter `5000`, and press **Enter**.
4. Copy the hostname from the **Local Address** URL. For example, for
   `https://abc123.prod.udacity-student-workspaces.com/proxy/5000/`, the hostname
   is `abc123.prod.udacity-student-workspaces.com`.
5. In the terminal, from the same repository directory, run:

   ```bash
   mlflow ui --port 5000 \
     --backend-store-uri sqlite:///mlflow.db \
     --allowed-hosts <workspace-hostname>
   ```

   Replace `<workspace-hostname>`, including the angle brackets, with your
   hostname. Keep the terminal running.
6. In **Ports**, hover over port `5000` and select **Open in Browser**.
   Open **Experiments > Dataset Drift Analysis with Evidently** to inspect
   or compare runs.

If you see **Invalid Host header**, check the hostname passed to
`--allowed-hosts`; omit `https://` and `/proxy/5000/`. If the experiment is
missing, start MLflow from the directory containing the generated `mlflow.db`.
After a workspace timeout, check the forwarded address and restart MLflow.

## Optional: Run locally

Source repository: [udacity/cd0583-diagnose-and-fix](https://github.com/udacity/cd0583-diagnose-and-fix).

Clone the repository, then create and activate a Python virtual environment.
The script has been tested with Python 3.11 and 3.13.

```bash
git clone https://github.com/udacity/cd0583-diagnose-and-fix.git
cd cd0583-diagnose-and-fix
python -m venv .venv
```

Activate it with `source .venv/bin/activate` on macOS/Linux, or
`.venv\Scripts\Activate.ps1` in Windows PowerShell. Then run:

```bash
python -m pip install -r requirements.txt
python train.py
mlflow ui --port 5000 --backend-store-uri sqlite:///mlflow.db
```

Keep the terminal running and open [http://localhost:5000](http://localhost:5000).
The workspace Ports panel and hostname steps do not apply locally.

## Verification

```bash
python -m unittest discover -s tests -v
```

The regression tests cover dataset drift ratios, p-value and distance-based
feature drift decisions, and missing drift test results.

## How to fix Data Drift issues
* If there is substantial data drift then you should reweigh samples in the training data, giving more importance to newer patterns.
* Identify new segments where the model fails, and create a different model for it. Consider using an ensemble of several models for different segments of the data.
* Change the prediction target. For example, switch from weekly to daily forecast or replace the regression model with classification into categories from "high" to "low."
* Pick a different model architecture to account for ongoing drift. You can consider incremental or online learning, where the model continuously adapts to new data.
* Apply domain adaptation strategies. There are a number of approaches to help the model better generalize to a new target domain.

## Extra Resources
* [Incremental (Online) Learning](https://towardsdatascience.com/incremental-online-learning-with-scikit-multiflow-6b846913a50b)
* [Understanding Domain Adaptation](https://towardsdatascience.com/understanding-domain-adaptation-5baa723ac71f)
