# Diagonsing and Fixing Operational Problems

## How to diagnose and fix problems in a production deployed code?

To understand a model's functionality and find its underlying problems, we need to take care of the following things:

* Periodic training
    * Figure out an optimal retraining strategy.
* Monitor model performance
* Analyse
    * Clear visibility of the model helps us and guides the model performance.

## Summary

In this section we will diagnose and fix problems in a production deployed code. To that end, we will:

* Use **evidently** and **mlflow** libraries.
* Deploy a machine learning model in **Heroku**.
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

## Environment setup

#### Prerequisites
* Heroku account
* GitHub account

#### Project Structure

#TODO: Add the project tree here.

#### Dependencies
* All the dependencies are listed in the `requirements.txt` file. You can setup a virtual environment using [Anaconda](https://www.anaconda.com/products/distribution) and install the required dependencies there.
* `runtime.txt` contains the python version that is used for this tutorial.

## Steps
Follow the steps below for deploying this model:

* Ensure that all the dependencies listed in the `requirements.txt` file are installed.
* Run the `train.py` file to log experiments in **mlflow** <br />
* View the results in the **mlflow webui** <br />
















