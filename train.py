import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

import json
import json
import pandas as pd
import requests
import zipfile
import io

from evidently.model_profile import Profile
from evidently.model_profile.sections import DataDriftProfileSection
from evidently.pipeline.column_mapping import ColumnMapping

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

#load data
content = requests.get("https://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip").content

with zipfile.ZipFile(io.BytesIO(content)) as arc:
    raw_data = pd.read_csv(arc.open("day.csv"), header=0, sep=',', parse_dates=['dteday'], index_col='dteday')

#set column mapping for Evidently Profile
data_columns = ColumnMapping()
data_columns.numerical_features = ['weathersit', 'temp', 'atemp', 'hum', 'windspeed']
data_columns.categorical_features = ['holiday', 'workingday']

#evaluate data drift with Evidently Profile
def eval_drift(reference, production, column_mapping):
    data_drift_profile = Profile(sections=[DataDriftProfileSection()])
    data_drift_profile.calculate(reference, production, column_mapping=column_mapping)
    report = data_drift_profile.json()
    json_report = json.loads(report)

    drifts = []

    for feature in column_mapping.numerical_features + column_mapping.categorical_features:
        drifts.append((feature, json_report['data_drift']['data']['metrics'][feature]['drift_score']))

    return drifts
#set reference dates
reference_dates = ('2011-01-01 00:00:00','2011-01-28 23:00:00')

#set experiment batches dates
experiment_batches = [
    ('2011-02-01 00:00:00','2011-02-28 23:00:00'),
    ('2011-03-01 00:00:00','2011-03-31 23:00:00'),
    ('2011-04-01 00:00:00','2011-04-30 23:00:00'),
    ('2011-05-01 00:00:00','2011-05-31 23:00:00'),  
    ('2011-06-01 00:00:00','2011-06-30 23:00:00'), 
    ('2011-07-01 00:00:00','2011-07-31 23:00:00'), 
]

#log into MLflow
client = MlflowClient()

#set experiment
mlflow.set_experiment('Data Drift Evaluation with Evidently')

#start new run
for date in experiment_batches:
    with mlflow.start_run() as run: #inside brackets run_name='test'
        
        # Log parameters
        mlflow.log_param("begin", date[0])
        mlflow.log_param("end", date[1])

        # Log metrics
        metrics = eval_drift(raw_data.loc[reference_dates[0]:reference_dates[1]], 
                            raw_data.loc[date[0]:date[1]], 
                            column_mapping=data_columns)
        for feature in metrics:
            mlflow.log_metric(feature[0], round(feature[1], 3))

        print(run.info)