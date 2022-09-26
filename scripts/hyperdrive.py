import azureml.core
from azureml.core import Workspace, Environment, Experiment, Datastore, Dataset, Run

import os
import shutil
import urllib
import numpy as np
import matplotlib.pyplot as plt
import argparse
import argparse
import os
import glob
import shutil
import joblib
import numpy as np
import pandas as pd
from pandas import read_csv

import mlflow
import mlflow.sklearn

from sklearn import __version__ as sklearnver
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OrdinalEncoder, LabelEncoder
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LogisticRegression

def featurize_data(preped_data):
    preped_data = preped_data.copy()
    
    X = preped_data.drop(['Survived'],axis =1)   #dropped unnecessary columns
    y_train = preped_data['Survived']
    num_columns = list(X.columns)

    ct = make_column_transformer(
        (MinMaxScaler(), num_columns),
        (StandardScaler(), num_columns),
        remainder='passthrough'
    )

    X_features = ct.fit_transform(X)
    return preped_data, X_features, y_train

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--titanic_processed', type=str, dest='titanic_processed')
    parser.add_argument('--solver', type=str, dest='solver')
    parser.add_argument('--max_iter', type=int, dest='max_iter')
    parser.add_argument('--penalty', type=str, dest='penalty')
    parser.add_argument('--tol', type=float, dest='tol')

    args = parser.parse_args()
    print(f'args={args}')

    run = Run.get_context()
    print(f'run.input_datasets={run.input_datasets}')
    
    ds_titanic_raw = run.input_datasets['titanic_processed']
    pdf_titanic_raw = ds_titanic_raw.to_pandas_dataframe()

    print(f'pdf_titanic_raw.head()')
    print(pdf_titanic_raw.head())

    preped_data, X_features, y_train = featurize_data(pdf_titanic_raw)

    mlflow.sklearn.autolog()

    with mlflow.start_run() as run:
        log_reg = LogisticRegression(solver=args.solver, max_iter=args.max_iter, penalty=args.penalty, tol=args.tol)
        fitted_model = log_reg.fit(X_features, y_train)

        isdir = os.path.isdir("outputs")
        if isdir:
            shutil.rmtree("outputs")
        mlflow.sklearn.save_model(fitted_model, "outputs")
