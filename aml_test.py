import argparse
import os
import glob
import numpy as np
import pandas as pd
from pandas import read_csv

from sklearn import __version__ as sklearnver
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OrdinalEncoder, LabelEncoder
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LogisticRegression
from azureml.core.model import Model
from azureml.core.resource_configuration import ResourceConfiguration

from azureml.core import Dataset, Run
from packaging.version import Version
if Version(sklearnver) < Version("0.23.0"):
    from sklearn.externals import joblib
else:
    import joblib

def featurize_data(preped_data):
    preped_data = preped_data.drop(['id'],axis =1)

    print("preped_data.head()")
    print(preped_data.head())

    X = preped_data.drop(['Survived'],axis =1)   #dropped unnecessary columns
    y_train = preped_data['Survived']
    num_columns = list(X.columns)

    ct = make_column_transformer(
        (MinMaxScaler(), num_columns),
        (StandardScaler(), num_columns),
        remainder='passthrough'
    )

    X_features = ct.fit_transform(X)
    return X_features, y_train

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', type=str, dest='data_folder', default='data', help='data folder mounting point')
    parser.add_argument('--model_name', type=str, dest='model_name', default='data', help='data folder mounting point')
    parser.add_argument('--featureset_name', type=str, dest='featureset_name', default='data', help='data folder mounting point')
    args = parser.parse_args()

    run = Run.get_context()
    parent_id = run.parent.id

    print('run.input_datasets')
    print(run.input_datasets)

    print('args.data_folder')
    print(args.data_folder)

    dataset = run.input_datasets['data_folder']
    print("type(dataset)")
    print(type(dataset))

    print('os.listdir(dataset)')
    print(os.listdir(dataset))

    data_folder = args.data_folder

    print(f"data_folder: [{data_folder}]")

    print("reading from parquet")
    preped_data = pd.read_parquet(data_folder)

    X_features, y_train = featurize_data(preped_data)

    log_reg = LogisticRegression(solver='liblinear')
    log_reg.fit(X_features, y_train)

    model_file_name = 'titanic.pkl'
    file_path = os.path.join('./outputs/', model_file_name)

    os.makedirs('./outputs/', exist_ok=True)
    # save model in the outputs folder so it automatically get uploaded
    with open(model_file_name, "wb") as file:
        joblib.dump(value=log_reg, filename=file_path)

    ws = run.experiment.workspace

    ds_feature = Dataset.get_by_name(ws, name=args.featureset_name)

    model = Model.register(model_path=file_path,
                            model_name=args.model_name,
                            datasets=[('featurized data', ds_feature)],
                            tags={'run_id': parent_id},
                            description="Ridge regression model to predict diabetes",
                            resource_configuration=ResourceConfiguration(cpu=1, memory_in_gb=0.5),
                            workspace=ws)
