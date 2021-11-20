import argparse
import json
import os
import azureml.core
from azureml.core import Workspace, Experiment, Model
from azureml.core import Run
from azureml.train.hyperdrive import HyperDriveRun
from shutil import copy2

parser = argparse.ArgumentParser()
parser.add_argument('--saved-model', type=str, dest='saved_model', help='path to saved model file')
parser.add_argument('--metrics_data', type=str, dest='metrics_data', help='metrics')
parser.add_argument('--model_name', type=str, dest='model_name', help='model name')
args = parser.parse_args()

saved_model = args.saved_model
print('saved_model')
print(saved_model)

metrics_data = args.metrics_data
print('metrics_data')
print(metrics_data)

model_name = args.model_name
print('model_name')
print(model_name)


model_output_dir = './model/'

os.makedirs(model_output_dir, exist_ok=True)
copy2(args.saved_model, model_output_dir)

ws = Run.get_context().experiment.workspace

model = Model.register(workspace=ws, model_name=model_name, model_path=model_output_dir)
