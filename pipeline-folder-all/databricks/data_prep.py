from azureml.core import Run
import argparse
import os

def populate_environ():
    parser = argparse.ArgumentParser(description='Process arguments passed to script')
    parser.add_argument('--AZUREML_SCRIPT_DIRECTORY_NAME')
    parser.add_argument('--AZUREML_RUN_TOKEN')
    parser.add_argument('--AZUREML_RUN_TOKEN_EXPIRY')
    parser.add_argument('--AZUREML_RUN_ID')
    parser.add_argument('--AZUREML_ARM_SUBSCRIPTION')
    parser.add_argument('--AZUREML_ARM_RESOURCEGROUP')
    parser.add_argument('--AZUREML_ARM_WORKSPACE_NAME')
    parser.add_argument('--AZUREML_ARM_PROJECT_NAME')
    parser.add_argument('--AZUREML_SERVICE_ENDPOINT')
    
    parser.add_argument("--input", type=str)
    parser.add_argument("--input_filename", type=str)
    parser.add_argument("--processed_data", type=str)
    parser.add_argument("--output_filename", type=str)

    # parser.add_argument("--datastore_name", type=str)
    # parser.add_argument("--output_aml_dataset_name", type=str)

    args, unknown = parser.parse_known_args()
    # print('unknown')
    # print(unknown)
    
    os.environ['AZUREML_SCRIPT_DIRECTORY_NAME'] = args.AZUREML_SCRIPT_DIRECTORY_NAME
    os.environ['AZUREML_RUN_TOKEN'] = args.AZUREML_RUN_TOKEN
    os.environ['AZUREML_RUN_TOKEN_EXPIRY'] = args.AZUREML_RUN_TOKEN_EXPIRY
    os.environ['AZUREML_RUN_ID'] = args.AZUREML_RUN_ID
    os.environ['AZUREML_ARM_SUBSCRIPTION'] = args.AZUREML_ARM_SUBSCRIPTION
    os.environ['AZUREML_ARM_RESOURCEGROUP'] = args.AZUREML_ARM_RESOURCEGROUP
    os.environ['AZUREML_ARM_WORKSPACE_NAME'] = args.AZUREML_ARM_WORKSPACE_NAME
    os.environ['AZUREML_ARM_PROJECT_NAME'] = args.AZUREML_ARM_PROJECT_NAME
    os.environ['AZUREML_SERVICE_ENDPOINT'] = args.AZUREML_SERVICE_ENDPOINT

    return args

args = populate_environ()

input_path = args.input
input_filename = args.input_filename
output_path = args.processed_data
output_filename = args.output_filename

run = Run.get_context(allow_offline=False)
print(run._run_dto["parent_run_id"])

# input_fullpath = input_path + f'/{input_filename}'
# df = spark.read.csv(input_fullpath)
# 
# print(display(df))
# 
# total_rows = df.count()
# run.log('total_rows', total_rows)
# 
# onput_fullpath = output_path + f"/{output_filename}"
# df.write.parquet(onput_fullpath)

run.log('Loss', 1.2)
run.log('Loss', 1.8)
run.log('Loss', 0.9)

run.log('Metric2', 8)
run.log('Metric2', 5)
run.log('Metric2', 9)
run.log('Metric2', 6)
run.log('Metric2', 2)

# ws = run.experiment.workspace
# datastore = Datastore.get(ws, datastore_name)
# 
# # create a TabularDataset from 3 file paths in datastore
# datastore_paths = [(datastore, f'{output_filename}/*.parquet')]
# 
# from azureml.core import Dataset
# amlds_output = Dataset.Tabular.from_delimited_files(path=datastore_paths)
# amlds_output.register(workspace=ws,
#                       name='titanic_ds',
#                       description='titanic training data')

# pdf = dataset.to_pandas_dataframe()

# i = args.input

# df = spark.read.parquet(f'{i}/test.parquet')

print('*************************** End ***************************')