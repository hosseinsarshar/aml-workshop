import argparse
import os
import re
import pandas as pd

from azureml.core import Run

print('Importing DeltaTable')
from deltalake import DeltaTable
print('DeltaTable Imported')

parser = argparse.ArgumentParser()
parser.add_argument('--data-folder', type=str, dest='data_folder', default='data', help='data folder mounting point')

args = parser.parse_args()
data_folder = args.data_folder

print(f"data_folder: [{data_folder}]")

# Read the Delta Table using the Rust API
dt = DeltaTable(f"{data_folder}/temp_delta")

print('getting py_arrow_tb')
py_arrow_tb = dt.to_pyarrow_table()

print('converting py_arrow_tb to pandas')
pdf = py_arrow_tb.select(['home_type',
    'address_area',
    'SubCity',
    'PostalCodeNeighborhood',
    'address_neighborhood']).to_pandas(strings_to_categorical=True)

print("pdf.head()")
print(pdf.head())
