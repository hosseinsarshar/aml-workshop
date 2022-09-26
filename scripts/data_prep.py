from azureml.core import Run
import argparse
import os


def featurize_data(base_data):
    preped_data = base_data.copy()

    preped_data['Sex'] = preped_data['Sex'].replace({'male': 0, 'female': 1})

    preped_data = preped_data.drop(
        ['Name', 'Ticket', 'Cabin', 'Embarked'], axis=1)
    preped_data['Age'] = preped_data['Age'].fillna(preped_data['Age'].mean())

    return preped_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data', type=str, dest='input_data')
    parser.add_argument('--processed_data', type=str, dest='processed_data')

    args = parser.parse_args()

    print(f'args={args}')

    run = Run.get_context()
    ds_titanic_raw = run.input_datasets['input_data']
    pdf_titanic_raw = ds_titanic_raw.to_pandas_dataframe()

    output_data = featurize_data(pdf_titanic_raw)
    output_path = os.path.join(args.processed_data, 'output.csv')

    print(f'Output path: [{output_path}]')
    output_data.to_csv(output_path)


print('*************************** End ***************************')
