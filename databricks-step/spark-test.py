from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()


import pandas as pd
pdf_titanic_csv = pd.read_csv('https://dprepdata.blob.core.windows.net/demo/Titanic.csv')

df = spark.createDataFrame(pdf_titanic_csv)

df.write \
  .format("delta") \
  .option("overwriteSchema", "true") \
  .mode("overwrite") \
  .save('/mnt/delta/titanic_dataset_2')