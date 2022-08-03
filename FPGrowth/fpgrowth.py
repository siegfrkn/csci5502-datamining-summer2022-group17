import pandas as pd
import numpy as np
from mlxtend.preprocessing import TransactionEncoder
from pathlib import Path 
from mlxtend.frequent_patterns import fpgrowth
from pyspark.sql.functions import split
from pyspark.sql import SparkSession
import pyspark.sql.functions as f
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder

spark = SparkSession.builder.master("local[1]").appName("fpgrowth").getOrCreate()

def csvLoad(csv_name):
	print("Loading in file \"" + str(csv_name) + "\"...")
	# df = pd.read_csv(csv_name, low_memory=False)
	# df = pd.read_csv(csv_name, low_memory=False, nrows=100)
	df = spark.read.load(csv_name, format="csv", sep=";", inferSchema="true", header="true")
	# df = df.sample(frac = 0.001, replace = False)
	df = df.sample(False, fraction=0.001, seed=3)
	# df = df.astype(str)
	for col in df.columns:
		df = df.withColumn(col,f.lower(f.col(col)))
	# df = df.applymap(lambda s: s.lower() if type(s) == str else s)
	return df

def printSummary(df_pd):
	print(df_pd.summary())

def removePreCodedCols(df_pd):
	print("Removing pre-coded columns...")
	for (columnName, columnData) in df_pd.toPandas().iteritems():
		if 'Code' in columnName:
			df_pd.toPandas().drop([columnName], axis=1, inplace=True)
			# print("DROP COLUMN")
		# else:
			# print("\nColumnName: " + str(columnName))
			# print(columnData.unique())
	return df_pd



# def dropCols():
	# df.drop(['Railroad Code','Report Year'.'Incident Number','Other Railroad Name','Other Incident Number', \
	# 	     'Other Incident Year','Other Incident Month','Maintenance Railroad Name', \
	# 	     'Maintenance Indcident Number','Grade Crossing ID','Date','County Name', \
	# 	     'State Name','Equipment Struck','Involvement','Visibility','Weather Condition', \
	# 	     'Equipment Type','Track Type','Train Direction','Crossing Warning Expanded Code 8', \
	# 	     'Crossing Warning Expanded Code 9','Crossing Warning Expanded Code 10', \
	# 	     'Crossing Warning Expanded Code 11','Crossing Warning Expanded 1', \
	# 	     'Crossing Warning Expanded 2','Crossing Warning Expanded 3', \
	# 	     'Crossing Warning Expanded 4','Crossing Warning Expanded 5', \
	# 	     'Crossing Warning Expanded 6','Crossing Warning Expanded 7', \
	# 	     'Crossing Warning Expanded 8','Crossing Warning Expanded 9', \
	# 	     'Crossing Warning Expanded 10','Crossing Warning Expanded 11', \
	# 	     'Crossing Warning Expanded 12','Signaled Crossing Warning', \
	# 	     'Crossing Warning Explanation','Roadway Condition'], axis=1)


def printSize(df_pd):
	num_attributes = len(df_pd.columns)
	print("Number of attributes: " + repr(num_attributes) + "\n")
	num_records = len(df)
	print("Number of records: " + repr(num_records) + "\n")


def transcodeData(df_pd):
	print("Transcoding data...")
	# df_out = df_pd.apply(lambda x: list(x.dropna().values), axis=1).tolist()
	# df_pd = df_pd.astype(str)
	# te = TransactionEncoder()
	# fitted = te.fit(df_out)
	# te_ary = fitted.transform(df_out, sparse=True)
	# df_transcoded = pd.DataFrame.sparse.from_spmatrix(te_ary, columns=te.columns_)

	#create a list of the columns that are string typed
	categoricalColumns = [item[0] for item in df_pd.dtypes if item[1].startswith('string') ]

	#define a list of stages in your pipeline. The string indexer will be one stage
	stages = []

	for categoricalCol in categoricalColumns:
		#create a string indexer for those categorical values and assign a new name including the word 'Index'
		stringIndexer = StringIndexer(inputCol = categoricalCol, outputCol = categoricalCol + 'Index')
		encoder = OneHotEncoder(dropLast=False, inputCol=categoricalCol + 'Index', outputCol=categoricalCol + 'Encoded')
		#append the string Indexer to our list of stages
		stages += [stringIndexer]
		stages += [encoder]

	#Create the pipeline. Assign the satges list to the pipeline key word stages
	pipeline = Pipeline(stages = stages)
	#fit the pipeline to our dataframe
	pipelineModel = pipeline.fit(df_pd)
	#transform the dataframe
	df_pd = pipelineModel.transform(df_pd)
	return df_pd

def saveDataFrame(df_pd, filename):
	print("Saving dataframe to file \"" + filename + "\"...")
	df_pd.toPandas().to_csv(filename)


def calcFPGrowth(df_pd, supp):
	print("Calculating frequent patterns...")
	df_pd = df_pd.astype(bool)
	freq_patterns = fpgrowth(df_pd, min_support=supp, use_colnames=True,verbose=1)
	print(freq_patterns)
	return freq_patterns


### INSTRUCTIONS


### USE ORIGINAL DATA
df = csvLoad("../Dataset/Highway-Rail_Grade_Crossing_Accident_Data.csv")
# printSize(df)
df_cleaned = removePreCodedCols(df)
saveDataFrame(df_cleaned, "cleaned.csv")
df_transcoded = transcodeData(df_cleaned)
saveDataFrame(df_transcoded, "transcoded.csv")
calcFPGrowth(df_transcoded, 0.6)

### USE CLEANED DATA
# df_cleaned = csvLoad("./cleaned.csv")
# df_transcoded = transcodeData(df_cleaned)
# saveDataFrame(df_transcoded, "transcoded.csv")
# calcFPGrowth(df_transcoded, 0.6)

### USE TRANSCODED DATA
# df_transcoded = csvLoad("./transcoded.csv")
# # calcFPGrowth(df_transcoded, 0.6)
# df_transcoded = df_transcoded.astype(bool)
# fpgrowth(df_transcoded, min_support=0.01, use_colnames=True)
