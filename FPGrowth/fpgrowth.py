import pandas as pd
import numpy as np
import sys
from mlxtend.preprocessing import TransactionEncoder
from pathlib import Path 
from mlxtend.frequent_patterns import fpgrowth
from pyspark.sql.functions import split, col, lower, element_at, isnan
from pyspark.sql import SparkSession
# import pyspark.sql.functions as f
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import StringIndexer, Imputer, QuantileDiscretizer, OneHotEncoder
from pyspark.ml import Pipeline
from pyspark.ml.fpm import FPGrowth
from pyspark.sql.types import IntegerType
from pyspark import SparkFiles

from pyspark.sql.functions import row_number, array, col, explode, lit, concat, \
  collect_list, struct, array_contains, create_map, size, regexp_replace
  
from pyspark.sql.window import Window

import pandas as pd
import json
from itertools import chain

spark = SparkSession.builder.master("local[1]").appName("fpgrowth").getOrCreate()

# general purpose function for obtain a list of column names from a dataframe that begin with `pattern`
def list_cols_starting_with(df, pattern):
	print(pattern)
	out = list(filter(lambda x: x.startswith(pattern), df.columns))
	return out

# general purpose function to display Spark DF as a pandas DF for improved printing
def display_pandas(df, limit=5):
	return df.limit(limit).toPandas()

def csvLoad(csv_name):
	print("\nLoading in file \"" + str(csv_name) + "\"...")
	df = spark.read.options(header='True', inferSchema='False', delimiter=',').csv(csv_name)
	df = df.sample(False, fraction=0.001, seed=3)
	return df

def printSummary(df_pd):
	print(df_pd.summary())

def removePreCodedCols(self):
	print("\nRemoving pre-coded columns...")
	newSelf = self
	for (columnName, columnData) in self.toPandas().iteritems():
		if 'Code' in columnName:
			newSelf = newSelf.drop(columnName)
			# print("DROP COLUMN")
		# else:
		# 	print("ColumnName: " + str(columnName))
			# print(columnData.unique())
	return newSelf

def dropCols(self):
	print("\nDropping uneccessary columns...")
	newSelf = self.drop('Report Year','Incident Number','Other Railroad Name','Other Incident Number', \
			 'Other Incident Year','Other Incident Month','Maintainance Railroad Name', \
			 'Maintainance Incident Number','Maintainance Incident Year','Date','County Name', \
			 'Maintainance Incident Month','Incident Year','Incident Month','Minute','Time', \
			 'Hazmat Released by','Video Taken','Video Used','Form 54 Filed','Video Used', \
			 'Crossing Warning Expanded Code 8','Special Study 1', 'Special Study 2', \
			 'Crossing Warning Expanded Code 9','Crossing Warning Expanded Code 10', \
			 'Crossing Warning Expanded Code 11','Crossing Warning Expanded 1', \
			 'Crossing Warning Expanded 2','Crossing Warning Expanded 3','Hazmat Released Quantity', \
			 'Crossing Warning Expanded 4','Crossing Warning Expanded 5', 'Hazmat Released Name', \
			 'Crossing Warning Expanded 6','Crossing Warning Expanded 7', 'Nearest Station', \
			 'Crossing Warning Expanded 8','Crossing Warning Expanded 9','Other Railroad Grouping', \
			 'Crossing Warning Expanded 10','Crossing Warning Expanded 11', \
			 'Crossing Warning Expanded 12','Signaled Crossing Warning','Report Key', \
			 'Crossing Warning Explanation','Roadway Condition','Narrative','Time', \
			 'Total Killed Form 57','Total Injured Form 57','Grade Crossing ID')
	return newSelf

def castCols(self):
	print("\nCasting numerical columns...")
	newSelf = self.withColumn("Estimated Vehicle Speed",col("Estimated Vehicle Speed").cast(IntegerType())) \
	.withColumn("Railroad Car Unit Position",col("Railroad Car Unit Position").cast(IntegerType())) \
	.withColumn("Temperature",col("Temperature").cast(IntegerType())) \
	.withColumn("Number of Locomotive Units",col("Number of Locomotive Units").cast(IntegerType())) \
	.withColumn("Number of Cars",col("Number of Cars").cast(IntegerType())) \
	.withColumn("Train Speed",col("Train Speed").cast(IntegerType())) \
	.withColumn("User Age",col("User Age").cast(IntegerType())) \
	.withColumn("Crossing Users Killed For Reporting Railroad",col("Crossing Users Killed For Reporting Railroad").cast(IntegerType())) \
	.withColumn("Crossing Users Injured For Reporting Railroad",col("Crossing Users Injured For Reporting Railroad").cast(IntegerType())) \
	.withColumn("Vehicle Damage Cost",col("Vehicle Damage Cost").cast(IntegerType())) \
	.withColumn("Number Vehicle Occupants",col("Number Vehicle Occupants").cast(IntegerType())) \
	.withColumn("Employees Killed For Reporting Railroad",col("Employees Killed For Reporting Railroad").cast(IntegerType())) \
	.withColumn("Employees Injured For Reporting Railroad",col("Employees Injured For Reporting Railroad").cast(IntegerType())) \
	.withColumn("Number People On Train",col("Number People On Train").cast(IntegerType())) \
	.withColumn("Passengers Killed For Reporting Railroad",col("Passengers Killed For Reporting Railroad").cast(IntegerType())) \
	.withColumn("Passengers Injured For Reporting Railroad",col("Passengers Injured For Reporting Railroad").cast(IntegerType())) \
	.withColumn("Total Killed Form 55A",col("Total Killed Form 55A").cast(IntegerType())) \
	.withColumn("Total Injured Form 55A",col("Total Injured Form 55A").cast(IntegerType()))
	return newSelf

# TODO: keep this or lose this?
def cleanStrings(self):
	print("\nCleaning strings...")
	# cast all strings to lower and remove whitespace
	newSelf = self
	newColumns=(column.replace(' ', '') for column in newSelf.columns) # this works but taking out for the time being
	newSelf = newSelf.toDF(*newColumns)
	return newSelf

def selectColumns(self, isString):
	if isString == True:
		selectedColumns = [item[0] for item in self.dtypes if item[1].startswith('string') ]
	else:
		selectedColumns = [item[0] for item in self.dtypes if not item[1].startswith('string') ]
	print(selectedColumns)
	print(len(selectedColumns))
	return selectedColumns

def countNull(self):
	return df.filter((df[self] == "") | df[self].isNull() | isnan(df[self])).count()

def handleNullValues(self):
	print("\nAddressing null values...")
	# Numerical null values replaced with mean
	mean_subset = ['Estimated Vehicle Speed','Temperature','Train Speed', \
				   'User Age','Vehicle Damage Cost','Number People On Train']
	imputer_mean = Imputer(inputCols=mean_subset,
					  outputCols=[col_ for col_ in mean_subset]
					  ).setStrategy("mean")
	newSelf = imputer_mean.fit(self).transform(self)
	# Numerical null values relpaced with mode
	mode_subset = ['Number of Cars','Number of Locomotive Units','Railroad Car Unit Position','Number of Locomotive Units', \
	'Crossing Users Killed For Reporting Railroad', 'Crossing Users Injured For Reporting Railroad', \
	'Number Vehicle Occupants', 'Employees Killed For Reporting Railroad', 'Employees Injured For Reporting Railroad', \
	'Number People On Train', 'Passengers Killed For Reporting Railroad', 'Passengers Injured For Reporting Railroad', \
	'Total Killed Form 55A', 'Total Injured Form 55A']
	imputer_mode = Imputer(inputCols=mean_subset,
					  outputCols=[col_ for col_ in mean_subset]
					  ).setStrategy("mean")
	newSelf = imputer_mode.fit(self).transform(self)
	# Categorical null values replaced with most frequent
	column_subset = ['Railroad Name', 'Maintenance Railroad Name', 'Maintenance Incident Year', \
					 'Maintenance Incident Month', 'Month', 'Day', 'Hour', 'AM/PM', \
					 'Division', 'Subdivision', 'State Name', 'City Name', 'Highway Name', 'Public/Private', \
					 'Highway User', 'Vehicle Direction', 'Highway User Position', 'Equipment Involved', \
					 'Equipment Struck', 'Hazmat Involvement', \
					 'Hazmat Released Measure', 'Visibility', 'Weather Condition', 'Equipment Type', \
					 'Track Type', 'Track Name', 'Track Class', 'Estimated/Recorded Speed', 'Train Direction', \
					 'Crossing Warning Location', 'Warning Connected To Signal', 'Crossing Illuminated', \
					 'User Gender', 'User Struck By Second Train', 'Highway User Action', 'Driver Passed Vehicle', \
					 'View Obstruction', 'Driver Condition', 'Driver In Vehicle', 'Railroad Type', 'District', \
					 'Whistle Ban', 'Reporting Railroad/Company Grouping', 'Reporting Railroad Class', \
					 'Reporting Railroad SMT Grouping', 'Reporting Parent Railroad Name', \
					 'Reporting Railroad Holding Company', 'Other Railroad Class', \
					 'Other Railroad SMT Grouping', 'Other Parent Railroad Name', 'Other Railroad Holding Company', \
					 'Maintenance Railroad Grouping', 'Maintenance Railroad Class', \
					 'Maintenance Railroad SMT Grouping', 'Maintenance Parent Railroad Name', \
					 'Maintenance Railroad Holding Company']
	null_count = 0
	for col_ in column_subset:
		if countNull(col_) > 0:
			temp_col = newSelf.groupBy(col_).count()
			temp_col = temp_col.dropna(subset=col_)
			frequent_category=temp_col.orderBy(
							 temp_col['count'].desc()).collect()[0][0]
			newSelf = newSelf.fillna(frequent_category, subset=col_)
		# print the number of unique values
		# print(str(col_) + " : "  + str(df.select(col_).distinct().count()))
	return newSelf

def printSize(df_pd):
	num_attributes = len(df_pd.columns)
	print("\nNumber of attributes: " + repr(num_attributes))
	num_records = df_pd.count()
	print("Number of records: " + repr(num_records))

def saveDataFrame(df_pd, filename):
	print("\nSaving dataframe to file \"" + filename + "\"...")
	df_pd.toPandas().to_csv(filename)

def discretizeCol(df_pd):
	# get features list
	feature_list = selectColumns(df_pd, 'False')
	# select features
	features_df = df_pd.select(feature_list)
	# fit QuantileDiscretizer to all and transform
	discretizer = [QuantileDiscretizer(inputCol=x, outputCol="Quantile_"+x, numBuckets=10) for x in feature_list]
	discretizer_results = Pipeline(stages=discretizer).fit(features_df).transform(features_df)
	print(discretizer_results)
	# select transformed columns
	discrete_cols = list_cols_starting_with(discretizer_results, "Quantile_")
	discrete_df = discretizer_results.select(discrete_cols)
	return discrete_df

def to_explode(df, by):
	cols, dtypes = zip(*((c, t) for (c, t) in df.dtypes if c not in by))
	kvs = explode(array([struct(lit(c).alias("feature"), col(c).alias("value")) for c in cols])).alias("kvs")
	return df.select(by + [kvs]).select(by + ["kvs.feature", "kvs.value"])

def append_row_number(df):
	w = Window().orderBy(lit('A'))
	out = df.withColumn("row_number", row_number().over(w))
	return(out)

# cast from wide to long
def wide_to_long(df):
	df_numbered = append_row_number(df)
	df_long = to_explode(df_numbered, by=['row_number'])
	return(df_long)

mapping = {
	0.0: 'Extremely_Low',
	1.0: 'Very_Low',
	2.0: 'Moderately_Low',
	3.0: 'Moderately_Low',
	4.0: 'Moderate',
	5.0: 'Moderate',
	6.0: 'Moderately_High',
	7.0: 'Moderately_High',
	8.0: 'Very_High',
	9.0: 'Extremely_High'
}

def int_to_labels(column, mapping):
	mapping_expr = create_map([lit(x) for x in chain(*mapping.items())])
	labels = mapping_expr.getItem(col(column))
	return(labels)

def calcFPGrowth(newdf, supp, conf):
	print("\nCalculating frequent patterns...")
	print("\nITEMSETS")
	with pd.option_context('display.max_colwidth', 100):
		print(display_pandas(newdf))

	fpGrowth = FPGrowth(itemsCol="items", minSupport=supp, minConfidence=conf)
	model = fpGrowth.fit(newdf)
	model.freqItemsets.sort('freq', ascending=False).show(10, truncate=False)
	model.freqItemsets.show()
	return model.freqItemsets

def itemsets_to_json(df):
	itemsets_dict = df.toPandas().to_dict(orient='records')
	return(itemsets_dict)

# Retrieve itemsets
# -- parameters for the maximum number of items and the desired consequent
# -- results are ordered (descending) by lift score
def get_itemsets(max_items, consequent):
  out = model.associationRules \
    .orderBy('lift', ascending=False) \
    .where(col('lift') > 1) \
    .where(size(col('antecedent')) == max_items-1) \
    .where(array_contains(col("consequent"), consequent))
  return(out)


### INSTRUCTIONS


### USE ORIGINAL DATA

# Load the csv
df = csvLoad("../Dataset/Highway-Rail_Grade_Crossing_Accident_Data.csv")

# cleaning and preprocessing
df = dropCols(df)
df = removePreCodedCols(df)
df = castCols(df)
df = handleNullValues(df)
saveDataFrame(df, "cleaned.csv")

# discretize the numerical columns
# discrete_numerical_df = discretizeCol(df)
discrete_df = discretizeCol(df)
# print("\ndiscrete_numerical_df shape: " + str(discrete_numerical_df.count()) + "," + str(len(discrete_numerical_df.columns)))
# # recombine categorical columns to make a full dataset
# print("TEST:")
# print(df)
# categorical_df = df.select(selectColumns(df, 'True'))
# saveDataFrame(categorical_df, "categorical.csv")
# print("\ncategorical_df shape: " + str(categorical_df.count()) + "," + str(len(categorical_df.columns)))
# # discrete_df = discrete_numerical_df.union(categorical_df)
# discrete_df = discrete_numerical_df.select(concat(categorical_df))
# print("\discrete_df shape: " + str(discrete_df.count()) + "," + str(len(discrete_df.columns)))
# saveDataFrame(discrete_df, "discretized.csv")

# transform to vertical format to increase processing speed
long_df = wide_to_long(df = discrete_df)
long_df_labeled = long_df.withColumn("label", int_to_labels("value", mapping)).drop("value")

itemset_df = long_df_labeled.withColumn('feature', regexp_replace('feature', 'Quantile_', '')) \
	.withColumn("item", concat(col("feature"), lit("_"), col("label"))) \
	.drop("feature", "label") \
	.groupBy("row_number") \
	.agg(collect_list("item").alias("items"))

saveDataFrame(itemset_df, "transcoded.csv")
frequentItemsets = calcFPGrowth(itemset_df, 0.6, 0.6)
file_path = 'frequentItemsets.txt'
sys.stdout = open(file_path, "w")
# print(frequentItemsets.show())


#TODO
# General cleanup
# README
# combine categorical back into discrete bucketed items before itemset creation
# null value casting based on type of incident?
# feature scalaing?
# outliers?
# user defined consequents
# change to jupyter notebook?