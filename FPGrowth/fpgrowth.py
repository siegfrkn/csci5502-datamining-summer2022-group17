"""
FPGrowth
Analysis Of The Effects Of Railroad Safety
CSCI 5502 - Group 17
Written by Katrina Siegfried, Summer 2022
"""


"""
REFERENCES

Association Rules from Continuous Features
https://abndistro.com/post/2021/05/20/using-pyspark-and-mllib-to-generate-association-rules-from-continuous-features/

Market Basket Analysis Using pyspark FPGrowth
https://towardsdatascience.com/market-basket-analysis-using-pysparks-fpgrowth-55c37ebd95c0

Machine Learning with pyspark and mllib
https://towardsdatascience.com/machine-learning-with-pyspark-and-mllib-solving-a-binary-classification-problem-96396065d2aa

Apache Spark Docs
https://spark.apache.org/docs/latest/index.html

Association Rule Mining
https://notebook.community/donaghhorgan/COMP9033/labs/10%20-%20Association%20rule%20mining
"""


"""
IMPORT MODULES
Import the modules required in this file, this requires modules pandas, numpy, sys, json,
itertools, and pyspark are installed prior to running
"""
import pandas as pd
import numpy as np
import sys
import json
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, Imputer, QuantileDiscretizer, OneHotEncoder
from pyspark.ml import Pipeline
from pyspark.ml.fpm import FPGrowth
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import row_number, array, col, explode, lit, concat, \
	collect_list, struct, array_contains, create_map, size, regexp_replace, split, \
	element_at, isnan
from pyspark.sql.window import Window
from itertools import chain


"""
listColsStartingWith
Description: helper function to list columns matching a pattern
Input: dataframe, pattern of columns to list
Output: list of columns from dataframe matching pattern
""" 
def listColsStartingWith(df, pattern):
	print(pattern)
	out = list(filter(lambda x: x.startswith(pattern), df.columns))
	return out

"""
displayPandas
Description: helper function to display the top rows of a dataframe
Input: dataframe, optional limit for number of rows to return default = 5
Output: the top number of rows passed from dataframe
"""
def displayPandas(df, limit=5):
	return df.limit(limit).toPandas()

"""
csvLoad
Description: load a csv 
Input: filename of csv file to load as string
Output: pyspark dataframe from file contents
"""
def csvLoad(csv_name):
	print("\nLoading in file \"" + str(csv_name) + "\"...")
	df = spark.read.options(header='True', inferSchema='False', delimiter=',').csv(csv_name)
	# uncomment the line below to only run on a sample of the data
	# df = df.sample(False, fraction=0.001, seed=3)
	return df


"""
removePreCodedCols
Description: remove columns in the dataset that were already precoded copies of data from
other columns
Input: dataframe
Output: dataframe without precoded duplicate columns
"""
def removePreCodedCols(self):
	print("\nRemoving pre-coded columns...")
	newSelf = self
	for (columnName, columnData) in self.toPandas().iteritems():
		if 'Code' in columnName:
			newSelf = newSelf.drop(columnName)
	return newSelf

"""
dropCols
Description: drop a selection of columns that do not encode meaningful information for frequent
pattern mining, these columns are either unique identifiers for each record or are not populated
Input: dataframe
Output: dataframe without the columns expliditly dropped in this function
"""
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

"""
castCols
Description: cast numerical columns (called out explicitly) from string to integer
Input: dataframe
Output: original dataframe with called out numerical columns casted to integer
"""
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

"""
selectColumns
Description: helper function select columns of either string type or non-string type
Input: dataframe and bool where 'True' means get string types and 'False' means get non-string type
Output: list of columns of either string or non-string type
"""
def selectColumns(self, isString):
	if isString == True:
		selectedColumns = [item[0] for item in self.dtypes if item[1].startswith('string') ]
	else:
		selectedColumns = [item[0] for item in self.dtypes if not item[1].startswith('string') ]
	return selectedColumns

"""
countNull
Description: count the number of null entries in a column
Input: column from dataframe
Output: the count of null entries in the column
"""
def countNull(self):
	return df.filter((df[self] == "") | df[self].isNull() | isnan(df[self])).count()

"""
handleNullValues
Description: handle the null values in the dataframe by populating by mean/mode for numerical or by
most frequent category for categorical
Input: dataframe
Output: dataframe with null values populated accordingly
"""
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
	return newSelf

"""
printSize
Description: helper function to print the size of a dataframe
Input: dataframe
Output: print statement of number of attributes and records
"""
def printSize(df_pd):
	num_attributes = len(df_pd.columns)
	print("\nNumber of attributes: " + repr(num_attributes))
	num_records = df_pd.count()
	print("Number of records: " + repr(num_records))

"""
saveDataFrame
Description: helper function to save a dataframe to a csv file
Input: dataframe and filename to save dataframe to
Output: csv file saved to either working directory or directory specified as part of filename
"""
def saveDataFrame(df_pd, filename):
	print("\nSaving dataframe to file \"" + filename + "\"...")
	df_pd.toPandas().to_csv(filename)

"""
discretizeCol
Description: put numerical continuous columns into a max of 10 discrete buckets, easier to find patterns
Input: dataframe
Output: dataframe with numerical continuous columns in bins
"""
def discretizeCol(df_pd):
	# get features and select
	feature_list = selectColumns(df_pd, False)
	features_df = df_pd.select(feature_list)
	# fit QuantileDiscretizer to all and transform
	discretizer = [QuantileDiscretizer(inputCol=x, outputCol="Quantile_"+x, numBuckets=10) for x in feature_list]
	discretizer_results = Pipeline(stages=discretizer).fit(features_df).transform(features_df)
	print("\nDISCRETIZER RESULTS")
	# select transformed columns
	discrete_cols = listColsStartingWith(discretizer_results, "Quantile_")
	print("Total Injured Form 55A: " + str(discretizer_results.select("Total Injured Form 55A")))
	# recombine discrete columns with categorical columns
	categorical_cols = selectColumns(df_pd, True)
	total_cols = discrete_cols + categorical_cols
	print(total_cols)
	discrete_df = discretizer_results.select(discrete_cols)
	total_df = discrete_df.select(["*"] + [lit(f"{x}").alias(f"ftr{x}") for x in categorical_cols])
	return total_df

"""
toExplode
Description: helper function convert dataframe to vertical format to increase processing speed of
fpgrowth
Input: dataframe and column name by which to pivot
Output: dataframe in vertical format
"""
def toExplode(df, by):
	cols, dtypes = zip(*((c, t) for (c, t) in df.dtypes if c not in by))
	kvs = explode(array([struct(lit(c).alias("feature"), col(c).alias("value")) for c in cols])).alias("kvs")
	return df.select(by + [kvs]).select(by + ["kvs.feature", "kvs.value"])

"""
appendRowNumber
Description: helper function to add a column of row numbers
Input: dataframe
Output: return dataframe with appended column corresponding to row number
"""
def appendRowNumber(df):
	w = Window().orderBy(lit('A'))
	out = df.withColumn("row_number", row_number().over(w))
	return(out)
"""
wideToLong
Description: calls the helper functions to pivor a dataframe to verical format
Input: dataframe
Output: pivoted dataframe in vertical format
"""
def wideToLong(df):
	df_numbered = appendRowNumber(df)
	df_long = toExplode(df_numbered, by=['row_number'])
	return(df_long)

"""
intToLabels
Description: map the bucketed numerical values to strings which have more meaning
Input: column to map, the mapping schema below
Output: the column remapped
"""
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
def intToLabels(column, mapping):
	mapping_expr = create_map([lit(x) for x in chain(*mapping.items())])
	labels = mapping_expr.getItem(col(column))
	return(labels)

"""
calcFPGrowth
Description: calculate frequent patters from a dataframe given constraints
Input: dataframe, minimum support, minimum confidence
Output: fpgrowth model of the dataframe
"""
def calcFPGrowth(newdf, supp, conf):
	print("\nCalculating frequent patterns...")
	fpGrowth = FPGrowth(itemsCol="items", minSupport=supp, minConfidence=conf)
	model = fpGrowth.fit(newdf)
	model.freqItemsets.sort('freq', ascending=False).show(25, truncate=False)
	return model

"""
itemsetsToJson
Description: helper function to convert frequent itemsets to dictionary to make them more readable
Input: dataframe of frequent itemsets
Output: dictionary of frequent itemsets
"""
def itemsetsToJson(df):
	itemsets_dict = df.toPandas().to_dict(orient='records')
	return(itemsets_dict)

"""
getItemsets
Description: retrieve itemsets which contain the maximum number of items and desirec consequent
Input: max number of items in the pattern, consequent(s) of interest, the fpgrowth model
Output: itemsets which match the consequent and max number of items
"""
def getItemsets(max_items, consequent, model):
	print("\nSorting frequent patterns by consequents...")
	out = model.associationRules \
		.orderBy('lift', ascending=False) \
		.where(col('lift') > 1) \
		.where(size(col('antecedent')) == max_items-1) \
		.where(array_contains(col("consequent"), consequent))
	out.show(25, truncate=False)
	return(out)





"""
MAIN
The following is the primary code that drives the pipeline for generating frequent patterns
from the given data
"""

# create the spark session
spark = SparkSession.builder.master("local[1]").appName("fpgrowth").getOrCreate()

# Load the csv
df = csvLoad("../Dataset/Highway-Rail_Grade_Crossing_Accident_Data.csv")

# cleaning and preprocessing
df = dropCols(df)
df = removePreCodedCols(df)
df = castCols(df)
df = handleNullValues(df)
discrete_df = discretizeCol(df)

# transform to vertical format to increase processing speed
long_df = wideToLong(df = discrete_df)
long_df_labeled = long_df.withColumn("label", intToLabels("value", mapping)).drop("value")
itemset_df = long_df_labeled.withColumn('feature', regexp_replace('feature', 'Quantile_', '')) \
	.withColumn("item", concat(col("feature"), lit("_"), col("label"))) \
	.drop("feature", "label") \
	.groupBy("row_number") \
	.agg(collect_list("item").alias("items"))

# calculate frequent itemsets
fpGrowthModel = calcFPGrowth(itemset_df, 0.1, 0.1)
frequentItemsets = fpGrowthModel.freqItemsets

# retrieve frequent itemsets related to specific attributes of interest i.e. consequents
consequents = ["Total Injured Form 55A_Moderate"]
itemsets = [getItemsets(2, i, fpGrowthModel ) for i in consequents]
itemsetsJson = [itemsetsToJson(i) for i in itemsets]

# show association rules
fpGrowthModel.associationRules.sort("antecedent", "consequent").show(100, truncate=False)
