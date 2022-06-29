import pandas as pd
import numpy as np

df = pd.read_csv("./GCIS_Published_Crossing_Data/PublishedCrossingData-05-31-2022.csv", low_memory=False)

print(df.describe())

num_attributes = len(df.columns)
num_records = len(df)
num_nans = df.isnull().sum().sum()
num_total_entries = num_attributes * num_records - num_nans

print("Number of attributes: " + repr(num_attributes) + "\n")
print("Number of records: " + repr(num_records) + "\n")
print("Number of NaN's: " + repr(num_nans) + "\n")
print("Total Number of Records: " + repr(num_total_entries) + "\n")

print("Year Min: " + str( min(df.ReportYear)))
print("Year Max: " + str(max(df.ReportYear)) + "\n")

print("Number of Minor Incidents: " + str(len(df[df.ReportType == "Minor"])) + "\n")
print("Number of Major Incidents: " + str(len(df[df.ReportType == "Major"])) + "\n")

print(df.SpselIDs.unique())

# create subset of the data using simple random sample without replacement
df_sample = df.sample(frac = 0.10, replace = False)
print(df_sample.describe())