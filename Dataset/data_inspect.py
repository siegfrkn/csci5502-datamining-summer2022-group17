import pandas as pd
import numpy as np

df = pd.read_csv("./GCIS_Published_Crossing_Data/PublishedCrossingData-05-31-2022.csv", low_memory=False)

print(df.describe())

print("Number of attributes: " + repr(len(df.columns)) + "\n")
print("Number of records: " + repr(len(df)) + "\n")
print("Attributes: " + repr(df.info()) + "\n")

print("Year Min: " + str( min(df.ReportYear)))
print("Year Max: " + str(max(df.ReportYear)) + "\n")

print("Number of Minor Incidents: " + str(len(df[df.ReportType == "Minor"])) + "\n")
print("Number of Major Incidents: " + str(len(df[df.ReportType == "Major"])) + "\n")

print(df.EmrgncySrvc.unique())

# create subset of the data using simple random sample without replacement
df_sample = df.sample(frac = 0.10, replace = False)
print(df_sample.describe())