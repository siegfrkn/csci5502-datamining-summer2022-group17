import os
import pandas
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn import metrics
import matplotlib.pyplot as plt

# Variables
origData = "Highway-Rail_Grade_Crossing_Accident_Data.csv"
latLongData = "us-county-boundaries.csv" # https://public.opendatasoft.com/explore/dataset/us-county-boundaries/table/?disjunctive.statefp&disjunctive.countyfp&disjunctive.name&disjunctive.namelsad&disjunctive.stusab&disjunctive.state_name

# Utility
def printUniques(df, colName):
    print(df[colName].unique())

def countNA(df, colName):
    print(df[colName].isna().sum())

# Helper
def sameDirAbsolutePath(fileName):
    return os.path.join(os.path.dirname(__file__), fileName)

# Data Processing
def readCrossingData(fileName):
    return pandas.read_csv(sameDirAbsolutePath(fileName), dtype=str)

latLongDf = pandas.read_csv(sameDirAbsolutePath(latLongData), sep=";", dtype=str)

def stripOneColThenSave(df, colName, saveName):
    df[colName] = df[colName].str.strip()
    df.to_csv(saveName)

def oneHotEncodeThenSave(df, colName, saveName):
    df = pandas.get_dummies(df, columns=[colName])
    df.to_csv(saveName)

def dropThenSave(df, colNames, saveName):
    df.dropna(subset=colNames, inplace=True)
    df.to_csv(saveName)

def findLatLong(row):
    stateCode = int(row["State Code"])
    countyCode = int(row["County Code"])
    if stateCode == 12 and countyCode == 25: # Dade County, Florida
        countyCode = 86 # Renamed to Miami-Dade County, Florida
    elif stateCode == 51:
        if countyCode == 515: # Bedford City, Virginia
            countyCode = 19 # Merged into Bedford County, Virginia
        elif countyCode == 560: # Clifton Forge City, Virginia
            countyCode = 5 # Merged into Alleghany County, Virginia
        elif countyCode == 780: # South Boston City, Virginia
            countyCode = 83 # Merged into Halifax County, Virginia
    elif stateCode == 2 and countyCode == 231: # Skagway-Yakutat-Angoon Census Area, Alaska
        countyCode = 230 # Split out to Skagway Municipality, Alaska; Note that Skagway-Yakutat-Angoon Census Area covers multiple modern FIPS codes but all incidents recorded happened in Skagway Borough

    return latLongDf[(latLongDf["STATEFP"].astype(int) == int(row["State Code"])) & (latLongDf["COUNTYFP"].astype(int) == countyCode)].iloc[0]["Geo Point"]

def fipsToLatLong(df, saveName):
    df["County Center Point"] = df.apply(findLatLong, axis=1)
    df.to_csv(saveName)

if __name__ == "__main__":
    #dropThenSave(readCrossingData(origData), ["County Code", "County Name", "State Code", "State Name"], "CleanCountyData.csv") # Drop data without county and state info, that cannot be used in clustering
    #fipsToLatLong(readCrossingData("CleanCountyData.csv"), "LatLong.csv") # Add columns for latitude and longitude based on county info
    #latLongCrossingData = readCrossingData("LatLong.csv")
    #latLongCrossingData[["Latitude", "Longitude"]] = latLongCrossingData["County Center Point"].str.split(",", expand=True)
    #latLongCrossingData.to_csv("SplitLatLong.csv")
    df = readCrossingData("SplitLatLong.csv")
    df.dropna(subset=["Temperature", "Weather Condition Code"], inplace=True)

    kmeans = KMeans(n_clusters=4)
    scalerTemp = StandardScaler()
    df["Temperature"] = scalerTemp.fit_transform(df[["Temperature"]])
    scalerWeather = StandardScaler()
    df["Weather Condition Code"] = scalerWeather.fit_transform(df[["Weather Condition Code"]])
    kmeans.fit(df[["Temperature", "Weather Condition Code"]])
    df["Cluster"] = kmeans.labels_
    colors = ["red", "green", "blue", "yellow"]
    plt.figure(0, figsize=(50,30))
    for i in range(4):
        label = f"Temperature: {scalerTemp.inverse_transform([[kmeans.cluster_centers_[i][0]]])}, Weather Condition Code: {scalerWeather.inverse_transform([[kmeans.cluster_centers_[i][1]]])}"
        plt.scatter(df.loc[df["Cluster"].astype(int) == i, ["Longitude"]].astype(float), df.loc[df["Cluster"].astype(int) == i, ["Latitude"]].astype(float), 100, colors[i], label=label)
        plt.legend()
    plt.savefig(f"Temperature-Weather Clusters.png")

    print(f"Silhouette Score: {metrics.silhouette_score(df[['Temperature', 'Weather Condition Code']], kmeans.labels_, metric='euclidean')}")
