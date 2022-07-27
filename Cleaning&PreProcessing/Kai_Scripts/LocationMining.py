import os
import pandas

# Variables
origData = "PublishedCrossingData-06-30-2022.csv"

# Utility
def printUniques(df, colName):
    print(df[colName].unique())

def countNA(df, colName):
    print(df[colName].isna().sum())

# Helper
def sameDirAbsolutePath(fileName):
    return os.path.join(os.path.dirname(__file__), fileName)

# Data Processing
def readData(fileName):
    return pandas.read_csv(sameDirAbsolutePath(fileName), dtype=str, index_col=[0])

def stripOneColThenSave(df, colName, saveName):
    df[colName] = df[colName].str.strip()
    df.to_csv(saveName)

def oneHotEncodeThenSave(df, colName, saveName):
    df = pandas.get_dummies(df, columns=[colName])
    df.to_csv(saveName)

def dropThenSave(df, colName, saveName):
    df.dropna(subset=[colName], inplace=True)
    df.to_csv(saveName)

if __name__ == "__main__":
    printUniques(readData("DroppedNALandUse.csv"), "DevelTypID")

# Notes
#
# Created DroppedNACity.csv with dropThenSave(readData(origData), "CityName", "DroppedNACity.csv")
#
# Created DroppedNAState.csv with dropThenSave(readData(origData), "StateName", "DroppedNAState.csv")
#
# Created DroppedNALandUse.csv with dropThenSave(readData("DroppedNACity.csv"), "DevelTypID", "DroppedNALandUse.csv")
#
# Created OneHotLandUse-DroppedNACity.csv with:
# stripOneColThenSave(readData("DroppedNALandUse.csv"), "DevelTypID", "DroppedNALandUse.csv")
# oneHotEncodeThenSave(readData("DroppedNALandUse.csv"), "DevelTypID", "OneHotLandUse-DroppedNACity.csv")