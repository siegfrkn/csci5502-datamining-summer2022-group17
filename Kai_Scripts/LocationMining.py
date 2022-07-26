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
    return pandas.read_csv(sameDirAbsolutePath(fileName), dtype=str)


def dropThenSave(df, colName, saveName):
    df.dropna(subset=[colName], inplace=True)
    df.to_csv(saveName)

if __name__ == "__main__":
    dropThenSave(readData(origData), "StateName", "DroppedNAState.csv")

# Notes
# Created DroppedNACity.csv with dropThenSave(readData(origData), "CityName", "DroppedNACity.csv")
# Created DroppedNAState.csv with dropThenSave(readData(origData), "CityName", "DroppedNAState.csv")