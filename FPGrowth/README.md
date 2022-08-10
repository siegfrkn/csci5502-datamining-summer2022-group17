# Frequent Itemset Generation
Using FPGrowth, the frequent itemsets are generated for the dataset. Attributes of interest, called consequents, are passed so that patterns containing the attributes of interest can be identified. The generation of these itemsets is completed in several stages, detailed below.

## Running the Code

### Required Version
This code was written and tested using Python version 3.8.10 on x86_64 GNU/Linux. While Python should run on other distrubutions and operating systems, only the aforementioned have been tested.

### Required Modules
The following python modules are required to run the code: pandas, numpy, sys, json, itertools, and pyspark are installed prior to running

### To Run
In the directory of the file fpgrowth.py, run the command `python3 fpgrowth.py`.

## Implementation Description

### Pyspark
The implementation of PFGrowth used in this analysis is from the Apache pyspark library. The pyspark library is the python API for the Apache Spark open-source, distributed computing framework that is optimized for real-time, large-scale data processing. Initially, a different implementation of FPGrowth was used from the mlxtend library, but the implementation was not parallelized and not optimized to handle extremely large quantities of data.

### Cleaning and Preprocessing
The first step before any analysis can be done is cleaning the data. First, columns containing data that is unique to every record or empty is manually removed. The dataset also contains "pre-coded" columns where string columns were coded into a numerical value. However, these bins were often too coarse or too fine and duplicated the data encoded in other columns, so these columns were also manually removed. The entire dataset was read in as strings, so numerical attributes were manually identified and casted to integer type. Null values for all columns were then addressed. Numerical null values were replaced with either the attribute mode or mean, whichever was assessed to be most appropriate. Categorical null values were replaced with the most frequent value in the column. Finally, the numerical columns are discretized into at most 10 bins so that pattern generation will be more meaningful and easier to interperet.

### Transform to Verical Format
The amount of data being processes is very large, and so the amount of memory and time required to process this data is also very large. Transforming the data from the horizontal format (records on the vertical and attributes on the horizontal) to vertical (attributes on the vertical and records on the horizontal) increases run time efficiency.

### Calculate Frequent Itemsets
Now that the data has been cleaned, preprocessed, and tranformed into a more efficient format, the frequent patterns can be generated. The function to generate the FPGrowth model takes values for minimum support and confidence. The complete set of frequent patterns is returned. In addition, the user can specify an attribute of interest, referred to as a consequent, and a maximum number of items in the final pattern, such that patterns that are deemed of interest to the user are identified.

### Show Association Rules
The FPGrowth model developed is then used to show the association rules for the given consequents, and the association rules are reported with metrics lift, confidence, and support.
