#Using New Dataset (Pre-Processing)
import pandas as pd
import numpy as np
df = pd.read_csv('FullCrossingData.csv')

import sklearn
from sklearn.preprocessing import OneHotEncoder
#create instance of one-hot-encoder
encoder = OneHotEncoder(handle_unknown='ignore')
def OneHotFix(attribute,labels):
    colmns = df.columns.values.tolist()
    encoder_df = pd.DataFrame(encoder.fit_transform(df[[attribute]]).toarray())
    final_df = df.join(encoder_df)
    final_df.drop(attribute,axis=1, inplace=True)
    newcolms = [x for x in colmns if x != attribute]
    for x in labels:
        newcolms.append(labels[labels.index(x)])
    final_df.columns = newcolms
    return final_df

df['View Obstruction Code'] = df['View Obstruction Code'].replace(9,np.nan)
df = OneHotFix('View Obstruction Code',['Permanent Structure','Railroad Equipment','Passing Train','Topography','Vegetation','Highway Vehicles','Other Obstruction','Not Obstructed','Unknown Obstruction'])

df['Crossing Illuminated'] = df['Crossing Illuminated'].replace(np.nan,'Unknown')
df = OneHotFix('Crossing Illuminated',['Illuminated-No','Illuminated-Yes','Illuminated-Unknown'])

df['Crossing Warning Location Code'] = df['Crossing Warning Location Code'].replace([np.nan,0,'N'],4.0)
df['Crossing Warning Location Code'] = df['Crossing Warning Location Code'].astype(int)
df = OneHotFix('Crossing Warning Location Code',['Both Sides','Side of Vehicle Approach','Opposite Side of Vehicle Approach','Unknown Side'])

df['Warning Connected To Signal'] = df['Warning Connected To Signal'].replace(np.nan,'Unknown')
df = OneHotFix('Warning Connected To Signal',['Connected To Signal', 'Not Connected To Signal','Unknown If Connected To Signal'])

df = OneHotFix('Public/Private Code',['Private','Public'])

df['Crossing Warning Expanded Code 12'] = df['Crossing Warning Expanded Code 12'].replace(np.nan,11)
df = OneHotFix('Crossing Warning Expanded Code 12',['Gates','Cantilever FLS','Standard FLS','Wig wags','Traffic Signals','Audible','Crossbucks','Stop Signs','Watchman','Flagged','Other Warning','No Warning'])

df.to_csv('RevisedData.csv')

#Decision Tree Classification (Crossing Characterisitics)
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, precision_recall_fscore_support
from sklearn import metrics
# identify attributes desired to study
col_names = ['Total Injured Form 55A','Train Speed','Public','Private',
            'Connected To Signal', 'Not Connected To Signal','Unknown If Connected To Signal',
            'Both Sides','Side of Vehicle Approach','Opposite Side of Vehicle Approach','Unknown Side',
            'Illuminated-Yes','Illuminated-No','Illuminated-Unknown','Permanent Structure','Railroad Equipment',
            'Passing Train','Topography','Vegetation','Highway Vehicles','Other Obstruction','Not Obstructed',
            'Unknown Obstruction','Gates','Cantilever FLS','Standard FLS','Wig wags','Traffic Signals','Audible',
            'Crossbucks','Stop Signs','Watchman','Flagged','Other Warning','No Warning']
# load dataset
data = pd.read_csv('RevisedData.csv',usecols=col_names)
# split dataset between features and target variable
col_names.remove('Total Injured Form 55A')
data = data.dropna(axis=0,subset=col_names)
sortedData = data[col_names] # Features
targetVariables = data['Total Injured Form 55A'] # Target Variable

X_train, X_test, y_train, y_test = train_test_split(sortedData,
                                 targetVariables, test_size=0.2, 
                                 random_state=1) # 80% training and 20% test

# Create Decision Tree classifier object
clf = DecisionTreeClassifier()
# Train Decision Tree Classifier
clf = clf.fit(X_train,y_train)
#Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

# Generate image of decison tree
from sklearn.tree import export_graphviz
from six import StringIO  
from IPython.display import Image  
import pydotplus
dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,max_depth = 4,
                feature_names = col_names)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
#graph.write_png('C:\\Users\\andre\\OneDrive\\Documents\\CU Boulder\\Summer 2022\\CSCI 5502 - Data Mining\\Final Project\\Train_DecisionTree.png')
#Image(graph.create_png())

print("Confusion Matrix Tree : \n", confusion_matrix(y_test, y_pred),"\n")
print("The precision for Tree is ", precision_score(y_test, y_pred, average = 'micro'))
print("The recall for Tree is ", recall_score(y_test, y_pred, average = 'micro'),"\n")  
print(precision_recall_fscore_support(y_test, y_pred, average = 'micro'))