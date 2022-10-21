from typing import List, Any

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_absolute_error

pd.options.display.max_columns = 20
pd.options.display.max_rows = 200

""" 
What is outliers ?
    Outlier is an observation in a given dataset that lies far from the rest of the observations
    
Why it is important to handle with outliers ? 
    Outliers can skew statistical measures and data distributions, providing a misleading representation of the underlying
    data and relationships. Removing outliers from training data prior to modeling can result in a better fit of the data and,
    in turn, more skillful predictions.
    
How to Detect ?
   1. BoxPlots (visualization method  to detect outliers)
   
   2. Z-score
   
   3. Inter Quantile Range(IQR)
   
   4. Isolation Forest (automatic method to detect outliers)
      https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html
   
   5. Minimum Covariance Determinant (automatic method to detect outliers)
   
   6. Local Outlier Factor (automatic method to detect outliers)
   
   7. One-Class SVM (automatic method to detect outliers)
   
How to handle ? 

   
   
"""

# load dataset
df = pd.read_csv('data\california_housing.csv')

# check first 5 rows
print(df.head())

# drop missing values as we are focusing outliers
df.dropna(inplace=True)

# checking number of missing values
print(df.isnull().sum())

df = df.drop('ocean_proximity', axis=1)

# split into target(y) and features(X)
X, y = df.iloc[:, :-1], df.iloc[:, -1:]

# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)

# summarize the shape of the train and test dataset after splitting
print('train dataset has {} rows \ntest dataset has {} rows.'.format(X_train.shape[0], X_test.shape[0]))

# 1. Detect outliers with BoxPlots
# BoxPlot methods used to see outliers visually

for col in [col for col in df.columns if (df[col].dtype in ['int64', 'float64'])]:
    sns.boxplot(data=df, x=col)
    # plt.show()


# 2. Detect outliers with Z-score
# logic :  any data point whose Z-score falls out of the 3rd standard deviation is considered an outlier.

def detect_outliers_zscore(df, col, threshold=3):
    outliers = []
    threshold = threshold
    mean = np.mean(df[col])
    std = np.std(df[col])
    for i in df[col]:
        z_score = (i - mean) / std
        if np.abs(z_score) > threshold:
            outliers.append(i)
    return outliers


# detecting outliers for total_rooms with z_score approaches
total_rooms_outliers_z_score = detect_outliers_zscore(df, 'total_rooms')
print('Column : {}\nMethod: {}\nNumber of outliers detected : {}\nOutliers: \n{}\n'.format('total_rooms','Z-SCORE',len(total_rooms_outliers_z_score),total_rooms_outliers_z_score))

# 3. Detect outliers with IQR
def detect_outliers_iqr(df, col):
    outliers = []

    # calculate 1. and 3. quartiles
    q1 = np.percentile(df[col], 25)
    q3 = np.percentile(df[col], 75)

    # calculates IQR = q3-q1
    IQR = q3 - q1

    # calculate upper and lower bounds
    lower_bound = q1 - (1.5 * IQR)
    upper_bound = q3 + (1.5 * IQR)

    # find outliers which are below lower bound or above upper bound
    for i in df[col]:
        if i < lower_bound or i > upper_bound:
            outliers.append(i)

    return outliers


total_rooms_outliers_iqr = detect_outliers_iqr(df, 'total_rooms')
print('Column : {}\nMethod: {}\nNumber of outliers detected : {}\nOutliers: \n{}\n'.format('total_rooms','IQR',len(total_rooms_outliers_iqr),total_rooms_outliers_iqr))


# 4. Detect outliers with Isolation Forest
# tree-based anomaly detection algorithm
# logic : based on modeling the normal data in such a way as to isolate anomalies that
# are both few and different in the feature space.

#import IsolationForest
from sklearn.ensemble import IsolationForest
iso = IsolationForest(contamination=0.1)
X_train['is_outlier'] = iso.fit_predict(X_train)
print(X_train)

sns.scatterplot(data = X_train, x="total_rooms", y= 'population', hue='is_outlier')
plt.show()