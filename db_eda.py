from ucimlrepo import fetch_ucirepo

#load and pre-process Data

# fetch dataset
heart_disease = fetch_ucirepo(id=45)

# data (as pandas dataframes)
X = heart_disease.data.features
y = heart_disease.data.targets
# metadata
print(heart_disease.metadata)
# variable information

print(heart_disease.variables)

print(X.head())
print(y.head())
print(f"For X, there are: {X.isnull().sum()} null values")
print (f"For y, there are : { y.isnull()} null values")

#missing values: we have 6 missing values in total.
x_col = list(X.columns)
y_col = list(y.columns)
print (x_col)
print(y_col)
print(X.describe())
print(y.describe())
print (f"there are {y['num'].unique()} unique values in this column")

#visualise columns
#import seaborn as sns





