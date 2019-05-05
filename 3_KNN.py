import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report



crimes = pd.read_csv("Crimes.csv")  # csv file renamed as Crimes.csv

# Only need 6 columns for this project
col_list = ['Primary Type','Location Description','Arrest','Domestic','District','Community Area']
df = crimes[col_list]
# Drop rows with null entry
df2=df.dropna()

# Factorize every variable and create an index reference
Type_var = pd.factorize(df2['Primary Type']) 
df2['Primary Type'] = Type_var[0]
definition_list_Type = Type_var[1] 

Location_var = pd.factorize(df2['Location Description'])
df2['Location Description'] = Location_var[0]
definition_list_Location = Location_var[1] 

Arrest_var = pd.factorize(df2['Arrest'])
df2['Arrest'] = Arrest_var[0]
definition_list_Arrest = Arrest_var[1] 

Domestic_var = pd.factorize(df2['Domestic'])
df2['Domestic'] = Domestic_var[0]
definition_list_Domestic = Domestic_var[1]

District_var = pd.factorize(df2['District'])
df2['District'] = District_var[0]
definition_list_District = District_var[1] 

Community_var = pd.factorize(df2['Community Area'])
df2['Community Area'] = Community_var[0]
definition_list_Community = Community_var[1] 

# Set up X as the independent variables and y as the dependent variable
X = df2.drop(['Primary Type'],axis=1).values

y = df2['Primary Type'].values

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 1)

# Run randomforestclassifier and get predict results
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print(accuracy_score(y_test, y_pred)) 
print(confusion_matrix(y_test, y_pred)) 
print(classification_report(y_test,y_pred, target_names=definition_list_Type)) 
