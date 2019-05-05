import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from sklearn.preprocessing import OneHotEncoder

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

# Use OneHotEncoder for all independent variables
binary_encoder = OneHotEncoder(sparse=False)
encoded_X = binary_encoder.fit_transform(X)

X_train_hot, X_test_hot, y_train_hot, y_test_hot = train_test_split(encoded_X, y, test_size = 0.25, random_state = 1)

# Use OneHot encoded variables
classifier = RandomForestClassifier(n_estimators = 64, criterion = 'entropy', random_state = 11)
classifier.fit(X_train_hot, y_train_hot)
y_pred_hot = classifier.predict(X_test_hot)

print(accuracy_score(y_test_hot, y_pred_hot)) 
print(confusion_matrix(y_test_hot, y_pred_hot)) 
print(classification_report(y_test_hot,y_pred_hot, target_names=definition_list_Type)) 
