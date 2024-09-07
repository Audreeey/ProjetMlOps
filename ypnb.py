import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

data = pd.read_csv('Loan_Data.csv')

data = data.drop('customer_id', axis=1)

X = data.drop('default', axis=1)  
y = data['default']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

with open('random_forest_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Modèle random forest entraîné")
print(data)
