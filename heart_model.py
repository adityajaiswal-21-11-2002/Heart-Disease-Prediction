import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
df = pd.read_csv("heart_disease_uci.csv")

# Fill numeric columns
num_cols = ['trestbps', 'chol', 'thalch', 'oldpeak', 'ca']
for col in num_cols:
    df[col].fillna(df[col].median(), inplace=True)

# Fill categorical columns
cat_cols = ['fbs', 'restecg', 'exang', 'slope', 'thal']
for col in cat_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
for col in cat_cols + ['sex', 'cp', 'dataset']:
    df[col] = le.fit_transform(df[col])


from sklearn.model_selection import train_test_split

X = df.drop(['id', 'num'], axis=1)
y = df['num']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Random Forest (good for tabular data)
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


import joblib

joblib.dump(model, 'heart_disease_model.pkl')

