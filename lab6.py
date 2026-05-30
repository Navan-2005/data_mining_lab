import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

# Load dataset
data = pd.read_csv("dataset.csv")

# Remove extra spaces from column names
data.columns = data.columns.str.strip()

# Encode ALL columns
for col in data.columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))

# Select attributes
selected_columns = [
    'credit_history',
    'purpose',
    'employment',
    'housing',
    'residence_since'
]

X = data[selected_columns]
y = data['class']

# Verify encoding
print("Feature Data Types:")
print(X.dtypes)

# Create model
model = DecisionTreeClassifier(
    criterion='entropy',
    random_state=42
)

# 10-fold cross validation
scores = cross_val_score(
    model,
    X,
    y,
    cv=10
)

print("\nAccuracy for each fold:")
print(scores)

print("\nAverage Accuracy:")
print(scores.mean() * 100)
