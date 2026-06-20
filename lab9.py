import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix

# Load dataset
dataset = pd.read_csv("dataset.csv")

# Encode categorical columns
for col in dataset.columns:
    if not pd.api.types.is_numeric_dtype(dataset[col]):
        dataset[col] = LabelEncoder().fit_transform(dataset[col].astype(str))

# Features and target
X = dataset.drop("class", axis=1)
y = dataset["class"]

# Decision Tree
model = DecisionTreeClassifier(random_state=42)

# Cross-validation predictions (10-fold like WEKA default)
y_pred = cross_val_predict(model, X, y, cv=10)

# Classification Report
print("\n=== Detailed Accuracy By Class ===")
print(classification_report(y, y_pred, target_names=["bad", "good"]))

# Confusion Matrix
cm = confusion_matrix(y, y_pred)

print("\n=== Confusion Matrix ===")
print(cm)
