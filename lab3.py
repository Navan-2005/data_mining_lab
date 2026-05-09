import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
dataset = pd.read_csv('dataset.csv')

# Display first 5 rows
print("\nDataset Preview:\n")
print(dataset.head())

# Separate features and target
X = dataset.iloc[:, :-1].copy()
y = dataset.iloc[:, -1].copy()

# Encode categorical columns
for column in X.columns:
    try:
        X[column] = pd.to_numeric(X[column])
    except:
        le = LabelEncoder()
        X[column] = le.fit_transform(X[column].astype(str))

# Encode target column
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y.astype(str))

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Create Decision Tree model
clf = DecisionTreeClassifier(
    criterion='entropy',
    max_depth=4,
    random_state=42
)

# Train model
clf.fit(X_train, y_train)

# Predict test data
y_pred = clf.predict(X_test)

# Print Accuracy
print("\nAccuracy:")
print(accuracy_score(y_test, y_pred))

# Print Classification Report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Plot Decision Tree
plt.figure(figsize=(20,10))

plot_tree(
    clf,
    feature_names=X.columns,
    class_names=target_encoder.classes_,
    filled=True,
    rounded=True,
    fontsize=8
)

# Save tree image
plt.savefig("decision_tree.png")

# Show tree
plt.show()
