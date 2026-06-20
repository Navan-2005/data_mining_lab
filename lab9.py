import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import cross_val_score

# Load Dataset
dataset = pd.read_csv("dataset.csv")

print("Original Data Types:")
print(dataset.dtypes)

# Encode all non-numeric columns
for col in dataset.columns:
    if not pd.api.types.is_numeric_dtype(dataset[col]):
        le = LabelEncoder()
        dataset[col] = le.fit_transform(dataset[col].astype(str))

print("\nData Types After Encoding:")
print(dataset.dtypes)

# Features and Target
X = dataset.drop("class", axis=1)
y = dataset["class"]

print("\nChecking X data types:")
print(X.dtypes)

# Decision Tree
model = DecisionTreeClassifier(random_state=42)

accuracy_scores = cross_val_score(
    model,
    X,
    y,
    cv=5
)

print("\nAccuracy Scores:")
print(accuracy_scores)

print("\nAverage Accuracy:")
print(accuracy_scores.mean())

# Pruned Tree
pruned_model = DecisionTreeClassifier(
    ccp_alpha=0.01,
    random_state=42
)

accuracy_scores_pruned = cross_val_score(
    pruned_model,
    X,
    y,
    cv=5
)

print("\nPruned Accuracy Scores:")
print(accuracy_scores_pruned)

print("\nAverage Pruned Accuracy:")
print(accuracy_scores_pruned.mean())

# Train final model
model.fit(X, y)

# Plot Tree
plt.figure(figsize=(25, 12))

plot_tree(
    model,
    feature_names=X.columns,
    class_names=["bad", "good"],
    filled=True,
    rounded=True,
    fontsize=8
)

plt.title("Decision Tree - German Credit Dataset")
plt.show()
