import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

# ==========================
# Load Dataset
# ==========================
dataset = pd.read_csv("dataset.csv")

# Encode categorical attributes
for col in dataset.columns:
    if not pd.api.types.is_numeric_dtype(dataset[col]):
        dataset[col] = LabelEncoder().fit_transform(
            dataset[col].astype(str)
        )

# Features and target
X = dataset.drop("class", axis=1)
y = dataset["class"]

# ==========================
# 1. Unpruned Decision Tree
# ==========================
unpruned_tree = DecisionTreeClassifier(random_state=42)

unpruned_scores = cross_val_score(
    unpruned_tree,
    X,
    y,
    cv=10,
    scoring='accuracy'
)

print("=== Unpruned Tree Accuracy ===")
print("Mean Accuracy: {:.4f}".format(unpruned_scores.mean()))

# Train on full dataset for tree display
unpruned_tree.fit(X, y)

# ==========================
# 2. Find Best Pruning Level
# ==========================
path = unpruned_tree.cost_complexity_pruning_path(X, y)

ccp_alphas = path.ccp_alphas

best_alpha = 0
best_accuracy = 0

for alpha in ccp_alphas:

    tree = DecisionTreeClassifier(
        random_state=42,
        ccp_alpha=alpha
    )

    scores = cross_val_score(
        tree,
        X,
        y,
        cv=10,
        scoring='accuracy'
    )

    mean_acc = scores.mean()

    if mean_acc > best_accuracy:
        best_accuracy = mean_acc
        best_alpha = alpha

print("\nBest Alpha:", best_alpha)

# ==========================
# 3. Train Pruned Tree
# ==========================
pruned_tree = DecisionTreeClassifier(
    random_state=42,
    ccp_alpha=best_alpha
)

pruned_tree.fit(X, y)

# ==========================
# 4. Evaluate Pruned Tree
# ==========================
pruned_scores = cross_val_score(
    pruned_tree,
    X,
    y,
    cv=10,
    scoring='accuracy'
)

print("\n=== Pruned Tree Accuracy ===")
print("Mean Accuracy: {:.4f}".format(pruned_scores.mean()))

# Predictions for report
from sklearn.model_selection import cross_val_predict

y_pred = cross_val_predict(
    pruned_tree,
    X,
    y,
    cv=10
)

print("\n=== Classification Report ===")
print(classification_report(y, y_pred))

print("\n=== Confusion Matrix ===")
print(confusion_matrix(y, y_pred))

# ==========================
# 5. Print Pruned Tree
# ==========================
print("\n=== Pruned Decision Tree ===")
tree_rules = export_text(
    pruned_tree,
    feature_names=list(X.columns)
)

print(tree_rules)

# ==========================
# 6. Accuracy Comparison
# ==========================
print("\n=== Accuracy Comparison ===")
print("Unpruned Accuracy : {:.4f}".format(
    unpruned_scores.mean()
))

print("Pruned Accuracy   : {:.4f}".format(
    pruned_scores.mean()
))

if pruned_scores.mean() > unpruned_scores.mean():
    print("\nAccuracy increased after pruning.")
elif pruned_scores.mean() < unpruned_scores.mean():
    print("\nAccuracy decreased after pruning.")
else:
    print("\nAccuracy remained the same.")
