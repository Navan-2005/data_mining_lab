import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import export_text

# Load dataset
dataset = pd.read_csv("dataset.csv")

selected_attributes = [
    "checking_status",
    "duration",
    "credit_history",
    "credit_amount",
    "class"
]

dataset_selected = dataset[selected_attributes].copy()

# Encode all non-numeric columns
for col in dataset_selected.columns:
    if not pd.api.types.is_numeric_dtype(dataset_selected[col]):
        dataset_selected[col] = LabelEncoder().fit_transform(
            dataset_selected[col].astype(str)
        )

# Features and target
X = dataset_selected.drop("class", axis=1)
y = dataset_selected["class"]

print("\nData Types After Encoding:")
print(X.dtypes)

# Train tree
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X, y)

# Print tree
print("\n=== Decision Tree ===\n")
print(export_text(model, feature_names=list(X.columns)))

# Convert tree to IF-THEN rules
def tree_to_rules(tree, feature_names):
    rules = []

    def recurse(node, rule):
        if tree.children_left[node] == tree.children_right[node]:
            prediction = tree.value[node][0].argmax()
            rules.append(
                "IF " + " AND ".join(rule) +
                f" THEN class = {prediction}"
            )
            return

        feature = feature_names[tree.feature[node]]
        threshold = tree.threshold[node]

        recurse(
            tree.children_left[node],
            rule + [f"{feature} <= {threshold:.2f}"]
        )

        recurse(
            tree.children_right[node],
            rule + [f"{feature} > {threshold:.2f}"]
        )

    recurse(0, [])
    return rules

print("\n=== IF-THEN RULES ===\n")

rules = tree_to_rules(model.tree_, list(X.columns))

for i, rule in enumerate(rules, 1):
    print(f"Rule {i}: {rule}")
