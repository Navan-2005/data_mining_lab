import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

# Load dataset
dataset = pd.read_csv('dataset.csv')

# Print columns
print("Columns in dataset:")
print(dataset.columns)

# Separate features and target
X = dataset.drop('class', axis=1)
y = dataset['class']

# Convert categorical columns into numeric
X = pd.get_dummies(X)

# Encode target column
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Create Decision Tree model
model = DecisionTreeClassifier(
    criterion='entropy',
    max_depth=5,
    random_state=42
)

# Perform 5-fold cross-validation
accuracy_scores = cross_val_score(model, X, y, cv=10)

# Print results
print("\nAccuracy Scores for each fold:")
print(accuracy_scores)

print("\nAverage Accuracy:")
print(accuracy_scores.mean())

# Train model on full dataset
model.fit(X, y)

# Plot Decision Tree
plt.figure(figsize=(20,10))

plot_tree(
    model,
    feature_names=X.columns,
    class_names=label_encoder.classes_,
    filled=True,
    rounded=True,
    fontsize=8
)

plt.title("Decision Tree")
plt.show()
