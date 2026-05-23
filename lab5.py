import pandas as pd
from sklearn.tree import DecisionTreeClassifier
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

# Convert categorical feature columns into numeric
X = pd.get_dummies(X)

# Encode target column
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Create Decision Tree model
model = DecisionTreeClassifier(random_state=42)

# Perform 5-fold cross-validation
accuracy_scores = cross_val_score(model, X, y, cv=5)

# Print results
print("\nAccuracy Scores for each fold:")
print(accuracy_scores)

print("\nAverage Accuracy:")
print(accuracy_scores.mean())
