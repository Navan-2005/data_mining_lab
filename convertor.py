from scipy.io import arff
import pandas as pd

# Load ARFF file
data, meta = arff.loadarff('dataset.arff')

# Convert to pandas DataFrame
df = pd.DataFrame(data)

# Decode byte strings into normal strings
for col in df.select_dtypes([object]).columns:
    df[col] = df[col].str.decode('utf-8')

# Save as CSV
df.to_csv('dataset.csv', index=False)

print("ARFF converted to CSV successfully!")
