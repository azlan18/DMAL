import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv("Book1.csv")

# Standardize the features
scaler = StandardScaler()
df[['R&D Spend', 'Marketing Spend']] = scaler.fit_transform(df[['R&D Spend', 'Marketing Spend']])

# Identify outliers
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1

outliers = df[((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
num_outliers = len(outliers)

print("Number of outliers:", num_outliers)

# Identify the most extreme outlier
if num_outliers > 0:
    most_extreme_outlier = outliers.abs().max().max()
    print("Most extreme outlier:", most_extreme_outlier)

# Partition the dataset
X = df[['R&D Spend', 'Marketing Spend']]  # Features
y = df['Profit']  # Target variable

# Split the dataset into training and testing sets (75% training, 25% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Now you can use X_train and y_train for training your model, and X_test and y_test for testing it.
