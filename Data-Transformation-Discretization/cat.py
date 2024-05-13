import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv("cat.csv")

# One-hot encode the 'State' column
df = pd.get_dummies(df, columns=['State'], prefix='State', dtype=int)

# Separate features (X) and target variable (y)
X = df.drop(columns=['Profit'])
y = df['Profit']

# Standardize the numerical variables
scaler = StandardScaler()
numerical_cols = ['R&D Spend', 'Administration', 'Marketing Spend']
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

# Identify outliers
Q1 = X.quantile(0.25)
Q3 = X.quantile(0.75)
IQR = Q3 - Q1

outliers = X[((X < (Q1 - 1.5 * IQR)) | (X > (Q3 + 1.5 * IQR))).any(axis=1)]
num_outliers = len(outliers)

print("Number of outliers:", num_outliers)

# Identify the most extreme outlier
if num_outliers > 0:
    most_extreme_outlier = outliers.abs().max().max()
    print("Most extreme outlier:", most_extreme_outlier)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Now you can use X_train and y_train for training your model, and X_test and y_test for testing it.
