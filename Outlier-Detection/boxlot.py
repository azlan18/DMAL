import pandas as pd
import matplotlib.pyplot as plt

# Load the data into a DataFrame
df = pd.read_csv('exp5_outlier/data.csv')

# Calculate quartiles and IQR
Q1 = df['area'].quantile(0.25)
Q2 = df['area'].median()  # Median (Q2)
Q3 = df['area'].quantile(0.75)
IQR = Q3 - Q1

# Determine outlier thresholds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Find outliers
outliers = df[(df['area'] < lower_bound) | (df['area'] > upper_bound)]

# Print statistics
print("Statistics for 'area' field:")
print("Q1 (25th percentile):", Q1)
print("Q2 (Median):", Q2)
print("Q3 (75th percentile):", Q3)
print("Min:", df['area'].min())
print("Max:", df['area'].max())
print("IQR:", IQR)
print("Lower Outlier Bound:", lower_bound)
print("Upper Outlier Bound:", upper_bound)

# Print the listing of outliers
print("\nListing of Outliers:")
print(outliers)

# Print the 10 largest values for the field
top_5_largest = df.nlargest(5, 'area')
print("\nTop 5 Largest Values for 'area':")
print(top_5_largest)

# Create a boxplot for the 'area' field
plt.figure(figsize=(8, 6))
plt.boxplot(df['area'])
plt.title('Boxplot for area')
plt.ylabel('Area')
plt.show()
