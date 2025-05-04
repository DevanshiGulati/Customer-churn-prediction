import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("DATA/raw_customer_data.csv")

# Quick overview
print("🔹 Dataset shape:", df.shape)
print("\n🔹 First 5 rows:\n", df.head())

# Check for null values
print("\n🔹 Null values:\n", df.isnull().sum())
00
# Churn value distribution
print("\n🔹 Churn distribution:\n", df['Churn'].value_counts(normalize=True))

# Visualize churn distribution
sns.countplot(x="Churn", data=df)
plt.title("Churn Distribution")
plt.savefig("visuals/churn_distribution1.png")
plt.close()

# Correlation matrix
corr = df.drop("CustomerID", axis=1).select_dtypes(include='number').corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.savefig("visuals/correlation_heatmap1.png")
plt.close()

print("✅ Preprocessing and EDA completed.")