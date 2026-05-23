import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("titanic.csv")

# Display first rows
print("FIRST 5 ROWS")
print(df.head())

# Dataset information
print("\nDATASET INFO")
print(df.info())

# Missing values
print("\nMISSING VALUES")
print(df.isnull().sum())

# Statistical summary
print("\nSTATISTICAL SUMMARY")
print(df.describe())

# -----------------------------
# GRAPH 1 : Survival Count
# -----------------------------
plt.figure(figsize=(6,4))
sns.countplot(x='Survived', data=df)
plt.title("Survival Count")
plt.xlabel("Survival Status")
plt.ylabel("Count")
plt.show()

# -----------------------------
# GRAPH 2 : Gender Distribution
# -----------------------------
plt.figure(figsize=(6,4))
sns.countplot(x='Sex', data=df)
plt.title("Gender Distribution")
plt.show()

# -----------------------------
# GRAPH 3 : Passenger Class
# -----------------------------
plt.figure(figsize=(6,4))
sns.countplot(x='Pclass', data=df)
plt.title("Passenger Class Distribution")
plt.show()

# -----------------------------
# GRAPH 4 : Age Distribution
# -----------------------------
plt.figure(figsize=(8,5))
sns.histplot(df['Age'].dropna(), bins=20)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.show()

# -----------------------------
# GRAPH 5 : Fare Distribution
# -----------------------------
plt.figure(figsize=(8,5))
sns.histplot(df['Fare'], bins=20)
plt.title("Fare Distribution")
plt.xlabel("Fare")
plt.show()

# -----------------------------
# GRAPH 6 : Heatmap
# -----------------------------
plt.figure(figsize=(10,8))

numeric_df = df.select_dtypes(include=['number'])

sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')

plt.title("Correlation Heatmap")
plt.show()

print("\nEDA PROJECT COMPLETED SUCCESSFULLY")