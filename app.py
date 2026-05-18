import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load dataset
data = pd.read_csv("dataset/Mall_Customers.csv")

# Select features
x = data[["Annual Income (k$)", "Spending Score (1-100)"]]

# Create KMeans model
kmeans = KMeans(n_clusters=5, random_state=42)

# Train model
kmeans.fit(x)

# Predict clusters
data["Cluster"] = kmeans.labels_

# Print first rows
print(data.head())

# Visualize clusters
plt.scatter(
    data["Annual Income (k$)"],
    data["Spending Score (1-100)"],
    c=data["Cluster"]
)

plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("Customer Segmentation")

plt.show()