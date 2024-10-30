from pyspark.sql import SparkSession
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import ClusteringEvaluator
import matplotlib.pyplot as plt
import numpy as np

# Initialize Spark session
spark = SparkSession.builder \
    .appName("KMeans Clustering Analysis") \
    .getOrCreate()

# Load the dataset
file_path = "fb_live_thailand.csv"  # Update the path accordingly
df = spark.read.csv(file_path, header=True, inferSchema=True)

# Select relevant columns and cast them to DoubleType
df = df.select(df.num_sads.cast(DoubleType()), df.num_reactions.cast(DoubleType()))

# Feature engineering and scaling
vec_assembler = VectorAssembler(inputCols=["num_sads", "num_reactions"], outputCol="features")
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=False)

# Finding the optimal number of clusters (k) using silhouette score
k_values = []
for i in range(2, 5):
    kmeans = KMeans(featuresCol="scaledFeatures", predictionCol="prediction_col", k=i)
    pipeline = Pipeline(stages=[vec_assembler, scaler, kmeans])
    model = pipeline.fit(df)
    output = model.transform(df)
    evaluator = ClusteringEvaluator(predictionCol="prediction_col", featuresCol="scaledFeatures", metricName="silhouette")
    score = evaluator.evaluate(output)
    k_values.append(score)
    print(f"Silhouette Score for k = {i}: {score}")

# Select the best k
best_k = k_values.index(max(k_values)) + 2
print(f"The best k: {best_k} with Silhouette Score: {max(k_values)}")

# Run KMeans with the best k
kmeans = KMeans(featuresCol="scaledFeatures", predictionCol="prediction_col", k=best_k)
pipeline = Pipeline(stages=[vec_assembler, scaler, kmeans])
model = pipeline.fit(df)
predictions = model.transform(df)

# Convert to Pandas DataFrame for plotting
clustered_data_pd = predictions.select("num_reactions", "num_sads", "prediction_col").toPandas()

# Plotting
plt.figure(figsize=(10, 5))
plt.scatter(clustered_data_pd["num_reactions"], clustered_data_pd["num_sads"], c=clustered_data_pd["prediction_col"])
plt.xlabel("num_reactions")
plt.ylabel("num_sads")
plt.title("K-means Clustering")
plt.colorbar().set_label("Cluster")
plt.show()

# Bar chart of cluster counts
unique, counts = np.unique(clustered_data_pd["prediction_col"], return_counts=True)
plt.bar(unique, counts, color='skyblue')
plt.xlabel("Cluster ID")
plt.ylabel("Number of Data Points")
plt.title("Number of Data Points per Cluster")

# Display counts above bars
for i, count in zip(unique, counts):
    plt.text(i, count + 0.5, str(count), ha='center')

plt.show()


# Plot Bubble Plot
plt.figure(figsize=(10, 6))
plt.scatter(df["num_reactions"], df["num_sads"], s=df["num_reactions"], alpha=0.5)  # Using 'num_reactions' for bubble size
plt.xlabel("Number of Reactions")
plt.ylabel("Number of Sads")
plt.title("Bubble Plot of Reactions vs. Sads with Reactions as Bubble Size")
plt.colorbar(label="Number of Reactions")
plt.show()


# Plot Line Chart
plt.figure(figsize=(10, 6))
plt.plot(df.index, df["num_reactions"], label="Reactions", marker="o")
plt.plot(df.index, df["num_sads"], label="Sads", marker="x")
plt.xlabel("Index (or Time)")
plt.ylabel("Count")
plt.title("Line Chart of Reactions and Sads")
plt.legend()
plt.show()


# Stop Spark session
spark.stop()
