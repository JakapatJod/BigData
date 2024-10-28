from pyspark.sql import SparkSession
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml import Pipeline
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
import matplotlib.pyplot as plt

# Start Spark session
spark = SparkSession.builder.appName("testClustering").getOrCreate()

# Load and preprocess data
df = spark.read.csv("fb_live_thailand.csv", header=True, inferSchema=True)
df = df.select(df.num_sads.cast(DoubleType()), df.num_reactions.cast(DoubleType()))

# Set up feature vector
vec_assembler = VectorAssembler(inputCols=["num_sads", "num_reactions"], outputCol="features")
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=False)

# Find the optimal k using silhouette score
k_values = []
for i in range(2, 6):
    kmeans = KMeans(featuresCol="scaledFeatures", predictionCol="prediction", k=i)
    pipeline = Pipeline(stages=[vec_assembler, scaler, kmeans])
    model = pipeline.fit(df)
    output = model.transform(df)
    evaluator = ClusteringEvaluator(predictionCol="prediction", featuresCol="scaledFeatures",
                                    metricName="silhouette", distanceMeasure="squaredEuclidean")

    score = evaluator.evaluate(output)
    k_values.append(score)
    print('Silhouette Score for k =', i, ':', score)

# Adjust `best_k` to actual k value
best_k = k_values.index(max(k_values)) + 2
print("The best k is:", best_k, "with a Silhouette Score of:", max(k_values))

# Initialize KMeans with the best k
kmeans = KMeans(featuresCol="scaledFeatures", predictionCol="prediction", k=best_k)
pipeline = Pipeline(stages=[vec_assembler, scaler, kmeans])

# Fit model
model = pipeline.fit(df)

# Predictions
predictions = model.transform(df)

# Convert predictions to pandas for plotting
clustered_data_pd = predictions.toPandas()

# Plot each cluster
plt.figure(figsize=(10, 6))
for cluster in clustered_data_pd["prediction"].unique():
    clustered_subset = clustered_data_pd[clustered_data_pd["prediction"] == cluster]
    plt.scatter(clustered_subset["num_reactions"], 
                clustered_subset["num_sads"], label=f"Cluster {cluster}")

plt.xlabel("num_reactions")
plt.ylabel("num_sads")
plt.title("K-Means Clustering")
plt.legend(title="Clusters")
plt.show()

# Count the number of points in each cluster
cluster_counts = clustered_data_pd["prediction"].value_counts().sort_index()

# Plotting the bar chart for cluster sizes
plt.figure(figsize=(8, 6))
bars = plt.bar(cluster_counts.index, cluster_counts.values, color='skyblue')
plt.xlabel("Cluster")
plt.ylabel("Count")
plt.title("Number of Points in Each Cluster")
plt.xticks(cluster_counts.index)

# Add count labels on top of each bar
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval, int(yval), ha='center', va='bottom')

plt.show()
