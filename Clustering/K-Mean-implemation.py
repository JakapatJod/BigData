from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml import Pipeline
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

import matplotlib.pyplot as plt
import pandas as pd

# Creating the SparkSession
spark = SparkSession.builder.appName("KMeans").getOrCreate()

# Loading and casting data
df = spark.read.format("csv").option("header",True).load("fb_live_thailand.csv")
df = df.select(df.num_sads.cast(DoubleType()), df.num_reactions.cast(DoubleType()))

# Concatenate input columns to the output "features"
vec_assembler = VectorAssembler(inputCols=["num_sads", "num_reactions"], outputCol="features")

# Scaling for making columns comparable
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=False)

# Initialise k_values list
k_values = []
print('='*50)

# Loop for finding the optimal k in range 2 to 5
for i in range(2, 5):
    kmeans = KMeans(featuresCol="scaledFeatures", predictionCol="prediction", k=i)
    pipeline = Pipeline(stages=[vec_assembler, scaler, kmeans])
    model = pipeline.fit(df)
    output = model.transform(df)

    # Correct column name and evaluator setup
    evaluator = ClusteringEvaluator(predictionCol="prediction", featuresCol="scaledFeatures",
                                     metricName="silhouette", distanceMeasure="squaredEuclidean")

    # Evaluate silhouette score
    score = evaluator.evaluate(output)
    k_values.append(score)
    print(f'Silhouette Score for k={i}:', score)

# Get the best K
best_k = k_values.index(max(k_values)) + 2
print("The best k is:", best_k, max(k_values))

# Initialise KMeans with the best k
kmeans = KMeans(featuresCol="scaledFeatures", predictionCol="prediction", k=best_k)

# Create pipeline
pipeline = Pipeline(stages=[vec_assembler, scaler, kmeans])

# Fit model
model = pipeline.fit(df)

# Predictions
predictions = model.transform(df)

# Evaluate
evaluator = ClusteringEvaluator(predictionCol="prediction", featuresCol="scaledFeatures", 
                                metricName="silhouette", distanceMeasure="squaredEuclidean")
silhouette = evaluator.evaluate(predictions)
print(f"Silhouette with best k={best_k}: {silhouette}")
print('='*50)

# Visualizing the results
clustered_data_pd = predictions.select("num_reactions", "num_sads", "prediction").toPandas()

plt.scatter(clustered_data_pd["num_reactions"], clustered_data_pd["num_sads"],
             c=clustered_data_pd["prediction"])
plt.xlabel("num_reactions")
plt.ylabel("num_sads")
plt.title("K-Means Clustering")
plt.colorbar().set_label("Cluster")
plt.show()
