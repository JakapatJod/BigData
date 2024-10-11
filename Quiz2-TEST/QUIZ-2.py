from pyspark.sql import SparkSession
from graphframes import GraphFrame
from pyspark.sql.functions import desc, col, lit

# Create SparkSession
spark = SparkSession.builder \
    .appName("CyclingRoutesGraph") \
    .config("spark.jars.packages", "graphframes:graphframes:0.8.2-spark3.0-s_2.12") \
    .getOrCreate()

# Read the data from cycling.csv
cycling_routes_df = spark.read.csv("cycling.csv", header=True, inferSchema=True)

# Create the vertices DataFrame
vertices = cycling_routes_df.select("FromStationName").withColumnRenamed("FromStationName", "id").distinct()

# Create the edges DataFrame
edges = cycling_routes_df.select("FromStationName", "ToStationName") \
    .withColumnRenamed("FromStationName", "src") \
    .withColumnRenamed("ToStationName", "dst")

# Create the GraphFrame
graph = GraphFrame(vertices, edges)

# Group the edges by source and destination, filter by count >= 3
grouped_edges = graph.edges.groupBy("src", "dst").count() \
    .filter(col("count") >= 3) \
    .withColumn("source_color", lit("#FF3F33")) \
    .withColumn("destination_color", lit("#3358FF"))  

# Write the result to CSV without header and including only the necessary columns
grouped_edges.select("src", "dst", "source_color", "destination_color") \
    .write.csv("grouped_cycling_routes_no_header.csv", mode="overwrite", header=False)

grouped_edges.show()

print("Processing and writing to file completed.")
