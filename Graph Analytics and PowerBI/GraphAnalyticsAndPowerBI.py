from pyspark.sql import SparkSession
from graphframes import GraphFrame
from pyspark.sql.functions import desc, col, lit

spark = SparkSession.builder \
    .appName("AirlineRoutesGraph") \
    .getOrCreate()

# Read data from airline_routes.csv
airline_routes_df = spark.read.csv("airline_routes.csv", header=True, inferSchema=True)

# Show the DataFrame
airline_routes_df.show()

# Create the vertice DataFrame using withColumnRenamed() and source_airport as id
vertices = airline_routes_df.select("source_airport").withColumnRenamed("source_airport", "id").distinct()

# Create the edge DataFrame using withColumnRenamed() with source_airport as src and destination_airport as dst
edges = airline_routes_df.select("source_airport", "destination_airport") \
    .withColumnRenamed("source_airport", "src") \
    .withColumnRenamed("destination_airport", "dst")

# Show vertice DataFrame
vertices.show()

# Show edge DataFrame
edges.show()

# Create GraphFrame using the created vertice and edge DataFrames
graph = GraphFrame(vertices, edges)

# Show the number of vertices
print("Number of vertices:", graph.vertices.count())

# Show the number of edges
print("Number of edges:", graph.edges.count())

# Group the edges based on src and dst, filter by count > 5, add source_color and destination_color columns
grouped_edges = graph.edges.groupBy("src", "dst").count() \
    .filter(col("count") > 5) \
    .orderBy(desc("count")) \
    .withColumn("source_color", lit("#3358FF")) \
    .withColumn("destination_color", lit("#FF3F33"))

# Show the grouped data
grouped_edges.show()

# Write the grouped data into a CSV file with overwrite mode and header set to True
grouped_edges.write.csv("grouped_airline_routes.csv", mode="overwrite", header=True)
