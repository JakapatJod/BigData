from pyspark.sql import SparkSession
from graphframes import GraphFrame

spark = SparkSession.builder \
    .appName("Graph Analytics") \
    .config("spark.jars.packages", "graphframes:graphframes:0.8.2-spark3.0-s_2.12") \
    .getOrCreate()

vertices_data = [("Alice", 45), ("Bob", 30)]
edges_data = [("Alice", "Bob", "Friend")]

vertices_df = spark.createDataFrame(vertices_data, ["id", "age"])
edges_df = spark.createDataFrame(edges_data, ["src", "dst", "relationship"])

graph = GraphFrame(vertices_df, edges_df)

print("Vertices:")
graph.vertices.show()
print("Edges:")
graph.edges.show()

spark.stop()
