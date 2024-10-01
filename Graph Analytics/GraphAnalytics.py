# Import Libraries
from pyspark.sql import SparkSession
from graphframes import GraphFrame
from pyspark.sql.functions import desc

# Initialize Spark Session with GraphFrames package
spark = SparkSession.builder \
    .appName("Graph Analytics") \
    .config("spark.jars.packages", "graphframes:graphframes:0.8.2-spark3.0-s_2.12") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()


# Create vertices and edges DataFrames
vertices_data = [
    ("Alice", 45),
    ("Jacob", 43),
    ("Roy", 21),
    ("Ryan", 49),
    ("Emily", 24),
    ("Sheldon", 52)
]
edges_data = [
    ("Sheldon", "Alice", "Sister"),
    ("Alice", "Jacob", "Husband"),
    ("Emily", "Jacob", "Father"),
    ("Ryan", "Alice", "Friend"),
    ("Alice", "Emily", "Daughter"),
    ("Alice", "Roy", "Son"),
    ("Jacob", "Roy", "Son")
]

# Create DataFrames
vertices_df = spark.createDataFrame(vertices_data, ["id", "age"])
edges_df = spark.createDataFrame(edges_data, ["src", "dst", "relationship"])

# Create GraphFrame
graph = GraphFrame(vertices_df, edges_df)

# 1. Grouped and Ordered Edges
print("Grouped and Ordered Edges:")
graph.edges.groupBy("src", "dst").count().orderBy(desc("count")).show(truncate=False)

# 2. Filtered Edges where src or dst is 'Alice'
print("Filtered Edges where src or dst is 'Alice':")
graph.edges.where("src = 'Alice' OR dst = 'Alice'").groupBy("src", "dst").count().orderBy(desc("count")).show(truncate=False)

# 3. Subgraph where 'Alice' is involved
print("Subgraph where 'Alice' is involved:")
subgraph_edges = graph.edges.where("src = 'Alice' OR dst = 'Alice'")
subgraph = GraphFrame(graph.vertices, subgraph_edges)
subgraph.edges.show(truncate=False)

# 4. Motifs in the Graph (connections involving Alice)
print("Motifs in the Graph (connections involving Alice):")
motifs = graph.find("(a)-[ab]->(b)")
motifs_filtered = motifs.filter("ab.relationship = 'Friend' OR ab.relationship = 'Daughter'")
motifs_filtered.show(truncate=False)

# 5. PageRank Results
print("PageRank Results:")
page_rank = graph.pageRank(resetProbability=0.15, maxIter=5)
page_rank.vertices.orderBy(desc("pagerank")).show(truncate=False)

# 6. In-Degree of Each Vertex
print("In-Degree of Each Vertex:")
in_degree = graph.inDegrees
in_degree.orderBy(desc("inDegree")).show(truncate=False)

# 7. Connected Components
spark.sparkContext.setCheckpointDir("/tmp/checkpoints")  # Set checkpoint directory
print("Connected Components:")
connected_components = graph.connectedComponents()
connected_components.show(truncate=False)

# 8. Strongly Connected Components
print("Strongly Connected Components:")
strongly_connected_components = graph.stronglyConnectedComponents(maxIter=5)
strongly_connected_components.show(truncate=False)

# 9. Breadth-First Search (BFS)
print("Breadth-First Search (BFS):")
bfs_results = graph.bfs(fromExpr="id = 'Alice'", toExpr="id = 'Roy'", maxPathLength=2)
bfs_results.show(truncate=False)

print('END')

# Stop the Spark session
spark.stop()
