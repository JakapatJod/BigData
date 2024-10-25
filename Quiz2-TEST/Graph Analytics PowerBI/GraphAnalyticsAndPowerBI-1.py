from pyspark.sql import SparkSession
from graphframes import GraphFrame
from pyspark.sql.functions import desc, col, lit

# สร้าง SparkSession พร้อมกับเพิ่ม GraphFrames package
spark = SparkSession.builder \
    .appName("AirlineRoutesGraph") \
    .config("spark.jars.packages", "graphframes:graphframes:0.8.2-spark3.0-s_2.12") \
    .getOrCreate()

# อ่านข้อมูลจากไฟล์ airline_routes.csv
airline_routes_df = spark.read.csv("airline_routes.csv", header=True, inferSchema=True)

# แสดง DataFrame
airline_routes_df.show()

# สร้าง DataFrame สำหรับ vertices โดยเลือก source_airport และ destination_airport เป็น id ที่ไม่ซ้ำกัน
vertices = airline_routes_df.select("source_airport").union(airline_routes_df.select("destination_airport")) \
    .distinct().withColumnRenamed("source_airport", "id")

# สร้าง DataFrame สำหรับ edges โดยให้ source_airport เป็น src และ destination_airport เป็น dst
edges = airline_routes_df.select("source_airport", "destination_airport") \
    .withColumnRenamed("source_airport", "src") \
    .withColumnRenamed("destination_airport", "dst")

# สร้าง GraphFrame โดยใช้ vertices และ edges ที่สร้างขึ้น
graph = GraphFrame(vertices, edges)

# แสดงจำนวน vertices และ edges
print("Number of vertices:", graph.vertices.count())
print("Number of edges:", graph.edges.count())

# การคำนวณ Degree Centrality (inDegree, outDegree, degree)
in_degree = graph.inDegrees
out_degree = graph.outDegrees
degree = in_degree.join(out_degree, "id", "outer").na.fill(0)

# แสดง Top 10 สนามบินที่มี Degree สูงสุด
degree.withColumn("degree", col("inDegree") + col("outDegree")) \
    .orderBy(desc("degree")).show(10)

# การคำนวณ PageRank
pagerank = graph.pageRank(resetProbability=0.15, maxIter=10)
pagerank.vertices.select("id", "pagerank").orderBy(desc("pagerank")).show(10)

# การตรวจหากลุ่มสนามบิน (Community Detection) โดยใช้ Label Propagation Algorithm (LPA)
communities = graph.labelPropagation(maxIter=5)
communities.show()

# ส่งออกข้อมูล Degree Centrality ไปยัง CSV สำหรับการนำเข้าใน Power BI
degree.write.csv("degree_centrality.csv", mode="overwrite", header=True)

# ส่งออกข้อมูล PageRank ไปยัง CSV สำหรับการนำเข้าใน Power BI
pagerank.vertices.write.csv("pagerank.csv", mode="overwrite", header=True)

# ส่งออกข้อมูล Communities ไปยัง CSV สำหรับการนำเข้าใน Power BI
communities.write.csv("airport_communities.csv", mode="overwrite", header=True)
