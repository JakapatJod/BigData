import os
os.environ["JAVA_HOME"] = r"C:\Program Files\Java\jdk-22"
os.environ["SPARK_HOME"] = r"C:\spark\spark-3.5.1-bin-hadoop3"

from pyspark.sql import SparkSession
from graphframes import GraphFrame
from pyspark.sql.functions import desc

# สร้าง SparkSession สำหรับใช้งาน PySpark
spark = SparkSession.builder \
    .appName("Graph Analytics") \
    .config("spark.jars.packages", "graphframes:graphframes:0.8.2-spark3.0-s_2.12") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "4g") \
    .config("spark.pyspark.python", "python") \
    .config("spark.pyspark.driver.python", "python") \
    .getOrCreate()  # สร้างหรือดึง SparkSession

# ข้อมูลเวอร์เท็กซ์ (จุดยอด) ที่มีชื่อและอายุ
vertices_data = [
    ("Alice", 45),
    ("Jacob", 43),
    ("Roy", 21),
    ("Ryan", 49),
    ("Emily", 24),
    ("Sheldon", 52)
]
# ข้อมูลเอดจ์ (เชื่อมโยง) ที่มีความสัมพันธ์ระหว่างเวอร์เท็กซ์
edges_data = [
    ("Sheldon", "Alice", "Sister"),
    ("Alice", "Jacob", "Husband"),
    ("Emily", "Jacob", "Father"),
    ("Ryan", "Alice", "Friend"),
    ("Alice", "Emily", "Daughter"),
    ("Alice", "Roy", "Son"),
    ("Jacob", "Roy", "Son")
]

# สร้าง DataFrame สำหรับเวอร์เท็กซ์
vertices_df = spark.createDataFrame(vertices_data, ["id", "age"])
# สร้าง DataFrame สำหรับเอดจ์
edges_df = spark.createDataFrame(edges_data, ["src", "dst", "relationship"])

# สร้างกราฟจาก DataFrame ของเวอร์เท็กซ์และเอดจ์
graph = GraphFrame(vertices_df, edges_df)

# แสดงเอดจ์ที่จัดกลุ่มและเรียงตามจำนวน
print("Grouped and Ordered Edges:")
graph.edges.groupBy("src", "dst").count().orderBy(desc("count")).show(truncate=False)

# แสดงเอดจ์ที่เกี่ยวข้องกับ 'Alice'
print("Filtered Edges where src or dst is 'Alice':")
graph.edges.where("src = 'Alice' OR dst = 'Alice'").groupBy("src", "dst").count().orderBy(desc("count")).show(truncate=False)

# สร้างซับกราฟที่มี 'Alice' เป็นส่วนหนึ่ง
print("Subgraph where 'Alice' is involved:")
subgraph_edges = graph.edges.where("src = 'Alice' OR dst = 'Alice'")
subgraph = GraphFrame(graph.vertices, subgraph_edges)
subgraph.edges.show(truncate=False)

# แสดง motifs ในกราฟที่เกี่ยวข้องกับ 'Alice'
print("Motifs in the Graph (connections involving Alice):")
motifs = graph.find("(a)-[ab]->(b)")
motifs_filtered = motifs.filter("ab.relationship = 'Friend' OR ab.relationship = 'Daughter'")
motifs_filtered.show(truncate=False)

# คำนวณ PageRank
print("PageRank Results:")
page_rank = graph.pageRank(resetProbability=0.15, maxIter=5)
page_rank.vertices.orderBy(desc("pagerank")).show(truncate=False)

# แสดง In-Degree ของแต่ละเวอร์เท็กซ์
print("In-Degree of Each Vertex:")
in_degree = graph.inDegrees
in_degree.orderBy(desc("inDegree")).show(truncate=False)

# ตั้งค่าพ้อยเช็คพ้อยสำหรับการประมวลผล
spark.sparkContext.setCheckpointDir("/tmp/checkpoints")
# แสดง Connected Components
print("Connected Components:")
connected_components = graph.connectedComponents()
connected_components.show(truncate=False)

# แสดง Strongly Connected Components
print("Strongly Connected Components:")
strongly_connected_components = graph.stronglyConnectedComponents(maxIter=5)
strongly_connected_components.show(truncate=False)

# ค้นหาเส้นทางจาก 'Alice' ไป 'Roy' โดยใช้ BFS
print("Breadth-First Search (BFS):")
bfs_results = graph.bfs(fromExpr="id = 'Alice'", toExpr="id = 'Roy'", maxPathLength=2)
bfs_results.show(truncate=False)

# สรุปว่าโค้ดนี้ทำการวิเคราะห์กราฟด้วย GraphFrames โดยเริ่มจากการสร้างกราฟจากข้อมูลเวอร์เท็กซ์และเอดจ์ แล้วดำเนินการวิเคราะห์ต่างๆ เช่น การค้นหาเอดจ์ที่เกี่ยวข้องกับ 'Alice', คำนวณ PageRank, และหาส่วนประกอบที่เชื่อมโยงกันในกราฟ
print('END')

# ปิด Spark session
spark.stop()
