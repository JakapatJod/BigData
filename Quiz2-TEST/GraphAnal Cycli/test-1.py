from pyspark.sql import SparkSession
from graphframes import GraphFrame
from pyspark.sql.functions import desc, col, lit

# สร้าง SparkSession
spark = SparkSession.builder \
    .appName("CyclingRoutesGraph") \
    .config("spark.jars.packages", "graphframes:graphframes:0.8.2-spark3.0-s_2.12") \
    .getOrCreate()

# อ่านข้อมูลจาก cycling dataset (ปรับตามพาธจริงของไฟล์)
cycling_df = spark.read.csv("cycling.csv", header=True, inferSchema=True)

# แสดงข้อมูล
cycling_df.show()

# สร้างโหนด (Vertices) จากสถานี
# แต่ละสถานีเป็นโหนด จะมีทั้งสถานีต้นทางและปลายทาง
vertices = cycling_df.select(col("FromStationName").alias("id")).distinct() \
    .union(cycling_df.select(col("ToStationName").alias("id")).distinct()) \
    .distinct()

# แสดงข้อมูลโหนด
vertices.show()

# สร้างขอบ (Edges) จากเส้นทางการเดินทางระหว่างสถานี
edges = cycling_df.select(
    col("FromStationName").alias("src"),
    col("ToStationName").alias("dst")
)

# แสดงข้อมูลขอบ
edges.show()

# สร้างกราฟจากโหนดและขอบ
cycling_graph = GraphFrame(vertices, edges)

# แสดงข้อมูลกราฟ
cycling_graph.vertices.show()
cycling_graph.edges.show()

# การวิเคราะห์เส้นทาง
# ค้นหาสถานีที่มีการเชื่อมต่อมากที่สุด (degree centrality)
station_degrees = cycling_graph.degrees.sort(desc("degree"))

# แสดงสถานีที่มีการเชื่อมต่อมากที่สุด
station_degrees.show()

# หยุดการทำงานของ Spark
spark.stop()
