from pyspark.sql import SparkSession
from graphframes import GraphFrame
from pyspark.sql.functions import col

# สร้าง SparkSession
spark = SparkSession.builder \
    .appName("CyclingRoutesGraph") \
    .config("spark.jars.packages", "graphframes:graphframes:0.8.2-spark3.0-s_2.12") \
    .getOrCreate()

# อ่านข้อมูลจาก cycling dataset (ปรับตามพาธจริงของไฟล์)
cycling_df = spark.read.csv("cycling.csv", header=True, inferSchema=True)

# แสดงข้อมูลเพื่อเช็ค
cycling_df.show()

# สร้างโหนด (Vertices) จากสถานี (ต้นทางและปลายทาง)
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

# นับจำนวนการใช้งานแต่ละสถานี (ทั้งสถานีเริ่มต้นและสถานีปลายทาง)
station_usage = cycling_graph.edges.groupBy("src").count().alias("start_count") \
    .union(cycling_graph.edges.groupBy("dst").count().alias("end_count")) \
    .groupBy("src").sum("count").alias("usage_count")

# กรองเฉพาะสถานีที่มีการใช้งานมากกว่า 100 ครั้ง
high_usage_stations = station_usage.filter(col("sum(count)") > 100)

# แสดงสถานีที่มีการใช้งานสูง
high_usage_stations.show()

# กรองเส้นทางตามสถานีที่มีการใช้งานสูง
# Alias both DataFrames to avoid ambiguous column errors
edges_alias = cycling_graph.edges.alias("edges")
stations_alias = high_usage_stations.alias("stations")

# Use the aliases and join on specific columns
filtered_edges = edges_alias.join(
    stations_alias, 
    (edges_alias.src == stations_alias.src) | (edges_alias.dst == stations_alias.src)
)

# แสดงผลลัพธ์ของเส้นทางที่มีการใช้งานสูง
filtered_edges.show()

# เขียนผลลัพธ์ออกเป็นไฟล์ CSV
filtered_edges.write.csv("cycling_high_usage_routes.csv", mode="overwrite", header=True)

# ปิดการทำงานของ Spark
spark.stop()
