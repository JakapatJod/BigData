from pyspark.sql import SparkSession
from graphframes import GraphFrame
from pyspark.sql.functions import desc, col, lit

# สร้าง SparkSession
spark = SparkSession.builder \
    .appName("CyclingRoutesGraph") \
    .config("spark.jars.packages", "graphframes:graphframes:0.8.2-spark3.0-s_2.12") \
    .getOrCreate()

# อ่านข้อมูลจากไฟล์ cycling.csv
cycling_routes_df = spark.read.csv("cycling.csv", header=True, inferSchema=True)

# แสดง DataFrame
cycling_routes_df.show()

# สร้าง DataFrame สำหรับ vertices โดยใช้ withColumnRenamed() และ FromStationName เป็น id
vertices = cycling_routes_df.select("FromStationName").withColumnRenamed("FromStationName", "id").distinct()

# สร้าง DataFrame สำหรับ edges โดยใช้ withColumnRenamed() โดยให้ FromStationName เป็น src และ ToStationName เป็น dst
edges = cycling_routes_df.select("FromStationName", "ToStationName") \
    .withColumnRenamed("FromStationName", "src") \
    .withColumnRenamed("ToStationName", "dst")

# แสดง DataFrame สำหรับ vertices
vertices.show()

# แสดง DataFrame สำหรับ edges
edges.show()

# สร้าง GraphFrame โดยใช้ vertices และ edges ที่สร้างขึ้น
graph = GraphFrame(vertices, edges)

# แสดงจำนวน vertices
print("Number of vertices:", graph.vertices.count())

# แสดงจำนวน edges
print("Number of edges:", graph.edges.count())

# กลุ่ม edges โดยใช้ src และ dst, กรองตาม count > 5, เพิ่มคอลัมน์ source_color และ destination_color
grouped_edges = graph.edges.groupBy("src", "dst").count() \
    .filter(col("count") > 5) \
    .orderBy(desc("count")) \
    .withColumn("source_color", lit("#3358FF")) \
    .withColumn("destination_color", lit("#FF3F33"))

# คำสั่ง groupBy("src", "dst").count() ใช้ในการนับจำนวนการเชื่อมต่อระหว่างสนามบินต้นทางและปลายทาง
# มีการใช้คำสั่ง .filter(col("count") > 5) เพื่อกรองเส้นทางที่มีการบินซ้ำมากกว่า 5 ครั้ง
# คำสั่ง .orderBy(desc("count")) ทำการเรียงลำดับจากมากไปน้อย
# ส่วนสี (source_color, destination_color) ถูกเพิ่มเข้าไปใน DataFrame เพื่อใช้ในงานแสดงผล เช่น การวิเคราะห์กราฟขั้นสูง หรือการ visualizing กราฟ.

# ASC เรียงจากน้อยไปหามาก
# DESC เรียงจากมากไปหาน้อย
# orderBY เรียง

# .filter(col("distance") > 10)  # ตัวอย่างกรองระยะทางมากกว่า 10 กิโลเมตร
# .filter(col("trip_date").between('2023-01-01', '2023-12-31'))  # ตัวอย่างกรองการเดินทางในปี 2023
# .filter(col("src") != col("dst"))  # ตัวอย่างกรองเส้นทางที่มีจุดเริ่มต้นและปลายทางไม่ซ้ำกัน
# hub_stations = graph.edges.groupBy("src").count().orderBy(desc("count"))  # ตัวอย่างหาสถานีที่มีการออกเดินทางมากที่สุด
# .filter(col("trip_type") == "leisure")  # ตัวอย่างกรองเฉพาะการเดินทางเพื่อพักผ่อน

# grouped_edges = graph.edges.groupBy("src", "dst").count() \
#    .filter(col("count") > 5) \
#    .filter(col("src") != col("dst"))  # กรองเฉพาะเส้นทางที่ไม่มีการย้อนกลับ \
#    .filter(col("distance") > 10)  # กรองเฉพาะเส้นทางที่มีระยะทางมากกว่า 10 กิโลเมตร \
#    .orderBy(desc("count")) \
#    .withColumn("source_color", lit("#3358FF")) \
#    .withColumn("destination_color", lit("#FF3F33"))


# หาสถานีที่มีการเดินทางออกจากมากที่สุด
# popular_start_stations = graph.edges.groupBy("src").count().orderBy(desc("count"))

# หาสถานีที่มีการเดินทางเข้ามามากที่สุด
# popular_end_stations = graph.edges.groupBy("dst").count().orderBy(desc("count"))

# popular_start_stations.show()
# popular_end_stations.show()

# popular_routes = graph.edges.groupBy("src", "dst").count().orderBy(desc("count"))

# หาสถานีที่มีเส้นทางเชื่อมโยงกับหลายสถานีที่สุด
# station_connectivity = graph.degrees.orderBy(desc("degree"))

# แสดงข้อมูลที่ถูกจัดกลุ่ม
grouped_edges.show()

# เขียนข้อมูลที่ถูกจัดกลุ่มลงในไฟล์ CSV โดยใช้โหมด overwrite และตั้ง header เป็น True
grouped_edges.write.csv("grouped_cycling_routes.csv", mode="overwrite", header=True)

print('='*80)
print('')
