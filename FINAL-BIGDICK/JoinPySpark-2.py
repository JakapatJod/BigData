from pyspark.sql import SparkSession
from pyspark.sql.functions import trim, col  # ใช้ในการตัดช่องว่างและเลือกคอลัมน์
from pyspark.sql.types import IntegerType  # ใช้สำหรับแปลงข้อมูลให้เป็นชนิด Integer

# สร้าง SparkSession
spark = SparkSession.builder.appName("OuterJoinFillNA").getOrCreate()

# โหลดข้อมูลจากไฟล์ CSV
file_path_1 = "fb_live_thailand2.csv"
file_path_2 = "fb_live_thailand3.csv"

df1 = spark.read.csv(file_path_1, header=True, inferSchema=True)
df2 = spark.read.csv(file_path_2, header=True, inferSchema=True)

join_key = "status_id"  # แทนที่ด้วยชื่อคอลัมน์ที่ใช้เป็น key

data = df1.join(df2, on=join_key, how="outer") # inner , left , right , outer

print(data)

fill_data = data.fillna({"num_comments": 0},{"num_shares": 0}) 

# แสดงข้อมูลที่เติมค่า null เป็น 0
fill_data.show()

